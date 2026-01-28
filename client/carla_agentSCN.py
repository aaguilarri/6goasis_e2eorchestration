#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
from datetime import datetime
#import numpy as np
import json
import math
import os
import random
import sys
import threading
import time
import weakref
import traceback
import queue
import copy
import carla
from carla import ColorConverter as cc
import cv2
from kafka import KafkaProducer
from kafka.errors import KafkaError
#import paho.mqtt.client as mqtt
from PIL import Image
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import ultralytics
from ultralytics import YOLO
#import matplotlib.pyplot as plt


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass



# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================




try:
    import pygame
    #from pygame.locals import K_ESCAPE
    #from pygame.locals import K_SPACE
    #from pygame.locals import K_a
    #from pygame.locals import K_d
    #from pygame.locals import K_s
    #from pygame.locals import K_w
    
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH =  800
VIEW_HEIGHT = 600
VIEW_FOV = 90
LAT0 = 0.
LON0 = 0.
CARLAIP="127.0.0.1"
cam_x = 0. 
cam_y = 0.
cam_z = 0.
cam_r = 0.
cam_p = 0.
cam_w = 0.
MY_GPU="0"
MY_TYPE='car'
STATION_ID='0000'
mqtt_port = 0
THE_THICK = False
BB_COLOR = (248, 64, 24)
tracklets = queue.Queue()
cam_tracklets = queue.Queue()
is_kafka = True
kafka_file = 'kafka_in.json'
kafka_IP = '172.17.0.1'
kafka_port = '9092'
docker_path = '/home/jsons/client_files/'
is_synch = False
is_autopilot = False
is_yolo = True
producer = None
MY_FPS = 0 #0 = A toda pastilla!


def invert_transform(transform):
    """Compute inverse of a carla.Transform (rotation + translation)."""
    # Extract translation
    tx = transform.location.x
    ty = transform.location.y
    tz = transform.location.z

    # Extract rotation (in degrees) and convert to radians
    yaw = np.radians(-transform.rotation.yaw)
    pitch = np.radians(transform.rotation.pitch)
    roll = np.radians(transform.rotation.roll)

    # Compute rotation matrix R (world → sensor)
    cy = np.cos(yaw); sy = np.sin(yaw)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cr = np.cos(roll); sr = np.sin(roll)

    # build rotation matrices
    Rz = np.array([[ cy, -sy, 0],
                   [ sy,  cy, 0],
                   [  0,   0, 1]])
    Ry = np.array([[ cp,  0, sp],
                   [  0,  1,  0],
                   [-sp,  0, cp]])
    Rx = np.array([[1,   0,    0],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])

    # Full rotation: yaw then pitch then roll
    R = Rz @ Ry @ Rx

    # Inverse rotation is transpose
    R_inv = R.T

    # Inverse translation:       t_inv = - R_inv * t
    t = np.array([tx, ty, tz])
    t_inv = - R_inv @ t

    return R_inv, t_inv


def world_to_camera(R_inv, t_inv, pw):
    """Apply inverse transform to get camera‐frame point (x, y, z)."""
    pw = np.array(pw)
    pc = R_inv @ pw + t_inv
    return pc

##### IOU ####
def iou_2d(boxA, boxB):
    # Coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Compute areas of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

###### KAFKA
def create_producer(server,port):
    port = str(port)
    print(f"server:{server}")
    print(f"port:{port}")
    bootstrap_servers=server+':'+port
    return KafkaProducer(
        acks=1,
        batch_size=16384,
        linger_ms=5,
        bootstrap_servers=bootstrap_servers,
        compression_type='gzip',  # Reduce network bandwidth usage
        retries=3,  # Allow some retries if Kafka is temporarily unreachable
        max_in_flight_requests_per_connection=10,  # Prevent blocking when using acks=0
        value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v  # Safer serialization
        #compression_type='gzip',
        #value_serializer=lambda v: v.encode('utf-8')
    )

def produce_messages(producer, messages, topic_name):
    for message in messages:
        try:
            future = producer.send(topic_name, value=message)
        except KafkaError as e:
            print(f"Failed to send message '{message}': {e}")
        def on_success(metadata):
            print(f"Message sent to {metadata.topic} at partition {metadata.partition}")
        def on_error(ex):
            print(f"Message failed: {ex}")
        future.add_callback(on_success)
        future.add_errback(on_error)
        time.sleep(0.01)
        try:
            future.get(timeout=10)
            #print(f"Message '{message}' sent successfully.")
        except KafkaError as e:
            print(f"Failed to send message '{message}': {e}")
###### MQTT Client
kafka_messages = []
KAFKA_TOPIC = "tracklets"

###### Thread function
def save_to_file(data, filename):
    global docker_path
    #print(f"This is save to file: path is {docker_path + filename}")
    try:
        with open(docker_path + filename, 'a') as file:  # Open file in append mode
            file.write(data + '\n')  # Add a newline between each JSON entry
        #print(f"Data appended to {docker_path + filename} successfully.")
    except Exception as e:
        print(f"Failed to append data to {docker_path + filename}: {e}")
        #try:
        #    with open(docker_path + filename, 'w') as file:
        #        file.write(data)
        #    print(f"Data saved to {docker_path + filename} successfully.")
        #except Exception as e:
        #    print(f"Failed to save data to {docker_path + filename}: {e}")
        traceback.print_exc()




def carla26g(producer):
    tmax = 1
    filename = ''
    #print("Greetings from carla thread!")
    #print(f"producer is {producer.config}")
    try:
        future = producer.send('test-topic', b'hello world')
        record_metadata = future.get(timeout=10)
        #print("Message sent to topic:", record_metadata.topic)
    except KafkaError as e:
        print("Failed to send message:", e)
    yolo_dict = get_classes()
    global KAFKA_TOPIC
    global mqtt_port
    global is_kafka
    global kafka_IP
    global kafka_port
    global kafka_file

    #client = mqtt.Client()
    #client.on_connect = on_connect
    #client.on_message = mqtt_callback
    #client.on_subscribe = on_subscribe
    #ip = "127.0.0.1"  # "localhost" #"192.168.1.156"  #"10.1.2.212"  # "192.168.55.1"  #
    #broker_url = ip  # "localhost"
    #broker_port = mqtt_port
    #my_topic = KAFKA_TOPIC
    #my_user = "carla_client"
    #my_pwd = "cttc"
    #broker_url= kafka_IP
    #broker_port= kafka_port
    filename = kafka_file
    #client.username_pw_set(my_user, my_pwd)
    #producer = None
    #if is_kafka is True:
    #    print('connecting to ', broker_url, ' port ', broker_port)
    #    producer = create_producer(broker_url,broker_port)
    #    #client.connect(broker_url, port=broker_port, keepalive=6000)
    #    print('Connected')
    #    #client.loop_start()
    #else:
    #    print(f"Kafka disabled, saving tracklets to ./{filename}")
    #t0 = time.time()
    #while True:
    global tracklets
    global cam_tracklets
    #print(f"Current tracklets list is length {tracklets.qsize()}")
    #print(f"Current tracklets list is length {cam_tracklets.qsize()}")    
    if tracklets.qsize() > 0:
        #print(f"Current tracklets list is length {tracklets.qsize()}")   
        a_tracklet = tracklets.get()
        #Transmission code
        my_tracklet = make_tracklet(a_tracklet, yolo_dict)
        my_tracklet_j = json.dumps(my_tracklet, indent=4)
        #print("Entering save to file")
        save_to_file(my_tracklet_j, filename)
        #print(f"Tracklet to transmit is {my_tracklet_j}. Tracklet saved.")
        #print(f"kafka is {is_kafka}")
        if is_kafka is True:
            try:
                produce_messages(producer, [my_tracklet_j], KAFKA_TOPIC)
                #client.publish(KAFKA_TOPIC, my_tracklet_j)
                print(f"-->T ({KAFKA_TOPIC})")
            except:
                traceback.print_exc()
                print("-->X...")
            #print("See you later!")
    #time.sleep(0.01)
    if cam_tracklets.qsize() > 0:
        cam_tracklet = cam_tracklets.get()
    #    cam_tracklet_f = make_tracklet(cam_tracklet, yolo_dict)
        cam_tracklet_j = json.dumps(cam_tracklet, indent=4)
        #print("Entering save to file")
        save_to_file(cam_tracklet_j, filename)
        #print(f"Cam Tracklet to transmit is {cam_tracklet_j}")
        if is_kafka is True:
            try:
                produce_messages(producer, [cam_tracklet_j], KAFKA_TOPIC)
                #client.publish(KAFKA_TOPIC, cam_tracklet_j)
                print("--->G")
            except:
                traceback.print_exc()
                print("--->X...")
    #time.sleep(0.01)
        #time.sleep(0.1)
    #client.loop.stop()

def make_tracklet(tracklet, yolo_dict):
    for item in tracklet.keys():
        if torch.is_tensor(tracklet[item]):
            tracklet[item] = tracklet[item].cpu().item()
        tracklet[item] = str(tracklet[item])
    tracklet['class_str'] = yolo_dict[int(float(tracklet['class']))]
    #print(f"Class is {tracklet['class_str']}")
    return tracklet




def get_classes():
    #[0,1,2,3,5,7,9,10,11,12]
    yolo_dict={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl', 46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot', 52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch', 58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop', 64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone', 68:'microwave', 69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock', 75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush', 999:'car hero camera'}
    return yolo_dict

def build_projection_matrix(w, h, fov):
    #print(f"in build w={w}, h={h}, fov={fov}")
    h = h
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)
        #print(f"point camera {point_camera}")
        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        #print(f"now point_ camera {point_camera}")
        # now project 3D->2D using the camera matrix
        #print(f"K={K}")
        point_img = np.dot(K, point_camera)
        #print(f"point imaga {point_img}")
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
        #print(f"finally point imaga {point_img}")
        return point_img[0:2]

def get_point_image(pt,K,c2w,zb):
    ptx = [pt[0]*zb, pt[1]*zb, zb]
    Ki = np.linalg.inv(K)
    pt0 = np.dot(Ki, ptx)
    #pt1 = [pt0[2], pt0[0], -pt0[1], 1]
    pt1 = [pt0[2], pt0[0], -pt0[1], 1]
    loc = np.dot(c2w,pt1)
    return loc

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.map = None
        self.depth_camera = None
        self.segmentation_camera = None
        self.car = None
        self.gps = None
        self.lat = None
        self.lon = None
        self.display = None
        self.image = None
        self.depth_image = None
        self.segmentation_image = None
        self.raw_image = None
        self.capture = True
        self.inv_cal = None
        self.boxes =  queue.Queue()
        self.controller = None
        self.autopilot = False
        # Check if CUDA is available
        #os.environ["CUDA_VISIBLE_DEVICES"] = MY_GPU
        if torch.cuda.is_available():
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()

            print(f"Number of available GPUs: {num_gpus}")

            # Iterate through each GPU and print its properties
            for gpu_id in range(num_gpus):
                gpu_properties = torch.cuda.get_device_properties(gpu_id)
                print(f"GPU {gpu_id} - {gpu_properties.name}")
                print(f"  Total Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB")
                print(f"  CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
        else:
            print("CUDA is not available. No GPUs found.")
        global MY_GPU
        print(f"loading model to device {MY_GPU}")
        self.model = YOLO("yolov8n.pt").to(f"cuda:{MY_GPU}")
        print("loaded")

    #### CARLA Bounding boxes ###
    def get_actor_box(self, actor, cam_transform):
        no_compute = ['hero', 'spectator', 'sensor']
        for id in no_compute: 
            if id in actor.type_id:
                return None
        global VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV

        img_w = VIEW_WIDTH
        img_h = VIEW_HEIGHT
        fov = VIEW_FOV

        # Build intrinsic matrix
        focal = img_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.array([[focal,    0, img_w / 2],
                    [   0, focal, img_h / 2],
                    [   0,    0,          1]])

        # 1. Get world‐vertices of bounding box (8 corners)
        # Use CARLA’s API: bounding_box.get_world_vertices()
        world_vertices = actor.bounding_box.get_world_vertices(actor.get_transform())

        # Manually compute inverse transform: world → camera coordinate frame
        R_inv, t_inv = invert_transform(cam_transform)
        #print(f"Rinv : {R_inv}")
        #print(f"tinv : {t_inv}")
        # 3. Project each world vertex into camera frame then to image plane
        corners_2d = []
        for v in world_vertices:
            pw = (v.x, v.y, v.z)
            pc = world_to_camera(R_inv, t_inv, pw)  # camera‐frame (x, y, z)
            # **Axis remapping** from UE4 to camera coordinate system
            cam_x = pc[1]
            cam_y = -pc[2]
            cam_z = pc[0]
            # Filter: only project if in front of camera
            if cam_z <= 0:
                continue
            # Keep only points in front of camera
            # Project using intrinsics
            proj = np.dot(K, np.array([cam_x, cam_y, cam_z]))
            u = proj[0] / proj[2]
            v = proj[1] / proj[2]
            corners_2d.append((u, v))
        #print("Actor:", actor.id)
        #print("Cam position:", cam_transform.location)
        #print("Actor position:", actor.get_transform().location)
        #print("Projected corners:", corners_2d)

        # If no points projected, return None or a default box
        if len(corners_2d) == 0:
            return None

        # 4. Compute 2D bounding box from projected points
        xs = [p[0] for p in corners_2d]
        ys = [p[1] for p in corners_2d]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)

        # Optionally clamp to image bounds
        x0 = max(0, min(img_w - 1, x0))
        y0 = max(0, min(img_h - 1, y0))
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))

        return [int(np.round(x0)),int(np.round(y0)),int(np.round(x1)),int(np.round(y1))]

    def haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        radius = 6371000  # approximately 6,371 km
        distance = radius * c
        return distance

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """
        global VIEW_FOV
        global VIEW_HEIGHT
        global VIEW_WIDTH
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp
    
    def camera_depth_blueprint(self):
        """
        Returns camera blueprint.
        """
        global VIEW_FOV
        global VIEW_HEIGHT
        global VIEW_WIDTH
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp


    def camera_segmentation_blueprint(self):
        """
        Returns camera blueprint.
        """
        global VIEW_FOV
        global VIEW_HEIGHT
        global VIEW_WIDTH
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        global is_synch
        if is_synch is False:
            print("Agent running on asynchronous mode.")
            return
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        #settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self, tipo):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_transform = None
        if 'car' in tipo: #tipo == 'car' or tipo == 'car2':
            camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
            self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        if tipo == 'red_light':
            global cam_x
            global cam_y
            global cam_z
            global cam_r
            global cam_p
            global cam_w
            camera_transform = carla.Transform(carla.Location(x=cam_x, y=cam_y, z=cam_z), carla.Rotation(pitch=cam_p, yaw=cam_w, roll=cam_r))
            self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))
        camera_bp = self.camera_blueprint()
        
    def setup_depth_camera(self, tipo):
        camera_transform = None
        if 'car' in tipo: #tipo == 'car' or tipo == 'car2':
            camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
            self.depth_camera = self.world.spawn_actor(self.camera_depth_blueprint(), camera_transform, attach_to=self.car)
        if tipo == 'red_light':
            global cam_x
            global cam_y
            global cam_z
            global cam_r
            global cam_p
            global cam_w
            camera_transform = carla.Transform(carla.Location(x=cam_x, y=cam_y, z=cam_z + 0.1), carla.Rotation(pitch=cam_p, yaw=cam_w, roll=cam_r))
            self.depth_camera = self.world.spawn_actor(self.camera_depth_blueprint(), camera_transform)
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda image: weak_self().set_depth_image(weak_self, image))
    
    def setup_segmentation_camera(self, tipo):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_transform = None
        if 'car' in tipo: #tipo == 'car' or tipo == 'car2':
            camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
            self.segmentation_camera = self.world.spawn_actor(self.camera_segmentation_blueprint(), camera_transform, attach_to=self.car)
        if tipo == 'red_light':
            global cam_x
            global cam_y
            global cam_z
            global cam_r
            global cam_p
            global cam_w
            camera_transform = carla.Transform(carla.Location(x=cam_x, y=cam_y, z=cam_z), carla.Rotation(pitch=cam_p, yaw=cam_w, roll=cam_r))
            self.camera = self.world.spawn_actor(self.camera_segmentation_blueprint(), camera_transform)
        weak_self = weakref.ref(self)
        self.segmentation_camera.listen(lambda image: weak_self().set_segmentation_image(weak_self, image))
        camera_bp = self.camera_blueprint()

    def setup_gps(self, parento):
        self.lat = 0.0
        self.lon = 0.0
        bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gps = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=0., z=0.1)), attach_to=parento)
        weak_self = weakref.ref(self)
        self.gps.listen(lambda event: weak_self().on_gnss_event(weak_self, event))
    
    def detect_objects_async(self, img, dimg, bboxes, actors):
        thread = threading.Thread(target=self.detect_objects, args=(img, dimg, bboxes, actors))
        thread.start()
    
    def detect_objects(self, rgb, dimg, simg, bboxes, actors):
        #print(f"rgb {rgb.shape}, depth {dimg.shape}, seg {simg.shape}")
        global MY_GPU
        global STATION_ID
        cam_trans = self.camera.get_transform()
        #print("before")
        t0 = time.time()
        try:
            results = self.model.track(rgb, conf=0.1, classes=[0,1,2,3,5,7,9,11], device=MY_GPU)
            #time.sleep(np.random.rand())
        except Exception as e:
            print(f"Error while tracking: {e}")
            return rgb
        t1 = time.time()
        #print("after")
        res = results[0].plot()
        #print(f"res is shape {res.shape}")
        #img = Image.fromarray(res[..., ::-1])
        img = cv2.cvtColor(res[..., ::-1], cv2.COLOR_BGR2RGB)
        #print(f"img is shape {img.shape}")
        boxes = results[0].boxes.xyxy
        clss =  results[0].boxes.cls
        confs = results[0].boxes.conf
        ids = results[0].boxes.id
        if ids is None:
            ids = torch.zeros_like(clss)
        ti = time.time()
        tracklet = dict()
        global tracklets
        #labels, counts = np.unique(simg, return_counts=True)
        #my_ix = 10
        #print(f"in this frame lables are {labels} with count {counts} is {my_ix} in lables? {my_ix in labels}")
        #if my_ix in labels:
        #    indices = np.argwhere(simg == my_ix)      # np.where returns a tuple, take [0] for the indices
        #    print(f"vehicle box is {np.min(indices[:,0])}, {np.min(indices[:,1])}, {np.max(indices[:,0])} {np.max(indices[:,1])} ")  
        print(f"we have {len(boxes)} boxes to process")
        
        for box, clase, conf, id  in zip(boxes, clss, confs, ids):
            #print(f"box is {box}")
            res = self.get_object_distance(box, bboxes, actors)
            #res = self.get_object_distance(box, cam_trans, dimg, simg, MY_TYPE)
            print(f"739. res is {res}")
            if res[0] is None:
                my_lat_lon = self.get_my_lat_lon()
                tracklet['timestamp'] = str(ti*10**7)
                tracklet['station_id'] = STATION_ID
                tracklet['station_lat'] = my_lat_lon.latitude
                tracklet['station_lon'] = my_lat_lon.longitude
                tracklet['ID'] = 123456789
                tracklet['class'] = 0
                tracklet['carla_class'] = 'clutter'
                tracklet['conf'] = 0
                tracklet['latitude'] = my_lat_lon.latitude
                tracklet['longitude'] = my_lat_lon.longitude
                tracklet['distance'] = 0
                #global tracklets 
                tracklets.put(tracklet) # to be furhter processed and transmitted on separated thread
                print("->->->?")
                continue
            else: 
                lat, lon, disto, type_id = res[0], res[1], res[2], res[3]
                my_lat_lon = self.get_my_lat_lon()
                tracklet['timestamp'] = str(ti*10**7)
                tracklet['station_id'] = STATION_ID
                tracklet['station_lat'] = my_lat_lon.latitude
                tracklet['station_lon'] = my_lat_lon.longitude
                tracklet['ID'] = id
                tracklet['class'] = clase
                tracklet['carla_class'] = type_id
                tracklet['conf'] = conf
                tracklet['latitude'] = lat
                tracklet['longitude'] = lon
                tracklet['distance'] = disto
                #global tracklets 
                tracklets.put(tracklet) # to be furhter processed and transmitted on separated thread
                print("->->->")
        # csimg = None
        # for box in boxes:
        #     x0, y0, x1, y1 = [int(v) for v in box]
        #     col = (255, 255, 0)
        #     csimg = simg.copy()
        #     cv2.rectangle(csimg, (x0, y0), (x1, y1), col, thickness=2)
        # if csimg is None:
        #     return np.array(simg) 
        # return np.array(csimg)
        return img, t1 - t0

    def make_camera_tracklet(self,t0,tmax=0.001):
        t1 = time.time()
        if np.abs(t1 - t0) < tmax:
            return t0
        global STATION_ID
        global cam_tracklets
        global MY_TYPE
        tracklet = dict()
        my_lat_lon = self.get_my_lat_lon()
        tracklet['timestamp'] = str(time.time()*10**7)
        tracklet['station_id'] = str(STATION_ID)
        tracklet['station_lat'] = str(my_lat_lon.latitude)
        tracklet['station_lon'] = str(my_lat_lon.longitude)
        tracklet['ID'] = str(STATION_ID)
        #tracklet['class'] = 0
        #if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2' :
        tracklet['class'] = str(999)
        my_actor = self.world.get_actor(self.car.id)
        tracklet['carla_class'] = my_actor.type_id
        tracklet['conf'] = "1"
        tracklet['latitude'] = str(my_lat_lon.latitude)
        tracklet['longitude'] = str(my_lat_lon.longitude)
        tracklet['distance'] = str(0)
        tracklet['class_str'] = MY_TYPE+'_camera'
        cam_tracklets.put(tracklet)
        return t1

    @staticmethod
    def on_gnss_event(weak_self, event):
        global LAT0
        global LON0
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude + LAT0
        self.lon = event.longitude + LON0
        #print('***GNSS***')
        #print(f"{event.latitude} + {LAT0} = {self.lat}")
        #print(f"{event.longitude} + {LON0} = {self.lon}")
        #print('***GNSS***')
    

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        #print(f"from set image h={img.height},w={img.width}, fov={img.fov}")
        use_yolo = True
        #img = None
        if img is not None and self.capture:
            raw_img = np.array(img.raw_data)
            image_data = np.frombuffer(raw_img, dtype=np.dtype("uint8"))
            image_data = np.reshape(image_data, (img.height, img.width, 4))
            image_data = image_data[:, :, :3]
            image_data = image_data[:, :, ::-1]
            self.image = image_data
            self.raw_image = image_data
            self.capture = False


    @staticmethod
    def set_depth_image(weak_self, img):
        self = weak_self()
        if img is not None:
            img.convert(cc.Depth)
            #self.depth_image= img
            #print(f"depth img is {type(img)}")
            array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (img.height, img.width, 4))
            array = array[:, :, :3]
            R = array[:,:,2]
            G = array[:,:,1]
            B = array[:,:,0]
            R = R.astype(np.uint32)
            G = G.astype(np.uint32)
            B = B.astype(np.uint32)
            #D = 10*(1/(256*256*256 - 1))*(R + 256*G + 256*256*B)
            D = (R + 256 * G + 256**2 * B) / (256**3 - 1)
            D = 1000 * D  # distance in meters

            #print(f"D is {D.shape}")
            self.depth_image = D

    @staticmethod
    def set_segmentation_image(weak_self, img):
        self = weak_self()
        if img is not None:
            array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (img.height, img.width, 4))
            #print(f"array is {array}")
            labels = array[:, :, 2]
            self.segmentation_image = labels
     

    def render(self, image):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        if image is not None:
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
    
    def get_object_distance(self, box, bboxes, actors):
       
        global LAT0
        global LON0
        global MY_TYPE
        box = box.cpu()
        x0, y0, x1, y1  = [int(x) for x in box]
        hero_box = [x0, y0, x1, y1]
        max_iou = 0
        my_actor = None
        my_box = []
        for boxi, actor in zip(bboxes, actors):
            if boxi is not None:
                iou = iou_2d(hero_box, boxi)
                if iou > max_iou:
                    max_iou = iou
                    my_actor = actor
                    my_box = boxi
        if my_actor is None:
            return [None, None, None, None]
        #print(f"my actors is {my_actor.type_id} with bbox {my_box} and iou {max_iou}")
        my_loc = my_actor.get_location()
        my_lalo = self.map.transform_to_geolocation(my_loc)
        my_lalo.latitude += LAT0
        my_lalo.longitude += LON0
        return [my_lalo.latitude, my_lalo.longitude, float(max_iou), my_actor.type_id]


    def get_object_distance0(self, box, trans, depth, segmentation, tipo):
        seg_labels = {'car':10, 'truck':10, 'bus':10, 'person':4, 'bicycle':10, 'motorcycle':10, 'traffic light':18, 'fire hydrant':19, 'stop sign':12, 'parking meter':19}
        inv_seg_labels = {
            0: "Unlabeled",
            1: "Building",
            2: "Fence",
            3: "Other",
            4: "Pedestrian",
            5: "Pole",
            6: "RoadLine",
            7: "Road",
            8: "SideWalk",
            9: "Vegetation",
            10: "Vehicles",
            11: "Wall",
            12: "TrafficSign",
            13: "Sky",
            14: "Ground",
            15: "Bridge",
            16: "RailTrack",
            17: "GuardRail",
            18: "TrafficLight",
            19: "Static",
            20: "Dynamic",
            21: "Water",
            22: "Terrain"
        }

        good_seg_labels = {
            4: "Pedestrian",
            10: "Vehicles",
            12: "TrafficSign",
            18: "TrafficLight",
        }

        #inv_seg_labels = {10:'car', 10:'truck', 10:'bus':10, 'person':4, 'bicycle':10, 'motorcycle':10, 'traffic light':18, 'fire hydrant':19, 'stop sign':12, 'parking meter':19}
        #print("************************************")
        #print("Player 1, Greetings from get_object_distance()")
        global LAT0
        global LON0
        global MY_TYPE
        my_id = 0
        if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2':
            my_id = self.car.id
            #print(f"Car id is {my_id}")
        depth = np.array(depth)
        camera_bp = self.camera_blueprint()
        w = camera_bp.get_attribute("image_size_x").as_int()
        h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        pos = self.camera.get_location()
        if self.depth_image is None:
            #print("Depth is None...")
            return [None, None, None, None]
        ptc = torch.div(box[2] - box[0], 2, rounding_mode='floor') + box[0]
        ptr = torch.div(box[3] - box[1], 2, rounding_mode='floor') + box[1]
        ptc = int(ptc.item())
        ptr = int(ptr.item())
        #actors = self.world.get_actors()
        box = box.cpu()
        x0, y0, x1, y1  = [int(x) for x in box]
        hero_box = [x0, y0, x1, y1]
        actors = self.world.get_actors()
        landmarks = self.map.get_all_landmarks()
        my_actor = None
        box_iou = 0
        my_dist = np.inf
        hero_loc = self.camera.get_location()
        hero_trans = self.camera.get_transform()
        for actor in actors:
            if 'sensor' in actor.type_id or 'spectator' in actor.type_id:
                continue
            
            actor_loc = actor.get_location()
            dist = hero_loc.distance(actor_loc)
            bb = self.get_actor_box(actor, hero_trans)
            #print(f"hero si {hero_box} bb is {bb}")
            if bb is None:
                continue
            iou = iou_2d(hero_box, bb)
            print(f"iou {iou}")
            if iou > box_iou and dist > 0 and dist < 20:
                my_actor = actor
                box_iou = iou
                my_dist = dist
        # for lm in landmarks:
        #     hero_loc = self.car.get_location()
        #     actor_loc = lm.transform.location #lm.get_location()
        #     dist = hero_loc.distance(actor_loc)
        #     bb = self.get_actor_box(lm, hero_trans)
        #     iou = iou_2d(hero_box, bb)
        #     if iou > box_iou and dist > 0 and dist < 20:
        #         my_actor = actor
        #         box_iou = iou
        #         my_dist = dist
        if my_actor is None:
            return [None, None, None, None]
        print(f"the actors is {my_actor.type_id}, dist {my_dist}, iou {box_iou}")
        my_loc = my_actor.get_location()
        #rint(f"loc is {caught_loc}")
        my_lalo = self.map.transform_to_geolocation(my_loc)
        my_lalo.latitude += LAT0
        my_lalo.longitude += LON0
        #print(f"lat {caught_lalo.latitude} lon {caught_lalo.longitude} dist {caught_dist} zb {zb}, delta {the_delta}")
        return [my_lalo.latitude, my_lalo.longitude, float(box_iou), my_actor.type_id]
        
        #print(f"box is {[x0,y0,x1,y1]}")
        #print(f"with center at {[ptc, ptr]}")
        zbs = depth[y0:y1, x0:x1]
        segs = segmentation[y0:y1, x0:x1]
        if zbs.size == 0 or segs.size == 0:
            return [None, None, None, None]
        #print(f"segs {segs.shape}")
        flat_zbs = zbs.flatten()
        flat_segs = segs.flatten()
        valid = np.isfinite(flat_zbs) & (flat_zbs > 0)
        zb = np.median(flat_zbs[valid])
        
        hero_loc = self.car.get_location()
        caught_actor = None
        caught_dist = None
        the_delta = np.inf
        for actor in actors:
            ac_loc = actor.get_location()
            dist = hero_loc.distance(ac_loc)
            delta = np.abs(dist - zb) 
            if delta > 0 and delta < the_delta:
                caught_actor = actor
                caught_dist = dist
                the_delta = delta
        if caught_actor is None:
            caught_actor = None
            caught_dist = None
            the_delta = np.inf
            for lm in landmarks:
                ac_loc = lm.get_location()
                dist = hero_loc.distance(ac_loc)
                delta = np.abs(dist - zb) 
                if delta < the_delta:
                    caught_actor = actor
                    caught_dist = dist
                    the_delta = delta
        if caught_actor is None:
            return [None, None, None, None]
        caught_loc = caught_actor.get_location()
        #rint(f"loc is {caught_loc}")
        caught_lalo = self.map.transform_to_geolocation(caught_loc)
        caught_lalo.latitude += LAT0
        caught_lalo.longitude += LON0
        #print(f"lat {caught_lalo.latitude} lon {caught_lalo.longitude} dist {caught_dist} zb {zb}, delta {the_delta}")
        return [caught_lalo.latitude, caught_lalo.longitude, float(the_delta), caught_actor.type_id]
        #     if actor.id != my_id and hvs < my_hvs and 'sensor' not in actor.type_id:
        #         my_actor = actor
        #         my_hvs = hvs
        # #if my_hvs > 0 and my_hvs < 4*zb:
        # my_id = my_actor.id
        # #print(f"actor is {self.world.get_actor(my_id)}")
        # my_actor = self.world.get_actor(my_id)
        # my_ac_trans = my_actor.get_transform()
        # #print(f"err = {obj_loc.distance(my_ac_trans.location)}")
        # obj_loc = my_ac_trans.location
        # #print(f"Actor location is {obj_loc}")
        # # 541-548 al if ->>>>>>>>>>>>>><
        # obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
        # obj_lat_lon.latitude += LAT0
        # obj_lat_lon.longitude += LON0
        # #hvs = my_actor.get_transform().location.distance(self.camera.get_transform().location)
        # hvs = obj_loc.distance(trans.location)
        # type_id = my_actor.type_id
        # #print(f"cam loc: {trans.location}")
        # #print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
        # #print(f"location {obj_loc}")
        # #print(f"haversine = {hvs}, val = {zb}")
        # return [obj_lat_lon.latitude, obj_lat_lon.longitude, my_seg, type_id]
        # #else:
        # #    #print("No ID was found...")
        # #    return [None,None,None, None]

        # # labels, counts = np.unique(flat_segs, return_counts=True)
        # # print(f"labels {labels}, counts {counts}")
        # # best_idx = np.argmax(counts)
        # # best_lbl = labels[best_idx]
        # # lbl_frac = counts[best_idx] / flat_segs.size
        # # #print(f"lbl {best_lbl}, {lbl_frac}")
        # # if lbl_frac < 0.25:
        # #     best_lbl = 'clutter'
        # # u = (x0 + x1) / 2
        # # v = (y0 + y1) / 2
        # # zc = [u, v]
        # # K = build_projection_matrix(w,h,fov)
        # # camera_2_world = trans.get_matrix()
        # # the_loc = get_point_image(zc, K, camera_2_world, zb)
        # # obj_loc = carla.Location(the_loc[0], the_loc[1], the_loc[2])
        # # obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
        # # obj_lat_lon.latitude += LAT0
        # # obj_lat_lon.longitude += LON0
        # # if best_lbl not in inv_seg_labels.keys():
        # #     best_lbl = 0
        # # return [obj_lat_lon.latitude, obj_lat_lon.longitude, float(lbl_frac), inv_seg_labels[best_lbl]]
        
        # # #print(f"Box location in the world is {obj_loc}")
        # # actors_ids = []
        # # cloc = trans.location #self.camera.get_location()
        # # my_id = None
        # # my_actor = None
        # # my_seg = np.inf
        # # my_hvs = np.inf
        # # dif_hvs = np.inf
        # # for actor in actors:
        # #     if 'sensor' in actor.type_id:
        # #         continue #ingnore 
        # #     ac_loc = actor.get_location()
        # #     hvs = obj_loc.distance(ac_loc)
        # #     #print(f"hvs {hvs} zbs {zb}")
        # #     if np.abs(hvs - zb) < dif_hvs:
        # #         my_actor = actor
        # #         my_hvs = hvs
        # #         my_seg = segsm
        # #         dif_hvs = np.abs(hvs - zb)
        # # #print(f"final difference = {dif_hvs}")
        # # # for actor in actors:
        # # #     ac_trans = actor.get_transform()
        # # #     ac_loc = actor.get_location()
        # # #     hvs = obj_loc.distance(ac_loc)
        # # #     if actor.id != my_id and hvs < my_hvs and 'sensor' not in actor.type_id:
        # # #         my_actor = actor
        # # #         my_hvs = hvs
        # # #if my_hvs > 0 and my_hvs < 4*zb:
        # # my_id = my_actor.id
        # # #print(f"actor is {self.world.get_actor(my_id)}")
        # # my_actor = self.world.get_actor(my_id)
        # # my_ac_trans = my_actor.get_transform()
        # # #print(f"err = {obj_loc.distance(my_ac_trans.location)}")
        # # obj_loc = my_ac_trans.location
        # # #print(f"Actor location is {obj_loc}")
        # # # 541-548 al if ->>>>>>>>>>>>>><
        # # obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
        # # obj_lat_lon.latitude += LAT0
        # # obj_lat_lon.longitude += LON0
        # # #hvs = my_actor.get_transform().location.distance(self.camera.get_transform().location)
        # # hvs = obj_loc.distance(trans.location)
        # # type_id = my_actor.type_id
        # # #print(f"cam loc: {trans.location}")
        # # #print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
        # # #print(f"location {obj_loc}")
        # # #print(f"haversine = {hvs}, val = {zb}")
        # # return [obj_lat_lon.latitude, obj_lat_lon.longitude, my_seg, type_id]
        # # #else:
        # # #    #print("No ID was found...")
        # # #    return [None,None,None, None]
       
    def get_my_lat_lon(self):
        global LAT0
        global LON0
        loc = self.camera.get_transform().location
        lat_lon = self.map.transform_to_geolocation(loc)
        lat_lon.latitude += LAT0
        lat_lon.longitude += LON0
        return lat_lon

    def game_loop(self, num_classes, input_size, graph, return_tensors):
        """
        Main program loop.
        """

        global LAT0
        global LON0
        global CARLAIP
        global THE_THICK
        global STATION_ID
        global MY_GPU
        global MY_TYPE
        global MY_FPS
        global is_synch
        global is_yolo
        global kafka_IP
        global kafka_port
        global is_kafka

        try:
            pygame.init()
            
            self.client = carla.Client(CARLAIP, 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            print(f"my type is {MY_TYPE}")
            print(f"Getting hero vehicle")
            actors = self.world.get_actors()
            the_actor = None
            for actor in actors:
                print(f"actor is {actor.attributes.get('role_name') }")
                if actor is None:
                    continue
                if 'hero' in [actor.attributes.get('role_name')]:
                    the_actor = actor
            if the_actor is None:
                print("No hero vehicle as found...")
                print("Making a hero...")
                #exit()
            print(f"the actor si {the_actor}")
            self.car = the_actor
            if the_actor is None:
                self.setup_car()
            print(f"the_actor is {self.car.attributes.get('role_name') }")
            self.setup_camera(MY_TYPE)
            self.setup_depth_camera(MY_TYPE)
            self.setup_segmentation_camera(MY_TYPE)
            self.setup_gps(self.camera)
            self.map=self.world.get_map()
            camera_bp = self.camera_blueprint()
            w = camera_bp.get_attribute("image_size_x").as_int()
            h = camera_bp.get_attribute("image_size_y").as_int()
            fov = camera_bp.get_attribute("fov").as_float()
            print(f"w={w}, h={h}")
            self.display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
            #init_image = np.random.randint(0,255,(h,w,3),dtype='uint8')
            #surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))
            #self.display.fill((0,0,0))
            #self.display.blit(surface, (0,0))
            #pygame.display.flip()
            pygame_clock = pygame.time.Clock()
            if is_synch:
                self.set_synchronous_mode(True)
            else:
                self.set_synchronous_mode(False) 
            print(f"Type is {MY_TYPE}")
            if 'car' in MY_TYPE: #MY_TYPE=='car' or MY_TYPE == 'car2':
                print("OK")
                self.car.set_autopilot(self.autopilot)
                print("Autopilot set")
                print(f"car is {self.car}")
                self.controller = KeyboardControl(self.car, self.world, self.autopilot)
            ti = time.time()
            kafka_producer = None
            if is_kafka:
                print('connecting to ', kafka_IP, ' port ', kafka_port)
                kafka_producer = create_producer(kafka_IP, kafka_port)
                print(f"conntected to {kafka_producer}")
            max_kafka = 10
            cont_kafka = max_kafka
            print("Warm up...")
            #time.sleep(30)
            # print("tags")
            # blueprints = self.world.get_blueprint_library()  
            # for bp in blueprints:
            #     tags = bp.get_attribute('semantic_tags') if bp.has_attribute('semantic_tags') else None
            #     print(f" bp is {bp}")
            #     if tags:
            #      print(bp.id, tags.as_string())
            timos = []
            while True:
                #t0 = time.time()
                #print("while")
                self.capture = True
                pygame_clock.tick_busy_loop(MY_FPS)
                print("---------------------------------")
                #if MY_TYPE == 'car':
                #    print("---------------------------------")
                #    print(f"Lat {self.lat}, Lon={self.lon}")
                img = self.image
                dimg = self.depth_image
                simg = self.segmentation_image
                if img is None or dimg is None or simg  is None:
                    print("one is none")
                    continue
                #print("detect")
                #self.detect_objects_async(img, dimg)
                #print(f"is yolo is {is_yolo}")
                hero_trans = self.camera.get_transform()
                bboxes = []
                actors = self.world.get_actors()
                for actor in actors:
                    bb = self.get_actor_box(actor, hero_trans)
                    #print(f"actor: {actor.type_id}, {bb}")
                    bboxes.append(bb)
                if is_yolo is True:
                    #print("in if!")
                    img, dt = self.detect_objects(img, dimg, simg, bboxes, actors)
                    timos.append(dt)
                ti = self.make_camera_tracklet(ti)
                to_6G = threading.Thread(target=carla26g, args=(kafka_producer,))
                to_6G.start()
                cont_kafka -= 1
                if cont_kafka < 1:
                    kafka_producer.flush()
                    cont_kafka = max_kafka


                #print("render")
                if 'car' in MY_TYPE and self.controller.parse_events(self.car, self.world, pygame_clock):
                    return
                self.render(img)
                pygame.display.flip()
                pygame.event.pump()
                #print(f" thick is {THE_THICK}")
                if THE_THICK is True:
                    #print(f"myType is {MY_TYPE}")
                #if is_synch == True:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                #print("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
                #t1 = time.time()
                #timos.append(t1 - t0)
        except Exception as e:
                        print(f"Game Loop Exeption {e}")
                        traceback.print_exc()
        finally:
            kafka_producer.close()
            self.camera.destroy()
            if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2':
                self.car.destroy()
            self.gps.destroy()
            percs = np.percentile(timos, [0, 25, 50, 75, 90, 99, 100])
            #print(timos)
            #plt.hist(timos)
            #plt.savefig()
            print(f"boxplot data: {percs}, std {np.std(percs)}")

            pygame.quit()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, car, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        #if isinstance(car, carla.Vehicle):
        self._control = carla.VehicleControl()
        self._ackermann_control = carla.VehicleAckermannControl()
        self._lights = carla.VehicleLightState.NONE
        car.set_autopilot(self._autopilot_enabled)
        car.set_light_state(self._lights)
        #elif isinstance(car, carla.Walker):
        #    self._control = carla.WalkerControl()
        #    self._autopilot_enabled = False
        #    self._rotation = world.player.get_transform().rotation
        #else:
        #    raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        #world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, car,  world, clock):
        #print("Hello from controller")
        if self._autopilot_enabled is True:
            return
        current_lights = self._lights
        #print(f"autopilot state is {car.get_autopilot()}")
        if self._autopilot_enabled is False:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    car.set_light_state(carla.VehicleLightState(current_lights))
                # Apply control
                if not self._ackermann_enabled:
                    car.apply_control(self._control)
                else:
                    car.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = car.get_control()
                    # Update hud with the newest ackermann control
                    #world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                pass
                #self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                #world.player.apply_control(self._control)

        self._lights = current_lights

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)
        if keys[K_q]:
            if not self._ackermann_enabled:
                self._control.gear = 1 if self._control.reverse else -1
            else:
                self._ackermann_reverse *= -1
                # Reset ackermann control
                self._ackermann_control = carla.VehicleAckermannControl()
    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
    
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    print(f"---------------------------------------------------------------------------------{datetime.now()}-----------------------------------------------------------------------------")
    try:
        global is_autopilot
        num_classes     = 80
        input_size      = 416
        global LAT0
        global LON0
        global CARLAIP
        global MY_GPU
        global MY_TYPE
        global THE_THICK
        global STATION_ID
        global cam_x
        global cam_y
        global cam_z
        global cam_r
        global cam_p
        global cam_w
        global mqtt_port
        global is_kafka
        global kafka_IP
        global kafka_port
        global kafka_file
        global docker_path
        global is_synch
        global is_yolo
        global MY_FPS
        #global producer
        args = sys.argv
        argnames = ['filename', 'LAT0', 'LON0', 'CARLA_IP', 'MY_GPU', 'THE_THICK', 'MY_TYPE', 'cam_x', 'cam_y', 'cam_z', 'cam_r', 'cam_p', 'cam_w','STATION_ID','mqtt_port', 'is_kafka', 'kafka_IP', 'kakfa_port', 'kafka_file',
                  'is_synch', 'is_autopilot', 'MY_FPS', 'is_yolo']
        print("Args are:")
        for nam, ar in zip(args, argnames):
            print(f"{nam}->{ar}")        
        LAT0 = float(args[1])
        LON0 = float(args[2])
        CARLAIP = str(args[3])
        MY_GPU = str(args[4])
        if str(args[5]) == 'yes':
            print("args[5] is" , args[5])
            THE_THICK = True
        MY_TYPE = str(args[6]) #car/red_light
        cam_x = float(args[7])
        cam_y = float(args[8])
        cam_z = float(args[9])
        cam_r = float(args[10])
        cam_p = float(args[11])
        cam_w = float(args[12])
        STATION_ID = str(args[13])
        mqtt_port = int(args[14])
        if str(args[15]) == 'no_kafka':
            is_kafka = False
        if is_kafka is True:
            kafka_IP = str(args[16])
            kafka_port = str(args[17])
            kafka_file = str(args[18])
            #print('connecting to ', kafka_IP, ' port ', kafka_port)
            #producer = create_producer(kafka_IP, kafka_port)
            #client.connect(broker_url, port=broker_port, keepalive=6000)
            #print('Connected')
            #client.loop_start()
        else:
            kafka_file = str(args[18])
            print(f"Kafka disabled, saving tracklets to ./{kafka_file}")
        if str(args[19]) == 'yes':
            is_synch = True
        if str(args[20]) == 'autopilot':
            is_autopilot = True
        MY_FPS = int(args[21])
        if str(args[22]) == 'no':
            is_yolo = False
        print("--------- CONFIG SUMMARY----------")
        print(f"Connecting to server at IP {CARLAIP}")
        print(f"Running on GPU {MY_GPU}")
        print(f"Taking the following role: {MY_TYPE}")
        print(f"Reference coordiantes: {LAT0},{LON0}")
        print(f"World-ticker Script: {str(THE_THICK)}")
        if MY_TYPE == 'red_light':
            print(f"Camera location (XYZ)= ({cam_x},{cam_y},{cam_z})")
            print(f"Camera transform (RPY)= ({cam_r},{cam_p},{cam_w})")
        else:
            print("Agent's transform will be chosen by the simulator")
        if is_kafka is True:
            print(f"Tracklets transmited via Kafka to {kafka_IP}:{kafka_port}")
        else:
            print(f"Tracklets will be saved to file {docker_path+kafka_file}")
        print(" That is all! Happy Simulation!")
        print("-------------------------------------")
        carla_client = BasicSynchronousClient()
        if 'car' in MY_TYPE:
            carla_client.autopilot = is_autopilot
        graph = []
        return_tensors = []
        # Threading and MQTT
        #to_6G = threading.Thread(target=carla26g)
        #to_6G.start()
        carla_client.game_loop(num_classes, input_size, graph, return_tensors)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
