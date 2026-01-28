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
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#VIEW_WIDTH = 1920//2
#VIEW_HEIGHT = 1080//2
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
docker_path = './jsons'
is_synch = False


###### KAFKA
def create_producer(server,port):
    port = str(port)
    print(f"server:{server}")
    print(f"port:{port}")
    bootstrap_servers=server+':'+port
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v.encode('utf-8')
    )

def produce_messages(producer, messages, topic_name):
    for message in messages:
        future = producer.send(topic_name, value=message)
        try:
            future.get(timeout=10)
            print(f"Message '{message}' sent successfully.")
        except KafkaError as e:
            print(f"Failed to send message '{message}': {e}")
###### MQTT Client
kafka_messages = []
KAFKA_TOPIC = "tracklets"

###### Thread function
def save_to_file(data, filename):
    try:
        with open(filename, 'w') as file:  # Open file in append mode
            file.write(data + '\n')  # Add a newline between each JSON entry
        print(f"Data appended to {filename} successfully.")
    except Exception as e:
        print(f"Failed to append data to {filename}: {e}")
        try:
            with open(filename, 'w') as file:
                file.write(data)
            print(f"Data saved to {filename} successfully.")
        except Exception as e:
            print(f"Failed to save data to {filename}: {e}")
            traceback.print_exc()

def carla26g():
    tmax = 1
    filename = ''
    print("Greetings from carla thread!")
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
    ip = "127.0.0.1"  # "localhost" #"192.168.1.156"  #"10.1.2.212"  # "192.168.55.1"  #
    #broker_url = ip  # "localhost"
    #broker_port = mqtt_port
    my_topic = KAFKA_TOPIC
    my_user = "carla_client"
    my_pwd = "cttc"
    broker_url= kafka_IP
    broker_port= kafka_port
    filename = kafka_file
    #client.username_pw_set(my_user, my_pwd)
    producer = None
    if is_kafka is True:
        print('connecting to ', broker_url, ' port ', broker_port)
        producer = create_producer(broker_url,broker_port)
        #client.connect(broker_url, port=broker_port, keepalive=6000)
        print('Connected')
        #client.loop_start()
    else:
        print(f"Kafka disabled, saving tracklets to ./{filename}")
    t0 = time.time()
    while True:
        global tracklets
        global cam_tracklets
        if tracklets.qsize() > 0:
            print(f"Current tracklets list is length {tracklets.qsize()}")   
            a_tracklet = tracklets.get()
            #Transmission code
            my_tracklet = make_tracklet(a_tracklet, yolo_dict)
            my_tracklet_j = json.dumps(my_tracklet, indent=4)
            save_to_file(my_tracklet_j, filename)
            print(f"Tracklet to transmit is {my_tracklet_j}. Tracklet saved.")
            if is_kafka is True:
                try:
                    produce_messages(producer, [my_tracklet_j], KAFKA_TOPIC)
                    #client.publish(KAFKA_TOPIC, my_tracklet_j)
                    print("Tracklet transmitted succesfully.")
                except:
                    print("Tracklet transmission failed...")
                print("See you later!")
        #if cam_tracklets.qsize() > 0:
        #    cam_tracklet = cam_tracklets.get()
        #    cam_tracklet_j = json.dumps(cam_tracklet, indent=4)
        #    save_to_file(cam_tracklet_j, filename)
        #    print(f"Tracklet to transmit is {cam_tracklet_j}")
        #    try:
        #        produce_messages(producer, [cam_tracklet_j], KAFKA_TOPIC)
        #        #client.publish(KAFKA_TOPIC, cam_tracklet_j)
        #        print("Tracklet transmitted succesfully.")
        #    except:
        #        print("Tracklet transmission failed...")
        time.sleep(0.1)
    #client.loop.stop()

def make_tracklet(tracklet, yolo_dict):
    for item in tracklet.keys():
        if torch.is_tensor(tracklet[item]):
            tracklet[item] = tracklet[item].cpu().item()
        tracklet[item] = str(tracklet[item])
    tracklet['class_str'] = yolo_dict[int(float(tracklet['class']))]
    print(f"Class is {tracklet['class_str']}")
    return tracklet




def get_classes():
    #[0,1,2,3,5,7,9,10,11,12]
    yolo_dict={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl', 46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot', 52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch', 58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop', 64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone', 68:'microwave', 69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock', 75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush'}
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
        self.car = None
        self.gps = None
        self.lat = None
        self.lon = None
        self.display = None
        self.image = None
        self.depth_image = None
        self.raw_image = None
        self.capture = True
        self.inv_cal = None
        self.boxes =  queue.Queue()
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
        self.model = YOLO("yolov8n.pt") 

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

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        #camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        #camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        #camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp
    
    def camera_depth_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        #camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        #camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        #camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        global is_sync
        if is_sync is False:
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
            camera_transform = carla.Transform(carla.Location(x=1.6, z=1.8), carla.Rotation(pitch=0))
            self.depth_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        if tipo == 'red_light':
            global cam_x
            global cam_y
            global cam_z
            global cam_r
            global cam_p
            global cam_w
            camera_transform = carla.Transform(carla.Location(x=cam_x, y=cam_y, z=cam_z + 0.1), carla.Rotation(pitch=cam_p, yaw=cam_w, roll=cam_r))
            self.depth_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform)
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda image: weak_self().set_depth_image(weak_self, image))
    

    def setup_gps(self, parento):
        self.lat = 0.0
        self.lon = 0.0
        bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gps = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=0., z=0.1)), attach_to=parento)
        weak_self = weakref.ref(self)
        self.gps.listen(lambda event: weak_self().on_gnss_event(weak_self, event))

    def detect_objects(self, rgb, dimg):
        global MY_GPU
        global STATION_ID
        cam_trans = self.camera.get_transform()
        #print("before")
        try:
            results = self.model.track(rgb, conf=0.1, classes=[0,1,2,3,5,7,9,10,11,12], device=MY_GPU)
            time.sleep(np.random.rand())
        except Exception as e:
            print(f"Error while tracking: {e}")
        #print("after")
        res = results[0].plot()
        #img = Image.fromarray(res[..., ::-1])
        img = cv2.cvtColor(res[..., ::-1], cv2.COLOR_BGR2RGB)
        boxes = results[0].boxes.xyxy
        clss =  results[0].boxes.cls
        confs = results[0].boxes.conf
        ids = results[0].boxes.id
        if ids is None:
            ids = torch.zeros_like(clss)
        ti = time.time()
        tracklet = dict()
        for box, clase, conf, id  in zip(boxes, clss, confs, ids):
            #print(f"box is {box}")
            res = self.get_object_distance(box, cam_trans, dimg, MY_TYPE)
            if res[0] is None:
                print(f"res is {res}, not good")
                continue
            else: 
                lat, lon, disto = res[0], res[1], res[2]
                my_lat_lon = self.get_my_lat_lon()
                tracklet['timestamp'] = ti
                tracklet['station_id'] = STATION_ID
                tracklet['station_lat'] = my_lat_lon.latitude
                tracklet['station_lon'] = my_lat_lon.longitude
                tracklet['ID'] = id
                tracklet['class'] = clase
                tracklet['conf'] = conf
                tracklet['latitude'] = lat
                tracklet['longitude'] = lon
                tracklet['distance'] = disto
                global tracklets 
                tracklets.put(tracklet) # to be furhter processed and transmitted on separated thread
        return np.array(img)

    def make_camera_tracklet(self,t0,tmax=1):
        t1 = time.time()
        if np.abs(t1 - t0) < tmax:
            return t0
        global STATION_ID
        global cam_tracklets
        global MY_TYPE
        tracklet = dict()
        my_lat_lon = self.get_my_lat_lon()
        tracklet['timestamp'] = time.time()
        tracklet['station_id'] = STATION_ID
        tracklet['station_lat'] = my_lat_lon.latitude
        tracklet['station_lon'] = my_lat_lon.longitude
        tracklet['ID'] = STATION_ID
        tracklet['class'] = 0
        if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2' :
            tracklet['class'] = 2
        tracklet['conf'] = 1
        tracklet['latitude'] = my_lat_lon.latitude
        tracklet['longitude'] = my_lat_lon.longitude
        tracklet['distance'] = 0
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
        print(f"from set image h={img.height},w={img.width}, fov={img.fov}")
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
            array = np.reshape(array, (img.width, img.height, 4))
            array = array[:, :, :3]
            R = array[:,:,2]
            G = array[:,:,1]
            B = array[:,:,0]
            D = 10*(1/(256*256*256 - 1))*(R + 256*G + 256*256*B)
            print(f"D is {D.shape}")
            self.depth_image = D
     

    def render(self, image):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        if image is not None:
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
    

    def get_object_distance(self, box, trans, depth, tipo):
        print("************************************")
        print("Player 1, Greetings from get_object_distance()")
        global LAT0
        global LON0
        global MY_TYPE
        my_id = 0
        if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2':
            my_id = self.car.id
            print(f"Car id is {my_id}")
        depth = np.array(depth)
        print(f"box is {box}")
        print(f"tranas is {trans}")
        print(f"depth is {np.isnan(depth).any()}")
        print(f"tipo is {tipo}")
        camera_bp = self.camera_blueprint()
        w = camera_bp.get_attribute("image_size_x").as_int()
        h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        pos = self.camera.get_location()
        cam_lat_lon = self.map.transform_to_geolocation(pos)
        if self.depth_image is None:
            print("Depth is None...")
            return [None, None, None]
        ptc = torch.div(box[2] - box[0], 2, rounding_mode='floor') + box[0]
        ptr = torch.div(box[3] - box[1], 2, rounding_mode='floor') + box[1]
        ptc = int(ptc.item())
        ptr = int(ptr.item())
        
        actors = self.world.get_actors()
        box = box.cpu()
        box = [int(x) for x in box]
        print(f"box is {box}")
        print(f"with center at {[ptc, ptr]}")
        zbs = depth[box[0]:box[2],box[1]:box[3]]
        bct = [int((box[2]-box[0])/2), int((box[3]-box[1])/2)]
        print(f"test is {depth[800-1, 600-1]}")
        zb = np.mean(zbs)
        if np.isnan(zb):
            print("Zb is None...")
            return [None, None, None]
        print(f"zb = {zb}")
        zc = [ptc, ptr]
        K = build_projection_matrix(w,h,fov)
        #world_2_camera = trans.get_inverse_matrix()
        camera_2_world = trans.get_matrix()
        #ac_loc = actor.get_location()
        #ac_px = get_image_point(ac_loc, K, world_2_camera)
        the_loc = get_point_image(zc, K, camera_2_world, zb)
        obj_loc = carla.Location(the_loc[0], the_loc[1], the_loc[2])
        print(f"Box location in the world is {obj_loc}")
        actors_ids = []
        cloc = trans.location #self.camera.get_location()
        my_id = None
        my_actor = None
        my_hvs = np.Inf
        for actor in actors:
            ac_trans = actor.get_transform()
            ac_loc = actor.get_location()
            hvs = obj_loc.distance(ac_loc)
            if actor.id != my_id and hvs < my_hvs:
                my_actor = actor
                my_hvs = hvs
        if my_hvs > 0 and my_hvs < 4*zb:
            my_id = my_actor.id
            print(f"actor is {self.world.get_actor(my_id)}")
            my_actor = self.world.get_actor(my_id)
            my_ac_trans = my_actor.get_transform()
            print(f"err = {obj_loc.distance(my_ac_trans.location)}")
            obj_loc = my_ac_trans.location
            print(f"Actor location is {obj_loc}")
            # 541-548 al if ->>>>>>>>>>>>>><
            obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
            obj_lat_lon.latitude += LAT0
            obj_lat_lon.longitude += LON0
            #hvs = my_actor.get_transform().location.distance(self.camera.get_transform().location)
            hvs = obj_loc.distance(trans.location)

            print(f"cam loc: {trans.location}")
            print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
            print(f"location {obj_loc}")
            print(f"haversine = {hvs}, val = {zb}")
            return [obj_lat_lon.latitude, obj_lat_lon.longitude, hvs]
        else:
            print("No ID was found...")
            return [None,None,None]
        #for actor in actors:
        #    ac_trans = actor.get_transform()
        #    ac_loc = ac_trans.location
        #    forward_vec = ac_trans.get_forward_vector()
        #    ray = ac_loc - cloc
        #    disto = cloc.distance(ac_loc)
        #    doto = forward_vec.dot(ray)
        #    if doto > 1 and disto < 50:
        #        K = build_projection_matrix(w,h,fov)
        #        world_2_camera = trans.get_inverse_matrix()
        #        camera_2_world = trans.get_matrix()
        #        
        #        ac_px = get_image_point(ac_loc, K, world_2_camera)
        #        the_loc = get_point_image(zc, K, camera_2_world, doto)
        #        if ac_px[0] > min(box[0],box[2]) and ac_px[0] < max(box[0], box[2]): # or (ac_px[1] > box[1] and ac_px[1] < box[3]):
        #            #print(f"K is {K}")
        #            #print(f"w2c is {world_2_camera}")
        #            print(f"box is {box}")
        #            print(f"cmara is at {trans.location}")
        #            #print(f"but currently is {self.camera.get_transform().location}")
        #            print(f"actor location is {ac_loc}")
        #            print(f"zb = {zb}")
        #            print(f"doto = {doto}")
        #            print(f"In pixels {ac_px}")
        #            print(f"and box center in the world {the_loc}")
        #            #print("this is it")
        #            my_id = actor.id
        #            break
        #if my_id is not None:
        #    print(f"actor is {self.world.get_actor(my_id)}")
        #    my_actor = self.world.get_actor(my_id)
        #    my_ac_trans = my_actor.get_transform()
        #    print(f"err = {obj_loc.distance(my_ac_trans.location)}")
        #    obj_loc = my_ac_trans.location
        #    print(f"Actor location is {obj_loc}")
        #    # 541-548 al if ->>>>>>>>>>>>>><
        #    obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
        #    obj_lat_lon.latitude += LAT0
        #    obj_lat_lon.longitude += LON0
        #    #hvs = my_actor.get_transform().location.distance(self.camera.get_transform().location)
        #    hvs = obj_loc.distance(trans.location)
        #
        #    print(f"cam loc: {trans.location}")
        #    print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
        #    print(f"location {obj_loc}")
        #    print(f"haversine = {hvs}, val = {zb}")
        #    return [obj_lat_lon.latitude, obj_lat_lon.longitude, hvs]
        #else:
        #    print("No ID was found...")
        #    return [None,None,None]


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

        try:
            pygame.init()
            
            self.client = carla.Client(CARLAIP, 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            print(f"my type is {MY_TYPE}")
            if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2':
                self.setup_car()
            self.setup_camera(MY_TYPE)
            self.setup_depth_camera(MY_TYPE)
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
            self.set_synchronous_mode(True)
            print(f"Type is {MY_TYPE}")
            if 'car' in MY_TYPE: #MY_TYPE=='car' or MY_TYPE == 'car2':
                print("OK")
                self.car.set_autopilot(True)
                print("Autopilot set")
            ti = time.time()
            while True:
                #print("while")
                self.capture = True
                pygame_clock.tick_busy_loop(0)
                print("---------------------------------")
                #if MY_TYPE == 'car':
                #    print("---------------------------------")
                #    print(f"Lat {self.lat}, Lon={self.lon}")
                img = self.image
                dimg = self.depth_image
                if img is None or dimg is None:
                    print("one is none")
                    continue
                #print("detect")
                img = self.detect_objects(img, dimg)
                ti = self.make_camera_tracklet(ti)
                #print("render")
                self.render(img)
                pygame.display.flip()
                pygame.event.pump()
                #print(f" thick is {THE_THICK}")
                if is_sync == True:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                #print("lets go")
        except Exception as e:
                        print(f"Game Loop Exeption {e}")
                        traceback.print_exc()
        finally:
            self.camera.destroy()
            if 'car' in MY_TYPE: #MY_TYPE == 'car' or MY_TYPE == 'car2':
                self.car.destroy()
            self.gps.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
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
        global is_sync
        args = sys.argv
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
        if str(args[19]) == 'yes':
            is_sync = True
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
        graph = []
        return_tensors = []
        # Threading and MQTT
        to_6G = threading.Thread(target=carla26g)
        to_6G.start()
        carla_client.game_loop(num_classes, input_size, graph, return_tensors)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
