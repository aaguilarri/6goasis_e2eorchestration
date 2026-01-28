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
import paho.mqtt.client as mqtt
from PIL import Image
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
STATION_ID='8855'
THE_THICK = False
BB_COLOR = (248, 64, 24)
tracklets = []



###### MQTT Client
mqtt_messages = []
MQTT_TOPIC = "tracklets"
def on_connect(client, userdata, flags, rc):
    #    global loop_flat
    print('Connected to MQTT broker')
#    loop_flag=0


def on_subscribe(mosq, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))


def mqtt_callback(client, user_data, message):
    global mqtt_messages
    mqtt_messages.append(message)
    mqtt_messages

###### Thread function
def save_to_file(data, filename):
    try:
        with open(filename, 'a') as file:  # Open file in append mode
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
    filename = 'oasis.txt'
    print("Greetings from carla thread!")
    yolo_dict = get_classes()
    global MQTT_TOPIC
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = mqtt_callback
    client.on_subscribe = on_subscribe
    ip = "127.0.0.1"  # "localhost" #"192.168.1.156"  #"10.1.2.212"  # "192.168.55.1"  #
    broker_url = ip  # "localhost"
    broker_port = 1884
    my_topic = MQTT_TOPIC
    my_user = "carla_client"
    my_pwd = "cttc"
    client.username_pw_set(my_user, my_pwd)
    print('connecting to ', broker_url, ' port ', broker_port)
    client.connect(broker_url, port=broker_port, keepalive=600)
    print('Connected')
    client.loop_start()
    while True:
        global tracklets
        if len(tracklets) > 0:
            print(f"Current tracklets list is length {len(tracklets)}")   
            a_tracklet = tracklets.pop()
            #Transmission code
            my_tracklet = make_tracklet(a_tracklet, yolo_dict)
            my_tracklet_j = json.dumps(my_tracklet, indent=4)
            #save_to_file(my_tracklet_j, filename)
            print(f"Tracklet to transmit is {my_tracklet_j}")
            try:
                client.publish(MQTT_TOPIC, my_tracklet_j)
                print("Tracklet transmitted succesfully.")
            except:
                print("Tracklet transmission failed...")
            print("See you later!")
        time.sleep(0.5)
    client.loop.stop()

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
        print(f"point camera {point_camera}")
        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        print(f"now point_ camera {point_camera}")
        # now project 3D->2D using the camera matrix
        print(f"K={K}")
        point_img = np.dot(K, point_camera)
        print(f"point imaga {point_img}")
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
        print(f"finally point imaga {point_img}")
        return point_img[0:2]
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
        if tipo == 'car':
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
        w = camera_bp.get_attribute("image_size_x").as_int()
        h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        f = w / (2.0 * np.tan(fov * np.pi / 360.0))
        cx= w /2
        cy = h / 2
        calibration = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
        #calibration = np.identity(3)
        #calibration[0, 2] = VIEW_WIDTH / 2.0
        #calibration[1, 2] = VIEW_HEIGHT / 2.0
        #calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        
    def setup_depth_camera(self, tipo):
        camera_transform = None
        if tipo == 'car':
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

        #f = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        #cx= VIEW_WIDTH /2
        #cy = VIEW_HEIGHT / 2
        #calibration = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
        #calibration = np.identity(3)
        #calibration[0, 2] = VIEW_WIDTH / 2.0
        #calibration[1, 2] = VIEW_HEIGHT / 2.0
        #calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        #self.camera.calibration = calibration
    
    

    def setup_gps(self, parento):
        self.lat = 0.0
        self.lon = 0.0
        bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.gps = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=0., z=0.1)), attach_to=parento)
        weak_self = weakref.ref(self)
        self.gps.listen(lambda event: weak_self().on_gnss_event(weak_self, event))
    
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
    

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

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
            #print(f"image is {type(img)}")
            raw_img = np.array(img.raw_data)
            image_data = np.frombuffer(raw_img, dtype=np.dtype("uint8"))
            image_data = np.reshape(image_data, (img.height, img.width, 4))
            image_data = image_data[:, :, :3]
            image_data = image_data[:, :, ::-1]
            img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            #print(f"image size is {img.shape}")
            #print("lets try yolo")
            if use_yolo:
                #print("1")
                my_trans = self.camera.get_transform()
                results = self.model.track(image_data, conf=0.1, classes=[0,1,2,3,5,7,9,10,11,12], device=MY_GPU)
                #print("2")
                #print(results)
                res = results[0].plot()
                #print("3")
                img = Image.fromarray(res[..., ::-1])
                #print("4")
                print(f"yolo image size is {img.width}, {img.height}")
                self.boxes.put([results, my_trans])
                #print("5")
                #if len(self.boxes) > 10:
                #    self.boxes.pop(0) # discards oldest
                #print("6")
            self.image = img
            self.raw_image = img
            self.capture = False
            #print("ok")

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
            B = array[:, :, 0].astype(np.float32)
            G = array[:, :, 1].astype(np.float32)
            R = array[:, :, 2].astype(np.float32)
            #R = array[:,:,2]
            #G = array[:,:,1]
            #B = array[:,:,0]
            D = 10*(1/(256*256*256 - 1))*(R + 256*G + 256*256*B)
            print(f"D is {D.shape}")
            self.depth_image = D
     

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
    
    def render2(self, display,modo='original'):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            if modo == 'original':
                array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (self.image.height, self.image.width, 4))
            #else:     
            #    array = np.array(self.raw_image)
            #    array = np.reshape(array, (self.image.height, self.image.width, 3))
            #array = array[:, :, :3]
            #array = array[:, :, ::-1]
            #self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            array = self.raw_image
            array = np.array(array)
            array = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def get_object_distance(self, box, trans, tipo):
        print("************************************")
        print("Player 1, Greetings from get_object_distance()")
        global LAT0
        global LON0
        camera_bp = self.camera_blueprint()
        w = camera_bp.get_attribute("image_size_x").as_int()
        h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        pos = self.camera.get_location()
        cam_lat_lon = self.map.transform_to_geolocation(pos)
        #print(f"box is {box}")
        if self.depth_image is None:
            return
        ptc = torch.div(box[2] - box[0], 2, rounding_mode='floor') + box[0]
        ptr = torch.div(box[3] - box[1], 2, rounding_mode='floor') + box[1]
        ptc = int(ptc.item())
        ptr = int(ptr.item())
        depth = self.depth_image
        depth = np.array(depth)
        actors = self.world.get_actors()
        box = box.cpu()
        box = [int(x) for x in box]
        zbs = depth[box[1]:box[3],box[0]:box[2]]
        #print(vals)
        zb = np.mean(zbs)
        #print(f"zb = {zb}")
        # Iterate over the actors to find traffic lights
        actors_ids = []
        cloc = self.camera.get_location()
        my_id = None
        for actor in actors:
            #print(f"possibly is actor: {actor.type_id}")
            #print(f"distance: {actor.get_location().distance(cloc)}")
            forward_vec = actor.get_transform().get_forward_vector()
            ray = actor.get_transform().location - cloc
            doto = forward_vec.dot(ray)
            #print(f"doto is {doto}")
            #print(".")
            if doto > 1 and doto < 2*zb:
                #print(f"possibly is actor: {actor.type_id}")
                #print(f"distance: {actor.get_location().distance(cloc)}")
                #print(f"doto is {doto}")
                K = build_projection_matrix(w,h,fov)
                world_2_camera = trans.get_inverse_matrix()
                ac_loc = actor.get_location()
                #print(f"loc is: {ac_loc}")
                ac_px = get_image_point(ac_loc, K, world_2_camera)
                #print(f"in pixels is {ac_px}")
                #print(f"and box is {box}")
                if ac_px[0] > min(box[0],box[2]) and ac_px[0] < max(box[0], box[2]): # or (ac_px[1] > box[1] and ac_px[1] < box[3]):
                    print(f"K is {K}")
                    print(f"w2c is {world_2_camera}")
                    print(f"box is {box}")
                    print(f"cmara is at {trans.location}")
                    print(f"but currently is {self.camera.get_transform().location}")
                    print(f"actor location is {ac_loc}")
                    print(f"In pixels {ac_px}")
                    print("this is it")
                    my_id = actor.id
                    break
        if my_id is not None:
            print(f"actor is {self.world.get_actor(my_id)}")
            my_actor = self.world.get_actor(my_id)
            obj_loc = my_actor.get_transform().location
            obj_lat_lon = self.map.transform_to_geolocation(obj_loc)
            obj_lat_lon.latitude += LAT0
            obj_lat_lon.longitude += LON0
            hvs = my_actor.get_transform().location.distance(self.camera.get_transform().location)
            print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
            print(f"haversine = {hvs}, val = {zb}")
            return [obj_lat_lon.latitude, obj_lat_lon.longitude, hvs]
                #traffic_lights_ids.append(actor.id)
        else:
            return [None,None,None]




        my_light = self.world.get_actor(traffic_lights_ids[0])
        pos = my_light.get_location()
        print(f"light is {my_light}")
        pt = get_image_point(pos,K,world_2_camera)
        print(f"pt is {pt}")

        # Print the IDs of all traffic lights
        print("IDs of all traffic lights:", traffic_lights_ids)
        #vals = 
        box = np.array(box.cpu())
        box = [int(x) for x in box]
        print(f"box is {box}")
        vals = depth[box[1]:box[3],box[0]:box[2]]
        #print(vals)
        val = np.mean(vals)
        print(val)
        inv_cal = self.inv_cal
        my_tensor = torch.tensor([[ptx], [pty], [1.]]).to(torch.double)
        cam_t = self.camera.get_transform()
        trans = cam_t.rotation
        loc = cam_t.location
        Rm = self.euler_to_rotation_matrix(trans.roll, trans.pitch, trans.yaw)
        C = torch.tensor([[loc.x], [loc.y], [loc.z]])
        xyz0 = torch.mm(inv_cal, my_tensor)
        print(f"normalized coords {xyz0}")
        zxy = torch.tensor([[xyz0[2]],[-xyz0[0]],[xyz0[1]]])
        print(f"derotated coords {zxy}")
        xyzr = torch.mm(Rm.t(), zxy)
        print(f"rotated coords {xyzr}")
        xyzb = val * xyzr
        print(f"scaled coords {xyzb}")
        xyz = xyzb + C
        print(f"final coords {xyz}")
        rx = xyz.cpu()
        print(f"camera coords: {cam_t.location}")
        obj_transform = carla.Transform(carla.Location(x=float(rx[0]), y=float(rx[1]), z=float(rx[2])), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        obj_lat_lon = self.map.transform_to_geolocation(obj_transform.location)
        obj_lat_lon.latitude += LAT0
        obj_lat_lon.longitude += LON0
        hvs = self.haversine(cam_lat_lon.latitude, cam_lat_lon.longitude, obj_lat_lon.latitude, obj_lat_lon.longitude)
        print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
        print(f"haversine = {hvs}, val = {val}")
        #print(my_tensor)
        #print(f"inv cal is {self.inv_cal.device}")
        #print(f"my_tensor is {my_tensor.device}")
        #coords = val * torch.mm(self.inv_cal, my_tensor.unsqueeze(dim=0).t())
        #cam_t = self.camera.get_transform()
        #loc = cam_t.location
        #trans = cam_t.rotation
        #Em = self.euler_to_rotation_matrix(trans.roll, trans.pitch, trans.yaw)
        #cp = torch.mm(Em.to(float),coords.to(float))
        #fv = trans.get_forward_vector()
        #rv = trans.get_right_vector()
        #uv = trans.get_up_vector()
        #print(f"fv is {fv}")
        #print(f"rv is {rv}")
        #print(f"uv is {uv}")
        #print(f"camara loc is {loc}")
        #cam_lat_lon = self.map.transform_to_geolocation(loc)
        #cam_lat_lon.latitude += LAT0
        #cam_lat_lon.longitude += LON0
        #print(f"equivalent to {cam_lat_lon.latitude}, {cam_lat_lon.longitude}")
        #print(f"from gps: {self.lat}, {self.lon}")
        #print(f"coords are: {coords}")
        #fvx2 = fv.x * fv.x
        #fvy2 = fv.y * fv.y
        #fvz2 = fv.z * fv.z
        #rvx2 = rv.x * rv.x
        #rvy2 = rv.y * rv.y
        #rvz2 = rv.z * rv.z
        #uvx2 = uv.x * uv.x
        #uvy2 = uv.y * uv.y
        #uvz2 = uv.z * uv.z
        #cam_coords = self.depth_camera.get_transform().location
        #obj_x = float(loc.x + coords[2]*fvx2 + coords[0]*rvx2 + coords[1]*uvx2) 
        #obj_y = float(loc.y + coords[2]*fvy2 + coords[0]*rvy2 + coords[1]*uvy2)
        #obj_z = float(loc.z + coords[2]*fvz2 + coords[0]*rvz2 + coords[1]*uvz2)
        #obj_x = float(loc.x + val*fvx2 + val*rvx2 + val*uvx2) 
        #obj_y = float(loc.y + val*fvy2 + val*rvy2 + val*uvy2)
        #obj_z = float(loc.z + val*fvz2 + val*rvz2 + val*uvz2)
        #print(f"obj loc is {obj_x},{obj_y},{obj_z}")
        #obj_transform = carla.Transform(carla.Location(x=obj_x, y=obj_y, z=obj_z), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        #obj_lat_lon = self.map.transform_to_geolocation(obj_transform.location)
        #obj_lat_lon.latitude += LAT0
        #obj_lat_lon.longitude += LON0
        #hvs = self.haversine(cam_lat_lon.latitude, cam_lat_lon.longitude, obj_lat_lon.latitude, obj_lat_lon.longitude)
        #print(f"object: {obj_lat_lon.latitude}, {obj_lat_lon.longitude}")
        #print(f"haversine = {hvs}, val = {val}")
        #return [self.lat, self.lon, 0.]
        return [obj_lat_lon.latitude, obj_lat_lon.longitude, val]

    def rotation_matrix_z(self, psi):
        """
        Create a 3D rotation matrix for rotation about the z-axis (yaw).
        """
        c = torch.cos(psi)
        s = torch.sin(psi)
        return torch.tensor([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]], dtype=torch.double)

    def rotation_matrix_y(self, theta):
        """
        Create a 3D rotation matrix for rotation about the y-axis (pitch).
        """
        c = torch.cos(theta)
        s = torch.sin(theta)
        return torch.tensor([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]], dtype=torch.double)

    def rotation_matrix_x(self, phi):
        """
        Create a 3D rotation matrix for rotation about the x-axis (roll).
        """
        c = torch.cos(phi)
        s = torch.sin(phi)
        return torch.tensor([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]], dtype=torch.double)

    def euler_to_rotation_matrix(self, phi, theta, psi):
        """
        Create a 3D rotation matrix from roll (phi), pitch (theta), and yaw (psi) angles.
        """
        f = 2*math.pi
        print("rx")
        R_x = self.rotation_matrix_x(torch.tensor(phi/f))
        print("ry")
        R_y = self.rotation_matrix_y(torch.tensor(theta/f))
        print("rz")
        R_z = self.rotation_matrix_z(torch.tensor(psi/f))
        return torch.matmul(R_z, torch.matmul(R_y, R_x))  # Multiply in the order of z, y, x

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

        try:
            pygame.init()
            
            self.client = carla.Client(CARLAIP, 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()
            if MY_TYPE == 'car':
                self.setup_car()
            self.setup_camera(MY_TYPE)
            self.setup_depth_camera(MY_TYPE)
            self.setup_gps(self.camera)
            self.inv_cal = torch.linalg.inv(torch.tensor(self.camera.calibration))
            self.map=self.world.get_map()
            camera_bp = self.camera_blueprint()
            w = camera_bp.get_attribute("image_size_x").as_int()
            h = camera_bp.get_attribute("image_size_y").as_int()
            fov = camera_bp.get_attribute("fov").as_float()
            print(f"w={w}, h={h}")
            self.display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
            #self.display2 = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()
            self.set_synchronous_mode(True)
            if MY_TYPE=='car':
                self.car.set_autopilot(True)
            t0 = time.time()
            while True:
                self.capture = True
                pygame_clock.tick_busy_loop(1)
                #image_data = self.image
                #if self.image is not None:
                #    print(f"image data is {image_data.shape}")
                #    print(f"now image data is {image_data.shape}")
                print("---------------------------------")
                #print(f"Lat {self.lat}, Lon={self.lon}")
                t1 = time.time()
                if t1 - t0 > 1:
                    
                    try:
                        #results = self.model.track(image_data, conf=0.1, classes=[0,1,2,3,5,7,9,10,11,12], device=MY_GPU)
                        if MY_TYPE == 'car':
                            print("---------------------------------")
                            print(f"Lat {self.lat}, Lon={self.lon}")
                        dist = []
                        pos = []
                        if self.boxes.qsize() > 0:
                            pack = self.boxes.get()
                            detection = pack[0]
                            cam_trans = pack[1]
                            boxes = detection[0].boxes.xyxy
                            clss = detection[0].boxes.cls
                            confs = detection[0].boxes.conf
                            ids = detection[0].boxes.id
                            if ids is None:
                                ids = torch.zeros_like(clss)
                            ti = time.time()
                            tracklet = dict()
                            for box, clase, conf, id  in zip(boxes, clss, confs, ids):
                                res = self.get_object_distance(box, cam_trans, MY_TYPE)
                                if res[0] is None:
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
                                tracklets.append(tracklet) # to be furhter processed and transmitted on separated thread
                        self.render2(self.display,modo='yolo')
                    except Exception as e:
                        print(f"CUDA Exeption {e}")
                        traceback.print_exc()
                    t0 = t1

                pygame.display.flip()
                pygame.event.pump()
                print(f" thick is {THE_THICK}")
                if THE_THICK == True:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                if MY_TYPE == 'car' and self.control(self.car):
                    return

        finally:
            self.camera.destroy()
            if MY_TYPE == 'car':
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
        global cam_x
        global cam_y
        global cam_z
        global cam_r
        global cam_p
        global cam_w
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
