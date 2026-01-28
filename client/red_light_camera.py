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


import cv2
import time
import numpy as np
import ultralytics
import time
import torch
from ultralytics import YOLO
#import tensorflow_yolov3.carla.utils as utils

#import tensorflow as tf
from PIL import Image


import glob
import os
import sys

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

import carla
from carla import ColorConverter as cc
import weakref
import random

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

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90
LAT0 = 0.
LON0 = 0.
CARLAIP = "127.0.0.1"
BB_COLOR = (248, 64, 24)
cam_x = 0. 
cam_y = 0.
cam_z = 0.
cam_r = 0.
cam_p = 0.
cam_w = 0.

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
        self.model = YOLO("yolov8n.pt") 

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp
    
    def camera_depth_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        global cam_x
        global cam_y
        global cam_z
        global cam_r
        global cam_p
        global cam_w
        print("camera x = ", cam_x)
        print("camera y = ", cam_y)
        print("camera z = ", cam_z)
        print("camera r = ", cam_r)
        print("camera p = ", cam_p)
        print("camera y = ", cam_w)
        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        #First person view transform settings
        #expectator is  Transform(Location(x=-106.573982, y=21.803450, z=6.657751), Rotation(pitch=-25.463066, yaw=89.280281, roll=0.000021))
        #camera is  Transform(Location(x=1.600000, y=0.000000, z=1.700000), Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))  
        #camera_transform = carla.Transform(carla.Location(x=-106.573982, y=21.803450, z=6.657751), carla.Rotation(pitch=-25.463066, yaw=89.280281, roll=0.000021)) #crossroad
        #camera_transform = carla.Transform(carla.Location(x=-55.037571, y=144.485336, z=6.085244), carla.Rotation(pitch=-27.175135, yaw=-73.018288, roll=0.000021)) # near brigde 
        camera_transform = carla.Transform(carla.Location(x=cam_x, y=cam_y, z=cam_z), carla.Rotation(pitch=cam_p, yaw=cam_w, roll=cam_r))
        #camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform)
        spectator = self.world.get_spectator()
        #self.camera.set_transform(spectator.get_transform())
        #print("spectator is ", spectator.get_transform())
        #self.camera.set_transform(camera_transform)
        #print("The transform is ", camera_transform)
        #print("camera is ", self.camera.get_transform())
        #print(self.camera)
        #print("is everything ok?")
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        
    def setup_depth_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = self.camera.get_transform()
        camera_transform.location.y += 0.1
        self.depth_camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform)
        spectator = self.world.get_spectator()
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda image: weak_self().set_depth_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

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
        if self.capture:
            self.image = img
            self.capture = False

    @staticmethod
    def set_depth_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if img is not None:
            img.convert(cc.Depth)
            array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (img.height, img.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            R = array[:,:,0]
            G = array[:,:,1]
            B = array[:,:,2]
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            in_meters = 1000 * normalized
            self.depth_image = in_meters


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
            else:     
                array = np.array(self.raw_image)
                array = np.reshape(array, (self.image.height, self.image.width, 3))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def get_object_distance(self, box):
        print("************************************")
        #print(f"dept image is {self.depth_image}")
        global cam_x
        global cam_y
        global cam_z
        global LAT0
        global LON0
        if self.depth_image is None:
            return
        ptx = (box[2] - box[0]) // 2
        pty = (box[3] - box[1]) // 2
        ptx = int(ptx.item())
        pty = int(pty.item())
        val = self.depth_image[ptx][pty]
        my_tensor = torch.tensor([ptx, pty, 1]).to(torch.double)
        coords = val * torch.mm(self.inv_cal, my_tensor.unsqueeze(dim=0).t())
        cam_coords = self.depth_camera.get_transform().location
        obj_x = float(cam_x - coords[0])
        obj_y = float(cam_y + 0.1 - coords[1])
        obj_z = float(cam_z - coords[2])
        obj_transform = carla.Transform(carla.Location(x=obj_x, y=obj_y, z=obj_z), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        obj_lat_lon = self.map.transform_to_geolocation(obj_transform.location)
        obj_lat_lon.latitude += LAT0
        obj_lat_lon.longitude += LON0
        #print(f"ptx is {ptx}, type {type(ptx)}")
        #print(f"pty is {pty}, type {type(pty)}")
        #print(f"depth is {val}, type {type(val)}")
        #print (f"Distance is  {self.depth_image[ptx][pty]}")
        #print(f"Coordinates: {coords}")
        #print(f"Camera coordinates: {[cam_x, cam_y+0.1, cam_z]}")
        #print(f"Obj coordinates: {obj_transform.location}")
        #print(f"Obj geolocation: {obj_lat_lon}")
        #print("****************************************")
        return [obj_lat_lon.latitude, obj_lat_lon.longitude, val]

    def get_object_location(self, dist):
        pass

    def game_loop(self, num_classes, input_size, graph, return_tensors):
        """
        Main program loop.
        """
        global CARLAIP
        
        try:
            pygame.init()
            
            self.client = carla.Client(CARLAIP, 2000)
            self.client.set_timeout(15.0)
            self.world = self.client.get_world()

            #self.setup_car()
            self.setup_camera()
            self.setup_depth_camera()
            self.setup_gps(self.camera)
            self.inv_cal = torch.linalg.inv(torch.tensor(self.camera.calibration))
            self.map=self.world.get_map()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()
            self.set_synchronous_mode(True)
            t0 = time.time()
            while True:
                #self.world.tick()  
                self.world.wait_for_tick()
                self.capture = True
                pygame_clock.tick_busy_loop(20)
                self.render(self.display)
                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                frame_size = self.raw_image.shape[:2]
                #print("sahpe ", self.raw_image.shape)
                image_data = np.frombuffer(self.raw_image, dtype=np.dtype("uint8"))
                #image_data = np.reshape(image_data, (input_size, input_size, 4))
                image_data = np.reshape(image_data, (540, 960, 3))
                image_data = image_data[:, :, :3]
                image_data = image_data[:, :, ::-1]
                cuda_error = True
                t1 = time.time()
                #pectator = self.world.get_spectator()
                #print("spectator is ", spectator.get_transform())
                #self.camera.set_transform(camera_transform)
                #print("The transform is ", camera_transform)
                #print("camera is ", self.camera.get_transform())
                #print(self.camera)
                #print("is everything ok?")
                if t1 - t0 > 1:
                    self.model = YOLO("yolov8n.pt") 
                    results = self.model.track(image_data, conf=0.1)
                    print("---------------------------------")
                    print(f"Lat {self.lat}, Lon={self.lon}")
                    res = results[0].plot()
                    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(res[..., ::-1])
                    #im.show() 
                    dist = []
                    pos = []
                    boxes = results[0].boxes.xyxy
                    clss = results[0].boxes.cls
                    confs = results[0].boxes.conf
                    ids = results[0].boxes.id
                    if ids is None:
                        ids = torch.zeros_like(clss)
                    for box, clase, conf, id  in zip(boxes, clss, confs, ids):
                        #print(f"box is {box}")
                        #print(f"class is {clase}")
                        #print(f"conf is {conf}")
                        #print(f"id is {id}")
                        [lat, lon, disto] = self.get_object_distance(box)
                        
                    self.raw_image = im
                    self.render2(self.display,modo='yolo')
                    cuda_error = False     
                    t0 = t1
                #else:
                #    self.render2(self.display, modo='original')
                
                #self.raw_image = np.array(results[0].plot(), dtype=np.dtype("uint8"))

                #array = np.asarray(annotated_frame, dtype=np.dtype("uint8"))
                #self.raw_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                #self.image.raw_data = array
                #self.display = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                #image_data = utils.image_preporcess(np.copy(self.raw_image), [input_size, input_size])
                #image_data = image_data[np.newaxis, ...]
            
                #pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                #    [return_tensors[1], return_tensors[2], return_tensors[3]],
                #            feed_dict={ return_tensors[0]: image_data})
            
                #pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                #                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                #                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            
                #bboxes =  utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
                #bboxes =  utils.nms(bboxes, 0.45, method='nms')
                #utils.draw_bounding_boxes(pygame, self.display,  self.raw_image, bboxes)
                    
                pygame.display.flip()
    
                pygame.event.pump()
                  
                #if self.control(self.car):
                #    return

        finally:
            #self.set_synchronous_mode(False)
            self.camera.destroy()
            self.depth_camera.destroy()
            self.gps.destroy()
            #self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        #return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        #pb_file         = "tensorflow_yolov3\yolov3_coco.pb"
        
        ## video_path      = 0
        num_classes     = 80
        input_size      = 416
        args = sys.argv
        global cam_x
        global cam_y
        global cam_z
        global cam_r
        global cam_p
        global cam_w
        cam_x = float(args[1])
        cam_y = float(args[2])
        cam_z = float(args[3])
        cam_r = float(args[4])
        cam_p = float(args[5])
        cam_w = float(args[6])
        print("camera x = ", cam_x)
        print("camera y = ", cam_y)
        print("camera z = ", cam_z)
        print("camera r = ", cam_r)
        print("camera p = ", cam_p)
        print("camera y = ", cam_w)
        global LAT0
        global LON0
        global CARLAIP
        args = sys.argv
        LAT0 = float(args[7])
        LON0 = float(args[8])
        CARLAIP = str(args[9])
        #graph           = tf.Graph()
        
        #THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        #my_file = os.path.join(THIS_FOLDER, pb_file)
        #print("my_file:", my_file)
        
        #return_tensors  =  utils.read_pb_return_tensors(graph, my_file, return_elements)
        graph = []
        return_tensors = []
        client = BasicSynchronousClient()
        client.game_loop(num_classes, input_size, graph, return_tensors)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()