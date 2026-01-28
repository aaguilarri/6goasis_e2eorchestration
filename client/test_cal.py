import math
import random
import time
import queue
import numpy as np

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = focal
    #focal = h / (2.0 * np.tan(fov * np.pi / 360.0))
    K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        pt_camera = [point_camera[0], point_camera[2], -point_camera[1]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, pt_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

K = build_projection_matrix(800,600,90)
print(f"K= {K}")
pt = np.array([7.3957, 2.6304, -1.7017,1])
print(f"pt is {pt}")
ptt = np.array([pt[1], -pt[2], pt[0]])
print(f"pt is {ptt}")
px = np.dot(K,ptt)
print(f"px = {px}")
px /= px[2]
print(600 - px[1])
print(f"px ={px}")
box = [528, 229, 546, 280]
print(f"box ={box}")
box = [528, 600-229, 546, 600-280]
print(f"box ={box}")

