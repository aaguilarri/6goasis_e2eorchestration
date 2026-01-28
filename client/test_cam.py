import carla
import math
import random
import time
import queue
import numpy as np
import pygame
import cv2

def build_projection_matrix(w, h, fov):
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

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def render(image, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

    
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        display.blit(surface, (0, 0))
    
pygame.init()
client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()
pygame_clock = pygame.time.Clock()
# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()



# Set up the set of bounding boxes from the level
# We filter for traffic lights and traffic signs
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# Remember the edge pairs
edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# spawn vehicle
car_bp = world.get_blueprint_library().filter('vehicle.*')[0]
location = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(car_bp, location)

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
VIEW_WIDTH = int(camera_bp.get_attribute('image_size_x'))
VIEW_HEIGHT = int(camera_bp.get_attribute('image_size_y'))
VIEW_FOV = float(camera_bp.get_attribute('fov'))
display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
print(f"Camera properties are {VIEW_WIDTH}, {VIEW_HEIGHT}, {VIEW_FOV}")
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)



while True:
    #print(f"queue is size {image_queue.qsize()}")
    # Retrieve and reshape the image
    world.tick()
    pygame_clock.tick_busy_loop()
    image = image_queue.get()
    imh = image.height
    imw = image.width
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = np.reshape(np.copy(image.raw_data), (imh, imw, 4))
    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    
    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
    for bb in bounding_box_set:

        # Filter for distance from ego vehicle
        if bb.location.distance(vehicle.get_transform().location) < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = bb.location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                # Cycle through the vertices
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                for edge in edges:
                    # Join the vertices into edges
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
                    # Draw the edges into the camera output
                    cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

    # Now draw the image into the OpenCV display window
    render(img, display)
    world.tick()
    pygame.display.flip()
    pygame.event.pump()
    # Break the loop if the user presses the Q key


# Close the OpenCV display window when the game loop stops
#cv2.destroyAllWindows()
pygame.quit()
vehicle.destroy()
camera.destroy()
print("This is the end of the route, goodbye!")