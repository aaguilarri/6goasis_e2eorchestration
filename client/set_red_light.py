import carla
import time
import sys

maxsst = 10
sstdt = 0.02
tsleep = 1

args = sys.argv
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(15.0)
world = client.get_world()
print(client.get_available_maps())
world_str = "Town01"
if len(args) > 1:
    world_str = args[1]
print("world string is ", world_str)
if world_str != "Town00":
    print("Loading world ", world_str)
    world = client.load_world(world_str)
    print("world ", world_str, " loaded")
print("Setting simulation parameters")
settings = world.get_settings()
settings.substepping = True
settings.max_substeps = maxsst
settings.max_substep_delta_time = sstdt
settings.fixed_delta_seconds = maxsst * sstdt
world.apply_settings(settings)
print(f"Settings applied to be {maxsst} x {sstdt} = {maxsst * sstdt}")
time.sleep(1)
print("That is all!")
while True:
    spectator = world.get_spectator()
    #self.camera.set_transform(spectator.get_transform())
    print("expectator is ", spectator.get_transform())
    #print("camera is ", self.camera.get_transform())
    world.tick()
    time.sleep(tsleep)