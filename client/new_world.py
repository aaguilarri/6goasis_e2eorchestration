#!/usr/bin/env python
import carla
import time
import sys

def main():
    maxsst = 10
    sstdt = 0.02
    tsleep = 1
    is_sync = False
    args = sys.argv
    carla_ip = str(args[2])
    maxsst = int(args[3])
    sstdt = float(args[4])
    print("Running with args:")
    for arg in args:
        print(args)
    if str(args[5]) == 'yes':
        is_sync = True
    client = carla.Client(carla_ip, 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    print(f"got world {world}")
    #print("getting maps")
    #mapas = client.get_available_maps()
    #time.sleep(3)
    #print(f"mapas are {mapas}")
    #world_str = "Town01"
    #if len(args) > 1:
    #    world_str = args[1]
    #if world_str != "Town00":
    #    print("Loading world ", world_str)
    #    world = client.load_world(world_str)
    #    print("world ", world_str, " loaded")
    settings = world.get_settings()
    #produ = settings.fixed_delta_seconds * settings.max_substeps
    #print(f"Settings applied to be {settings.max_substeps} x {settings.fixed_delta_seconds} = {produ}")
    #print("That is all!")
    print("Setting simulation parameters")
    if is_sync is True:
        print("Setting synchronous mode.")
        settings = world.get_settings()
        settings.substepping = True
        settings.max_substeps = maxsst
        settings.max_substep_delta_time = 0.05
        settings.fixed_delta_seconds = sstdt
        world.apply_settings(settings)
        print(f"Settings applied to be {maxsst} x {sstdt} = {maxsst * sstdt}")
    #time.sleep(1)
    print("That is all!")
    #while True:
    #    spectator = world.get_spectator()
    #    #self.camera.set_transform(spectator.get_transform())
    #    print("expectator is ", spectator.get_transform())
    #    #print("camera is ", self.camera.get_transform())
    #    world.tick()
    #    time.sleep(tsleep)

if __name__=='__main__':
    main()