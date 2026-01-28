import carla

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
spectator = world.get_spectator()
actors = world.get_actors()
for actor in actors:
    if 'traffic_light' in actor.type_id:
        loc = actor.get_transform().location
        rot = actor.get_transform().rotation
        print(actor.id)
        print(loc)
        print(rot)
        print("*")