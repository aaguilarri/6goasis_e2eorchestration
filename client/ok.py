        for actor in actors:
            forward_vec = actor.get_transform().get_forward_vector()
            ray = actor.get_transform().location - cloc
            doto = forward_vec.dot(ray)
            if doto > 1 and doto < 2*zb:
                K = build_projection_matrix(w,h,fov)
                world_2_camera = trans.get_inverse_matrix()
                camera_2_world = trans.get_matrix()
                ac_loc = actor.get_location()
                ac_px = get_image_point(ac_loc, K, world_2_camera)
                the_loc = get_point_image(zc, K, camera_2_world, zb)
                if ac_px[0] > min(box[0],box[2]) and ac_px[0] < max(box[0], box[2]): # or (ac_px[1] > box[1] and ac_px[1] < box[3]):
                    print(f"K is {K}")
                    print(f"w2c is {world_2_camera}")
                    print(f"box is {box}")
                    print(f"cmara is at {trans.location}")
                    print(f"but currently is {self.camera.get_transform().location}")
                    print(f"actor location is {ac_loc}")
        
                    print(f"In pixels {ac_px}")
                    print(f"and box center in the world {the_loc}")
                    print("this is it")
                    my_id = actor.id
                    break
        if my_id is not None:
            print(f"actor is {self.world.get_actor(my_id)}")
            my_actor = self.world.get_actor(my_id)
            act_loc = my_actor.get_transform().location
            print(f"Actor location is {act_loc}")