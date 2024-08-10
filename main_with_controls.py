import numpy as np
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import time
import random
import keyboard as kbd

from rigidbody import *
from model import *
from graphics import *
from camera import *
from terrain import *
from ui import *
from scenery_objects import *
from sound import *
from alerts import *
from weapons import *

def main():
    
    def window_resize(window, width, height):
        try:
            # glfw.get_framebuffer_size(window)
            glViewport(0, 0, width, height)
            glLoadIdentity()
            gluPerspective(fov, width/height, near_clip, far_clip)
            glTranslate(main_cam.pos[0], main_cam.pos[1], main_cam.pos[2])
            main_cam.orient = np.eye(3)
            main_cam.rotate([0, 180, 0])
        except ZeroDivisionError:
            # if the window is minimized it makes height = 0, but we don't need to update projection in that case anyway
            pass
        
    bodies = []
    untargeted_attackers = []
    missile_model = Model("missile")

    # SCENERY OBJECTS
    print("Initializing scenery objects...")
    scenery_objects = []

    # RANDOM BUILDINGS
    Nx = 100
    Nz = 100
    chance = 0.01
    building_spacing_x = 50
    building_spacing_z = 50

    building_area_corner_x = Nx / 2 * building_spacing_x
    building_area_corner_z = Nz / 2 * building_spacing_z

    buildings = []

    for idx_x in range(Nx):
        for idx_z in range(Nz):
            if random.uniform(0, 1) < chance:
                c_x = -building_area_corner_x + idx_x * building_spacing_x
                c_z = -building_area_corner_z + idx_z * building_spacing_z
                new_pos = np.array([c_x, 0, c_z])
                new_building = RandomBuilding(new_pos)
                scenery_objects.append(new_building)

    # RANDOM BUILDINGS 2
    Nx = 100
    Nz = 100
    chance = 0.01
    building_spacing_x = 50
    building_spacing_z = 50

    building_area_corner_x = Nx / 2 * building_spacing_x
    building_area_corner_z = Nz / 2 * building_spacing_z - 35000

    buildings = []

    for idx_x in range(Nx):
        for idx_z in range(Nz):
            if random.uniform(0, 1) < chance:
                c_x = -building_area_corner_x + idx_x * building_spacing_x
                c_z = -building_area_corner_z + idx_z * building_spacing_z
                new_pos = np.array([c_x, 0, c_z])
                new_building = RandomBuilding(new_pos)
                scenery_objects.append(new_building)

    # TERRAIN
    print("Initializing terrain...")
    floor = Flatland(0, Color(0.1, 0.8, 0.1))

    # MISC PHYSICS
    gravity = np.array([0.0, -9.81, 0])

    # GRAPHICS
    print("Initializing graphics (OpenGL, glfw)...")
    window_x, window_y = 1600, 900
    fov = 70
    near_clip = 0.1
    far_clip = 10e6
    
    glfw.init()
    window = glfw.create_window(window_x, window_y, "The Begum's Fortune by arda-guler", None, None)
    glfw.set_window_pos(window, 100, 100)
    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    gluPerspective(fov, window_x/window_y, near_clip, far_clip)
    glClearColor(0, 0, 0.3, 1)

    # SOUND
    print("Initializing sound (pygame.mixer)...")
    init_sound()

    # CAMERA
    cam_pos = np.array([0, 0, 0])
    cam_orient = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, -1]])
    main_cam = Camera("main_cam", cam_pos, cam_orient, True)

    glRotate(-180, 0, 1, 0)

    main_cam.move([-20000, -2000, 35000 * 0.5])
    main_cam.rotate([0, 90, 0])

    cam_lock_idx = -1

    def move_cam(movement):
        main_cam.move(movement)

    def rotate_cam(rotation):
        main_cam.rotate(rotation)

    # CAMERA CONTROLS
    cam_pitch_up = "K"
    cam_pitch_dn = "I"
    cam_yaw_left = "J"
    cam_yaw_right = "L"
    cam_roll_cw = "O"
    cam_roll_ccw = "U"

    cam_move_fwd = "W"
    cam_move_bck = "S"
    cam_move_left = "A"
    cam_move_right = "D"
    cam_move_up = "R"
    cam_move_dn = "F"

    cam_close_in = "T"
    cam_move_out = "G"

    cam_next_lock = "Z"
    cam_prev_lock = "X"
    cam_unlock = "C"

    cam_speed = 200
    cam_rot_speed = 100

    play_sfx("wind1", -1, 2, 0)
    play_sfx("rocket", -1, 3, 0)
    play_sfx("blue_feather", -1, 0, 1)

    print("Starting...")
    dt = 0
    cam_change_last_frame = False

    print("The Begum's Fortune by arda-guler, partially inspired by Jules Verne book of same (translated) title.")
    print("Music: Blue Feather by Kevin MacLeod, Creative Commons: By Attribution 4.0")
    
    while not glfw.window_should_close(window):
        t_cycle_start = time.perf_counter()
        glfw.poll_events() 

        # CONTROLS
        if kbd.is_pressed("Shift"):
            cam_speed_applied = cam_speed * 100
        else:
            cam_speed_applied = cam_speed
            
        if kbd.is_pressed(cam_move_fwd):
            move_cam([0, 0, cam_speed_applied * dt])
        if kbd.is_pressed(cam_move_bck):
            move_cam([0, 0, -cam_speed_applied * dt])
        if kbd.is_pressed(cam_move_up):
            move_cam([0, -cam_speed_applied * dt, 0])
        if kbd.is_pressed(cam_move_dn):
            move_cam([0, cam_speed_applied * dt, 0])
        if kbd.is_pressed(cam_move_right):
            move_cam([-cam_speed_applied * dt, 0, 0])
        if kbd.is_pressed(cam_move_left):
            move_cam([cam_speed_applied * dt, 0, 0])

        if kbd.is_pressed(cam_pitch_up):
            rotate_cam([cam_rot_speed * dt, 0, 0])
        if kbd.is_pressed(cam_pitch_dn):
            rotate_cam([-cam_rot_speed * dt, 0, 0])
        if kbd.is_pressed(cam_yaw_left):
            rotate_cam([0, cam_rot_speed * dt, 0])
        if kbd.is_pressed(cam_yaw_right):
            rotate_cam([0, -cam_rot_speed * dt, 0])
        if kbd.is_pressed(cam_roll_cw):
            rotate_cam([0, 0, -cam_rot_speed * dt])
        if kbd.is_pressed(cam_roll_ccw):
            rotate_cam([0, 0, cam_rot_speed * dt])

        if kbd.is_pressed(cam_unlock):
            main_cam.lock = None
            cam_lock_idx = -1

        if not cam_change_last_frame:
            if kbd.is_pressed(cam_next_lock):
                cam_lock_idx += 1
                cam_change_last_frame = True
                cam_lock_idx = min(max(-1, cam_lock_idx), len(bodies) - 1)
                try:
                    main_cam.lock_to_target(bodies[cam_lock_idx])
                except:
                    main_cam.lock = None
            elif kbd.is_pressed(cam_prev_lock):
                cam_lock_idx -= 1
                cam_change_last_frame = True
                cam_lock_idx = min(max(-1, cam_lock_idx), len(bodies) - 1)
                try:
                    main_cam.lock_to_target(bodies[cam_lock_idx])
                except:
                    main_cam.lock = None

            if cam_lock_idx == -1:
                main_cam.lock = None

        else:
            if not (kbd.is_pressed(cam_next_lock) or kbd.is_pressed(cam_prev_lock)):
                cam_change_last_frame = False

        if main_cam.lock:
            if kbd.is_pressed(cam_close_in):
                main_cam.offset_amount *= 0.95
            elif kbd.is_pressed(cam_move_out):
                main_cam.offset_amount *= 1.05

            if main_cam.offset_amount < 1:
                main_cam.offset_amount = 1

        # ATTACK MISSILE GENERATION
        missile_chance = 0.5
        if random.randint(0, 100) < missile_chance and len(bodies) < 30:
            init_pos = np.array([0.0 + random.uniform(-500, 500), 1.0, 35000.0 + random.uniform(-500, 500)])         # m
            init_vel = np.array([0 + random.uniform(-2, 2), 0, 0 + random.uniform(-2, 2)])            # m s-1
            init_accel = np.array([0, 0, 0])          # m s-2
            init_orient = [[1.0000000,  0.0000000,  0.0000000],
                           [0.0000000, -0.5885011, -0.8084964],
                           [0.0000000,  0.8084964, -0.5885011]]
            init_ang_vel = np.array([0, 0, 0])       # rad s-1
            init_ang_accel = np.array([0, 0, 0])      # rad s-2
            init_mass = 100                                # kg
            init_inertia = np.array([[500.0, 0.0, 0.0],
                                     [0.0, 500.0, 0.0],
                                     [0.0, 0.0, 1000.0]])     # kg m2
            max_thrust = 5e3                               # N
            throttle_range = [60, 100]                      # %
            throttle = 100                                  # %
            prop_mass = 40                                 # kg
            mass_flow = 3                                   # kg s-1

            Cds = np.array([0.3, 0.3, 0.05])
            Cdas = np.array([1, 1, 0.4])
            cross_sections = np.array([0.15, 0.15, 0.04])

            init_CoM = np.array([0.0, 0.0, 0.0])

            new_attacker = Missile(missile_model, init_CoM,
                         init_pos, init_vel, init_accel,
                         np.array(init_orient), init_ang_vel, init_ang_accel,
                         init_mass, init_inertia,
                         max_thrust, throttle_range, throttle,
                         prop_mass, mass_flow, Cds, Cdas, cross_sections)

            bodies.append(new_attacker)
            untargeted_attackers.append(new_attacker)

        # DEFENSE MISSILE GENERATION
        defense_missile_chance = 25
        if random.uniform(0, 100) < missile_chance and len(untargeted_attackers) > 0:
            init_pos = np.array([0.0 + random.uniform(-10, 10), 1.0, 0.0 + random.uniform(-10, 10)])         # m
            init_vel = np.array([0 + random.uniform(-2, 2), 0, 0 + random.uniform(-2, 2)])            # m s-1
            init_accel = np.array([0, 0, 0])          # m s-2
            init_orient = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            init_ang_vel = np.array([0, 0, 0])       # rad s-1
            init_ang_accel = np.array([0, 0, 0])      # rad s-2
            init_mass = 100 + random.uniform(0, 3)                            # kg
            init_inertia = np.array([[500.0, 0.0, 0.0],
                                     [0.0, 500.0, 0.0],
                                     [0.0, 0.0, 1000.0]])     # kg m2
            max_thrust = 15e3 + random.uniform(-50, 50)                               # N
            throttle_range = [60, 100]                      # %
            throttle = 100                                  # %
            prop_mass = 40 + random.uniform(-1, 3)                                 # kg
            mass_flow = 9 + random.uniform(-0.1, 0.1)                                  # kg s-1

            Cds = np.array([0.3, 0.3, 0.05])
            Cdas = np.array([1, 1, 0.4])
            cross_sections = np.array([0.15, 0.15, 0.04])

            init_CoM = np.array([0.0, 0.0, 0.0])

            new_defender = Missile(missile_model, init_CoM,
                         init_pos, init_vel, init_accel,
                         np.array(init_orient), init_ang_vel, init_ang_accel,
                         init_mass, init_inertia,
                         max_thrust, throttle_range, throttle,
                         prop_mass, mass_flow, Cds, Cdas, cross_sections)

            new_defender.target = untargeted_attackers[0]
            untargeted_attackers.remove(new_defender.target)
            bodies.append(new_defender)
            play_sfx("missile_launch", channel=random.randint(4, 7))

        # PHYSICS

        for b in bodies:
            if isinstance(b, Rocket):
                b.apply_accel(gravity)
                b.apply_drag()
                b.apply_aero_torque()
                b.drain_fuel(dt)
                b.apply_thrust()
                b.update(dt)

                if b.pos[1] < 0:
                    bodies.remove(b)
                    del b
                    play_sfx("strike", channel=random.randint(4, 7))

            if isinstance(b, Missile):
                b.update_trail()
                b.guidance(dt)
                b.apply_accel(gravity)
                b.apply_drag()
                b.apply_aero_torque()
                b.drain_fuel(dt)
                b.apply_thrust()
                b.update(dt)
                b.check_target(bodies)

                if b.pos[1] < floor.height:
                    bodies.remove(b)
                    del b
                    play_sfx("strike", channel=random.randint(4, 7))

        # hit flat ground
        for b in bodies:
            if b.pos[1] < floor.height:
                b.pos[1] = 0
                b.vel[1] = 0
                b.vel = b.vel - b.vel * 0.05 * dt

        main_cam.move_with_lock(dt)
        # main_cam.rotate_with_lock(dt)

        # GRAPHICS
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        drawScene(main_cam, floor, bodies, scenery_objects)
        
        glfw.swap_buffers(window)

        try:
            set_channel_volume(2, min(np.linalg.norm(main_cam.lock.vel) / 500, 1) * 0.5) # airflow

            if main_cam.lock.prop_mass > 0 and main_cam.lock.thrust > 0:
                set_channel_volume(3, 1)
            else:
                set_channel_volume(3, 0)
        except:
            set_channel_volume(2, 0) # airflow
            set_channel_volume(3, 0)

        dt = time.perf_counter() - t_cycle_start
        dt = min(dt, 0.04)

    glfw.destroy_window(window)
    fade_out_channel(0)
    stop_channel(1)
    stop_channel(2)
    stop_channel(3)
    stop_channel(4)
    stop_channel(5)
    stop_channel(6)
    stop_channel(7)

main()
