import numpy as np
import ctypes
import cv2
import time
from rendering import Renderer
from colliders import load_colliders, check_collision
from objects import (
    scan_obj_folder,
    load_scene_from_obj,
    get_loaded_meshes,
    scene_facets_raw,
    toggle_object,
    translate_object,
    rotate_object,
    keep_transformed_file
)
import keyboard
import sys
import atexit

WIDTH = 960
HEIGHT = 540
FOV_DEGREES = 75.0
MOUSE_SENSITIVITY = 0.12
NEAR_CLIP = 0.1
MOVE_SPEED = 2
MAX_FPS = 144
OBJ_FOLDER = 'obj_models'
GRAVITY = -20
AIRRESISTANCE = 0.01

class Player:
    def __init__(self, pos, velocity, mass, cam, grounded = True):
        self.position = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.cam = cam
        self.grounded = True

class Camera:
    def __init__(self, pos, yaw=0.0, pitch=0.0):
        self.position = np.array(pos, dtype=float)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
    def forward(self):
        y = np.radians(self.yaw)
        p = np.radians(self.pitch)
        return np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    def right(self):
        f = self.forward()
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        n = np.linalg.norm(r)
        return r / n if n != 0 else r

mouse_prev = None
mouse_locked = True

def toggle_mouse_lock():
    global mouse_locked, mouse_prev
    mouse_locked = not mouse_locked
    mouse_prev = None

def mouse_cb(event, x, y, flags, param):
    global mouse_prev
    if not mouse_locked:
        mouse_prev = None
        return
    if mouse_prev is None:
        mouse_prev = (x, y)
        return
    px, py = mouse_prev
    dx = x - px
    dy = y - py
    cam = param
    cam.yaw += dx * MOUSE_SENSITIVITY
    cam.pitch -= dy * MOUSE_SENSITIVITY
    if cam.pitch > 89.9:
        cam.pitch = 89.9
    if cam.pitch < -89.9:
        cam.pitch = -89.9
    mouse_prev = (x, y)

def run():
    cam = Camera([0.0, -4.0, 1.2], yaw=0.0, pitch=0.0)
    player = Player([0.0, -4.0, 1.2], [0.0, 0.0, 0.0], 1, cam)
    renderer = Renderer(WIDTH, HEIGHT, FOV_DEGREES, NEAR_CLIP)
    load_colliders()
    scanned = scan_obj_folder(OBJ_FOLDER)
    load_scene_from_obj(scanned)

    meshes = get_loaded_meshes()

    renderer.init_shader_cache([tri for mesh in meshes for tri in mesh['tris']])
    renderer.update_shader_cache(meshes)
    cv2.namedWindow('3D', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('3D', WIDTH, HEIGHT)
    cv2.setMouseCallback('3D', mouse_cb, cam)
    last = time.time()
    last_time = time.time()
    frame_time_target = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
    while True:
        frame_start = time.time()
        now = frame_start
        dt = max(1e-6, now - last)
        last = now
        frame = renderer.clear()
        meshes = get_loaded_meshes()
        renderer.render_scene(frame, meshes, cam)
        fps = 1.0 / max(1e-6, (time.time() - last_time))
        last_time = time.time()
        
        speed = MOVE_SPEED * dt
        fwd = cam.forward()
        rgt = cam.right()
        
        if keyboard.is_pressed('w'):
            player.position += fwd * speed
        if keyboard.is_pressed('s'):
            player.position -= fwd * speed
        if keyboard.is_pressed('a'):
            player.position -= rgt * speed
        if keyboard.is_pressed('d'):
            player.position += rgt * speed

        if keyboard.is_pressed('up'):
            rotate_object("monkey", rx=0.0, ry=1.0, rz=0.0)
            print("Up arrow pressed!")
        if keyboard.is_pressed('down'):
            rotate_object("monkey", rx=0.0, ry=-1.0, rz=0.0)
        
        player.cam.position = player.position.copy()
        check_collision(player, height=1.8, radius=0.3)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('3D', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('m'):
            toggle_mouse_lock()

        frame_end = time.time()
        elapsed = frame_end - frame_start

        if keyboard.is_pressed(' '):
            launchPlayer(player, np.array([0.0, 0.0, 6]))
        if keyboard.is_pressed('c'):
            player.position -= np.array([0.0, 0.0, speed])
        gravity(player, elapsed)

        elapsed = frame_end - frame_start
        if frame_time_target > 0 and elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)
    #cv2.destroyAllWindows()
    sys.exit()

def gravity(player, dtime):
    global GRAVITY, AIRRESISTANCE
    player.velocity[2] += GRAVITY * dtime
    player.velocity[0] *= (1 - AIRRESISTANCE * dtime)
    player.velocity[1] *= (1 - AIRRESISTANCE * dtime)
    player.position += player.velocity * dtime
    if player.position[2] <= 0:
        player.position[2] = 0
        player.velocity[2] = 0
        player.on_ground = True
    else:
        player.on_ground = False

def launchPlayer(player, jump_force):
    if player.on_ground:
        player.velocity += jump_force
        player.on_ground = False

if __name__ == '__main__':
    run()
