import numpy as np
import cv2
import time
from rendering import Renderer
from objects import scan_stl_folder, load_scene_from_facets, scene_facets_raw
import os
import keyboard
import sys

WIDTH = 960
HEIGHT = 540
FOV_DEGREES = 75.0
MOUSE_SENSITIVITY = 0.12
NEAR_CLIP = 0.1
MOVE_SPEED = 6
MAX_FPS = 0
STL_FOLDER = 'stl_models'
GRAVITY = 0.2
AIRRESISTANCE = 0.1

class Player:
    def __init__(self, pos, force, mass, cam):
        self.position = np.array(pos, dtype=float)
        self.force = np.array(force, dtype=float)
        self.mass = mass
        self.cam = cam

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
    scanned = scan_stl_folder(STL_FOLDER)
    facets_all = scene_facets_raw + scanned
    meshes = load_scene_from_facets(facets_all)
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
        
        player.cam.position = player.position.copy()
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
            if player.position[2] <= 0: launchPlayer(player, np.array([0.0, 0.0, 2]))
        if keyboard.is_pressed('c'):
            player.position -= np.array([0.0, 0.0, speed])
        gravity(player, elapsed)

        if frame_time_target > 0 and elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)
    cv2.destroyAllWindows()
    sys.exit()

def gravity(player, dtime):
    global GRAVITY
    acceleration = player.force * player.mass
    if player.position[2] > 0:
        force = np.array([0.0, 0.0, gravity])
        player.position -= acceleration * dtime
    if player.position[2] <= 0:
        acceleration = 0
        player.position[2] = 0

def launchPlayer(player, force):
    global AIRRESISTANCE
    if np.linalg.norm(force) > 0:
        player.position += force
        force -= AIRRESISTANCE

if __name__ == '__main__':
    run()