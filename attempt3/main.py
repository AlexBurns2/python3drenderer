import numpy as np
import cv2
import time
from rendering import Renderer
from objects import scan_stl_folder, load_scene_from_facets, scene_facets_raw
import os
import keyboard

WIDTH = 600
HEIGHT = 400
FOV_DEGREES = 75.0
MOUSE_SENSITIVITY = 0.12
NEAR_CLIP = 0.1
MOVE_SPEED = 1
MAX_FPS = 0
STL_FOLDER = 'stl_models'
GRAVITY = 0.2
AIRRESISTANCE = 0.1


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
    #with Listener(on_press=on_press, on_release=on_release) as listener:
    #    listener.join()
    
    cam = Camera([0.0, -4.0, 1.2], yaw=0.0, pitch=0.0)
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
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('3D', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('m'):
            toggle_mouse_lock()
        speed = MOVE_SPEED * dt
        fwd = cam.forward()
        rgt = cam.right()
        
        if keyboard.is_pressed('w'):
            cam.position += fwd * speed
        if keyboard.is_pressed('s'):
            cam.position -= fwd * speed
        if keyboard.is_pressed('a'):
            cam.position -= rgt * speed
        if keyboard.is_pressed('d'):
            cam.position += rgt * speed
        
        if key == ord(' '):
            if cam.position[2] <= 0: launchPlayer(cam, np.array([0.0, 0.0, 2]))
        if key == ord('c'):
            cam.position -= np.array([0.0, 0.0, speed])
        gravity(cam)
        frame_end = time.time()
        elapsed = frame_end - frame_start
        if frame_time_target > 0 and elapsed < frame_time_target:
            time.sleep(frame_time_target - elapsed)
    cv2.destroyAllWindows()

def gravity(cam):
    global GRAVITY
    acceleration = 0
    if cam.position[2] > 0:
        acceleration += GRAVITY
        cam.position -= np.array([0.0, 0.0, acceleration])
    if cam.position[2] <= 0:
        acceleration = 0
        cam.position[2] = 0

def launchPlayer(cam, force):
    global AIRRESISTANCE
    if np.linalg.norm(force) > 0:
        cam.position += force
        force -= AIRRESISTANCE

if __name__ == '__main__':
    run()