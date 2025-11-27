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
from fdobjects import (
    scan_fdo_folder,
    load_scene_from_fdo,
    get_loaded_4meshes,
    rotate_object_4d,
    translate_object_4d,
    define_cam
)
import keyboard
import sys
import atexit

WIDTH = 1920
HEIGHT = 1080
FOV_DEGREES = 75.0
mouse_sens = 0.12
NEAR_CLIP = 0.1
move_speed = 6
max_fps = 144
OBJ_FOLDER = 'obj_models'
FDO_FOLDER = '4d_models'
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

held = False

def mouse_cb(event, x, y, flags, param):
    global mouse_prev
    global held
    if not mouse_locked:
        mouse_prev = None
        return
    if mouse_prev is None:
        mouse_prev = (x, y)
        return
    if event == cv2.EVENT_LBUTTONUP and held:
        held = False
        mouse_prev = None
    elif event == cv2.EVENT_MOUSEMOVE and held:
        px, py = mouse_prev
        dx = x - px
        dy = y - py
        cam = param
        cam.yaw += dx * mouse_sens
        cam.pitch -= dy * mouse_sens
        if cam.pitch > 89.9:
            cam.pitch = 89.9
        if cam.pitch < -89.9:
            cam.pitch = -89.9
        mouse_prev = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        held = True
        print("click")
        mouse_prev = None

def run():
    cam = Camera([0.0, -4.0, 1.2], yaw=0.0, pitch=0.0)
    define_cam(cam)
    player = Player([0.0, -4.0, 1.2], [0.0, 0.0, 0.0], 1, cam)
    renderer = Renderer(WIDTH, HEIGHT, FOV_DEGREES, NEAR_CLIP)
    load_colliders()
    scanned = scan_obj_folder(OBJ_FOLDER)
    scanned4d = scan_fdo_folder(FDO_FOLDER)
    load_scene_from_obj(scanned)
    load_scene_from_fdo(scanned4d)
    opaque = get_loaded_meshes()[0] + get_loaded_4meshes()[0]
    transparent = get_loaded_meshes()[1] + get_loaded_4meshes()[1]
    print("opaque:", len(opaque), "transparent:", len(transparent))
    renderer.init_shader_cache([tri for mesh in (opaque+transparent) for tri in mesh['tris']])
    renderer.update_shader_cache(opaque + transparent)

    for m in get_loaded_4meshes()[0]:
        print(m['name'], "verts range", np.min(m['verts_world'], axis=0), np.max(m['verts_world'], axis=0))

    cv2.namedWindow('3D', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('3D', WIDTH, HEIGHT)
    cv2.setMouseCallback('3D', mouse_cb, cam)
    last = time.time()
    last_time = time.time()
    frame_time_target = 1.0 / max_fps if max_fps > 0 else 0.0
    while True:
        frame_start = time.time()
        now = frame_start
        dt = max(1e-6, now - last)
        last = now
        frame = renderer.clear()
        
        rotate_object_4d('hypercube', {'xw': 1})
        rotate_object_4d('hypercube2', {'yw': -1})
        rotate_object_4d('hypercube2', {'yx': 1})
        rotate_object_4d('duocylinderLP', {'yw': 1})
        rotate_object('monkey', 0, 0, 1)

        renderer.update_shader_cache(opaque + transparent)

        if keyboard.is_pressed('e'):
            rotate_object_4d('hypercube', {'yw': -1})
        if keyboard.is_pressed('q'):
            rotate_object_4d('hypercube', {'yw': 1})

        if keyboard.is_pressed('up'):
            translate_object_4d('hypercube', dx=0, dy=-0.1, dz=0, dw=0)
        if keyboard.is_pressed('down'):
            translate_object_4d('hypercube', dx=0, dy=0.1, dz=0, dw=0)

        opaque = get_loaded_meshes()[0] + get_loaded_4meshes()[0]
        transparent = get_loaded_meshes()[1] + get_loaded_4meshes()[1]
        renderer.render_scene(frame, opaque, transparent, cam)
        fps = 1.0 / max(1e-6, (time.time() - last_time))
        last_time = time.time()

        speed = move_speed * dt
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
        
        player.cam.position = player.position.copy() + np.array([0.0, 0.0, 3.5])
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


_title_click = {"x": None, "y": None, "clicked": False}
_title_click = {"x": None, "y": None, "clicked": False}

def title_mouse(event, x, y, flags, param):
    global _title_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _title_click = {"x": x, "y": y, "clicked": True}

def inside_button(x, y, rect):
    (x1, y1), (x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def centered_text(frame, text, y, scale, thickness, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness)

def title_screen():
    global _title_click

    cv2.namedWindow('3D', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('3D', WIDTH, HEIGHT)
    cv2.setMouseCallback('3D', title_mouse)

    btn_w = int(WIDTH * 0.25)
    btn_h = int(HEIGHT * 0.08)
    x1 = (WIDTH // 2) - (btn_w // 2)
    x2 = x1 + btn_w

    y_start = int(HEIGHT * 0.45)
    play_btn     = ((x1, y_start),                     (x2, y_start + btn_h))
    settings_btn = ((x1, y_start + btn_h + 20),        (x2, y_start + 2*btn_h + 20))
    quit_btn     = ((x1, y_start + 2*btn_h + 40),      (x2, y_start + 3*btn_h + 40))

    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        centered_text(
            frame,
            "MUSEUM OF 4D OBJECTS",
            int(HEIGHT * 0.25),
            scale = WIDTH / 1300,
            thickness = 3,
            color = (255, 255, 255)
        )

        cv2.rectangle(frame, play_btn[0], play_btn[1], (110, 106, 1), -1)
        cv2.rectangle(frame, settings_btn[0], settings_btn[1], (79, 76, 1), -1)
        cv2.rectangle(frame, quit_btn[0], quit_btn[1], (54, 52, 0), -1)

        btn_scale = WIDTH / 2200
        btn_thick = 2

        centered_text(frame, "START",
                      play_btn[0][1] + int(btn_h * 0.65),
                      btn_scale, btn_thick, (0, 0, 0))

        centered_text(frame, "SETTINGS",
                      settings_btn[0][1] + int(btn_h * 0.65),
                      btn_scale, btn_thick, (0, 0, 0))

        centered_text(frame, "QUIT",
                      quit_btn[0][1] + int(btn_h * 0.65),
                      btn_scale, btn_thick, (0, 0, 0))

        cv2.imshow("3D", frame)
        key = cv2.waitKey(1) & 0xFF

        if _title_click["clicked"]:
            mx, my = _title_click["x"], _title_click["y"]
            _title_click["clicked"] = False

            if inside_button(mx, my, play_btn):
                return "play"
            if inside_button(mx, my, settings_btn):
                show_settings_screen()
            if inside_button(mx, my, quit_btn):
                sys.exit()

        if key == 27:
            sys.exit()

def show_settings_screen():
    global max_fps, mouse_sens, move_speed, _title_click

    cv2.setMouseCallback('3D', title_mouse)

    btn_w = int(WIDTH * 0.12)
    btn_h = int(HEIGHT * 0.06)

    center_x = WIDTH // 2

    def make_button(cx, y):
        x1 = cx - btn_w // 2
        x2 = cx + btn_w // 2
        return ((x1, y), (x2, y + btn_h))

    y_start = int(HEIGHT * 0.30)

    fps_minus = make_button(center_x - int(WIDTH*0.15), y_start)
    fps_plus  = make_button(center_x + int(WIDTH*0.15), y_start)

    speed_minus = make_button(center_x - int(WIDTH*0.15), y_start + btn_h*2)
    speed_plus  = make_button(center_x + int(WIDTH*0.15), y_start + btn_h*2)

    sens_minus = make_button(center_x - int(WIDTH*0.15), y_start + btn_h*4)
    sens_plus  = make_button(center_x + int(WIDTH*0.15), y_start + btn_h*4)

    back_btn = make_button(center_x, int(HEIGHT * 0.80))

    while True:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        centered_text(frame, "Settings", int(HEIGHT*0.18),
                      WIDTH/1300, 3, (255,255,255))

        font_scale = WIDTH / 2700

        centered_text(frame, f"FPS Cap: {max_fps}",
                      y_start - 10, font_scale, 2, (255,255,255))
        centered_text(frame, f"Move Speed: {move_speed}",
                      y_start + btn_h*2 - 10, font_scale, 2, (255,255,255))
        centered_text(frame, f"Sensitivity: {mouse_sens:.2f}",
                      y_start + btn_h*4 - 10, font_scale, 2, (255,255,255))

        for rect, color, label in [
            (fps_minus,    (9, 9, 102), "-"),
            (fps_plus,     (0, 92, 21), "+"),
            (speed_minus,  (9, 9, 102), "-"),
            (speed_plus,   (0, 92, 21), "+"),
            (sens_minus,   (9, 9, 102), "-"),
            (sens_plus,    (0, 92, 21), "+"),
        ]:
            cv2.rectangle(frame, rect[0], rect[1], color, -1)

            cx = (rect[0][0] + rect[1][0]) // 2
            cy = (rect[0][1] + rect[1][1]) // 2

            cv2.putText(frame, label, (cx - 15, cy + int(btn_h*0.25)),
                cv2.FONT_HERSHEY_DUPLEX, font_scale*2.0,
                (0,0,0), 3, cv2.LINE_AA)


        cv2.imshow("3D", frame)
        key = cv2.waitKey(1) & 0xFF

        if _title_click["clicked"]:
            mx, my = _title_click["x"], _title_click["y"]
            _title_click["clicked"] = False

            if inside_button(mx, my, fps_minus): 
                max_fps = max(10, max_fps - 10)

            if inside_button(mx, my, fps_plus):
                max_fps = min(1000, max_fps + 10)

            if inside_button(mx, my, speed_minus):
                move_speed = max(0.1, move_speed - 0.5)

            if inside_button(mx, my, speed_plus):
                move_speed = min(100, move_speed + 0.5)

            if inside_button(mx, my, sens_minus):
                mouse_sens = max(0.01, mouse_sens - 0.01)

            if inside_button(mx, my, sens_plus):
                mouse_sens = min(2.0, mouse_sens + 0.01)

            if inside_button(mx, my, back_btn):
                return

        # ESC returns to title
        if key == 27:
            return

        
if __name__ == '__main__':
    action = title_screen()
    if action == "play":
        run()