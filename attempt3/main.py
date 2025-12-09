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
    keep_transformed_file,
    scale_object
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
mouse_sens = 0.10
NEAR_CLIP = 0.1
move_speed = 5
max_fps = 10
OBJ_FOLDER = 'obj_models'
FDO_FOLDER = '4d_models'
GRAVITY = -7.5
AIRRESISTANCE = 0.01
opendoor = False
opendoor2 = False
doorrange = 10

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

def lerp(a, b, t):
    return a + (b - a) * t

CUBE_CENTER = np.array([0.0, 0.0, 7.0])   # <-- adjust if needed
CUBE_SIZE = 3.0
CUBE_HALF = CUBE_SIZE / 2.0
POINT_OFFSETS = {
    "point1": np.array([-1,  1,  1]),
    "point2": np.array([ 1,  1,  1]),
    "point3": np.array([ -1, -1,  1]),
    "point4": np.array([1, -1,  1]),
    "point5": np.array([-1,  1, -1]),
    "point6": np.array([ 1,  1, -1]),
    "point7": np.array([ -1, -1, -1]),
    "point8": np.array([1, -1, -1]),
}

CUBE_SIZE = 6.0
THIN = 0.1

STATE_DIMS = {
    "cube":  np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
    "line":  np.array([CUBE_SIZE, THIN, THIN]),
    "plane": np.array([CUBE_SIZE, CUBE_SIZE, THIN]),
}

def cube_to_line(t, t_prev, size=6.0, thin=0.1):
    S_prev = np.array([size,
                       lerp(size, thin, t_prev),
                       lerp(size, thin, t_prev)])
    S_curr = np.array([size,
                       lerp(size, thin, t),
                       lerp(size, thin, t)])
    return S_curr / S_prev


def line_to_plane(t, t_prev, size=6.0, thin=0.1):
    S_prev = np.array([size,
                       lerp(thin, size, t_prev),
                       thin])
    S_curr = np.array([size,
                       lerp(thin, size, t),
                       thin])
    return S_curr / S_prev


def plane_to_cube(t, t_prev, size=6.0, thin=0.1):
    S_prev = np.array([size, size,
                       lerp(thin, size, t_prev)])
    S_curr = np.array([size, size,
                       lerp(thin, size, t)])
    return S_curr / S_prev

def cube_absolute_half(stage, t, size=6.0, thin=0.1):
    if stage == 1:  # cube → line
        dims = np.array([
            size,
            lerp(size, thin, t),
            lerp(size, thin, t)
        ])
    elif stage == 2:  # line → plane
        dims = np.array([
            size,
            lerp(thin, size, t),
            thin
        ])
    else:  # plane → cube
        dims = np.array([
            size,
            size,
            lerp(thin, size, t)
        ])

    return dims * 0.5

def run():
    global opendoor, opendoor2
    cam = Camera([0.0, 0.0, 0.0], yaw=0, pitch=0.0)
    define_cam(cam)
    player = Player([0, -5, 0], [0.0, 0.0, 0.0], 1, cam)
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
    ready = False
    scale_sequence = ["cube", "line", "plane", "cube"]
    scale_index = 0

    scale_active = False
    scale_t = 0.0
    scale_prev = 0.0
    scale_duration = 1.5

    scale_t = 0.0
    scale_prev = 0.0
    scale_duration = 1.5  # seconds
    cube_half = np.array([CUBE_HALF, CUBE_HALF, CUBE_HALF])

    point_positions = {}

    for name, offset in POINT_OFFSETS.items():
        pos = CUBE_CENTER + offset * cube_half
        point_positions[name] = pos.copy()

    while True:
        frame_start = time.time()
        now = frame_start
        dt = max(1e-6, now - last)
        last = now
        frame = renderer.clear()
        
        rotate_object_4d('hypercube', {'yw': 1})
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

        if keyboard.is_pressed('`'):
            ready = True

        if keyboard.is_pressed('q') and not scale_active:
            scale_active = True
            scale_t = 0.0
            scale_prev = 0.0

            from_state = scale_sequence[scale_index]
            scale_index = (scale_index + 1) % len(scale_sequence)
            to_state = scale_sequence[scale_index]

            scale_from = STATE_DIMS[from_state]
            scale_to   = STATE_DIMS[to_state]

            cube_half = scale_from * 0.5
            for name, offset in POINT_OFFSETS.items():
                point_positions[name] = CUBE_CENTER + offset * cube_half

            time.sleep(0.2)

        if player.position[0] > 36 - doorrange and not opendoor:
            opendoor = True
        elif player.position[0] <= 36 - doorrange and opendoor:
            opendoor = False

        if player.position[0] < -36 + doorrange and not opendoor2 and ready:
            opendoor2 = True
        elif player.position[0] >= -36 + doorrange and opendoor2 and ready:
            opendoor2 = False
        
        openDoor()
        openDoor2()

        if scale_active:
            dt_anim = (1.0 / max_fps) / scale_duration
            scale_prev = scale_t
            scale_t = min(1.0, scale_t + dt_anim)

            dims_prev = lerp(scale_from, scale_to, scale_prev)
            dims_curr = lerp(scale_from, scale_to, scale_t)

            s = dims_curr / dims_prev
            scale_object('cube', s[0], s[1], s[2])

            cube_half = dims_curr * 0.5

            for name, offset in POINT_OFFSETS.items():
                target = CUBE_CENTER + offset * cube_half
                delta = target - point_positions[name]
                translate_object(name, delta[0], delta[1], delta[2])
                point_positions[name] = target

            if scale_t >= 1.0:
                scale_active = False
        
        for name, offset in POINT_OFFSETS.items():
            target_pos = CUBE_CENTER + offset * cube_half
            delta = target_pos - point_positions[name]

            if np.any(delta != 0.0):
                translate_object(name, delta[0], delta[1], delta[2])
                point_positions[name] = target_pos

        player.cam.position = player.position.copy() + np.array([0.0, 0.0, 0])
        print(player.position)

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
 
doortimer = 0

def openDoor():
    global doortimer
    if opendoor == True:
        if doortimer < 90:
            doortimer += 1
            rotate_object('door1', -1, 0, 0, degrees=True)
            translate_object('door1', 0, 0.08, -0.08)
            rotate_object('door2', -1, 0, 0, degrees=True)
            translate_object('door2', 0, 0.08, 0.08)
            rotate_object('door3', -1, 0, 0, degrees=True)
            translate_object('door3', 0, -0.08, 0.08)
            rotate_object('door4', -1, 0, 0, degrees=True)
            translate_object('door4', 0, -0.08, -0.08)
    else:
        if doortimer > 0:
            doortimer -= 1
            rotate_object('door1', 1, 0, 0, degrees=True)
            translate_object('door1', 0, -0.08, 0.08)
            rotate_object('door2', 1, 0, 0, degrees=True)
            translate_object('door2', 0, -0.08, -0.08)
            rotate_object('door3', 1, 0, 0, degrees=True)
            translate_object('door3', 0, 0.08, -0.08)
            rotate_object('door4', 1, 0, 0, degrees=True)
            translate_object('door4', 0, 0.08, 0.08)

doortimer2 = 0
def openDoor2():
    global doortimer2
    if opendoor2 == True:
        if doortimer2 < 90:
            doortimer2 += 1
            rotate_object('door1 copy', -1, 0, 0, degrees=True)
            translate_object('door1 copy', 0, 0.08, -0.08)
            rotate_object('door2 copy', -1, 0, 0, degrees=True)
            translate_object('door2 copy', 0, 0.08, 0.08)
            rotate_object('door3 copy', -1, 0, 0, degrees=True)
            translate_object('door3 copy', 0, -0.08, 0.08)
            rotate_object('door4 copy', -1, 0, 0, degrees=True)
            translate_object('door4 copy', 0, -0.08, -0.08)
    else:
        if doortimer2 > 0:
            doortimer2 -= 1
            rotate_object('door1 copy', 1, 0, 0, degrees=True)
            translate_object('door1 copy', 0, -0.08, 0.08)
            rotate_object('door2 copy', 1, 0, 0, degrees=True)
            translate_object('door2 copy', 0, -0.08, -0.08)
            rotate_object('door3 copy', 1, 0, 0, degrees=True)
            translate_object('door3 copy', 0, 0.08, -0.08)
            rotate_object('door4 copy', 1, 0, 0, degrees=True)
            translate_object('door4 copy', 0, 0.08, 0.08)



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

        if key == 27:
            return

if __name__ == '__main__':
    action = title_screen()
    if action == "play":
        run()