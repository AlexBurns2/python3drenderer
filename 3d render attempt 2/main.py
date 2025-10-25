import numpy as np
import cv2
import math
import time
from mathfunctions import *

WIDTH, HEIGHT = 1280, 720 
FPS_SLEEP = 0.001 
FOV_DEGREES = 80.0 
NEAR_CLIP = 0.01
ZOOM = 1.0
LIGHT_DIR = np.array([0.5, 1.0, -0.7], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / (np.linalg.norm(LIGHT_DIR) + 1e-12)

vertices = np.array([
    (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
    (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)
], dtype=np.float32)

tris = np.array([
    (0,1,2),(0,2,3),
    (4,5,6),(4,6,7),
    (0,1,5),(0,5,4),
    (2,3,7),(2,7,6),
    (1,2,6),(1,6,5),
    (0,3,7),(0,7,4)
], dtype=np.int32)

colors = np.array([
    (255,0,0),(255,0,0),
    (0,255,0),(0,255,0),
    (0,0,255),(0,0,255),
    (255,255,0),(255,255,0),
    (255,0,255),(255,0,255),
    (0,255,255),(0,255,255)
], dtype=np.uint8)

playerUp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cam_pos = np.array([0.0, 0.0, -6.0], dtype=np.float32)   # moved back to see object
yaw = 0.0    # rotation around global Z (left/right)
pitch = math.pi/2  # rotation around local X (up/down)

mouse_last = None
mouse_sensitivity = 0.003

move_speed = 0.25

fov_rad = math.radians(FOV_DEGREES)
focal_px = (HEIGHT / 2.0) / math.tan(fov_rad / 2.0)

def camera_axes(yaw, pitch, playerUp):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    forward = np.array([sy * cp, cy * cp, sp], dtype=np.float32)
    forward = normalize(*forward)
    right = normalize(*crossProduct(*forward, *playerUp))
    up = normalize(*crossProduct(*right, *forward))

    return right, up, forward

def view_matrix(cam_pos, yaw, pitch, playerUp):
    right, up, forward = camera_axes(yaw, pitch, playerUp)
    R = np.stack((right, up, forward), axis=1)
    invR = R.T
    invT = -invR @ cam_pos
    viewMat = np.eye(4, dtype=np.float32)
    viewMat[:3, :3] = invR
    viewMat[:3, 3] = invT
    return viewMat, right, up, forward

def perspective_matrix(focal_px, width, height):
    global WIDTH, HEIGHT
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    perspMat = np.zeros([
        [focal_px, 0.0,      0.0,      0.0],
        [0.0,      focal_px, 0.0,      0.0],
        [0.0,      0.0,      1.0,      0.0],
        [0.0,      0.0,      0.0,      1.0]
    ], dtype=np.float32)
    return perspMat, cx, cy

def project_point(vertices, cam_pos, yaw, pitch, focalpx, playerUp):
    global WIDTH, HEIGHT
    viewMat, right, up, forward = view_matrix(cam_pos, yaw, pitch, playerUp)
    perspMat, cx, cy = perspective_matrix(focalpx, WIDTH, HEIGHT)

    verts = np.hstack([perspective_matrix(focalpx, WIDTH, HEIGHT)])
    cameraSpace = verts @ viewMat.T
    x_cam, y_cam, z_cam = cameraSpace[:, 0], cameraSpace[:, 1], cameraSpace[:, 2]
    depth = z_cam
    with np.errstate(divide='ignore', invalid='ignore'):
        x_proj = (focalpx * (x_cam / y_cam)) + cx
        y_proj = (focalpx * (z_cam / y_cam)) + cy
    
    valid = y_cam > NEAR_CLIP
    x_proj = np.where(valid, x_proj, np.nan)
    y_proj = np.where(valid, y_proj, np.nan)
    depth = np.where(valid, depth, np.nan)

    return np.column_stack((x_proj, y_proj, depth))

def zbuffer_render(vertices, tris, colors, cam_pos, yaw, pitch, playerUp):
    projected = project_point(vertices, cam_pos, yaw, pitch, focal_px, playerUp)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    zbuffer = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)

    for i, tri in enumerate(tris):
        col = colors[i]
        pts = projected[tri]
        if np.any(np.isnan(pts)):
            continue  # skip off-screen or invalid
        verts2d = pts[:, :2]
        depths = pts[:, 2]

        min_x = int(max(np.floor(verts2d[:,0].min()), 0))
        max_x = int(min(np.ceil(verts2d[:,0].max()), WIDTH - 1))
        min_y = int(max(np.floor(verts2d[:,1].min()), 0))
        max_y = int(min(np.ceil(verts2d[:,1].max()), HEIGHT - 1))
        if max_x < min_x or max_y < min_y:
            continue

        area = edgeFunction(verts2d[0], verts2d[1], verts2d[2])
        if abs(area) < 1e-5:
            continue

        invz = 1.0 / depths

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                w0 = edgeFunction(verts2d[1], verts2d[2], p) / area
                w1 = edgeFunction(verts2d[2], verts2d[0], p) / area
                w2 = edgeFunction(verts2d[0], verts2d[1], p) / area
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    w_interp = w0 * invz[0] + w1 * invz[1] + w2 * invz[2]
                    if w_interp <= 0: 
                        continue
                    z_p = 1.0 / w_interp
                    if z_p < zbuffer[y, x]:
                        img[y, x] = col
                        zbuffer[y, x] = z_p

    return img

def handle_input(key):
    global cam_pos, yaw, pitch, ZOOM
    forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)

    if key == ord('w'): cam_pos += forward * move_speed
    if key == ord('s'): cam_pos -= forward * move_speed
    if key == ord('a'): cam_pos -= right * move_speed
    if key == ord('d'): cam_pos += right * move_speed
    if key == ord(' '): cam_pos[1] += move_speed
    if key == ord('c'): cam_pos[1] -= move_speed
    if key == ord('=') or key == ord('+'): ZOOM *= 1.1
    if key == ord('-'): ZOOM = max(0.1, ZOOM / 1.1)

    if key == 81: yaw -= 0.05   # left
    if key == 83: yaw += 0.05   # right
    if key == 82: pitch -= 0.05 # up
    if key == 84: pitch += 0.05 # down


def on_mouse(event, x, y, flags, param):
    global mouse_last, yaw, pitch
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_last is not None:
            dx = x - mouse_last[0]
            dy = y - mouse_last[1]
            yaw -= dx * mouse_sensitivity
            pitch -= dy * mouse_sensitivity
            # clamp pitch to avoid flipping
            pitch = max(-math.pi/2 + 1e-3, min(math.pi/2 - 1e-3, pitch))
        mouse_last = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_last = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_last = None

cv2.namedWindow("cube", cv2.WINDOW_NORMAL)
cv2.resizeWindow("cube", WIDTH, HEIGHT)
cv2.setMouseCallback("cube", on_mouse)

last_time = time.time()
while True:
    t0 = time.time()
    frame = zbuffer_render(vertices, tris, colors, cam_pos, yaw, pitch, playerUp)
    # show small HUD fps
    fps = 1.0 / max(1e-6, (time.time() - last_time))
    last_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("cube", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if key != 255:
        handle_input(key)

    time.sleep(FPS_SLEEP)

cv2.destroyAllWindows()

