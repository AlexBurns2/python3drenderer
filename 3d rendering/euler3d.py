import numpy as np
import cv2
import math
import time
from itertools import product
from rendering import *
from vectormath import *
WIDTH, HEIGHT = 1920, 1080  # lower res for speed

vertices = [
    (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
    (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)
]

tris = [
    (0,1,2),(0,2,3),
    (4,5,6),(4,6,7),
    (0,1,5),(0,5,4),
    (2,3,7),(2,7,6),
    (1,2,6),(1,6,5),
    (0,3,7),(0,7,4)
]

colors = [
    (255,0,0),(255,0,0),
    (0,255,0),(0,255,0),
    (0,0,255),(0,0,255),
    (255,255,0),(255,255,0),
    (255,0,255),(255,0,255),
    (0,255,255),(0,255,255)
]

playerUp = np.array([0.0, 1.0, 0.0])
cam_pos = np.array([0.0, 0.0, 0.0])
cam_rot_x, cam_rot_y = 0.0, np.pi/2 # yaw, pitch
zoom = 200

last_mouse = None
mouse_down = False

def rotate_point(x,y,z,ax,ay):
    cosx, sinx = math.cos(ax), math.sin(ax)
    cosy, siny = math.cos(ay), math.sin(ay)

    y, z = y * cosx - z * sinx, y * sinx + z * cosx
    # rotate around Y
    x, z = x * cosy - z * siny, x * siny + z * cosy
    return np.array([x, y, z])

def world_to_view(v):
    rel = v - cam_pos
    return rel

def project_point(x,y,z):
    global cam_pos, cam_rot_x, cam_rot_y, playerUp
    fov, imagePlaneDist = 256, 1.0

    # DEFINITIONS
    origin = cam_pos # plane origin
    camvectx = math.cos(cam_rot_y) * math.sin(cam_rot_x) # convert cam rotation in rad to 3d vector
    camvecty = math.sin(cam_rot_y)
    camvectz = math.cos(cam_rot_y) * math.cos(cam_rot_x)

    camXaxis = crossProduct(camvectx, camvecty, camvectz, *playerUp) # camera X axis
    camYaxis = crossProduct(*camXaxis, camvectx, camvecty, camvectz) # camera Y axis

    dist = ((cam_pos[0] - x) ** 2 + (cam_pos[1] - y) ** 2 + (cam_pos[2] - z) ** 2) ** 0.5 # dist from cam to point

    dotprod = (camvectx * ((cam_pos[0] - x))) + (camvecty * ((cam_pos[1] - y))) + (camvectz * ((cam_pos[2] - z))) # step 1 for opp side length - dot prod to get angle between cam direction and point direction from cam
    angle = math.acos(dotprod / (vectormod(camvectx, camvecty, camvectz) * dist)) # angle between cam dir and point dir
    x = dist/math.sin(angle)

    cam2point = []
    cam2point[0], cam2point[1], cam2point[2] = (cam_pos[0] - x), (cam_pos[1] - y), (cam_pos[2] - z)
    
    projPoint = []
    projPoint[0] = ((dotProduct(cam2point, camvectx, camvecty, camvectz)) / vectormod(camvectx, camvecty, camvectz)**2)

    factor = fov / (dist + imagePlaneDist)
    return int(x * factor * zoom / 100 + WIDTH/2), int(-y * factor * zoom / 100 + HEIGHT/2), z

def backface_cull(vertices, tris, colors, camera_pos):
    unculled_tris = []
    unculled_colors = []
    for t in tris:
        v0, v1, v2 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        normal = np.cross(np.subtract(v1,v0), np.subtract(v2,v0))
        view_dir = camera_pos - v0
        if np.dot(normal, view_dir) <= 0:
            unculled_tris.append(t)
            unculled_colors.append(colors[tris.index(t)])
    return np.array(unculled_tris), np.array(unculled_colors)

def draw_scene():
    zbuf = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    transformed = [world_to_view(np.array(v)) for v in vertices]
    projected = [project_point(*v) for v in transformed]

    for tri, col in zip(backface_cull(vertices, tris, colors, cam_pos)[0], backface_cull(vertices, tris, colors, cam_pos)[1]):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = [projected[i] for i in tri]
        xmin, xmax = max(min(x0, x1, x2), 0), min(max(x0, x1, x2), WIDTH-1)
        ymin, ymax = max(min(y0, y1, y2), 0), min(max(y0, y1, y2), HEIGHT-1)
        area = (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0)
        if area == 0: 
            continue
        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):
                w0 = (x1 - x)*(y2 - y) - (y1 - y)*(x2 - x)
                w1 = (x2 - x)*(y0 - y) - (y2 - y)*(x0 - x)
                w2 = (x0 - x)*(y1 - y) - (y0 - y)*(x1 - x)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    w0, w1, w2 = w0 / area, w1 / area, w2 / area
                    z = w0*z0 + w1*z1 + w2*z2
                    if z < zbuf[y, x]:
                        zbuf[y, x] = z
                        img[y, x] = col
    return img

def handle_input(key):
    global cam_pos, cam_rot_x, cam_rot_y, zoom
    forward = np.array([math.sin(cam_rot_y), 0, math.cos(cam_rot_y)])
    right = np.array([math.cos(cam_rot_y), 0, -math.sin(cam_rot_y)])
    if key == ord('w'): cam_pos += forward * 0.2
    if key == ord('s'): cam_pos -= forward * 0.2
    if key == ord('a'): cam_pos -= right * 0.2
    if key == ord('d'): cam_pos += right * 0.2
    if key == ord(' '): cam_pos[1] += 0.2
    if key == ord('c'): cam_pos[1] -= 0.2
    if key == ord('='): zoom += 20
    if key == ord('-'): zoom = max(20, zoom - 20)
    if key == 81: cam_rot_y -= 0.05   # left
    if key == 83: cam_rot_y += 0.05   # right
    if key == 82: cam_rot_x -= 0.05   # up
    if key == 84: cam_rot_x += 0.05   # down

def on_mouse(event, x, y, flags, param):
    global last_mouse, cam_rot_x, cam_rot_y
    if last_mouse is not None:
        dx = x - last_mouse[0]
        dy = y - last_mouse[1]
        cam_rot_y -= dy * 0.005
        cam_rot_x -= dx * 0.005
        print(f"Camera Rotation: X={cam_rot_x:.2f}, Y={cam_rot_y:.2f}")
    last_mouse = (x, y)

cv2.namedWindow("CPU Renderer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CPU Renderer", WIDTH, HEIGHT)
cv2.setMouseCallback("CPU Renderer", on_mouse)

while True:
    frame = draw_scene()
    cv2.imshow("CPU Renderer", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC
        break
    handle_input(key)

cv2.destroyAllWindows()