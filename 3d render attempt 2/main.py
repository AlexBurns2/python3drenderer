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
yaw = 0.0    # rotation around Y (left/right)
piss = math.pi/2  # rotation around X (up/down)

mouse_last = None
mouse_sensitivity = 0.003

move_speed = 0.25

fov_rad = math.radians(FOV_DEGREES)
focal_px = (HEIGHT / 2.0) / math.tan(fov_rad / 2.0)

def worldtoscreen(vertices):
    vx = vertices[:, 0].astype(np.float32)
    vy = vertices[:, 1].astype(np.float32)
    vz = vertices[:, 2].astype(np.float32)
    global cam_pos, yaw, piss, playerUp

    camRotVectorX = math.cos(yaw) * math.cos(piss)
    camRotVectorY = math.sin(yaw) * math.cos(piss)
    camRotVectorZ = math.sin(piss)

    camXaxis = crossProduct(camRotVectorX, camRotVectorY, camRotVectorZ, *playerUp) # camera X axis global vector
    camYaxis = crossProduct(*camXaxis, camRotVectorX, camRotVectorY, camRotVectorZ) # camera Y axis global vector
    camXaxis = camXaxis / vectormod(camXaxis) #normalise
    camYaxis = camYaxis / vectormod(camYaxis)

    ray = np.array([vx - cam_pos[0], vy - cam_pos[1], vz - cam_pos[2]])




