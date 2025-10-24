import numpy as np
import math
import time
from itertools import product

def vectormod(x,y,z):
    return (x**2 + y**2 + z**2) ** 0.5

def dotProduct(x1, y1, z1, x2, y2, z2):
    return x1*x2 + y1*y2 + z1*z2

def crossProduct(x1, y1, z1, x2, y2, z2):
    cx = y1 * z2 - z1 * y2
    cy = z1 * x2 - x1 * z2
    cz = x1 * y2 - y1 * x2
    return np.array([cx, cy, cz])