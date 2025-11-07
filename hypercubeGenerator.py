import numpy as np
from itertools import product
import math
import os
import shutil
import atexit

vertices = np.array(list(product([-1, 1], repeat=4)))

print(vertices)
print(len(vertices))

edges = []
for i, p1 in enumerate(vertices):
    for j, p2 in enumerate(vertices):
        if i<j and np.sum(np.abs(p2 - p1)) == 2:
            edges.append((i,j))
edges = np.array(edges)


tris = []
for i, point in enumerate(vertices):
    for j, test1 in enumerate(vertices):
        if i<j and np.sum(np.abs(test1 - point)) == 2:
            for k, test2 in enumerate(vertices):
                if k<j and np.sum(np.abs(test2 - point)) == 2:
                    tris.append((i,j,k))
tris = np.array(tris)
print(tris)
print(len(tris))