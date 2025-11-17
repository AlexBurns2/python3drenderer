import numpy as np
from itertools import product
import math
import os
import shutil
import atexit

vertices = np.array(list(product([-1, 1], repeat=4)))

#print(vertices)
#print(len(vertices))

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
#print(tris)
#print(len(tris))


quads = []
cube1verts = np.array(list(product([-1, 1], repeat=3)))
cube2verts = np.array(list(product([-1, 1], repeat=3)))
cube1verts = np.c_[cube1verts, np.ones(len(cube1verts))]
#print(cube1verts)
cube2verts = np.c_[cube2verts, np.full(len(cube1verts), -1, dtype=int)]

cube1tris = []
cube2tris = []

for i, point in enumerate(cube1verts):
    for j, test1 in enumerate(cube1verts):
        if np.sum(np.abs(test1 - point)) == 2:
            print(str(point) + " " + str(test1))

for i, point in enumerate(cube1verts):
    for j, test1 in enumerate(cube1verts):
        if j>i and np.sum(np.abs(test1 - point)) == 2:
            for k, test2 in enumerate(cube1verts):
                if k>j and np.sum(np.abs(test2 - point)) == 2:
                    for l, test3 in enumerate(cube1verts):
                        if l != i and np.sum(np.abs(test3 - test1)) == 2 and np.sum(np.abs(test3 - test2)) == 2:
                            quads.append((i,j,k,l))
                            cube1tris.append((i,j,k))
                            cube1tris.append((j,k,l))

for i, point in enumerate(cube2verts):
    for j, test1 in enumerate(cube2verts):
        if j>i and np.sum(np.abs(test1 - point)) == 2:
                for k, test2 in enumerate(cube2verts):
                    if k>j and np.sum(np.abs(test2 - point)) == 2:
                        for l, test3 in enumerate(cube2verts):
                            if l != i and np.sum(np.abs(test3 - test1)) == 2 and np.sum(np.abs(test3 - test2)) == 2:
                                quads.append((i+8,j+8,k+8,l+8))
                                cube2tris.append((i+8,j+8,k+8))
                                cube2tris.append((j+8,k+8,l+8))

joiningtris = []

for i, point in enumerate(cube2verts):
    for j, test1 in enumerate(cube2verts):
        if i<j and np.sum(np.abs(test1 - point)) == 2:
                for k, test2 in enumerate(cube1verts):
                    if np.sum(np.abs(test2 - point)) == 2:
                        for l, test3 in enumerate(cube1verts):
                            if np.sum(np.abs(test3 - test1)) == 2 and np.sum(np.abs(test3 - test2)) == 2:
                                quads.append((i+8,j+8,k,l))
                                joiningtris.append((i+8,j+8,k))
                                joiningtris.append((j+8,k,l))
'''
print(len(cube1tris))
print(len(cube2tris))
print(len(joiningtris))
'''

alltris = np.concatenate((cube1tris, cube2tris, joiningtris), axis = 0)


for i, tri in enumerate(cube1tris):
    #print(str(tri[0]))
    #print(str(verts[tri[0]]))
    print("f " + str(tri[0] + 1) + " " + str(tri[1] + 1) + " " + str(tri[2] + 1))

print(len(cube1tris))

'''f 2/1/1 4/4/1 3/2/1
f 4/4/2 8/6/2 7/5/2'''

'''
f 2 3 1
f 4 7 3
f 7 1 3
f 2 4 3
f 4 8 7
f 7 5 1
f 8 5 7
f 6 1 5
f 4 6 8
f 8 6 5
f 6 2 1
f 4 2 6'''