import numpy as np
from math import cos, sin, pi

circle_res = 32
radius = 1.0
w_dist = 4.0

points = []
for i in range(circle_res):
    theta = 2 * pi * i / circle_res
    for j in range(circle_res):
        phi = 2 * pi * j / circle_res
        x = radius * cos(theta)
        y = radius * sin(theta)
        z = radius * cos(phi)
        w = radius * sin(phi)
        points.append(np.array([x, y, z, w]))

faces = []
tris = []

for i in range(circle_res):
    for j in range(circle_res):
        i1 = (i + 1) % circle_res
        j1 = (j + 1) % circle_res

        a = i * circle_res + j
        b = i1 * circle_res + j
        c = i1 * circle_res + j1
        d = i * circle_res + j1
        faces.append([a, b, c, d])
        tris.append((a, b, c))
        tris.append((a, c, d))

with open("duocylinder.txt", "w") as f:
    for p in points:
        f.write("v {} {} {} {}\n".format(p[0], p[1], p[2], p[3]))
    for i, tri in enumerate(tris):
        f.write("f " + str(tri[0] + 1) + " " + str(tri[1] + 1) + " " + str(tri[2] + 1) + "\n")