import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import math
from itertools import product

WIDTH, HEIGHT = 500, 500
zoom = 200
angle_x, angle_y, angle_z, angle_w = 0, 0, 0, 0

vertices = np.array(list(product([-1, 1], repeat=4)))

edges = []
for i, p1 in enumerate(vertices):
    for j, p2 in enumerate(vertices):
        if i<j and np.sum(np.abs(p2 - p1)) == 2:
            edges.append((i,j))
edges = np.array(edges)

print(edges)

def project_point(x, y, z, w, width, height):
    fov = 100
    dist = 10
    factor1 = dist / (w+dist)
    x,y,z = x*factor1, y*factor1, z*factor1
    factor2 = fov * dist / (z + dist)
    x_proj = x * factor2 * zoom/200 + width/2
    y_proj = -y * factor2 * zoom/200 + height/2
    return (x_proj, y_proj)

def rotate_point(x, y, z, w, ax, ay, az, aw):
    if aw !=0:
        y = math.cos(aw) - math.sin(aw)
        x = math.sin(aw) + math.cos(aw)
        z = z
        w = w
    elif ax !=0:
        y = math.cos(ax) - math.sin(ax)
        x = x
        z = z
        w = math.sin(ax) + math.cos(ax)

    return x, y, z, w

def draw_cube():
    canvas.delete("all")
    width, height = 600, 600
    projected = []
    for v in vertices:
        #x, y, z, w = v[0], v[1], v[2], v[3]
        x, y, z, w = rotate_point(v[0], v[1], v[2], v[3], angle_x, angle_y, angle_z, angle_w)
        projected.append(project_point(x, y, z, w, width, height))
    for e in edges:
        x1, y1 = projected[e[0]]
        x2, y2 = projected[e[1]]
        canvas.create_line(x1, y1, x2, y2, fill="black", width=2)
        print(projected[e[0]])

def update():
    draw_cube()
    root.after(20, update)

def on_key(event):
    global angle_x, angle_y, angle_w, angle_z, zoom
    if event.keysym == "Left":
        angle_w -= math.pi/10
    elif event.keysym == "Right":
        angle_w += math.pi/10
    elif event.keysym == "Up":
        angle_x -= math.pi/10
    elif event.keysym == "Down":
        angle_x += math.pi/10
    elif event.keysym == "plus":
        zoom += 10
    elif event.keysym == "minus":
        zoom -= 10

root = tk.Tk()
root.title("Simple Cube Renderer")
canvas = tk.Canvas(root, width=600, height=600, bg="white")
canvas.pack()

root.bind("<Key>", on_key)
update()
root.mainloop()