import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import math
from itertools import product

WIDTH, HEIGHT = 500, 500
zoom = 200
azw, ayw, ayz, axw, axz, axy, = 0, 0, 0, 0, 0, 0

vertices = np.array(list(product([-1, 1], repeat=4)))

edges = []
for i, p1 in enumerate(vertices):
    for j, p2 in enumerate(vertices):
        if i<j and np.sum(np.abs(p2 - p1)) == 2:
            edges.append((i,j))
edges = np.array(edges)

angles = {
    "xy": 0.0, "xz": 0.0, "xw": 0.0,
    "yz": 0.0, "yw": 0.0, "zw": 0.0
}

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

def rotate_point(x, y, z, w, angles):
    cos, sin = math.cos, math.sin
    # XY-plane
    c,s = cos(angles.get("xy",0)), sin(angles.get("xy",0))
    x,y = c*x - s*y, s*x + c*y
    # XZ-plane
    c,s = cos(angles.get("xz",0)), sin(angles.get("xz",0))
    x,z = c*x - s*z, s*x + c*z
    # XW-plane
    c,s = cos(angles.get("xw",0)), sin(angles.get("xw",0))
    x,w = c*x - s*w, s*x + c*w
    # YZ-plane
    c,s = cos(angles.get("yz",0)), sin(angles.get("yz",0))
    y,z = c*y - s*z, s*y + c*z
    # YW-plane
    c,s = cos(angles.get("yw",0)), sin(angles.get("yw",0))
    y,w = c*y - s*w, s*y + c*w
    # ZW-plane
    c,s = cos(angles.get("zw",0)), sin(angles.get("zw",0))
    z,w = c*z - s*w, s*z + c*w
    return x,y,z,w

def draw_cube():
    canvas.delete("all")
    width, height = 600, 600
    projected = []
    for v in vertices:
        #x, y, z, w = v[0], v[1], v[2], v[3]
        x, y, z, w = rotate_point(*v, angles)
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
    global azw, ayz, zoom
    if event.keysym == "Left":
        angles["zw"] -= math.pi/10
    elif event.keysym == "Right":
        angles["zw"] += math.pi/10
    elif event.keysym == "Up":
        angles["yz"] -= math.pi/10
    elif event.keysym == "Down":
        angles["yz"] += math.pi/10
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