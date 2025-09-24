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
        canvas.create_line(x1, y1, x2, y2, fill="black", width=3)
        print(projected[e[0]])

def update():
    draw_cube()
    root.after(20, update)

def on_key(event):
    global azw, ayz, zoom
    step = 40
    if event.keysym == "w":
        angles["zw"] -= math.pi/step
    if event.keysym == "s":
        angles["zw"] += math.pi/step
    if event.keysym == "a":
        angles["xw"] -= math.pi/step
    if event.keysym == "d":
        angles["xw"] += math.pi/step
    if event.keysym == "q":
        angles["yw"] -= math.pi/step
    if event.keysym == "e":
        angles["yw"] += math.pi/step
    if event.keysym == "plus":
        zoom += 10
    if event.keysym == "minus":
        zoom -= 10

def on_mouse_down(event):
    global last_mouse
    last_mouse = (event.x, event.y)

def on_mouse_drag(event):
    global last_mouse, angles
    if last_mouse is not None:
        dx = event.x - last_mouse[0]
        dy = event.y - last_mouse[1]
        angles["yz"] -= dy * 0.005 # up down
        angles["xz"] += dx * 0.005 #side to side
        last_mouse = (event.x, event.y)

def on_mouse_up(event):
    global last_mouse
    last_mouse = None

def on_mouse_scroll(event):
    global angles
    if event.delta > 0:   # roll
        angles["xy"] += math.pi/40
    else:
        angles["xy"] -= math.pi/40

root = tk.Tk()
root.title("Simple Cube Renderer")
canvas = tk.Canvas(root, width=600, height=600, bg="white")
canvas.pack()
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
canvas.bind("<MouseWheel>", on_mouse_scroll)

root.bind("<Key>", on_key)
update()
root.mainloop()