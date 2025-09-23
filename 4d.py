import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import math
from itertools import product

WIDTH, HEIGHT = 500, 500
zoom = 200
angle_x, angle_y = 0, 0

vertices = [
    [-1, -1, -1, -1],
    [ 1, -1, -1, -1],
    [ 1,  1, -1, -1],
    [-1,  1, -1, -1],
    [-1, -1,  1, -1],
    [ 1, -1,  1, -1],
    [ 1,  1,  1, -1],
    [-1,  1,  1, -1],
    [-1, -1, -1,  1],
    [ 1, -1, -1,  1],
    [ 1,  1, -1,  1],
    [-1,  1, -1,  1],
    [-1, -1,  1,  1],
    [ 1, -1,  1,  1],
    [ 1,  1,  1,  1],
    [-1,  1,  1,  1]

]

edges = [
    for i in range(32):
        
]


def project_point(x, y, z, width, height):
    fov = 256
    dist = 4
    factor = fov / (z + dist*2)
    x_proj = x * factor * zoom/200 + width/2
    y_proj = -y * factor * zoom/200 + height/2
    return (x_proj, y_proj)

def rotate_point(x, y, z, ax, ay):
    cosx, sinx = math.cos(ax), math.sin(ax)
    y, z = y*cosx - z*sinx, y*sinx + z*cosx
    cosy, siny = math.cos(ay), math.sin(ay)
    x, z = x*cosy + z*siny, -x*siny + z*cosy
    return x, y, z

def draw_cube():
    canvas.delete("all")
    width, height = 600, 600
    projected = []
    for v in vertices:
        x, y, z = rotate_point(v[0], v[1], v[2], angle_x, angle_y)
        projected.append(project_point(x, y, z, width, height))
    for e in edges:
        x1, y1 = projected[e[0]]
        x2, y2 = projected[e[1]]
        canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

def update():
    draw_cube()
    root.after(20, update)

def on_key(event):
    global angle_x, angle_y, zoom
    if event.keysym == "Left":
        angle_y -= 0.1
    elif event.keysym == "Right":
        angle_y += 0.1
    elif event.keysym == "Up":
        angle_x -= 0.1
    elif event.keysym == "Down":
        angle_x += 0.1
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