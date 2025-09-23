import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import math
from itertools import product

WIDTH, HEIGHT = 300, 300  # lower res for speed

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

angle_x, angle_y = 0.5, 0.5
zoom = 200
last_mouse = None

def rotate_point(x,y,z,ax,ay):
    cosx,sinx = math.cos(ax), math.sin(ax)
    y,z = y*cosx - z*sinx, y*sinx + z*cosx #x axis rotation matrix
    cosy,siny = math.cos(ay), math.sin(ay)
    x,z = x*cosy + z*siny, -x*siny + z*cosy #y axis rotation matrix
    return x,y,z

def project_point(x,y,z):
    fov,dist = 256,4 #abritrary fov
    factor = fov/(z+ dist)
    return int(x * factor * zoom / 100 + WIDTH/2), int(-y * factor * zoom / 100 + HEIGHT/2), z #scaled by zoom and perspective distort

def draw_scene():
    global angle_x, angle_y
    zbuf = np.full((HEIGHT,WIDTH), np.inf, dtype=np.float32) #z buffer definition
    img = np.zeros((HEIGHT,WIDTH,3), dtype=np.uint8) #all black

    transformed = [rotate_point(*v,angle_x,angle_y) for v in vertices]
    projected = [project_point(*v) for v in transformed]

    for tri,col in zip(tris,colors):
        (x0,y0,z0),(x1,y1,z1),(x2,y2,z2) = [projected[i] for i in tri] #screen coords
        xmin,xmax = max(min(x0,x1,x2),0), min(max(x0,x1,x2),WIDTH-1)
        ymin,ymax = max(min(y0,y1,y2),0), min(max(y0,y1,y2),HEIGHT-1)
        area = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0) 
        if area==0: continue #point offscreen
        for y in range(ymin,ymax+1): 
            for x in range(xmin,xmax+1):
                w0 = (x1-x)*(y2-y)-(y1-y)*(x2-x)
                w1 = (x2-x)*(y0-y)-(y2-y)*(x0-x)
                w2 = (x0-x)*(y1-y)-(y0-y)*(x1-x)
                if (w0>=0 and w1>=0 and w2>=0) or (w0<=0 and w1<=0 and w2<=0):
                    w0,w1,w2 = w0/area, w1/area, w2/area
                    z = w0*z0+w1*z1+w2*z2
                    if z < zbuf[y,x]:
                        zbuf[y,x] = z
                        img[y,x] = col

    return img

def update():
    img_array = draw_scene()
    image = Image.fromarray(img_array, "RGB")
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0,0,image=photo,anchor="nw")
    canvas.photo = photo
    root.after(20,update)


def on_mouse_down(event):
    global last_mouse
    last_mouse = (event.x, event.y)

def on_mouse_drag(event):
    global last_mouse, angle_x, angle_y
    if last_mouse is not None:
        dx = event.x - last_mouse[0]
        dy = event.y - last_mouse[1]
        angle_x -= dy * 0.005
        angle_y -= dx * 0.005
        last_mouse = (event.x, event.y)

def on_mouse_up(event):
    global last_mouse
    last_mouse = None

def on_mouse_scroll(event):
    global zoom
    if event.delta > 0:   # zoom in
        zoom += 20
    else:                 # zoom out
        zoom = max(20, zoom-20)


root = tk.Tk()
canvas = tk.Canvas(root,width=WIDTH,height=HEIGHT)
canvas.pack()
canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
canvas.bind("<MouseWheel>", on_mouse_scroll)
update()
root.mainloop()
