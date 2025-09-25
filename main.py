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
qw, qx, qy, wz = 0,0,0,0

def rotate_point(x,y,z,ax,ay):
    cosx,sinx = math.cos(ax), math.sin(ax)
    y,z = y*cosx - z*sinx, y*sinx + z*cosx #x axis rotation matrix
    cosy,siny = math.cos(ay), math.sin(ay)
    x,z = x*cosy + z*siny, -x*siny + z*cosy #y axis rotation matrix
    return x,y,z

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(a, b):
        w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
        x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
        y = a.w*b.y + a.x*b.z + a.y*b.w - a.z*b.x
        z = a.w*b.z + a.x*b.y + a.y*b.x - a.z*b.w
        return w,x,y,z
    
    def conjugate(self):
        return self.w, -self.x, -self.y, -self.z

def quaternion_rotation(x,y,z,ax, ay, az, t):
    v = Quaternion(0, x, y, z)

    q = Quaternion(
        math.cos(t/2),
        ax * math.sin(t/2),
        ay * math.sin(t/2),
        az * math.sin(t/2)
    )
    qprime = q.conjugate
    out = q.multiply(v).multiply(qprime)
    return out.x, out.y, out.z



    '''
    cx = math.cos(ax/2)
    sx = math.sin(ax/2)
    cy = math.cos(ay/2)
    sy = math.sin(ay/2)
    cz = math.cos(az/2)
    sz = math.sin(az/2)

    q = {
        "w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0
        }

    q["w"] = cx*cy*cz + sx*sy*sz
    q["x"] = sx*cy*cz - cx*sy*sz
    q["y"] = cx*sy*cz + sx*cy*sz
    q["z"] = cx*cy*sz - sx*sy*cz

    qprime = {
        "w": q["w"], "x": -q["x"], "y": -q["y"], "z": -q["z"]
        }
    
    out = {
        "x": 0.0, "y": 0.0, "z": 0.0
        }
    
    #hamilton product
    for key in q:
        for keyprime in qprime:
            out["x"] += q[key] * qprime[keyprime] * x
            out["y"] += q[key] * qprime[keyprime] * y
            out["z"] += q[key] * qprime[keyprime] * z
    return out["x"], out["y"], out["z"] '''

def project_point(x,y,z):
    fov,dist = 256, 4 #abritrary fov
    factor = fov/(z + dist)
    return int(x * factor * zoom / 100 + WIDTH/2), int(-y * factor * zoom / 100 + HEIGHT/2), z #scaled by zoom and perspective distort

def draw_scene():
    global angle_x, angle_y
    zbuf = np.full((HEIGHT,WIDTH), np.inf, dtype=np.float32) #z buffer definition
    img = np.zeros((HEIGHT,WIDTH,3), dtype=np.uint8) #all black

    transformed = [quaternion_rotation(*v,angle_x,angle_y,0) for v in vertices]
    #transformed = [rotate_point(*v,angle_x,angle_y) for v in vertices]
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
        angle_x += dy * 0.005
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
