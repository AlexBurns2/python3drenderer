import tkinter as tk
import math

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def multiply(self, other):
        if not isinstance(other, Quaternion):
            raise TypeError("Can only multiply by another Quaternion")
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def __mul__(self, other):
        return self.multiply(other)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        L = math.sqrt(self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z)
        if L == 0:
            return Quaternion(1,0,0,0)
        return Quaternion(self.w/L, self.x/L, self.y/L, self.z/L)

    def rotate_vector(self, v):
        qv = Quaternion(0, v[0], v[1], v[2])
        r = self * qv * self.conjugate()
        return (r.x, r.y, r.z)

    def as_tuple(self):
        return (self.w, self.x, self.y, self.z)

    def __repr__(self):
        return f"Quaternion({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


vertices = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
]
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

orientation = Quaternion(1, 0, 0, 0)

root = tk.Tk()
root.title("cube")

control_frame = tk.Frame(root)
control_frame.pack(side="left", fill="y", padx=6, pady=6)

canvas = tk.Canvas(root, width=600, height=600, bg="white")
canvas.pack(side="right", expand=True, fill="both")

CX, CY = 300, 300 
FOV = 300          
Z_OFFSET = 5.0     

def project(pt):
    x, y, z = pt
    z = z + Z_OFFSET
    if z <= 0.1:
        z = 0.1
    s = FOV / z
    sx = CX + x * s * 40
    sy = CY - y * s * 40
    return sx, sy

def on_mouse_wheel(event):
    global Z_OFFSET
    if event.delta > 0:
        Z_OFFSET = max(1.0, Z_OFFSET - 0.5) 
    else:
        Z_OFFSET += 0.5 
    draw_cube()

canvas.bind("<MouseWheel>", on_mouse_wheel) 
tk.Label(control_frame, text="Quaternion (w, x, y, z)").pack(pady=(0,4))

w_var = tk.StringVar()
x_var = tk.StringVar()
y_var = tk.StringVar()
z_var = tk.StringVar()

def update_entry_vars():
    w_var.set(f"{orientation.w:.4f}")
    x_var.set(f"{orientation.x:.4f}")
    y_var.set(f"{orientation.y:.4f}")
    z_var.set(f"{orientation.z:.4f}")

entry_w = tk.Entry(control_frame, textvariable=w_var, width=12)
entry_x = tk.Entry(control_frame, textvariable=x_var, width=12)
entry_y = tk.Entry(control_frame, textvariable=y_var, width=12)
entry_z = tk.Entry(control_frame, textvariable=z_var, width=12)

tk.Label(control_frame, text="w").pack()
entry_w.pack()
tk.Label(control_frame, text="x").pack()
entry_x.pack()
tk.Label(control_frame, text="y").pack()
entry_y.pack()
tk.Label(control_frame, text="z").pack()
entry_z.pack()

def apply_entries():
    global orientation
    try:
        w = float(w_var.get())
        x = float(x_var.get())
        y = float(y_var.get())
        z = float(z_var.get())
    except ValueError:
        return
    orientation = Quaternion(w, x, y, z).normalize()
    update_entry_vars()
    draw_cube()

apply_btn = tk.Button(control_frame, text="Apply", command=apply_entries)
apply_btn.pack(pady=6)

def reset_orientation():
    global orientation
    orientation = Quaternion(1,0,0,0)
    update_entry_vars()
    draw_cube()

reset_btn = tk.Button(control_frame, text="Reset", command=reset_orientation)
reset_btn.pack(pady=(0,10))

hint = tk.Label(control_frame, text="Keys:\n w/W x/X y/Y z/Z\n(increase/decrease)\nDrag mouse to rotate", justify="left")
hint.pack(pady=(10,0))

quat_label = tk.Label(control_frame, text=str(orientation), justify="left")
quat_label.pack(pady=(12,0))

def project(pt):
    x, y, z = pt
    z = z + Z_OFFSET
    if z <= 0.1:
        z = 0.1
    s = FOV / z
    sx = CX + x * s * 40
    sy = CY - y * s * 40
    return sx, sy

def draw_cube():
    canvas.delete("all")
    rotated = [orientation.rotate_vector(v) for v in vertices]
    for a, b in edges:
        x1, y1 = project(rotated[a])
        x2, y2 = project(rotated[b])
        canvas.create_line(x1, y1, x2, y2, width=2)
    quat_label.config(text=f"{orientation}")
    update_entry_vars()

last_mouse = None

def on_mouse_press(event):
    global last_mouse
    last_mouse = (event.x, event.y)

def on_mouse_drag(event):
    global last_mouse, orientation
    if last_mouse is None:
        last_mouse = (event.x, event.y)
        return
    x0, y0 = last_mouse
    dx = event.x - x0
    dy = event.y - y0
    last_mouse = (event.x, event.y)

    sensitivity = 0.007 
    angle = math.sqrt(dx*dx + dy*dy) * sensitivity
    if angle == 0:
        return

    ax = dy
    ay = -dx
    az = 0.0
    L = math.sqrt(ax*ax + ay*ay + az*az)
    if L == 0:
        return
    ax /= L; ay /= L; az /= L

    dq = Quaternion(math.cos(angle/2),
                    ax * math.sin(angle/2),
                    ay * math.sin(angle/2),
                    az * math.sin(angle/2))

    orientation = (dq * orientation).normalize()
    draw_cube()

def on_mouse_release(event):
    global last_mouse
    last_mouse = None

canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_release)

STEP = 0.05

def modify_component(comp, delta):
    global orientation
    w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z
    if comp == 'w':
        w += delta
    elif comp == 'x':
        x += delta
    elif comp == 'y':
        y += delta
    elif comp == 'z':
        z += delta
    orientation = Quaternion(w, x, y, z).normalize()
    draw_cube()

root.bind("<KeyPress-w>", lambda e: modify_component('w', +STEP))
root.bind("<KeyPress-W>", lambda e: modify_component('w', -STEP))
root.bind("<KeyPress-x>", lambda e: modify_component('x', +STEP))
root.bind("<KeyPress-X>", lambda e: modify_component('x', -STEP))
root.bind("<KeyPress-y>", lambda e: modify_component('y', +STEP))
root.bind("<KeyPress-Y>", lambda e: modify_component('y', -STEP))
root.bind("<KeyPress-z>", lambda e: modify_component('z', +STEP))
root.bind("<KeyPress-Z>", lambda e: modify_component('z', -STEP))

root.bind("<space>", lambda e: reset_orientation())

update_entry_vars()
draw_cube()
root.mainloop()