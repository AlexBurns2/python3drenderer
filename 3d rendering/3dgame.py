import numpy as np
import cv2
import math
import time


WIDTH, HEIGHT = 1280, 720 
FPS_SLEEP = 0.001 
FOV_DEGREES = 60.0 
NEAR_CLIP = 0.01
ZOOM = 1.0
LIGHT_DIR = np.array([0.5, 1.0, -0.7], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / (np.linalg.norm(LIGHT_DIR) + 1e-12)

vertices = np.array([
    (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
    (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)
], dtype=np.float32)

tris = np.array([
    (0,1,2),(0,2,3),
    (4,5,6),(4,6,7),
    (0,1,5),(0,5,4),
    (2,3,7),(2,7,6),
    (1,2,6),(1,6,5),
    (0,3,7),(0,7,4)
], dtype=np.int32)

colors = np.array([
    (255,0,0),(255,0,0),
    (0,255,0),(0,255,0),
    (0,0,255),(0,0,255),
    (255,255,0),(255,255,0),
    (255,0,255),(255,0,255),
    (0,255,255),(0,255,255)
], dtype=np.uint8)

playerUp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cam_pos = np.array([0.0, 0.0, -6.0], dtype=np.float32)   # moved back to see object
yaw = 0.0    # rotation around Y (left/right)
pitch = 0.0  # rotation around X (up/down)

mouse_last = None
mouse_sensitivity = 0.003

move_speed = 0.25

# Pre-calc indices for triangles to speed lookups
tri_v0_idx = tris[:, 0]
tri_v1_idx = tris[:, 1]
tri_v2_idx = tris[:, 2]

# Convert degrees FOV to focal length in pixels:
fov_rad = math.radians(FOV_DEGREES)
focal_px = (HEIGHT / 2.0) / math.tan(fov_rad / 2.0)  # standard pinhole model

def build_camera_rotation_matrix(yaw, pitch):

    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)

    # Yaw then pitch
    Ry = np.array([
        [ cy, 0.0, sy],
        [ 0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ], dtype=np.float32)

    Rx = np.array([
        [1.0, 0.0,  0.0],
        [0.0,  cx, -sx],
        [0.0,  sx,  cx]
    ], dtype=np.float32)

    return Rx @ Ry

def world_to_view_all(verts, cam_pos, yaw, pitch):
    rel = (verts - cam_pos).astype(np.float32)  # translate
    R = build_camera_rotation_matrix(yaw, pitch)  # 3x3
    return (R @ rel.T).T  # (N,3)

def backface_cull_and_normals(view_verts, tris):
    v0 = view_verts[tris[:, 0], :]
    v1 = view_verts[tris[:, 1], :]
    v2 = view_verts[tris[:, 2], :]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e2, e1)

    centroids = (v0 + v1 + v2) / 3.0

    dp = np.einsum('ij,ij->i', normals, centroids)
    keep_mask = dp < 0.0

    norms = np.linalg.norm(normals[keep_mask], axis=1, keepdims=True)
    normals_unit = normals[keep_mask] / np.maximum(norms, 1e-9)

    tri_indices_kept = np.nonzero(keep_mask)[0]
    return keep_mask, normals_unit, centroids[keep_mask], tri_indices_kept

def project_vertices(view_verts):
    vx = view_verts[:, 0].astype(np.float32)
    vy = view_verts[:, 1].astype(np.float32)
    vz = view_verts[:, 2].astype(np.float32)

    in_front = vz > NEAR_CLIP

    sx = np.empty_like(vx)
    sy = np.empty_like(vy)
    depth = np.empty_like(vz)

    focal = focal_px * ZOOM

    sx[~in_front] = np.nan
    sy[~in_front] = np.nan
    depth[~in_front] = np.inf

    sx[in_front] = (vx[in_front] * focal / vz[in_front]) + (WIDTH / 2.0)
    sy[in_front] = -(vy[in_front] * focal / vz[in_front]) + (HEIGHT / 2.0)
    depth[in_front] = vz[in_front]

    return np.stack([sx, sy, depth], axis=1)

def rasterize_triangles(projected, view_verts, tri_indices, tri_colors, normals_unit, zbuf, img):
    for idx, tri_idx in enumerate(tri_indices):
        t = tris[tri_idx]
        p0 = projected[t[0]]
        p1 = projected[t[1]]
        p2 = projected[t[2]]

        # skip triangles with all vertex behind camera
        if np.isnan(p0[0]) or np.isnan(p1[0]) or np.isnan(p2[0]):
            continue

        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil (max(x0, x1, x2))), WIDTH - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil (max(y0, y1, y2))), HEIGHT - 1)

        if xmax < xmin or ymax < ymin:
            continue

        area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area2) < 1e-8:
            continue

        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        XX, YY = np.meshgrid(xs, ys)

        w0 = (x1 - XX) * (y2 - YY) - (y1 - YY) * (x2 - XX)
        w1 = (x2 - XX) * (y0 - YY) - (y2 - YY) * (x0 - XX)
        w2 = (x0 - XX) * (y1 - YY) - (y0 - YY) * (x1 - XX)

        mask_pos = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        mask_neg = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        mask = mask_pos | mask_neg

        if not np.any(mask):
            continue

        w0_sel = w0[mask] / area2
        w1_sel = w1[mask] / area2
        w2_sel = w2[mask] / area2

        # depth interpolation
        z_vals = w0_sel * z0 + w1_sel * z1 + w2_sel * z2

        ys_masked = YY[mask].astype(np.int32)
        xs_masked = XX[mask].astype(np.int32)

        current_z = zbuf[ys_masked, xs_masked]
        nearer = z_vals < current_z

        if not np.any(nearer):
            continue

        ux = xs_masked[nearer]
        uy = ys_masked[nearer]
        uz = z_vals[nearer]

        tri_normal = normals_unit[idx]
        brightness = np.dot(tri_normal, LIGHT_DIR)
        brightness = float(max(0.0, min(1.0, brightness)))  # clamp

        base_col = tri_colors[idx].astype(np.float32)  # (3,)
        shaded_col = np.clip(base_col * (0.15 + 0.85 * brightness), 0, 255).astype(np.uint8)

        zbuf[uy, ux] = uz
        img[uy, ux] = shaded_col



def draw_scene():
    global vertices, tris, colors, cam_pos, yaw, pitch

    zbuf = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    view_verts = world_to_view_all(vertices, cam_pos, yaw, pitch)

    keep_mask, normals_unit, centroids, tri_indices_kept = backface_cull_and_normals(view_verts, tris)
    if tri_indices_kept.size == 0:
        return img

    projected = project_vertices(view_verts)

    kept_colors = colors[tri_indices_kept]

    tri_v0 = tris[tri_indices_kept, 0]
    tri_v1 = tris[tri_indices_kept, 1]
    tri_v2 = tris[tri_indices_kept, 2]
    z0 = projected[tri_v0, 2]
    z1 = projected[tri_v1, 2]
    z2 = projected[tri_v2, 2]
    valid_mask = (z0 < np.inf) & (z1 < np.inf) & (z2 < np.inf)
    #valid_mask = (z0 > 0) & (z1 > 0) & (z2 > 0)

    if not np.any(valid_mask):
        return img

    tri_indices_kept = tri_indices_kept[valid_mask]
    normals_unit = normals_unit[valid_mask]
    kept_colors = kept_colors[valid_mask]

    rasterize_triangles(projected, view_verts, tri_indices_kept, kept_colors, normals_unit, zbuf, img)

    return img


def handle_input(key):
    global cam_pos, yaw, pitch, ZOOM
    forward = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)

    if key == ord('w'): cam_pos += forward * move_speed
    if key == ord('s'): cam_pos -= forward * move_speed
    if key == ord('a'): cam_pos -= right * move_speed
    if key == ord('d'): cam_pos += right * move_speed
    if key == ord(' '): cam_pos[1] += move_speed
    if key == ord('c'): cam_pos[1] -= move_speed
    if key == ord('=') or key == ord('+'): ZOOM *= 1.1
    if key == ord('-'): ZOOM = max(0.1, ZOOM / 1.1)

    if key == 81: yaw -= 0.05   # left
    if key == 83: yaw += 0.05   # right
    if key == 82: pitch -= 0.05 # up
    if key == 84: pitch += 0.05 # down


def on_mouse(event, x, y, flags, param):
    global mouse_last, yaw, pitch
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_last is not None:
            dx = x - mouse_last[0]
            dy = y - mouse_last[1]
            yaw -= dx * mouse_sensitivity
            pitch -= dy * mouse_sensitivity
            # clamp pitch to avoid flipping
            pitch = max(-math.pi/2 + 1e-3, min(math.pi/2 - 1e-3, pitch))
        mouse_last = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_last = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_last = None


cv2.namedWindow("cube", cv2.WINDOW_NORMAL)
cv2.resizeWindow("cube", WIDTH, HEIGHT)
cv2.setMouseCallback("cube", on_mouse)

last_time = time.time()
while True:
    t0 = time.time()
    frame = draw_scene()
    # show small HUD fps
    fps = 1.0 / max(1e-6, (time.time() - last_time))
    last_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("cube", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if key != 255:
        handle_input(key)

    time.sleep(FPS_SLEEP)

cv2.destroyAllWindows()