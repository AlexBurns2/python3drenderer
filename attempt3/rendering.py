import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def world_to_camera(points, cam_pos, cam_yaw, cam_pitch):
    pts = points - cam_pos
    y = np.radians(cam_yaw)
    p = np.radians(cam_pitch)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    forward = np.array([np.sin(y) * cp, np.cos(y) * cp, sp])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right) if np.linalg.norm(right) != 0 else 1.0
    up = np.cross(right, forward)
    M = np.stack((right, forward, up), axis=1)
    return pts.dot(M)

@njit(cache=True, fastmath=True)
def backface_cull(tri_cam, cam_pos):
    v0, v1, v2 = tri_cam
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    dotProd = np.dot(n, (v0 + v1 + v2) / 3 - cam_pos)
    return dotProd <= 0, n

@njit(cache=True, fastmath=True)
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

@njit(cache=True, fastmath=True)
def project_point(focal, width, height, v):
    if v[1] <= 0:
        raise ValueError
    x = (v[0] * focal) / v[1]
    z = (v[2] * focal) / v[1]
    sx = int(width * 0.5 + x)
    sy = int(height * 0.5 - z)
    return sx, sy

@njit(cache=True, fastmath=True)
def rasterize_color(width, height, zbuffer, frame, p2, depths, color):
    xs = np.empty(3, dtype=np.int32)
    ys = np.empty(3, dtype=np.int32)
    for i in range(3):
        xs[i] = p2[i][0]
        ys[i] = p2[i][1]
    minx = max(min(xs[0], xs[1], xs[2]), 0)
    maxx = min(max(xs[0], xs[1], xs[2]), width - 1)
    miny = max(min(ys[0], ys[1], ys[2]), 0)
    maxy = min(max(ys[0], ys[1], ys[2]), height - 1)
    if minx > maxx or miny > maxy:
        return
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[1], ys[1]
    x2, y2 = xs[2], ys[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
            w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
            w2 = 1.0 - w0 - w1
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
                if depth < zbuffer[y, x]:
                    zbuffer[y, x] = depth
                    frame[y, x, 0] = color[0]
                    frame[y, x, 1] = color[1]
                    frame[y, x, 2] = color[2]

@njit(cache=True, fastmath=True)
def rasterize_textured(width, height, zbuffer, frame, p2, depths, uvs, texture, light):
    tex_h, tex_w, _ = texture.shape
    xs = np.empty(3, dtype=np.int32)
    ys = np.empty(3, dtype=np.int32)
    for i in range(3):
        xs[i] = p2[i][0]
        ys[i] = p2[i][1]
    minx = max(min(xs[0], xs[1], xs[2]), 0)
    maxx = min(max(xs[0], xs[1], xs[2]), width - 1)
    miny = max(min(ys[0], ys[1], ys[2]), 0)
    maxy = min(max(ys[0], ys[1], ys[2]), height - 1)
    if minx > maxx or miny > maxy:
        return
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[1], ys[1]
    x2, y2 = xs[2], ys[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if denom == 0:
        return
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
            w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
            w2 = 1.0 - w0 - w1
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
                if depth < zbuffer[y, x]:
                    zbuffer[y, x] = depth
                    u = w0 * uvs[0, 0] + w1 * uvs[1, 0] + w2 * uvs[2, 0]
                    v = w0 * uvs[0, 1] + w1 * uvs[1, 1] + w2 * uvs[2, 1]
                    tx = min(max(int(u * (tex_w - 1)), 0), tex_w - 1)
                    ty = min(max(int((1.0 - v) * (tex_h - 1)), 0), tex_h - 1)
                    color = texture[ty, tx]
                    # apply lighting
                    frame[y, x, 0] = min(255, int(color[0] * light))
                    frame[y, x, 1] = min(255, int(color[1] * light))
                    frame[y, x, 2] = min(255, int(color[2] * light))

class Renderer:
    def __init__(self, width, height, fov_degrees, near_clip):
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov_degrees)
        self.near = float(near_clip)
        self.focal = 0.5 * self.width / np.tan(np.radians(self.fov) * 0.5)
        self.zbuffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        self.light_dir_world = normalize(np.array([0.6, 0.7, -0.3], dtype=float))
        self.shader_colors = None
        self.shader_lights = None

    def clear(self):
        self.zbuffer.fill(np.inf)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def init_shader_cache(self, tris):
        n = len(tris)
        self.shader_colors = np.full((n, 3), 0, dtype=np.uint8)
        self.shader_lights = np.full(n, 0.0, dtype=np.float32)

    def update_shader_cache(self, meshes):
        offset = 0
        for mesh in meshes:
            tris = mesh['tris']
            normals = mesh['tri_normals_world']
            colors = mesh.get('colors', None)
            for i, t in enumerate(tris):
                base_col = colors[i] if colors is not None else (1.0, 1.0, 1.0)
                shaded_color, light = self.shade_triangle(normals[i], base_col)
                self.shader_colors[offset + i] = shaded_color
                self.shader_lights[offset + i] = light
            offset += len(tris)
    
    def shade_triangle(self, normal_world, base_color=(1.0, 1.0, 1.0)):
        n = normalize(normal_world)
        intensity = max(0.0, np.dot(n, -self.light_dir_world))
        shaded = np.clip(np.array(base_color) * (0.2 + 0.8 * intensity), 0, 1)
        return (int(shaded[0]*255), int(shaded[1]*255), int(shaded[2]*255)), 0.2 + 0.8 * intensity

    def render_scene(self, frame, meshes, cam):
        cam_pos = cam.position
        cam_yaw = cam.yaw
        cam_pitch = cam.pitch
        tri_index = 0
        for mesh in meshes:
            verts = mesh['verts_world']
            tris = mesh['tris']
            normals = mesh['tri_normals_world']
            texture = mesh.get('texture', None)
            uvs = mesh.get('uvs', None)
            verts_cam = world_to_camera(verts, cam_pos, cam_yaw, cam_pitch)
            for i, t in enumerate(tris):
                v0, v1, v2 = verts_cam[t[0]], verts_cam[t[1]], verts_cam[t[2]]
                if v0[1] <= self.near and v1[1] <= self.near and v2[1] <= self.near:
                    tri_index += 1
                    continue
                visible, _ = backface_cull((verts[t[0]], verts[t[1]], verts[t[2]]), cam_pos)
                if not visible:
                    tri_index += 1
                    continue
                try:
                    p0 = project_point(self.focal, self.width, self.height, v0)
                    p1 = project_point(self.focal, self.width, self.height, v1)
                    p2 = project_point(self.focal, self.width, self.height, v2)
                except Exception:
                    tri_index += 1
                    continue
                depths = np.array([v0[1], v1[1], v2[1]], dtype=np.float32)
                color = self.shader_colors[tri_index]
                light = self.shader_lights[tri_index]
                if texture is not None and uvs is not None and len(uvs) > i:
                    rasterize_textured(self.width, self.height, self.zbuffer, frame, (p0, p1, p2), depths, uvs[i], texture, light)
                else:
                    rasterize_color(self.width, self.height, self.zbuffer, frame, (p0, p1, p2), depths, color)
                tri_index += 1
