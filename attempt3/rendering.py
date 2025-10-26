import numpy as np

class Renderer:
    def __init__(self, width, height, fov_degrees, near_clip):
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov_degrees)
        self.near = float(near_clip)
        self.focal = 0.5 * self.width / np.tan(np.radians(self.fov) * 0.5)
        self.zbuffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        self.light_dir_world = self._normalize(np.array([0.6, -0.7, -0.3], dtype=float))
    def _normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v
    def clear(self):
        self.zbuffer.fill(np.inf)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    def world_to_camera(self, points, cam_pos, cam_yaw, cam_pitch):
        pts = points - cam_pos
        y = np.radians(cam_yaw)
        p = np.radians(cam_pitch)
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        forward = np.array([np.sin(y) * cp, np.cos(y) * cp, sp])
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right) if np.linalg.norm(right) != 0 else 1.0
        up = np.cross(right, forward)
        M = np.stack([right, forward, up], axis=1)
        return pts.dot(M)
    def backface_cull(self, tri_cam):
        v0, v1, v2 = tri_cam
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        return n[1] <= 0, n
    def project_point(self, v):
        if v[1] <= 0:
            raise ValueError
        x = (v[0] * self.focal) / v[1]
        z = (v[2] * self.focal) / v[1]
        sx = int(self.width * 0.5 + x)
        sy = int(self.height * 0.5 - z)
        return sx, sy
    def shade_triangle(self, normal_world):
        n = self._normalize(normal_world)
        intensity = max(0.0, np.dot(n, -self.light_dir_world))
        base = 20
        g = int(base + 235 * intensity)
        print(g, flush=True)
        return (g, g, g)
    def rasterize_numpy(self, frame, p2, depths, color):
        xs = np.array([p2[0][0], p2[1][0], p2[2][0]])
        ys = np.array([p2[0][1], p2[1][1], p2[2][1]])
        minx = max(int(np.clip(xs.min(), 0, self.width - 1)), 0)
        maxx = min(int(np.clip(xs.max(), 0, self.width - 1)), self.width - 1)
        miny = max(int(np.clip(ys.min(), 0, self.height - 1)), 0)
        maxy = min(int(np.clip(ys.max(), 0, self.height - 1)), self.height - 1)
        if minx > maxx or miny > maxy:
            return
        px = np.arange(minx, maxx + 1)
        py = np.arange(miny, maxy + 1)
        gx, gy = np.meshgrid(px, py)
        x0, y0 = xs[0], ys[0]
        x1, y1 = xs[1], ys[1]
        x2, y2 = xs[2], ys[2]
        denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
        if denom == 0:
            return
        w0 = ((y1 - y2)*(gx - x2) + (x2 - x1)*(gy - y2)) / denom
        w1 = ((y2 - y0)*(gx - x2) + (x0 - x2)*(gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            return
        depth_map = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
        gy_mask = gy[mask].astype(int)
        gx_mask = gx[mask].astype(int)
        zbuf_vals = self.zbuffer[gy_mask, gx_mask]
        depth_sub = depth_map[mask]
        replace = depth_sub < zbuf_vals
        if not np.any(replace):
            return
        write_y = gy_mask[replace]
        write_x = gx_mask[replace]
        self.zbuffer[write_y, write_x] = depth_sub[replace]
        frame[write_y, write_x] = color
    def render_scene(self, frame, meshes, cam):
        cam_pos = cam.position
        cam_yaw = cam.yaw
        cam_pitch = cam.pitch
        for mesh in meshes:
            verts = mesh['verts_world']
            tris = mesh['tris']
            normals = mesh['tri_normals_world']
            verts_cam = self.world_to_camera(verts, cam_pos, cam_yaw, cam_pitch)
            for i, t in enumerate(tris):
                v0 = verts_cam[t[0]]
                v1 = verts_cam[t[1]]
                v2 = verts_cam[t[2]]
                if v0[1] <= self.near and v1[1] <= self.near and v2[1] <= self.near:
                    continue
                visible, _ = self.backface_cull((v0, v1, v2))
                if not visible:
                    continue
                try:
                    p0 = self.project_point(v0)
                    p1 = self.project_point(v1)
                    p2 = self.project_point(v2)
                except Exception:
                    continue
                depths = np.array([v0[1], v1[1], v2[1]], dtype=np.float32)
                color = self.shade_triangle(normals[i])
                self.rasterize_numpy(frame, (p0, p1, p2), depths, color)
                print(color)