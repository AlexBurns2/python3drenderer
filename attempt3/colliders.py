import os
import numpy as np

COLLIDER_FOLDER = "obj_colliders"

collider_meshes = []

def load_colliders(folder=COLLIDER_FOLDER):
    global collider_meshes
    collider_meshes = []

    if not os.path.isdir(folder):
        return []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.obj'):
            continue
        path = os.path.join(folder, fn)
        tris = []

        verts = []
        faces = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    verts.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float))
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    idxs = []
                    for p in parts:
                        vi = int(p.split('/')[0]) - 1
                        idxs.append(vi)
                    # triangulate if >3 verts
                    if len(idxs) == 3:
                        faces.append((idxs[0], idxs[1], idxs[2]))
                    elif len(idxs) > 3:
                        for i in range(1, len(idxs) - 1):
                            faces.append((idxs[0], idxs[i], idxs[i+1]))

        for f in faces:
            tris.append((verts[f[0]], verts[f[1]], verts[f[2]]))
        collider_meshes.append(tris)

    return collider_meshes


def capsule_triangle_distance(capsule_start, capsule_end, radius, tri):
    def closest_point_on_triangle(p, tri):
        a, b, c = tri
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        d00 = np.dot(ab, ab)
        d01 = np.dot(ab, ac)
        d11 = np.dot(ac, ac)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return a
        v = (d11 * d1 - d01 * d2) / denom
        w = (d00 * d2 - d01 * d1) / denom
        u = 1 - v - w
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        u = 1 - v - w
        return u * a + v * b + w * c

    closest_points = []
    for p in [capsule_start, capsule_end]:
        cp = closest_point_on_triangle(p, tri)
        closest_points.append(cp)

    closest_point = min(closest_points, key=lambda cp: np.linalg.norm(cp - capsule_start))
    dir_vec = capsule_start - closest_point
    dist = np.linalg.norm(dir_vec)
    if dist < radius and dist > 1e-6:
        # return penetration vector
        return dir_vec * ((radius - dist) / dist)
    return None


def nudge_collision(player, height, radius):
    capsule_start = player.position
    capsule_end = player.position + np.array([0, 0, height], dtype=float)

    for mesh in collider_meshes:
        for tri in mesh:
            pen = capsule_triangle_distance(capsule_start, capsule_end, radius, tri)
            if pen is not None:
                # Nudge player
                player.position += pen


def check_collision(player, height=1.8, radius=0.3):
    nudge_collision(player, height, radius)
