import os
import numpy as np
import cv2
import math

scene_facets_raw = []

def parse_mtl(path):
    materials = {}
    current = None
    folder = os.path.dirname(path)
    if not os.path.exists(path):
        return materials
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('newmtl'):
                current = line.split()[1]
                materials[current] = {'Kd': (1.0, 1.0, 1.0), 'map_Kd': None}
            elif current and line.startswith('Kd'):
                parts = line.split()
                kd = tuple(float(p) for p in parts[1:4])
                materials[current]['Kd'] = kd
            elif current and line.startswith('map_Kd'):
                tex_path = os.path.join(folder, line.split()[1].strip())
                materials[current]['map_Kd'] = tex_path
    return materials

def parse_obj(path):
    verts, texs, norms = [], [], []
    faces = []
    materials = {}
    cur_mtl = None
    mtl_path = None
    folder = os.path.dirname(path)

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('mtllib'):
                mtl_path = os.path.join(folder, line.split()[1].strip())
            elif line.startswith('usemtl'):
                cur_mtl = line.split()[1].strip()
            elif line.startswith('v '):
                parts = line.split()
                verts.append(tuple(float(p) for p in parts[1:4]))
            elif line.startswith('vt'):
                parts = line.split()
                texs.append(tuple(float(p) for p in parts[1:3]))
            elif line.startswith('vn'):
                parts = line.split()
                norms.append(tuple(float(p) for p in parts[1:4]))
            elif line.startswith('f'):
                parts = line.strip().split()[1:]
                idxs = []
                for p in parts:
                    vals = p.split('/')
                    vi = int(vals[0]) - 1
                    ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    ni = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    idxs.append((vi, ti, ni))
                if len(idxs) == 3:
                    faces.append({'verts': idxs, 'material': cur_mtl})
                elif len(idxs) > 3:
                    for i in range(1, len(idxs)-1):
                        faces.append({'verts': [idxs[0], idxs[i], idxs[i+1]], 'material': cur_mtl})

    if mtl_path:
        materials = parse_mtl(mtl_path)

    facets = []
    for f in faces:
        v = [verts[i[0]] for i in f['verts']]
        uv = [texs[i[1]] if i[1] is not None and i[1] < len(texs) else (0.0, 0.0) for i in f['verts']]
        n = None
        if f['verts'][0][2] is not None:
            n = np.mean([np.array(norms[i[2]]) for i in f['verts']], axis=0)
        else:
            v0, v1, v2 = np.array(v[0]), np.array(v[1]), np.array(v[2])
            n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n) if np.linalg.norm(n) != 0 else n
        mat = materials.get(f['material'], {})
        col = mat.get('Kd', (1.0, 1.0, 1.0))
        tex = mat.get('map_Kd', None)
        facets.append({'verts': v, 'uvs': uv, 'normal': n, 'color': col, 'texture': tex})
    return facets

def scan_obj_folder(folder='obj_models'):
    objs = []
    if not os.path.isdir(folder):
        return objs
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.obj') or fn.endswith('.hidden.obj'):
            continue
        path = os.path.join(folder, fn)
        facets = parse_obj(path)
        objs.append({'name': fn, 'facets': facets})
    return objs

def load_scene_from_obj(objects_with_facets):
    meshes = []
    for obj in objects_with_facets:
        facets = obj.get('facets') if 'facets' in obj else obj
        vert_map = {}
        verts = []
        tris = []
        tri_normals = []
        tri_colors = []
        tri_uvs = []
        tex_path = None

        for f in facets:
            idxs = []
            for v in f['verts']:
                key = tuple(v)
                if key not in vert_map:
                    vert_map[key] = len(verts)
                    verts.append(np.array(key, dtype=float))
                idxs.append(vert_map[key])
            tris.append((idxs[0], idxs[1], idxs[2]))
            tri_normals.append(f['normal'])
            tri_colors.append(np.array(f['color'], dtype=float))
            tri_uvs.append(np.array(f['uvs'], dtype=float))
            if f['texture']:
                tex_path = f['texture']

        texture = None
        if tex_path and os.path.exists(tex_path):
            texture = cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB)

        meshes.append({
            'name': obj.get('name', 'unnamed'),
            'verts_world': np.array(verts, dtype=float),
            'tris': tris,
            'tri_normals_world': np.array(tri_normals, dtype=float),
            'colors': np.array(tri_colors, dtype=float),
            'uvs': tri_uvs,
            'texture': texture
        })
    return meshes


def translate_object(name, dx, dy, dz, folder='obj_models'):
    path = os.path.join(folder, name if name.endswith('.obj') else f"{name}.obj")
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        lines = f.readlines()
    with open(path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]) + dx, float(parts[2]) + dy, float(parts[3]) + dz
                f.write(f"v {x} {y} {z}\n")
            else:
                f.write(line)
    return True

import os
import numpy as np

def rotate_object(name, rx=0.0, ry=0.0, rz=0.0, degrees=True, folder='obj_models'):
    """
    Rotates an OBJ model about a single axis at a time.
    - rx: roll around X axis (left/right)
    - ry: pitch around Y axis (up/down)
    - rz: yaw around Z axis (side-to-side)
    If more than one axis is specified, only the first nonzero one (in order X, Y, Z) is applied.
    """
    path = os.path.join(folder, name if name.endswith('.obj') else f"{name}.obj")
    if not os.path.exists(path):
        return False

    # Convert to radians if needed
    if degrees:
        rx, ry, rz = np.radians([rx, ry, rz])

    # Choose which axis to rotate about (only one)
    if abs(rx) > 1e-8:
        angle = rx
        axis = 'x'
    elif abs(ry) > 1e-8:
        angle = ry
        axis = 'y'
    elif abs(rz) > 1e-8:
        angle = rz
        axis = 'z'
    else:
        # No rotation
        return True

    # Define single-axis rotation matrix
    if axis == 'x':
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s,  c]
        ])
    elif axis == 'y':
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    else:  # axis == 'z'
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    # Apply rotation
    with open(path, 'r') as f:
        lines = f.readlines()

    with open(path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                v = v @ R.T
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            else:
                f.write(line)

    return True


def toggle_object(name, folder='obj_models'):
    base = f"{name}.obj" if not name.endswith('.obj') else name
    hidden = base.replace('.obj', '.hidden.obj')
    path_base = os.path.join(folder, base)
    path_hidden = os.path.join(folder, hidden)
    if os.path.exists(path_base):
        os.rename(path_base, path_hidden)
        return "hidden"
    elif os.path.exists(path_hidden):
        os.rename(path_hidden, path_base)
        return "visible"
    return "not found"
