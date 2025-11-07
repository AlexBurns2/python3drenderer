import numpy as np
from itertools import product
import math
import os
import shutil
import atexit

_loaded_4meshes = []

azw, ayw, ayz, axw, axz, axy, = 0, 0, 0, 0, 0, 0

def parse_4do(path):
    verts = []
    faces = []
    materials = {}
    cur_4dt = None
    dt_path = None
    folder = os.path.dirname(path)

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('4dtlib'):
                parts = line.split()
                if len(parts) >= 2:
                    dt_path = os.path.join(folder, parts[1].strip())
            elif line.startswith('use4dt'):
                cur_4dt = line.split(maxsplit=1)[1].strip()
            elif line.startswith('v '):
                parts = line.split()
                verts.append(tuple(float(p) for p in parts[1:4]))
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs = []
                for p in parts:
                    vals = p.split('/')
                    vi = int(vals[0]) - 1
                    ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    ni = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    idxs.append((vi, ti, ni))
                # triangulate if necessary
                if len(idxs) == 3:
                    faces.append({'verts': idxs, 'material': cur_4dt})
                elif len(idxs) > 3:
                    for i in range(1, len(idxs)-1):
                        faces.append({'verts': [idxs[0], idxs[i], idxs[i+1]], 'material': cur_4dt})

    if dt_path:
        materials = parse_4dt(dt_path)

    facets = []
    for f in faces:
        v = [verts[i[0]] for i in f['verts']]
        v0, v1, v2 = np.array(v[0]), np.array(v[1]), np.array(v[2])
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n) if np.linalg.norm(n) != 0 else n
        mat = materials.get(f['material'], {})
        col = mat.get('Kd', (1.0, 1.0, 1.0))
        facets.append({'verts': v,  'color': col})
    return facets


def scan_4d_folder(folder='obj_models'):
    dos = []
    if not os.path.isdir(folder):
        return dos
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.obj'):
            continue
        if fn.lower().endswith('.hidden.obj'):
            continue
        obj_path = os.path.join(folder, fn)
        azb_path = _make_azb_copy(obj_path)
        facets = parse_obj(azb_path)
        objs.append({'name': fn, 'facets': facets, 'azb_path': azb_path})
    return objs

def load_scene_from_obj(objects_with_facets):
    global _loaded_meshes, _loaded_meshes_by_name
    _loaded_meshes = []
    _loaded_meshes_by_name = {}

    for obj in objects_with_facets:
        facets = obj.get('facets') if 'facets' in obj else obj
        azb_path = obj.get('azb_path', None)
        basename = os.path.splitext(obj.get('name', 'unnamed'))[0]

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

        mesh = {
            'name': obj.get('name', 'unnamed'),
            'basename': basename,
            'azb_path': azb_path,
            'verts_world': np.array(verts, dtype=float),
            'tris': tris,
            'tri_normals_world': np.array(tri_normals, dtype=float),
            'colors': np.array(tri_colors, dtype=float),
            'uvs': tri_uvs,
            'texture': texture
        }
        _loaded_meshes_by_name[basename.lower()] = len(_loaded_meshes)
        _loaded_meshes.append(mesh)

    return _loaded_meshes

def get_loaded_4meshes():
    global _loaded_4meshes
    _loaded_4meshes = []
    return _loaded_4meshes