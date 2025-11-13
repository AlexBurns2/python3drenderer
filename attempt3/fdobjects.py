import numpy as np
from itertools import product
import math
import os
import shutil
import atexit
import cv2

_loaded_4meshes = []
azb_files = []
_loaded_4meshes_by_name = {}

def parse_fdt(path):
    materials = {}
    current = None
    folder = os.path.dirname(path)
    if not os.path.exists(path):
        return materials
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('newfdt'):
                current = line.split()[1]
                materials[current] = {'Kd': (1.0, 1.0, 1.0), 'map_Kd': None, 'd': 1.0}
            elif current and line.startswith('Kd'):
                parts = line.split()
                kd = tuple(float(p) for p in parts[1:4])
                materials[current]['Kd'] = kd
            elif current and line.startswith('map_Kd'):
                # texture path relative to mtl file location
                tex_name = line.split(maxsplit=1)[1].strip()
                tex_path = os.path.join(folder, tex_name)
                materials[current]['map_Kd'] = tex_path
            elif current and line.startswith('d '):
                parts = line.split()
                materials[current]['d'] = float(parts[1])
    return materials

def parse_fdo(path):
    verts = []
    faces = []
    materials = {}
    cur_fdt = None
    fdt_path = None
    folder = os.path.dirname(path)

    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('fdt'):
                parts = line.split()
                if len(parts) >= 2:
                    fdt_path = os.path.join(folder, parts[1].strip())
            elif line.startswith('usefdt'):
                cur_fdt = line.split(maxsplit=1)[1].strip()
            elif line.startswith('v '):
                parts = line.split()
                verts.append(tuple(float(p) for p in parts[1:4]))
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs = []
                for p in parts:
                    vals = p.split('/')
                    vi = int(vals[0]) - 1
                    idxs.append((vi))
                # triangulate if necessary
                if len(idxs) == 3:
                    faces.append({'verts': idxs, 'material': cur_fdt})
                elif len(idxs) > 3:
                    for i in range(1, len(idxs)-1):
                        faces.append({'verts': idxs[0], 'material': cur_fdt})

    if fdt_path:
        materials = parse_fdt(fdt_path)

    facets = []
    for f in faces:
        v = [verts[i[0]] for i in f['verts']]
        v0, v1, v2, v3 = np.array(v[0]), np.array(v[1]), np.array(v[2]), np.array(v[3])
        mat = materials.get(f['material'], {})
        col = mat.get('Kd', (1.0, 1.0, 1.0))
        facets.append({'verts': v,  'color': col})
    return facets


def scan_fdo_folder(folder='4d_models'):
    fdos = []
    if not os.path.isdir(folder):
        return fdos
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.fdo'):
            continue
        if fn.lower().endswith('.hidden.fdo'):
            continue
        obj_path = os.path.join(folder, fn)
        azb_path = _make_azb_copy(obj_path)
        facets = parse_fdo(azb_path)
        fdos.append({'name': fn, 'facets': facets, 'azb_path': azb_path})
    return fdos

def _make_azb_copy(obj_path):
    azb_path = obj_path[:-4] + '.azb'
    if not os.path.exists(azb_path):
        shutil.copy2(obj_path, azb_path)
        azb_files.append(azb_path)
    return azb_path


def load_scene_from_fdo(objects_with_facets):
    global _loaded_meshes, _loaded_meshes_by_name
    _loaded_meshes = []
    _loaded_meshes_by_name = {}

    for fdo in objects_with_facets:
        facets = fdo.get('facets') if 'facets' in fdo else fdo
        azb_path = fdo.get('azb_path', None)
        basename = os.path.splitext(fdo.get('name', 'unnamed'))[0]

        vert_map = {}
        verts = []
        tris = []
        tri_colors = []

        for f in facets:
            idxs = []
            for v in f['verts']:
                key = tuple(v)
                if key not in vert_map:
                    vert_map[key] = len(verts)
                    verts.append(np.array(key, dtype=float))
                idxs.append(vert_map[key])
            tris.append((idxs[0], idxs[1], idxs[2]))
            tri_colors.append(np.array(f['color'], dtype=float))

        mesh = {
            'name': fdo.get('name', 'unnamed'),
            'basename': basename,
            'azb_path': azb_path,
            'verts_world': np.array(verts, dtype=float),
            'tris': tris,
            'tri_normals_world': np.array(None, dtype=float),
            'colors': np.array(tri_colors, dtype=float),
            'uvs': None,
            'texture': None
        }
        _loaded_meshes_by_name[basename.lower()] = len(_loaded_meshes)
        _loaded_meshes.append(mesh)

    return _loaded_meshes

def get_loaded_4meshes():
    global _loaded_4meshes
    _loaded_4meshes = []
    for mesh in _loaded_4meshes:
        for vert in mesh['verts']:
            vert[0] = vert[0] * 1/(1+vert[3])
            vert[1] = vert[1] * 1/(1+vert[3])
            vert[2] = vert[2] * 1/(1+vert[3])
    opaque = []
    transparent = []
    for m in _loaded_4meshes:
        if np.any(m['alpha'] < 1.0):
            transparent.append(m)
        else:
            opaque.append(m)
    return opaque, transparent