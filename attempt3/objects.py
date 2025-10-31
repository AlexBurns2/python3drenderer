import os
import numpy as np
import cv2
import shutil
import math
import atexit

scene_facets_raw = []

# keep azb files after exit
keep_transformed_file = False

azb_files = [] # paths of transformed files
_loaded_meshes = []
_loaded_meshes_by_name = {}

def _cleanup_azb():
    if keep_transformed_file:
        return
    for path in azb_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

atexit.register(_cleanup_azb)


def parse_mtl(path):
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
            if line.startswith('newmtl'):
                current = line.split()[1]
                materials[current] = {'Kd': (1.0, 1.0, 1.0), 'map_Kd': None}
            elif current and line.startswith('Kd'):
                parts = line.split()
                kd = tuple(float(p) for p in parts[1:4])
                materials[current]['Kd'] = kd
            elif current and line.startswith('map_Kd'):
                # texture path relative to mtl file location
                tex_name = line.split(maxsplit=1)[1].strip()
                tex_path = os.path.join(folder, tex_name)
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
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('mtllib'):
                parts = line.split()
                if len(parts) >= 2:
                    mtl_path = os.path.join(folder, parts[1].strip())
            elif line.startswith('usemtl'):
                cur_mtl = line.split(maxsplit=1)[1].strip()
            elif line.startswith('v '):
                parts = line.split()
                verts.append(tuple(float(p) for p in parts[1:4]))
            elif line.startswith('vt '):
                parts = line.split()
                texs.append(tuple(float(p) for p in parts[1:3]))
            elif line.startswith('vn '):
                parts = line.split()
                norms.append(tuple(float(p) for p in parts[1:4]))
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
            # average provided normals
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

def _make_azb_copy(obj_path):
    azb_path = obj_path[:-4] + '.azb'
    if not os.path.exists(azb_path):
        shutil.copy2(obj_path, azb_path)
        azb_files.append(azb_path)
    return azb_path

def scan_obj_folder(folder='obj_models'):
    objs = []
    if not os.path.isdir(folder):
        return objs
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

def get_loaded_meshes():
    return _loaded_meshes

def _azb_path(name, folder):
    base = f"{name}.obj" if not name.endswith('.obj') else name
    return os.path.join(folder, base[:-4] + '.azb')

def translate_object(name, dx, dy, dz, folder='obj_models'):
    azb = _azb_path(name, folder)
    if not os.path.exists(azb):
        return False
    with open(azb, 'r') as f:
        lines = f.readlines()
    with open(azb, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]) + dx, float(parts[2]) + dy, float(parts[3]) + dz
                f.write(f"v {x} {y} {z}\n")
            else:
                f.write(line)

    basename = os.path.splitext(name)[0]
    idx = _loaded_meshes_by_name.get(basename.lower())
    if idx is not None and 0 <= idx < len(_loaded_meshes):
        m = _loaded_meshes[idx]
        m['verts_world'] = m['verts_world'] + np.array([dx, dy, dz], dtype=float)
    return True

def rotate_object(name, rx=0.0, ry=0.0, rz=0.0, degrees=True, folder='obj_models'):
    azb = _azb_path(name, folder)
    if not os.path.exists(azb):
        return False

    if degrees:
        rx, ry, rz = np.radians([rx, ry, rz])

    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rx @ Ry @ Rz

    with open(azb, 'r') as f:
        lines = f.readlines()
    with open(azb, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                v_new = v @ R.T
                f.write(f"v {v_new[0]} {v_new[1]} {v_new[2]}\n")
            else:
                f.write(line)

    basename = os.path.splitext(name)[0]
    idx = _loaded_meshes_by_name.get(basename.lower())
    if idx is not None and 0 <= idx < len(_loaded_meshes):
        m = _loaded_meshes[idx]
        m['verts_world'] = (m['verts_world'] @ R.T).astype(float)
        if 'tri_normals_world' in m and m['tri_normals_world'] is not None:
            m['tri_normals_world'] = (m['tri_normals_world'] @ R.T).astype(float)
    return True

def toggle_object(name, folder='obj_models'):
    base = f"{name}.obj" if not name.endswith('.obj') else name
    hidden_name = base.replace('.obj', '.hidden.obj')
    obj_path = os.path.join(folder, base)
    azb_path = os.path.join(folder, base[:-4] + '.azb')
    hidden_obj_path = os.path.join(folder, hidden_name)
    hidden_azb_path = os.path.join(folder, hidden_name[:-4] + '.azb')  # .hidden.azb

    if os.path.exists(obj_path):
        os.rename(obj_path, hidden_obj_path)
        if os.path.exists(azb_path):
            os.rename(azb_path, hidden_azb_path)
            try:
                azb_files.remove(azb_path)
            except ValueError:
                pass
        basename = os.path.splitext(base)[0]
        idx = _loaded_meshes_by_name.pop(basename.lower(), None)
        if idx is not None:
            _loaded_meshes.pop(idx)
            _loaded_meshes_by_name.clear()
            for i, m in enumerate(_loaded_meshes):
                _loaded_meshes_by_name[m['basename'].lower()] = i
        return "hidden"

    if os.path.exists(hidden_obj_path):
        os.rename(hidden_obj_path, obj_path)
        if os.path.exists(hidden_azb_path):
            os.rename(hidden_azb_path, azb_path)
            azb_files.append(azb_path)
        if not os.path.exists(azb_path):
            shutil.copy2(obj_path, azb_path)
            azb_files.append(azb_path)
        facets = parse_obj(azb_path)
        new_obj = {'name': base, 'facets': facets, 'azb_path': azb_path}
        meshes_before = len(_loaded_meshes)
        loaded = load_scene_from_obj([new_obj])
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
            'name': base,
            'basename': os.path.splitext(base)[0],
            'azb_path': azb_path,
            'verts_world': np.array(verts, dtype=float),
            'tris': tris,
            'tri_normals_world': np.array(tri_normals, dtype=float),
            'colors': np.array(tri_colors, dtype=float),
            'uvs': tri_uvs,
            'texture': texture
        }
        _loaded_meshes_by_name[mesh['basename'].lower()] = len(_loaded_meshes)
        _loaded_meshes.append(mesh)
        return "visible"

    return "not found"
