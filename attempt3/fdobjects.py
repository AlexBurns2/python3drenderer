import numpy as np
import os
import shutil
import atexit
import math

_loaded_4meshes = []
_loaded_4meshes_by_name = {}
azb_files = []
cam = None

def _cleanup_azb():
    for path in azb_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
atexit.register(_cleanup_azb)

def define_cam(camera):
    global cam
    cam = camera

def compute_tri_normals(verts_world, tris):
    normals = []
    center = [0, 0, 0]
    for v in verts_world:
        center += v
    center /= len(verts_world)
    print(center)
    for t in tris:
        v0, v1, v2 = verts_world[t[0]], verts_world[t[1]], verts_world[t[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        n = n / norm if norm != 0 else n
        if np.dot(n, (v0 + v1 + v2) / 3 - center) > 0:
            n = -n
        normals.append(n)

    return np.array(normals, dtype=float)

def compute_cam_normals(verts_world, tris):
    global cam
    normals = []
    for t in tris:
        v0, v1, v2 = verts_world[t[0]], verts_world[t[1]], verts_world[t[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        n = n / norm if norm != 0 else n
        if np.dot(n, (v0 + v1 + v2) / 3 - cam.position) <= 0:
            n = -n
        normals.append(n)
    return np.array(normals, dtype=float)


def backface_cull(tri_cam, cam_pos):
    v0, v1, v2 = tri_cam
    
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    dotProd = np.dot(n, (v0 + v1 + v2) / 3 - cam_pos)
    return dotProd <= 0, n

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
            elif current and line.startswith('d '):
                parts = line.split()
                materials[current]['d'] = float(parts[1])
    return materials

def parse_fdo(path):
    verts = []
    faces = []
    origin = []
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
            elif line.startswith('c '):
                parts = line.split()
                origin = tuple(float(p) for p in parts[1:5])
            elif line.startswith('usefdt'):
                cur_fdt = line.split(maxsplit=1)[1].strip()
            elif line.startswith('v '):
                parts = line.split()
                verts.append(tuple(float(p) for p in parts[1:5]))
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs = [int(p) - 1 for p in parts]
                if len(idxs) == 3:
                    faces.append({'verts': idxs, 'material': cur_fdt})
                elif len(idxs) > 3:
                    for i in range(1, len(idxs) - 1):
                        faces.append({'verts': [idxs[0], idxs[i], idxs[i + 1]], 'material': cur_fdt})

    if fdt_path:
        materials = parse_fdt(fdt_path)

    facets = []
    for f in faces:
        v4 = [verts[i] for i in f['verts']]
        mat = materials.get(f['material'], {})
        col = mat.get('Kd', (1.0, 1.0, 1.0))
        alpha = mat.get('d', 1.0)
        facets.append({
            'origin': origin,
            'verts4d': v4,
            'uvs': [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            'normal': None,
            'color': col,
            'texture': None,
            'alpha': alpha
        })
    return facets

def _make_azb_copy(fdo_path):
    azb_path = fdo_path[:-4] + '.azb'
    if not os.path.exists(azb_path):
        shutil.copy2(fdo_path, azb_path)
        azb_files.append(azb_path)
    return azb_path

def project_4d_to_3d_array(verts4d, dist=10.0):
    v4 = np.array(verts4d, dtype=float)
    w = v4[:, 3]
    factor1 = dist / (w + dist)
    factor1 = np.where(np.abs(w + dist) < 1e-6, 1.0, factor1)
    scaled = v4[:, :3] * factor1[:, np.newaxis]
    return scaled

def load_scene_from_fdo(objects_with_facets):
    global _loaded_4meshes, _loaded_4meshes_by_name
    _loaded_4meshes = []
    _loaded_4meshes_by_name = {}

    for fdo in objects_with_facets:
        facets = fdo.get('facets') if 'facets' in fdo else fdo
        azb_path = fdo.get('azb_path', None)
        basename = os.path.splitext(fdo.get('name', 'unnamed'))[0]

        vert_map = {}
        verts4d = []
        tris = []
        tri_normals = []
        tri_colors = []
        tri_alphas = []
        tri_uvs = []
        origins = []

        for f in facets:
            idxs = []
            for v4 in f['verts4d']:
                key = tuple(v4)
                if key not in vert_map:
                    vert_map[key] = len(verts4d)
                    verts4d.append(np.array(key, dtype=float))
                    origins.append(np.array(f['origin'], dtype=float))
                idxs.append(vert_map[key])
            tris.append((idxs[0], idxs[1], idxs[2]))
            tri_colors.append(np.array(f['color'], dtype=float))
            tri_alphas.append(float(f.get('alpha', 1.0)))
            tri_uvs.append(np.array(f['uvs'], dtype=float))

        verts4d_arr = np.array(verts4d, dtype=float)
        verts_world = project_4d_to_3d_array(verts4d_arr, dist=10.0)
        tri_normals = compute_cam_normals(verts_world, tris)

        mesh = {
            'name': fdo.get('name', 'unnamed'),
            'basename': basename,
            'azb_path': azb_path,
            'origin': origins,
            'verts4d': verts4d_arr,
            'verts_world': verts_world,
            'tris': tris,
            'tri_normals_world': np.array(tri_normals, dtype=float),
            'colors': np.array(tri_colors, dtype=float),
            'alpha': np.array(tri_alphas, dtype=float),
            'uvs': tri_uvs,
            'texture': None
        }

        _loaded_4meshes_by_name[basename.lower()] = len(_loaded_4meshes)
        _loaded_4meshes.append(mesh)

    print("Loaded 4D meshes:", len(_loaded_4meshes))
    return _loaded_4meshes

def scan_fdo_folder(folder='4d_models'):
    fdos = []
    if not os.path.isdir(folder):
        return fdos
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.fdo') or fn.lower().endswith('.hidden.fdo'):
            continue
        fdo_path = os.path.join(folder, fn)
        azb_path = _make_azb_copy(fdo_path)
        facets = parse_fdo(azb_path)
        fdos.append({'name': fn, 'facets': facets, 'azb_path': azb_path})
    return fdos

def get_loaded_4meshes():
    global _loaded_4meshes
    opaque, transparent = [], []
    for m in _loaded_4meshes:
        if np.any(m['alpha'] < 1.0):
            transparent.append(m)
        else:
            opaque.append(m)
    for m in transparent:
        m['verts4d'] = m['verts4d'] + np.array(m['origin'], dtype=float)
        m['verts_world'] = project_4d_to_3d_array(m['verts4d'], dist=10.0)
        tris = m['tris']
        tri_normals = compute_cam_normals(m['verts_world'], tris)
        m['tri_normals_world'] = np.array(tri_normals, dtype=float)
    for m in opaque:
        m['verts4d'] = m['verts4d'] + m['origin']
        m['verts_world'] = project_4d_to_3d_array(m['verts4d'], dist=10.0)
        tris = m['tris']
        tri_normals = compute_tri_normals(m['verts_world'], tris)
        m['tri_normals_world'] = np.array(tri_normals, dtype=float)
    return opaque, transparent

def _azb_path(name, folder='4d_models'):
    base = f"{name}.fdo" if not name.endswith('.fdo') else name
    azb = os.path.join(folder, base[:-4] + '.azb')
    return azb

def translate_object_4d(name, dx=0, dy=0, dz=0, dw=0, folder='4d_models'):
    azb = _make_azb_copy(os.path.join(folder, f"{name}.fdo"))
    if not os.path.exists(azb):
        return False
    with open(azb, 'r') as f:
        lines = f.readlines()
    with open(azb, 'w') as f:
        for line in lines:
            if line.startswith('c '):
                parts = line.split()
                x, y, z, w = map(float, parts[1:5])
                x += dx; y += dy; z += dz; w += dw
                f.write(f"c {x} {y} {z} {w}\n")
            else:
                f.write(line)
    basename = os.path.splitext(name)[0]
    idx = _loaded_4meshes_by_name.get(basename.lower())
    if idx is not None:
        m = _loaded_4meshes[idx]
        '''
        m['verts4d'] = m['verts4d'] + np.array([dx, dy, dz, dw], dtype=float)
        m['verts_world'] = project_4d_to_3d_array(m['verts4d'], dist=10.0)
        tris = m['tris']
        tri_normals = []
        for t in tris:
            v0 = m['verts_world'][t[0]]
            v1 = m['verts_world'][t[1]]
            v2 = m['verts_world'][t[2]]
            n = np.cross(v1 - v0, v2 - v0)
            n = n / np.linalg.norm(n) if np.linalg.norm(n) != 0 else n
            tri_normals.append(n)
        m['tri_normals_world'] = np.array(tri_normals, dtype=float)
        '''
        m['origin'] = m["origin"] + np.array([dx, dy, dz, dw], dtype=float)
    return True

def rotate_point_4d(x, y, z, w, angles):
    cos, sin = math.cos, math.sin
    c,s = cos(angles.get("xy",0)), sin(angles.get("xy",0)); x,y = c*x - s*y, s*x + c*y
    c,s = cos(angles.get("xz",0)), sin(angles.get("xz",0)); x,z = c*x - s*z, s*x + c*z
    c,s = cos(angles.get("xw",0)), sin(angles.get("xw",0)); x,w = c*x - s*w, s*x + c*w
    c,s = cos(angles.get("yz",0)), sin(angles.get("yz",0)); y,z = c*y - s*z, s*y + c*z
    c,s = cos(angles.get("yw",0)), sin(angles.get("yw",0)); y,w = c*y - s*w, s*y + c*w
    c,s = cos(angles.get("zw",0)), sin(angles.get("zw",0)); z,w = c*z - s*w, s*z + c*w
    return x, y, z, w

def rotate_object_4d(name, angles=None, degrees=True, folder='4d_models'):
    if angles is None:
        return False
    azb = _make_azb_copy(os.path.join(folder, f"{name}.fdo"))
    if not os.path.exists(azb):
        return False
    if degrees:
        angles = {k: math.radians(v) for k, v in angles.items()}
    with open(azb, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            x, y, z, w = map(float, parts[1:5])
            x, y, z, w = rotate_point_4d(x, y, z, w, angles)
            new_lines.append(f"v {x} {y} {z} {w}\n")
        else:
            new_lines.append(line)
    with open(azb, 'w') as f:
        f.writelines(new_lines)
    basename = os.path.splitext(name)[0]
    idx = _loaded_4meshes_by_name.get(basename.lower())
    if idx is not None:
        m = _loaded_4meshes[idx]
        verts4d = m['verts4d']
        rotated = np.empty_like(verts4d)
        for i, (x, y, z, w) in enumerate(verts4d):
            rotated[i] = rotate_point_4d(x, y, z, w, angles)
        m['verts4d'] = rotated
        m['verts_world'] = project_4d_to_3d_array(m['verts4d'], dist=10.0)
        tris = m['tris']
        tri_normals = []
        tri_normals = compute_cam_normals(m['verts_world'], tris)
        m['tri_normals_world'] = np.array(tri_normals, dtype=float)
    return True
