import numpy as np
import os
import shutil
import atexit
import math
from rendering import Renderer
from numba.typed import List as NList

from numba import njit, types
from numba.typed import Dict, List

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


@njit(cache=True)
def _tri_centroid(verts, t0, t1, t2):
    c0 = (verts[t0,0] + verts[t1,0] + verts[t2,0]) / 3.0
    c1 = (verts[t0,1] + verts[t1,1] + verts[t2,1]) / 3.0
    c2 = (verts[t0,2] + verts[t1,2] + verts[t2,2]) / 3.0
    c = np.empty(3, dtype=np.float64)
    c[0] = c0; c[1] = c1; c[2] = c2
    return c

@njit(cache=True)
def _tri_raw_normal_by_indices(verts, i0, i1, i2):
    v0 = verts[i0]; v1 = verts[i1]; v2 = verts[i2]
    e1x = v1[0] - v0[0]; e1y = v1[1] - v0[1]; e1z = v1[2] - v0[2]
    e2x = v2[0] - v0[0]; e2y = v2[1] - v0[1]; e2z = v2[2] - v0[2]
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    n = np.empty(3, dtype=np.float64)
    n[0] = nx; n[1] = ny; n[2] = nz
    return n

@njit(cache=True)
def _normalize_inplace(v):
    l = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if l != 0.0:
        v[0] /= l; v[1] /= l; v[2] /= l
    return v

@njit(cache=True)
def compute_and_orient_triangles(verts, tris, center):
    M = tris.shape[0]
    normals = np.zeros((M,3), dtype=np.float64)
    oriented = np.empty((M,3), dtype=np.int64)

    for i in range(M):
        oriented[i,0] = tris[i,0]
        oriented[i,1] = tris[i,1]
        oriented[i,2] = tris[i,2]

    if M == 0:
        return normals, oriented

    E = 3 * M
    edges_a = np.empty(E, dtype=np.int64)
    edges_b = np.empty(E, dtype=np.int64)
    edges_tri = np.empty(E, dtype=np.int64)
    edges_local = np.empty(E, dtype=np.int64)
    e_idx = 0
    for t in range(M):
        a = oriented[t,0]; b = oriented[t,1]; c = oriented[t,2]
        edges_a[e_idx] = a; edges_b[e_idx] = b; edges_tri[e_idx] = t; edges_local[e_idx] = 0; e_idx += 1
        edges_a[e_idx] = b; edges_b[e_idx] = c; edges_tri[e_idx] = t; edges_local[e_idx] = 1; e_idx += 1
        edges_a[e_idx] = c; edges_b[e_idx] = a; edges_tri[e_idx] = t; edges_local[e_idx] = 2; e_idx += 1

    neigh_count = np.zeros(M, dtype=np.int64)
    neigh_tri = np.full((M,3), -1, dtype=np.int64)
    neigh_shared_a = np.full((M,3), -1, dtype=np.int64)
    neigh_shared_b = np.full((M,3), -1, dtype=np.int64)

    for i in range(E):
        a_i = edges_a[i]; b_i = edges_b[i]; tri_i = edges_tri[i]
        for j in range(E):
            if i == j:
                continue
            if edges_a[j] == b_i and edges_b[j] == a_i:
                tri_j = edges_tri[j]
                idx = neigh_count[tri_i]
                if idx < 3:
                    neigh_tri[tri_i, idx] = tri_j
                    neigh_shared_a[tri_i, idx] = a_i
                    neigh_shared_b[tri_i, idx] = b_i
                    neigh_count[tri_i] += 1

    maxd = -1.0
    seed = 0
    for t in range(M):
        t0 = oriented[t,0]; t1 = oriented[t,1]; t2 = oriented[t,2]
        c = _tri_centroid(verts, t0, t1, t2)
        dx = c[0] - center[0]; dy = c[1] - center[1]; dz = c[2] - center[2]
        d = dx*dx + dy*dy + dz*dz
        if d > maxd:
            maxd = d
            seed = t

    visited = np.zeros(M, dtype=np.int64)
    queue = np.full(M, -1, dtype=np.int64)
    q_head = 0; q_tail = 0

    sc = _tri_centroid(verts, oriented[seed,0], oriented[seed,1], oriented[seed,2])
    rn = _tri_raw_normal_by_indices(verts, oriented[seed,0], oriented[seed,1], oriented[seed,2])
    dot_seed = rn[0] * (sc[0] - center[0]) + rn[1] * (sc[1] - center[1]) + rn[2] * (sc[2] - center[2])
    if dot_seed < 0.0:
        tmp = oriented[seed,1]; oriented[seed,1] = oriented[seed,2]; oriented[seed,2] = tmp

    # enqueue seed
    visited[seed] = 1
    queue[q_tail] = seed; q_tail += 1

    while q_head < q_tail:
        cur = queue[q_head]; q_head += 1
        nc = neigh_count[cur]
        for ni in range(nc):
            nbr = neigh_tri[cur, ni]
            if nbr < 0:
                continue
            if visited[nbr]:
                continue
            sa = neigh_shared_a[cur, ni]; sb = neigh_shared_b[cur, ni]
            na0 = oriented[nbr,0]; na1 = oriented[nbr,1]; na2 = oriented[nbr,2]

            if (na0 == sa and na1 == sb) or (na1 == sa and na2 == sb) or (na2 == sa and na0 == sb):
                tmp = oriented[nbr,1]; oriented[nbr,1] = oriented[nbr,2]; oriented[nbr,2] = tmp

            ncent = _tri_centroid(verts, oriented[nbr,0], oriented[nbr,1], oriented[nbr,2])
            nr = _tri_raw_normal_by_indices(verts, oriented[nbr,0], oriented[nbr,1], oriented[nbr,2])
            dote = nr[0] * (ncent[0] - center[0]) + nr[1] * (ncent[1] - center[1]) + nr[2] * (ncent[2] - center[2])
            if dote < 0.0:
                tmp = oriented[nbr,1]; oriented[nbr,1] = oriented[nbr,2]; oriented[nbr,2] = tmp

            visited[nbr] = 1
            queue[q_tail] = nbr; q_tail += 1

    for t in range(M):
        if visited[t]:
            continue
        sc = _tri_centroid(verts, oriented[t,0], oriented[t,1], oriented[t,2])
        rn = _tri_raw_normal_by_indices(verts, oriented[t,0], oriented[t,1], oriented[t,2])
        dot_seed = rn[0] * (sc[0] - center[0]) + rn[1] * (sc[1] - center[1]) + rn[2] * (sc[2] - center[2])
        if dot_seed < 0.0:
            tmp = oriented[t,1]; oriented[t,1] = oriented[t,2]; oriented[t,2] = tmp
        visited[t] = 1
        queue[q_tail] = t; q_tail += 1
        while q_head < q_tail:
            cur = queue[q_head]; q_head += 1
            nc = neigh_count[cur]
            for ni in range(nc):
                nbr = neigh_tri[cur, ni]
                if nbr < 0:
                    continue
                if visited[nbr]:
                    continue
                sa = neigh_shared_a[cur, ni]; sb = neigh_shared_b[cur, ni]
                na0 = oriented[nbr,0]; na1 = oriented[nbr,1]; na2 = oriented[nbr,2]
                if (na0 == sa and na1 == sb) or (na1 == sa and na2 == sb) or (na2 == sa and na0 == sb):
                    tmp = oriented[nbr,1]; oriented[nbr,1] = oriented[nbr,2]; oriented[nbr,2] = tmp
                # extra outward check
                ncent = _tri_centroid(verts, oriented[nbr,0], oriented[nbr,1], oriented[nbr,2])
                nr = _tri_raw_normal_by_indices(verts, oriented[nbr,0], oriented[nbr,1], oriented[nbr,2])
                dote = nr[0] * (ncent[0] - center[0]) + nr[1] * (ncent[1] - center[1]) + nr[2] * (ncent[2] - center[2])
                if dote < 0.0:
                    tmp = oriented[nbr,1]; oriented[nbr,1] = oriented[nbr,2]; oriented[nbr,2] = tmp
                visited[nbr] = 1
                queue[q_tail] = nbr; q_tail += 1

    for t in range(M):
        i0 = oriented[t,0]; i1 = oriented[t,1]; i2 = oriented[t,2]
        n = _tri_raw_normal_by_indices(verts, i0, i1, i2)
        _normalize_inplace(n)
        normals[t,0] = n[0]; normals[t,1] = n[1]; normals[t,2] = n[2]

    return normals, oriented




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
    origin = (0.0, 0.0, 0.0, 0.0)
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
                # object-level translation stored here (but NOT applied now)
                origin = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
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
    # avoid divide by zero
    mask = np.abs(w + dist) < 1e-9
    factor1[mask] = 1.0
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
        tri_colors = []
        tri_alphas = []
        tri_uvs = []

        obj_origin = (0.0, 0.0, 0.0, 0.0)
        if len(facets) > 0:
            obj_origin = facets[0].get('origin', obj_origin)

        for f in facets:
            idxs = []
            for v4 in f['verts4d']:
                key = tuple(v4)
                if key not in vert_map:
                    vert_map[key] = len(verts4d)
                    verts4d.append(np.array(key, dtype=float))
                idxs.append(vert_map[key])
            tris.append((idxs[0], idxs[1], idxs[2]))
            tri_colors.append(np.array(f['color'], dtype=float))
            tri_alphas.append(float(f.get('alpha', 1.0)))
            tri_uvs.append(np.array(f['uvs'], dtype=float))

        verts4d_arr = np.array(verts4d, dtype=float)

        mesh = {
            'name': fdo.get('name', 'unnamed'),
            'basename': basename,
            'azb_path': azb_path,
            'origin': np.array(obj_origin, dtype=float), 
            'verts4d': verts4d_arr,
            'verts_world': None, 
            'tris': np.array(tris, dtype=np.int64),
            'tri_normals_world': None,
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
    opaque = []
    transparent = []

    for m in _loaded_4meshes:
        if m['verts4d'] is None or m['verts4d'].shape[0] == 0:
            verts_world = np.zeros((0,3), dtype=float)
        else:
            verts_world = project_4d_to_3d_array(m['verts4d'], dist=10.0)
        verts_world += m['origin'][:3]
        tris = m['tris']
        tri_normals = None
        if verts_world.shape[0] == 0 or tris.shape[0] == 0:
            tri_normals = np.zeros((len(tris), 3), dtype=float)
        else:
            center = (verts_world.max(axis=0) + verts_world.min(axis=0)) / 2.0
            if np.any(m['alpha'] < 1.0):
                tri_normals = compute_cam_normals(verts_world, tris)
            else:
                normals, oriented_tris = compute_and_orient_triangles(verts_world, tris, center)
                tri_normals = normals
                tris = oriented_tris

        mesh_copy = {
            'name': m['name'],
            'basename': m['basename'],
            'azb_path': m['azb_path'],
            'origin': m['origin'].copy(),
            'verts4d': m['verts4d'].copy(),
            'verts_world': verts_world,
            'tris': np.array(tris, dtype=np.int64),
            'tri_normals_world': np.array(tri_normals, dtype=float),
            'colors': m['colors'].copy(),
            'alpha': m['alpha'].copy(),
            'uvs': list(m['uvs']),
            'texture': m['texture']
        }

        if np.any(mesh_copy['alpha'] < 1.0):
            transparent.append(mesh_copy)
        else:
            opaque.append(mesh_copy)

    return opaque, transparent

def _azb_path(name, folder='4d_models'):
    base = f"{name}.fdo" if not name.endswith('.fdo') else name
    azb = os.path.join(folder, base[:-4] + '.azb')
    return azb

def _recompute_mesh_world_and_normals(m):
    verts4d = m['verts4d']
    origin = m['origin']
    if verts4d is None or verts4d.shape[0] == 0:
        m['verts_world'] = np.zeros((0,3), dtype=float)
        m['tri_normals_world'] = np.zeros((len(m['tris']),3), dtype=float)
        return
    v4_with_origin = verts4d + origin
    m['verts_world'] = project_4d_to_3d_array(v4_with_origin, dist=10.0)
    tris_arr = np.array(m['tris'], dtype=np.int64)
    if m['verts_world'].shape[0] == 0 or tris_arr.shape[0] == 0:
        m['tri_normals_world'] = np.zeros((len(tris_arr),3), dtype=float)
        return
    center = (m['verts_world'].max(axis=0) + m['verts_world'].min(axis=0)) / 2.0
    normals, oriented_tris = compute_and_orient_triangles(m['verts_world'], tris_arr, center)
    m['tri_normals_world'] = normals
    m['tris'] = oriented_tris

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
        m['origin'] = m['origin'] + np.array([dx, dy, dz, dw], dtype=float)
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
    if not os.path.exists(os.path.join(folder, f"{name}.fdo")):
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
            rx, ry, rz, rw = rotate_point_4d(x, y, z, w, angles)
            new_lines.append(f"v {rx} {ry} {rz} {rw}\n")
        else:
            new_lines.append(line)
    with open(azb, 'w') as f:
        f.writelines(new_lines)

    basename = os.path.splitext(name)[0]
    idx = _loaded_4meshes_by_name.get(basename.lower())
    if idx is not None:
        m = _loaded_4meshes[idx]
        rotate4d_numba(m, angles)
    return True

@njit(cache=True, fastmath=True)
def rotate4d_numba(m, angles):
    verts4d = m['verts4d']
    rotated = np.empty_like(verts4d)
    for i, (x, y, z, w) in enumerate(verts4d):
        cos, sin = math.cos, math.sin
        c,s = cos(angles.get("xy",0)), sin(angles.get("xy",0)); x,y = c*x - s*y, s*x + c*y
        c,s = cos(angles.get("xz",0)), sin(angles.get("xz",0)); x,z = c*x - s*z, s*x + c*z
        c,s = cos(angles.get("xw",0)), sin(angles.get("xw",0)); x,w = c*x - s*w, s*x + c*w
        c,s = cos(angles.get("yz",0)), sin(angles.get("yz",0)); y,z = c*y - s*z, s*y + c*z
        c,s = cos(angles.get("yw",0)), sin(angles.get("yw",0)); y,w = c*y - s*w, s*y + c*w
        c,s = cos(angles.get("zw",0)), sin(angles.get("zw",0)); z,w = c*z - s*w, s*z + c*w
        rotated[i,0] = rx; rotated[i,1] = ry; rotated[i,2] = rz; rotated[i,3] = rw
        rx = x; ry = y; rz = z; rw = w
    m['verts4d'] = rotated

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
