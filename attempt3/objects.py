import os
import numpy as np

scene_facets_raw = []

def parse_ascii_stl(path):
    facets = []
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('facet normal'):
            parts = line.split()
            nx, ny, nz = float(parts[2]), float(parts[3]), float(parts[4])
            i += 1
            verts = []
            while i < len(lines) and not lines[i].strip().startswith('endfacet'):
                l = lines[i].strip()
                if l.startswith('vertex'):
                    p = l.split()
                    verts.append((float(p[1]), float(p[2]), float(p[3])))
                i += 1
            if len(verts) == 3:
                facets.append({'normal': (nx, ny, nz), 'verts': verts})
        else:
            i += 1
    return facets

def scan_stl_folder(folder='stl_models'):
    objs = []
    if not os.path.isdir(folder):
        return objs
    for fn in os.listdir(folder):
        if not fn.lower().endswith('.stl'):
            continue
        path = os.path.join(folder, fn)
        facets = parse_ascii_stl(path)
        objs.append({'name': fn, 'facets': facets})
    return objs

def load_scene_from_facets(objects_with_facets):
    meshes = []
    for obj in objects_with_facets:
        facets = obj.get('facets') if 'facets' in obj else obj
        vert_map = {}
        verts = []
        tris = []
        tri_normals = []
        for f in facets:
            idxs = []
            for v in f['verts']:
                key = (float(v[0]), float(v[1]), float(v[2]))
                if key not in vert_map:
                    vert_map[key] = len(verts)
                    verts.append(np.array(key, dtype=float))
                idxs.append(vert_map[key])
            tris.append((idxs[0], idxs[1], idxs[2]))
            v0 = np.array(f['verts'][0], dtype=float)
            v1 = np.array(f['verts'][1], dtype=float)
            v2 = np.array(f['verts'][2], dtype=float)
            n = np.cross(v1 - v0, v2 - v0)
            nn = n / (np.linalg.norm(n) if np.linalg.norm(n) != 0 else 1.0)
            tri_normals.append(nn)
        meshes.append({'name': obj.get('name','unnamed'), 'verts_world': np.array(verts, dtype=float), 'tris': tris, 'tri_normals_world': np.array(tri_normals, dtype=float)})
    return meshes

if __name__ == '__main__':
    pass