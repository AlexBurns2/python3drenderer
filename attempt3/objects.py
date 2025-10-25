import numpy as np


scene_facets = [
{'normal': (0.0,0.0,1.0), 'verts': [(1,1,1),(-1,1,1),(-1,-1,1)]},
{'normal': (0.0,0.0,1.0), 'verts': [(1,1,1),(-1,-1,1),(1,-1,1)]},
{'normal': (0.0,-1.0,0.0), 'verts': [(1,-1,-1),(1,-1,1),(-1,-1,1)]},
{'normal': (0.0,-1.0,0.0), 'verts': [(1,-1,-1),(-1,-1,1),(-1,-1,-1)]},
{'normal': (-1.0,0.0,0.0), 'verts': [(-1,-1,-1),(-1,-1,1),(-1,1,1)]},
{'normal': (-1.0,0.0,0.0), 'verts': [(-1,-1,-1),(-1,1,1),(-1,1,-1)]},
{'normal': (0.0,0.0,-1.0), 'verts': [(-1,1,-1),(1,1,-1),(1,-1,-1)]},
{'normal': (0.0,0.0,-1.0), 'verts': [(-1,1,-1),(1,-1,-1),(-1,-1,-1)]},
{'normal': (1.0,0.0,0.0), 'verts': [(1,1,-1),(1,1,1),(1,-1,1)]},
{'normal': (1.0,0.0,0.0), 'verts': [(1,1,-1),(1,-1,1),(1,-1,-1)]},
{'normal': (0.0,1.0,0.0), 'verts': [(-1,1,-1),(-1,1,1),(1,1,1)]},
{'normal': (0.0,1.0,0.0), 'verts': [(-1,1,-1),(1,1,1),(1,1,-1)]}
]


def load_scene_from_facets(facets):
    vert_map = {}
    verts = []
    tris = []
    for f in facets:
        idxs = []
        for v in f['verts']:
            key = (float(v[0]), float(v[1]), float(v[2]))
            if key not in vert_map:
                vert_map[key] = len(verts)
                verts.append(np.array(key, dtype=float))
            idxs.append(vert_map[key])
        tris.append((idxs[0], idxs[1], idxs[2]))
    return [{'verts_world': np.array(verts, dtype=float), 'tris': tris}]