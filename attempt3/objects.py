import os
import re

STL_FOLDER = "stl_models"

def parse_ascii_stl(path):
    """
    Parse an ASCII STL file into a list of facet dicts like:
    {'normal': (nx, ny, nz), 'verts': [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]}
    """
    facets = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return facets

    normal_pattern = re.compile(r"facet normal\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")
    vertex_pattern = re.compile(r"vertex\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")

    current_normal = None
    current_verts = []

    for line in lines:
        line = line.strip()
        if line.startswith("facet normal"):
            m = normal_pattern.match(line)
            if m:
                current_normal = tuple(float(m.group(i)) for i in range(1, 4))
        elif line.startswith("vertex"):
            m = vertex_pattern.match(line)
            if m:
                v = tuple(float(m.group(i)) for i in range(1, 4))
                current_verts.append(v)
        elif line.startswith("endfacet"):
            if current_normal and len(current_verts) == 3:
                facets.append({"normal": current_normal, "verts": current_verts})
            current_normal = None
            current_verts = []

    print(f"[INFO] Loaded {len(facets)} facets from {os.path.basename(path)}")
    return facets


def scan_stl_folder(folder=STL_FOLDER):
    """
    Scan a folder for STL files and return a dict of objects:
    { 'filename': [facet_dicts], ... }
    """
    objs = {}
    if not os.path.exists(folder):
        print(f"[WARNING] STL folder '{folder}' not found.")
        return objs

    for fname in os.listdir(folder):
        if fname.lower().endswith(".stl"):
            fpath = os.path.join(folder, fname)
            facets = parse_ascii_stl(fpath)
            if facets:
                objs[fname] = facets

    print(f"[INFO] Total STL objects loaded: {len(objs)}")
    return objs


def load_scene_from_facets():
    """
    Convert all parsed STL files into one flat list (for rendering).
    """
    objs = scan_stl_folder()
    scene_facets = []
    for name, facets in objs.items():
        scene_facets.extend(facets)
    print(f"[INFO] Scene contains {len(scene_facets)} total facets across {len(objs)} object(s).")
    return scene_facets


# Preload scene
scene_facets = load_scene_from_facets()