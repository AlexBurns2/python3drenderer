import numpy as np
from itertools import product, combinations

# 1️⃣ Generate all 16 vertices of the 4D hypercube
vertices = np.array(list(product([-1, 1], repeat=4)))  # (16, 4)

# 2️⃣ Generate all square faces (2D faces)
faces = []

# Choose 2 axes to vary → each choice defines one family of 2D faces
for vary_axes in combinations(range(4), 2):  # pick 2 out of 4 axes to vary
    const_axes = [a for a in range(4) if a not in vary_axes]
    
    # For each combination of fixed coordinate signs (2^2 = 4)
    for fixed_vals in product([-1, 1], repeat=len(const_axes)):
        face_verts = []
        # For all 4 combinations of the varying coordinates
        for vary_vals in product([-1, 1], repeat=len(vary_axes)):
            v = np.zeros(4, dtype=int)
            v[vary_axes[0]] = vary_vals[0]
            v[vary_axes[1]] = vary_vals[1]
            v[const_axes[0]] = fixed_vals[0]
            v[const_axes[1]] = fixed_vals[1]
            # Find vertex index in main vertex list
            idx = np.where((vertices == v).all(axis=1))[0][0]
            face_verts.append(idx)
        faces.append(face_verts)

tris = []
for f in faces:
    tris.append((f[0], f[1], f[2]))
    tris.append((f[0], f[2], f[3]))
tris = np.array(tris)

print(f"Vertices: {len(vertices)}")
print(f"Faces (quads): {len(faces)}")
print(f"Triangles: {len(tris)}")

print("\n# OBJ faces:")
for tri in tris:
    print(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}")
