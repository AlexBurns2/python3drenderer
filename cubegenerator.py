import numpy as np
from itertools import product

# Generate cube vertices (1-based indexing for printing)
cubeverts = np.array(list(product([-1, 1], repeat=3)))
# Adjust indices for printing like your example (1..8)
vertex_indices = np.arange(1, 9)

# Generate triangles procedurally
def generate_cube_tris(vertices):
    tris = []

    # Map from vertex coordinates to index (for quick lookup)
    vert_map = {tuple(v): i+1 for i, v in enumerate(vertices)}  # 1-based

    # Each cube face is perpendicular to an axis
    axes = [0, 1, 2]  # x, y, z
    for axis in axes:
        # For each face on that axis (coord = -1 or +1)
        for coord in [-1, 1]:
            # Get the 4 vertices on this face
            face = [v for v in vertices if v[axis] == coord]

            # Pick a consistent diagonal to split into two triangles
            # Tri 1: v0, v1, v2 ; Tri 2: v0, v2, v3
            # Sort by remaining axes to maintain consistent winding
            other_axes = [a for a in range(3) if a != axis]
            face.sort(key=lambda v: (v[other_axes[0]], v[other_axes[1]]))

            v0, v1, v2, v3 = face
            tris.append([vert_map[tuple(v0)], vert_map[tuple(v1)], vert_map[tuple(v2)]])
            tris.append([vert_map[tuple(v0)], vert_map[tuple(v2)], vert_map[tuple(v3)]])
    return tris

cubetris = generate_cube_tris(cubeverts)

# Print triangles in "f i j k" format
for tri in cubetris:
    print("f " + " ".join(map(str, tri)))