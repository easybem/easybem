import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot3dSurface(P, T, C=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if C is not None:
        surf = ax.plot_trisurf(
            P[:, 0], P[:, 1], P[:, 2],
            triangles=T-1,
            cmap="viridis",
            array=C,
            linewidth=0.2,
            antialiased=True,
            shade=False
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    else:
        ax.plot_trisurf(
            P[:, 0], P[:, 1], P[:, 2],
            triangles=T-1,
            color="lightgray",
            edgecolor="k",
            linewidth=0.2,
            shade=False
        )

    ax.set_box_aspect([
        np.ptp(P[:,0]),
        np.ptp(P[:,1]),
        np.ptp(P[:,2])
    ])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

def next_nonempty(f):
    for line in f:
        if line.strip():
            return line
    raise StopIteration

def load3dSurface(filename):
    with open(filename, "r") as f:

        dim = int(next_nonempty(f))
        nvert_per_elem = int(next(f))

        if dim != 3 or nvert_per_elem != 3:
            raise ValueError("Expected a 3D triangular surface.")

        n = int(next_nonempty(f))

        P = np.array([
            list(map(float, next_nonempty(f).split()))
            for _ in range(n)
        ])

        m = int(next_nonempty(f))

        T = np.array([
            list(map(int, next_nonempty(f).split()))
            for _ in range(m)
        ])

        m2 = int(next_nonempty(f))
        if m2 != m:
            raise ValueError("Triangle count mismatch.")

        Tidx = np.array([
            int(next_nonempty(f).split()[1])
            for _ in range(m)
        ])

    return P, T, Tidx


def load3dSurfaceFromNetgen(file):
    """
    Load surface mesh (triangles) from Netgen .mesh3d/.vol file.

    Returns
    -------
    P : (N,3) float array
        Node coordinates
    T : (M,3) int array
        Triangle connectivity (0-based indexing)
    Tidx : (M,) int array
        Surface region / boundary id
    """

    with open(file, "r") as f:
        lines = f.readlines()

    # -----------------------------
    # helper: find section index
    # -----------------------------
    def find_section(name):
        for i, l in enumerate(lines):
            if l.strip().lower() == name.lower():
                return i
        return -1

    # -----------------------------
    # parse POINTS
    # -----------------------------
    i = find_section("points")
    n_points = int(lines[i + 1].strip())

    P = np.zeros((n_points, 3), dtype=float)
    for k in range(n_points):
        P[k] = list(map(float, lines[i + 2 + k].split()[:3]))

    # -----------------------------
    # parse surface elements
    # -----------------------------
    i = find_section("surfaceelementsgi")
    n_tri = int(lines[i + 1].strip())

    T = np.zeros((n_tri, 3), dtype=int)
    Tidx = np.zeros(n_tri, dtype=int)

    for k in range(n_tri):
        parts = lines[i + 2 + k].split()

        # format:
        # surfnr bcnr domin domout np p1 p2 p3 ...
        surfnr = int(parts[0])
        nverts = int(parts[4])  # should be 3 for triangles

        # node indices are next entries
        nodes = list(map(int, parts[5:5 + nverts]))

        # 1-based indexing
        T[k, :] = np.array(nodes[:3])
        Tidx[k] = surfnr

    return P, T, Tidx


def refineMesh(P, T, Tidx):
    """
    Uniform 1-to-4 refinement of a triangular surface mesh.

    Parameters
    ----------
    P : (n,3) array
        Vertex coordinates
    T : (m,3) array
        Triangle indices (0-based)
    Tidx : (m,) array
        Surface labels

    Returns
    -------
    P2 : (n2,3)
        Refined vertices
    T2 : (4m,3)
        Refined triangles
    Tidx2 : (4m,)
        Refined triangle labels
    """

    P = np.asarray(P)
    T = np.asarray(T)-1
    Tidx = np.asarray(Tidx)

    n = P.shape[0]

    # ---- 1. All triangle edges (vectorized) ----
    i0 = T[:, 0]
    i1 = T[:, 1]
    i2 = T[:, 2]

    edges = np.vstack([
        np.stack([i0, i1], axis=1),
        np.stack([i1, i2], axis=1),
        np.stack([i2, i0], axis=1)
    ])  # (3m,2)

    # sort endpoints so edges are undirected
    edges = np.sort(edges, axis=1)

    # ---- 2. Unique edges + inverse mapping ----
    edges_unique, inv = np.unique(edges, axis=0, return_inverse=True)

    # ---- 3. Midpoint coordinates (vectorized) ----
    midpoints = 0.5 * (P[edges_unique[:, 0]] + P[edges_unique[:, 1]])

    # new vertex indices for midpoints
    mid_idx = n + np.arange(len(edges_unique))

    # ---- 4. Build new vertex array ----
    P2 = np.vstack([P, midpoints])

    # ---- 5. Map old edges → midpoint indices ----
    # inv has size 3m, corresponds to edges in order above
    m01 = mid_idx[inv[:len(i0)]]
    m12 = mid_idx[inv[len(i0):2*len(i0)]]
    m20 = mid_idx[inv[2*len(i0):]]

    # ---- 6. Build 4 triangles per old triangle ----
    T0 = np.stack([i0,  m01, m20], axis=1)
    T1 = np.stack([m01, i1,  m12], axis=1)
    T2 = np.stack([m20, m12, i2 ], axis=1)
    T3 = np.stack([m01, m12, m20], axis=1)

    T2_new = np.vstack([T0, T1, T2, T3])

    # ---- 7. Propagate triangle labels ----
    Tidx2 = np.repeat(Tidx, 4)

    return P2, T2_new+1, Tidx2
