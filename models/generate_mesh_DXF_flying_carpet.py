"""
Generate a triangular mesh from a closed curve stored in a DXF file.
The mesh is symmetric about both the x and y axes of the bounding-box centroid.

Dependencies:
    pip install ezdxf triangle numpy matplotlib
"""

import numpy as np
import ezdxf
import triangle as tr
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Extract the closed boundary polyline from the DXF
# ----------------------------------------------------------------------
def extract_boundary(dxf_path, arc_seg=32, close_tol=1e-6):
    """
    Returns an ordered (N, 2) array of boundary vertices.
    Handles LWPOLYLINE, POLYLINE, LINE, ARC, CIRCLE, SPLINE.
    Picks the single longest closed loop found.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments = []  # list of (P0, P1) edges as 2-tuples of np.array

    def add_pts(pts):
        pts = [np.asarray(p[:2], float) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            segments.append((a, b))

    for e in msp:
        t = e.dxftype()
        if t == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points("xy")]
            if e.closed and len(pts) > 1:
                pts.append(pts[0])
            add_pts(pts)
        elif t == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if e.is_closed and len(pts) > 1:
                pts.append(pts[0])
            add_pts(pts)
        elif t == "LINE":
            add_pts([(e.dxf.start.x, e.dxf.start.y),
                     (e.dxf.end.x, e.dxf.end.y)])
        elif t == "ARC":
            a0, a1 = np.radians(e.dxf.start_angle), np.radians(e.dxf.end_angle)
            if a1 <= a0:
                a1 += 2 * np.pi
            ang = np.linspace(a0, a1, arc_seg)
            c, r = e.dxf.center, e.dxf.radius
            add_pts([(c.x + r*np.cos(t_), c.y + r*np.sin(t_)) for t_ in ang])
        elif t == "CIRCLE":
            ang = np.linspace(0, 2*np.pi, arc_seg + 1)
            c, r = e.dxf.center, e.dxf.radius
            add_pts([(c.x + r*np.cos(t_), c.y + r*np.sin(t_)) for t_ in ang])
        elif t == "SPLINE":
            pts = [(p[0], p[1]) for p in e.flattening(distance=0.01)]
            add_pts(pts)

    if not segments:
        raise ValueError("No usable boundary entities found in DXF.")

    # ---- stitch edges into an ordered loop ----
    def key(p):
        return (round(p[0] / close_tol), round(p[1] / close_tol))

    from collections import defaultdict
    adj = defaultdict(list)
    for i, (a, b) in enumerate(segments):
        adj[key(a)].append((key(b), b, a))
        adj[key(b)].append((key(a), a, b))

    used = set()
    loops = []
    for start in list(adj.keys()):
        if start in used:
            continue
        loop = []
        cur = start
        prev = None
        while True:
            used.add(cur)
            nbrs = [n for n in adj[cur] if n[0] != prev]
            if not nbrs:
                break
            nxt_key, nxt_pt, cur_pt = nbrs[0]
            loop.append(cur_pt)
            prev, cur = cur, nxt_key
            if cur == start:
                loop.append(nxt_pt)
                break
            if cur in used:
                break
        if len(loop) >= 3:
            loops.append(np.array(loop))

    # longest loop = outer boundary
    boundary = max(loops, key=lambda L: len(L))

    # drop duplicate closing point
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    return boundary


# ----------------------------------------------------------------------
# 2. Symmetric triangulation helpers
# ----------------------------------------------------------------------
def _flip_axis_edges(verts, tris, cx, cy, tol=1e-6):
    """
    For every interior edge lying exactly on x=cx or y=cy, replace it with
    the opposite diagonal of the surrounding quad.

    Edges where the flip would create a zero-area triangle (happens at the
    center vertex (cx,cy) where both axes intersect) are left untouched.
    """
    tris = tris.copy()

    # directed-edge (a→b) -> (triangle index, local index of opposite vertex)
    edge_map = {}
    for ti, tri in enumerate(tris):
        for loc in range(3):
            a = int(tri[loc])
            b = int(tri[(loc + 1) % 3])
            edge_map[(a, b)] = (ti, (loc + 2) % 3)

    done = set()
    for (a, b), (ti, opp_i) in list(edge_map.items()):
        if (a, b) in done:
            continue
        if (b, a) not in edge_map:
            continue  # boundary edge — skip

        va, vb = verts[a], verts[b]
        on_x = abs(va[0] - cx) < tol and abs(vb[0] - cx) < tol
        on_y = abs(va[1] - cy) < tol and abs(vb[1] - cy) < tol
        if not (on_x or on_y):
            continue

        tj, opp_j = edge_map[(b, a)]
        c = int(tris[ti, opp_i])   # opposite vertex in T1 (a→b side)
        d = int(tris[tj, opp_j])   # opposite vertex in T2 (b→a side)

        # Skip if the new triangles would be degenerate (zero area)
        if abs(np.cross(verts[d] - va, verts[c] - va)) < tol:
            continue
        if abs(np.cross(verts[c] - vb, verts[d] - vb)) < tol:
            continue

        # Flip: T1=(a,b,c) → (a,d,c)  and  T2=(b,a,d) → (b,c,d)
        tris[ti] = [a, d, c]
        tris[tj] = [b, c, d]
        done.add((a, b))
        done.add((b, a))

    return tris



def _clip_halfplane(polygon, axis, threshold):
    """Sutherland-Hodgman clip: keep points where polygon[:,axis] >= threshold."""
    result = []
    n = len(polygon)
    for i in range(n):
        A = polygon[i]
        B = polygon[(i + 1) % n]
        a_in = A[axis] >= threshold - 1e-10
        b_in = B[axis] >= threshold - 1e-10
        if a_in:
            result.append(A.copy())
        if a_in != b_in:
            t = (threshold - A[axis]) / (B[axis] - A[axis])
            result.append(A + t * (B - A))
    return np.array(result) if result else np.empty((0, 2))


def _dedup_boundary(pts, tol=1e-10):
    """Remove consecutive near-duplicate points from a closed boundary."""
    if len(pts) == 0:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    while len(out) > 1 and np.linalg.norm(np.array(out[0]) - np.array(out[-1])) < tol:
        out.pop()
    return np.array(out)


def _merge_close_vertices(verts, tris, tol=1e-6):
    """
    Merge coincident vertices (from mirrored quadrants sharing an axis),
    remove degenerate and duplicate triangles, compact the vertex array.
    """
    n = len(verts)
    canon = np.arange(n, dtype=int)

    inv_tol = 1.0 / tol
    grid = {}
    for i in range(n):
        key = (int(round(verts[i, 0] * inv_tol)),
               int(round(verts[i, 1] * inv_tol)))
        if key in grid:
            canon[i] = grid[key]
        else:
            grid[key] = i

    new_tris = canon[tris]

    v0, v1, v2 = new_tris[:, 0], new_tris[:, 1], new_tris[:, 2]
    valid = (v0 != v1) & (v1 != v2) & (v0 != v2)
    new_tris = new_tris[valid]

    sorted_t = np.sort(new_tris, axis=1)
    _, uid = np.unique(sorted_t, axis=0, return_index=True)
    new_tris = new_tris[np.sort(uid)]

    used = np.unique(new_tris)
    remap = np.full(n, -1, dtype=int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[new_tris]


# ----------------------------------------------------------------------
# 3. Triangulate with x-y symmetry
# ----------------------------------------------------------------------
def mesh_polygon_symmetric(boundary, max_area=None, min_angle=30):
    """
    Constrained Delaunay triangulation with exact x and y mirror symmetry.

    Meshes the first quadrant (x >= cx, y >= cy) with quality constraints,
    then mirrors into the other three quadrants and merges shared axis vertices.
    All four quadrants are identical, so triangle sizes are uniform throughout.
    """
    cx = (boundary[:, 0].min() + boundary[:, 0].max()) / 2.0
    cy = (boundary[:, 1].min() + boundary[:, 1].max()) / 2.0

    # Clip boundary to the first quadrant
    q_bnd = _clip_halfplane(boundary, axis=1, threshold=cy)
    q_bnd = _clip_halfplane(q_bnd, axis=0, threshold=cx)

    q_bnd[np.abs(q_bnd[:, 0] - cx) < 1e-6, 0] = cx
    q_bnd[np.abs(q_bnd[:, 1] - cy) < 1e-6, 1] = cy
    q_bnd = _dedup_boundary(q_bnd)

    if len(q_bnd) < 3:
        raise ValueError(
            "Quarter boundary has < 3 points after clipping. "
            "Check that the DXF boundary is symmetric about its centroid."
        )

    n = len(q_bnd)
    segs = np.array([[i, (i + 1) % n] for i in range(n)])

    opts = f"pq{min_angle}"
    if max_area is not None:
        opts += f"a{max_area}"
    result = tr.triangulate({"vertices": q_bnd, "segments": segs}, opts)
    q_verts = np.array(result["vertices"], dtype=float)
    q_tris  = np.array(result["triangles"], dtype=int)

    # Snap any Steiner points placed on constrained axis edges
    q_verts[np.abs(q_verts[:, 0] - cx) < 1e-6, 0] = cx
    q_verts[np.abs(q_verts[:, 1] - cy) < 1e-6, 1] = cy

    # Mirror into all four quadrants
    # Mirroring an odd number of axes flips CCW → CW; swap cols 1&2 to fix.
    mirrors = [
        ( 1,  1, False),  # Q1: original
        (-1,  1, True),   # Q2: mirror x         → flip orientation
        (-1, -1, False),  # Q3: mirror x and y   → two flips cancel
        ( 1, -1, True),   # Q4: mirror y         → flip orientation
    ]

    all_v, all_t = [], []
    offset = 0
    for sx, sy, flip in mirrors:
        v = q_verts.copy()
        v[:, 0] = cx + sx * (q_verts[:, 0] - cx)
        v[:, 1] = cy + sy * (q_verts[:, 1] - cy)
        t = q_tris.copy() + offset
        if flip:
            t = t[:, [0, 2, 1]]
        all_v.append(v)
        all_t.append(t)
        offset += len(v)

    all_verts = np.vstack(all_v)
    all_tris  = np.vstack(all_t)

    all_verts, all_tris = _merge_close_vertices(all_verts, all_tris)
    all_tris = _flip_axis_edges(all_verts, all_tris, cx, cy)
    return all_verts, all_tris


# ----------------------------------------------------------------------
# 4. Build rotation-free triangle matrix
# ----------------------------------------------------------------------
def build_RF_matrix(triangles):
    """
    Returns RF_triangle_matrix of shape (N_tri, 6).

    Columns 0-2 : node indices of the triangle  [n0, n1, n2]
    Column  3   : global node index opposite to edge 12 (n0-n1) in the neighbour triangle
    Column  4   : global node index opposite to edge 23 (n1-n2) in the neighbour triangle
    Column  5   : global node index opposite to edge 31 (n2-n0) in the neighbour triangle

    Boundary edges (no neighbour) are filled with -1.
    """
    # local edge definitions: (local_i, local_j) -> opposite local index = 3-i-j
    local_edges = [(0, 1), (1, 2), (2, 0)]

    # map frozenset({a,b}) -> list of (tri_idx, opposite_global_node)
    edge_map = {}
    for ti, tri in enumerate(triangles):
        for i, j in local_edges:
            a, b = int(tri[i]), int(tri[j])
            opp = int(tri[3 - i - j])
            key = frozenset((a, b))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append((ti, opp))

    rf = np.full((len(triangles), 6), -1, dtype=np.int32)
    rf[:, :3] = triangles

    for ti, tri in enumerate(triangles):
        for col, (i, j) in enumerate(local_edges):
            key = frozenset((int(tri[i]), int(tri[j])))
            for nti, nopp in edge_map[key]:
                if nti != ti:
                    rf[ti, 3 + col] = nopp
                    break

    return rf


# ----------------------------------------------------------------------
# 5. Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    DXF_PATH = "./models/flying_carpet/flying_carpet_tri.dxf"
    folder   = "./models/flying_carpet/"
    MAX_AREA = 50   # max triangle area in DXF units²; lower = finer mesh

    boundary = extract_boundary(DXF_PATH)
    V, T = mesh_polygon_symmetric(boundary, max_area=MAX_AREA, min_angle=30)
    RF = build_RF_matrix(T)

    print(f"Boundary points: {len(boundary)}")
    print(f"Mesh: {len(V)} vertices, {len(T)} triangles")
    boundary_edges = np.sum(RF[:, 3:] == -1)
    print(f"RF matrix: {RF.shape}, boundary edges: {boundary_edges}")

    # Verify symmetry
    cx = (V[:, 0].min() + V[:, 0].max()) / 2
    cy = (V[:, 1].min() + V[:, 1].max()) / 2
    tol = 1e-4
    not_mirrored_y = sum(
        np.min(np.linalg.norm(V - np.array([v[0], 2*cy - v[1]]), axis=1)) > tol
        for v in V
    )
    not_mirrored_x = sum(
        np.min(np.linalg.norm(V - np.array([2*cx - v[0], v[1]]), axis=1)) > tol
        for v in V
    )
    print(f"Symmetry check: {not_mirrored_y} vertices without y-mirror, "
          f"{not_mirrored_x} without x-mirror  (should be 0)")

    # Save
    np.save(folder + "mesh_vertices.npy", V)
    np.save(folder + "mesh_triangles.npy", T)
    np.save(folder + "mesh_RF_triangles.npy", RF)

    # Plot
    plt.figure(figsize=(7, 7))
    plt.triplot(V[:, 0], V[:, 1], T, lw=0.5, color="steelblue")
    plt.plot(np.append(boundary[:, 0], boundary[0, 0]),
             np.append(boundary[:, 1], boundary[0, 1]),
             "r-", lw=1.5, label="boundary")
    plt.axvline(cx, color="gray", lw=0.8, ls="--", label="symmetry axes")
    plt.axhline(cy, color="gray", lw=0.8, ls="--")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.savefig("mesh.png", dpi=150, bbox_inches="tight")
    plt.show()