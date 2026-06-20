"""
Generate a triangular mesh from a closed curve stored in a DXF file.

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
# 2. Triangulate the polygon
# ----------------------------------------------------------------------
def mesh_polygon(boundary, max_area=None, min_angle=30):
    """
    Constrained Delaunay triangulation of the interior.
    max_area: target max triangle area (None = no refinement).
    min_angle: minimum angle quality constraint.
    """
    n = len(boundary)
    segs = np.array([[i, (i + 1) % n] for i in range(n)])
    A = {"vertices": boundary, "segments": segs}

    opts = "p"            # planar straight line graph
    opts += f"q{min_angle}"
    if max_area is not None:
        opts += f"a{max_area}"
    B = tr.triangulate(A, opts)
    return B["vertices"], B["triangles"]


# ----------------------------------------------------------------------
# 3. Build rotation-free triangle matrix
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
# 4. Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DXF_PATH = "./models/flat_tri_surface/flat_tri_surface.dxf"          # <-- your file
    DXF_PATH = "./models/flying_carpet/flying_carpet_tri.dxf" 
    folder = "./models/flying_carpet/"
    MAX_AREA = 100            # e.g. 5.0 to control density; None = coarse

    boundary = extract_boundary(DXF_PATH)
    V, T = mesh_polygon(boundary, max_area=MAX_AREA, min_angle=30)
    RF = build_RF_matrix(T)
    print(f"Boundary points: {len(boundary)}")
    print(f"Mesh: {len(V)} vertices, {len(T)} triangles")
    boundary_edges = np.sum(RF[:, 3:] == -1)
    print(f"RF matrix: {RF.shape}, boundary edges: {boundary_edges}")
    print("mesh_vertices: ", V)
    # save
    np.save(folder + "mesh_vertices.npy", V)
    np.save(folder + "mesh_triangles.npy", T)
    np.save(folder + "mesh_RF_triangles.npy", RF)

    # plot
    plt.figure(figsize=(7, 7))
    plt.triplot(V[:, 0], V[:, 1], T, lw=0.5, color="steelblue")
    plt.plot(np.append(boundary[:, 0], boundary[0, 0]),
             np.append(boundary[:, 1], boundary[0, 1]),
             "r-", lw=1.5, label="boundary")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.savefig("mesh.png", dpi=150, bbox_inches="tight")
    plt.show()