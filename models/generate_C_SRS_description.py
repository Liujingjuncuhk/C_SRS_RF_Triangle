import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pyvista as pv

thickness = 0.005 # meter
Youngs_modulus = 4.2e6 # Pa
Poisson_ratio = 0.3
density = 961 # kg/m^3

folder = "./models/flat_tri_surface/"
mesh_vertices = np.load(folder + "mesh_vertices.npy") * 1e-3
mesh_triangles = np.load(folder + "mesh_triangles.npy")
mesh_RF_triangles = np.load(folder + "mesh_RF_triangles.npy")
# for i in range(mesh_RF_triangles.shape[0]):
#     for j in range(mesh_RF_triangles.shape[1]):
#         if mesh_RF_triangles[i,j] < 0:
#             print(f"Error: mesh_RF_triangles[{i},{j}] is negative, the value is {mesh_RF_triangles[i,j]}")
# exit(0)
print("mesh_vertices shape:", mesh_vertices.shape)
# extend 0 on the z-axis for 2D mesh
mesh_vertices = np.hstack((mesh_vertices, np.zeros((mesh_vertices.shape[0], 1))))
print("mesh_vertices shape:", mesh_vertices.shape)


def _cot(u, v):
    """Cotangent of the angle between vectors u and v."""
    dot = np.dot(u, v)
    sin = np.linalg.norm(np.cross(u, v))
    return dot / max(sin, 1e-12)


def build_bending_elements(mesh_vertices, mesh_triangles, k_bend=1.0):
    """
    Linear bending-curvature stencil + ShapeUp/Projective-Dynamics weight for every
    interior (non-boundary) edge of a flat-rest triangle mesh.
 
    Stencil: gradient of the dihedral angle at the flat configuration
    (Bridson et al. 2003; Wardetzky et al. 2007). For each interior edge:
        v1, v2  = shared edge endpoints
        v3      = apex of triangle A = (v1, v2, v3)
        v4      = apex of triangle B = (v1, v2, v4)
    The hinge bending curvature is linear in the four node positions:
        K = c0*x_v1 + c1*x_v2 + c2*x_v3 + c3*x_v4
    Each c-row sums to zero (translation invariance); K = 0 at the flat rest state.
 
    Weight: quadratic-bending stiffness of Bergou et al. 2006, as used by the
    ShapeUp / Projective Dynamics bending constraint (Bouaziz et al. 2012, 2014 §5.4):
        w_e = k_bend * 3 * ||e||^2 / Abar_e
    where ||e|| is the shared-edge rest length and Abar_e = (A_A + A_B)/3 is the
    combined (one-third) area of the two incident triangles. The bending energy of
    the hinge is  0.5 * w_e * ||K||^2, contributing w_e * c c^T to the global matrix.
 
    Parameters
    ----------
    mesh_vertices  : (N, 3) array of initial (rest) vertex positions
    mesh_triangles : (M, 3) int array of triangle vertex indices
    k_bend         : float, scalar bending stiffness (material/thickness factor)
 
    Returns
    -------
    edge_vertices : (n_inner, 4) int array    rows [v1, v2, v3, v4]
    c_val_matrix  : (n_inner, 4) float array  rows [c0, c1, c2, c3]
    weight_list   : (n_inner,)  float array   ShapeUp weights w_e
    """
    V = np.asarray(mesh_vertices, dtype=float)
    F = np.asarray(mesh_triangles, dtype=np.int64)
 
    edge_map = {}
    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b, apex in ((i, j, k), (j, k, i), (k, i, j)):
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append(apex)
 
    edge_vertices = []
    c_val_matrix = []
    weight_list = []
 
    for (v1, v2), apexes in edge_map.items():
        if len(apexes) != 2:
            continue  # boundary (1 triangle) or non-manifold (>2): no bending
        v3, v4 = apexes
        p1, p2, p3, p4 = V[v1], V[v2], V[v3], V[v4]
 
        e = p2 - p1
        Le = np.linalg.norm(e)
        if Le < 1e-14:
            continue
 
        # apex heights to the shared-edge line
        ee = np.dot(e, e)
        proj3 = p1 + (np.dot(p3 - p1, e) / ee) * e
        proj4 = p1 + (np.dot(p4 - p1, e) / ee) * e
        hA = np.linalg.norm(p3 - proj3)
        hB = np.linalg.norm(p4 - proj4)
 
        # cotangents of the angles at the edge endpoints, per triangle
        a1 = _cot(p3 - p1, p2 - p1)   # angle at v1 in A
        a2 = _cot(p3 - p2, p1 - p2)   # angle at v2 in A
        b1 = _cot(p4 - p1, p2 - p1)   # angle at v1 in B
        b2 = _cot(p4 - p2, p1 - p2)   # angle at v2 in B
 
        # barycentric split of each apex's contribution onto the two endpoints
        wA1 = a2 / (a1 + a2)
        wA2 = a1 / (a1 + a2)
        wB1 = b2 / (b1 + b2)
        wB2 = b1 / (b1 + b2)
 
        c3 = Le / hA
        c4 = Le / hB
        c1 = -(Le / hA) * wA1 - (Le / hB) * wB1
        c2 = -(Le / hA) * wA2 - (Le / hB) * wB2
 
        # combined hinge area Abar = (A_A + A_B)/3  and Bergou bending weight
        AA = 0.5 * Le * hA
        AB = 0.5 * Le * hB
        Abar = (AA + AB) / 3.0
        w_e = k_bend * 3.0 * Le * Le / Abar
 
        edge_vertices.append([v1, v2, v3, v4])
        c_val_matrix.append([c1, c2, c3, c4])
        weight_list.append(w_e)
 
    return (np.array(edge_vertices, dtype=np.int64),
            np.array(c_val_matrix, dtype=float),
            np.array(weight_list, dtype=float))

def get_boundary_vertex_indices(mesh_triangles):
    edge_count = defaultdict(int)
    for tri in mesh_triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_count[edge] += 1
    boundary_vertices = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertices.update(edge)
    return sorted(boundary_vertices)

def ebst_stiffness(indices, X, E, nu, t, tol=1e-12):
    """
    EBST elemental stiffness, 18x18, global coords.
    Rotation-free triangular shell: 3 translational DOFs/node, 6-node patch.

    indices : length-6 list of global node ids.
        [0,1,2] central triangle M (must be valid, != -1)
        [3,4,5] neighbour across side 0=edge(1,2), 1=edge(2,0), 2=edge(0,1).
                Entry -1 => ghost (boundary/free edge): that side contributes
                no bending AND its 3 rows/cols in K are explicitly zeroed.
    X    : (6,3) coordinates, same order as indices; ghost slots may be NaN.
    E,nu,t : material + thickness.

    Returns (K, dofmap):
        K      : (18,18) symmetric; ghost-slot rows/cols are exactly 0.
        dofmap : (18,) global DOF indices [3g,3g+1,3g+2] per slot; ghost slots -1.
    """
    indices = list(indices)
    X = np.asarray(X, float)
    if any(indices[k] == -1 for k in (0, 1, 2)):
        raise ValueError("central triangle nodes (slots 0,1,2) cannot be ghosts")

    C  = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2]])
    Dm = E * t    / (1 - nu**2)        * C
    Db = E * t**3 / (12 * (1 - nu**2)) * C

    x0, x1, x2 = X[0], X[1], X[2]
    v1, v2 = x1 - x0, x2 - x0
    n = np.cross(v1, v2); nn = np.linalg.norm(n)
    if nn < tol:
        raise ValueError("central triangle is degenerate")
    A_M = 0.5 * nn
    t3 = n / nn
    e1 = v1 / np.linalg.norm(v1)
    e2 = np.cross(t3, e1)

    def cst_grad(P3):
        """Shape-fn gradients of triangle P3 in the central (e1,e2) frame,
        computed in the triangle's own plane (fold-robust). None if degenerate."""
        u1, u2 = P3[1] - P3[0], P3[2] - P3[0]
        nt = np.cross(u1, u2); a2 = np.linalg.norm(nt)
        if a2 < tol:
            return None
        f1 = u1 / np.linalg.norm(u1)
        f3 = nt / a2
        f2 = np.cross(f3, f1)
        Q = np.vstack([f1, f2])
        p = (P3 - P3[0]) @ Q.T
        (xa, ya), (xb, yb), (xc, yc) = p
        det = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
        b = np.array([yb - yc, yc - ya, ya - yb]) / det
        c = np.array([xc - xb, xa - xc, xb - xa]) / det
        T = np.array([[f1 @ e1, f1 @ e2], [f2 @ e1, f2 @ e2]])
        return np.column_stack([b, c]) @ T

    dNm = cst_grad(X[0:3])

    # membrane: CST on central triangle
    Bm = np.zeros((3, 18))
    for I in range(3):
        Na, Nb = dNm[I]; col = slice(3 * I, 3 * I + 3)
        Bm[0, col] = Na * e1
        Bm[1, col] = Nb * e2
        Bm[2, col] = Na * e2 + Nb * e1

    # bending: skip sides whose neighbour is a ghost
    side_nodes = {0: (1, 2, 3), 1: (2, 0, 4), 2: (0, 1, 5)}
    Bb = np.zeros((3, 18))
    for i in range(3):
        nds = side_nodes[i]
        if indices[nds[2]] == -1 or np.any(~np.isfinite(X[nds[2]])):
            continue
        dNi = cst_grad(X[list(nds)])
        if dNi is None:
            continue
        NMa, NMb = dNm[i]
        for k, J in enumerate(nds):
            Ta, Tb = dNi[k]
            cc = np.array([NMa * Ta, NMb * Tb, NMa * Tb + NMb * Ta])
            Bb[:, 3 * J:3 * J + 3] += np.outer(cc, t3)

    K = A_M * (Bm.T @ Dm @ Bm) + A_M * (Bb.T @ Db @ Bb)

    # zero out ghost-slot rows/cols; build dofmap
    dofmap = np.empty(18, dtype=int)
    for slot, g in enumerate(indices):
        base = slot * 3
        if g == -1:
            K[base:base + 3, :] = 0.0
            K[:, base:base + 3] = 0.0
            dofmap[base:base + 3] = -1
        else:
            dofmap[base:base + 3] = [3 * g, 3 * g + 1, 3 * g + 2]
    return K

def generate_edge_matrix(mesh_vertices, mesh_triangles):
    edge_list = []
    weight_list = []
    edge_ele_list = []
    for tri in mesh_triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            if edge not in edge_list:
                edge_list.append(edge)
    # each edge is shared by 1 or 2 triangles, get the weight for each edge
    for edge in edge_list:
        count = 0
        for tri in mesh_triangles:
            if edge[0] in tri and edge[1] in tri:
                count += 1
        if count == 1:
            # find the triangle that contains this edge and get its area
            for tri in mesh_triangles:
                if edge[0] in tri and edge[1] in tri:
                    # find the angle of the other two edges in this triangle
                    v0, v1, v2 = tri
                    if v0 not in edge:
                        other_vertex = v0
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    elif v1 not in edge:
                        other_vertex = v1
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    elif v2 not in edge:
                        other_vertex = v2
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    weight_list.append(0.5 / np.tan(angle))
        elif count == 2:
            weight_this = 0
            for tri in mesh_triangles:
                # print("tri:", tri)
                if edge[0] in tri and edge[1] in tri:
                    # find the angle of the other two edges in this triangle
                    v0, v1, v2 = tri

                    if v0 not in edge:
                        other_vertex = v0
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    elif v1 not in edge:
                        other_vertex = v1
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    elif v2 not in edge:
                        other_vertex = v2
                        angle = np.arccos(np.dot(mesh_vertices[edge[0]] - other_vertex, mesh_vertices[edge[1]] - other_vertex) / (np.linalg.norm(mesh_vertices[edge[0]] - other_vertex) * np.linalg.norm(mesh_vertices[edge[1]] - other_vertex)))
                    weight_this += 0.5 / np.tan(angle)
            weight_list.append(weight_this)
        else:
            print("error")
    return edge_list, weight_list

def cal_area_list(mesh_vertices, mesh_triangles):
    area_list = []
    for tri in mesh_triangles:
        v0, v1, v2 = mesh_vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        area_list.append(area)
    return np.array(area_list)

def cal_mass_matrix(mesh_vertices, mesh_triangles, density, thickness):
    num_vertices = len(mesh_vertices)
    M = np.zeros((num_vertices * 3, num_vertices * 3))
    for tri in mesh_triangles:
        v0, v1, v2 = mesh_vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        mass = density * thickness * area
        for i in range(3):
            idx = tri[i] * 3
            M[idx:idx+3, idx:idx+3] += (mass / 3) * np.eye(3)
    return M

def generate_neighbour_list(mesh_triangles, num_vertices, edge_list):
    neighbour_list = [[] for _ in range(num_vertices)]
    neighbout_edge_list = [[] for _ in range(num_vertices)]
    for tri in mesh_triangles:
        for i in range(3):
            v = tri[i]
            neighbours = [tri[j] for j in range(3) if j != i]
            neighbour_list[v].extend(neighbours)
    # Remove duplicates
    neighbour_list = [list(set(neigh)) for neigh in neighbour_list]
    for i in range(num_vertices):
        # append the index of edge in edge_list for each neighbour
        for neigh in neighbour_list[i]:
            edge = tuple(sorted((i, neigh)))
            if edge in edge_list:
                edge_index = edge_list.index(edge)
                neighbout_edge_list[i].append(edge_index)
    return neighbour_list, neighbout_edge_list


def visualize_mesh_neighbour(mesh_vertices, mesh_triangles, neighbour_list, idx):
    fig, ax = plt.subplots()
    ax.triplot(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_triangles, color='lightgray')
    ax.scatter(mesh_vertices[idx, 0], mesh_vertices[idx, 1], color='red', label='Vertex {}'.format(idx))
    for neigh in neighbour_list[idx]:
        ax.scatter(mesh_vertices[neigh, 0], mesh_vertices[neigh, 1], color='blue', label='Neighbour {}'.format(neigh))
    ax.legend()
    plt.show()

def get_ARAP_initial_SK_list(mesh_vertices, mesh_triangles, edge_list, weight_list, neighbour_list, neighbour_edge_list):
    initial_SK_list = []
    for i in range(len(mesh_vertices)):
        SK_i = []
        for j, neigh in enumerate(neighbour_list[i]):
            edge_index = neighbour_edge_list[i][j]
            weight = weight_list[edge_index]
            SK_i.append(weight * (mesh_vertices[i] - mesh_vertices[neigh]))

        SK_i = np.array(SK_i)
        initial_SK_list.append(SK_i)
    return initial_SK_list

def visualize_3d_mesh(mesh_vertices, mesh_triangles, pp_idx, pulley_locations,ee_idx):
    # use pv to visualize the mesh and the pullpoints and pulleys
    mesh = pv.PolyData(mesh_vertices, np.hstack((np.full((mesh_triangles.shape[0], 1), 3), mesh_triangles)))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    plotter.add_points(mesh_vertices[pp_idx], color='blue', point_size=10
                        , label='Pullpoints')
    plotter.add_points(pulley_locations, color='blue', point_size=10
                        , label='Pulleys')
    # add lines between pullpoints and pulleys
    for i in range(len(pp_idx)):
        plotter.add_lines(np.array([mesh_vertices[pp_idx[i]], pulley_locations[i]]), color='blue', width=2)
    # annotate ee vertices
    plotter.add_points(mesh_vertices[ee_idx], color='red', point_size=10
                        , label='EE vertices')
    plotter.add_legend()
    plotter.show()


def build_RF_matrix(triangles):
    local_edges = [(0, 1), (1, 2), (2, 0)]
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


def find_containing_triangle(mesh_vertices, mesh_triangles, point):
    px, py = point[0], point[1]
    for ti, tri in enumerate(mesh_triangles):
        ax, ay = mesh_vertices[tri[0], 0], mesh_vertices[tri[0], 1]
        bx, by = mesh_vertices[tri[1], 0], mesh_vertices[tri[1], 1]
        cx, cy = mesh_vertices[tri[2], 0], mesh_vertices[tri[2], 1]
        d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
        d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
        d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        if not (has_neg and has_pos):
            return ti
    return -1


def generate_C_SRS_description(mesh_vertices, mesh_triangles, pullpoint_locations, pulley_locations, ee_vertices_list, density, thickness, Youngs_modulus, Poisson_ratio):
    k_m = Youngs_modulus * thickness / (1-Poisson_ratio**2)
    k_b = Youngs_modulus * thickness**3 / (12 * (1-Poisson_ratio**2))
    def K(Vx, ev2, c2):
        return np.array([c2[e,0]*Vx[ev2[e,0]]+c2[e,1]*Vx[ev2[e,1]]+
                        c2[e,2]*Vx[ev2[e,2]]+c2[e,3]*Vx[ev2[e,3]]
                        for e in range(len(ev2))])
    # Add a new vertex at each pullpoint by splitting the containing triangle into 3
    pp_idx = []
    mesh_triangles_list = mesh_triangles.tolist()
    for i in range(len(pullpoint_locations)):
        new_vert = pullpoint_locations[i].copy()
        ti = find_containing_triangle(mesh_vertices, np.array(mesh_triangles_list, dtype=int), new_vert)
        if ti == -1:
            # fallback: compute interior vertices on the fly and snap the closest one
            edge_count_tmp = defaultdict(int)
            for tri in mesh_triangles_list:
                for k in range(3):
                    edge = tuple(sorted((tri[k], tri[(k + 1) % 3])))
                    edge_count_tmp[edge] += 1
            boundary_tmp = set()
            for edge, count in edge_count_tmp.items():
                if count == 1:
                    boundary_tmp.update(edge)
            interior_tmp = np.array([j for j in range(len(mesh_vertices)) if j not in boundary_tmp])
            dists = np.linalg.norm(mesh_vertices[interior_tmp] - new_vert, axis=1)
            idx = interior_tmp[np.argmin(dists)]
            mesh_vertices[idx] = new_vert
            pp_idx.append(idx)
            print(f"Warning: pullpoint {i} not inside any triangle, falling back to closest interior vertex {idx}")
            continue
        new_idx = len(mesh_vertices)
        mesh_vertices = np.vstack([mesh_vertices, new_vert.reshape(1, 3)])
        v0, v1, v2 = mesh_triangles_list[ti][0], mesh_triangles_list[ti][1], mesh_triangles_list[ti][2]
        mesh_triangles_list[ti] = [v0, v1, new_idx]
        mesh_triangles_list.append([v1, v2, new_idx])
        mesh_triangles_list.append([v2, v0, new_idx])
        pp_idx.append(new_idx)
    mesh_triangles = np.array(mesh_triangles_list, dtype=int)
    mesh_RF_triangles_updated = build_RF_matrix(mesh_triangles)

    # identify boundary vertices after remeshing
    edge_count = defaultdict(int)
    for tri in mesh_triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_count[edge] += 1
    boundary_vertex_set = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertex_set.update(edge)

    ee_idx = []
    for i in range(len(ee_vertices_list)):
        idx = np.argmin(np.linalg.norm(mesh_vertices - ee_vertices_list[i], axis=1))
        mesh_vertices[idx] = ee_vertices_list[i]
        ee_idx.append(idx)
    print("ee_idx:", ee_idx)

    edge_list, weight_list = generate_edge_matrix(mesh_vertices, mesh_triangles)
    bending_ele_idx, bending_ele_param, bending_weight_list = build_bending_elements(mesh_vertices, mesh_triangles)
    bending_weight_list = [k_b * w for w in bending_weight_list]    
    # print("\nmulti-edge mesh: interior edges =", len(ev2))
    # print("flat-rest |K| max:", np.abs(K(mesh_vertices, ev2, c2)).max())

    neighbour_list, neighbour_edge_list = generate_neighbour_list(mesh_triangles, len(mesh_vertices), edge_list)
    area_list = cal_area_list(mesh_vertices, mesh_triangles)
    mem_weight_list = [k_m * area for area in area_list]
    initial_SK_list = get_ARAP_initial_SK_list(mesh_vertices, mesh_triangles, edge_list, weight_list, neighbour_list, neighbour_edge_list)
    stiffness_matrices = []
    for tri in mesh_RF_triangles_updated:
        X = mesh_vertices[tri]
        K = ebst_stiffness(tri, X, Youngs_modulus, Poisson_ratio, thickness)
        stiffness_matrices.append(K)
    mass_mat = cal_mass_matrix(mesh_vertices, mesh_triangles, density, thickness)
    visualize_3d_mesh(mesh_vertices, mesh_triangles, pp_idx, pulley_locations, ee_idx)
    with open(folder + "C_SRS_description.pkl", "wb") as f:
        pickle.dump({
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
            "mesh_RF_triangles": mesh_RF_triangles_updated,
            "edge_list": edge_list,
            "weight_list": weight_list,
            "neighbour_list": neighbour_list,
            "neighbour_edge_list": neighbour_edge_list,
            "area_list": area_list,
            'bending_ele_idx': bending_ele_idx,
            'bending_ele_param': bending_ele_param,
            'bending_weight_list': bending_weight_list,
            'mem_weight_list': mem_weight_list,
            "initial_ARAP_SK_list": initial_SK_list,
            "stiffness_matrices": stiffness_matrices,
            "mass_matrix": mass_mat,
            "pp_idx": pp_idx,
            "ee_idx": ee_idx,
            "pullpoint_locations": pullpoint_locations,
            "pulley_locations": pulley_locations,
            "density": density,
            "thickness": thickness,
            "Youngs_modulus": Youngs_modulus,
            "Poisson_ratio": Poisson_ratio
        }, f)
    return


boundary_vertices = get_boundary_vertex_indices(mesh_triangles)
print("boundary_vertices:", boundary_vertices)

pullpoint_locations = np.array([[215, 10, 0], [250,80,0], [215, 150, 0], [180, 10, 0], [145, 80, 0], [180, 150,0]]) * 1e-3
pulley_locations = np.array([[30, 10, 400], [30, 80, 400], [30, 150, 400], [10, 10, -259], [10, 80, -259], [10, 150, -259]]) * 1e-3
ee_vertices_list = np.array([[270, 80, 0]]) * 1e-3

generate_C_SRS_description(mesh_vertices, mesh_triangles, pullpoint_locations=pullpoint_locations, pulley_locations=pulley_locations, ee_vertices_list=ee_vertices_list, density=density, thickness=thickness, Youngs_modulus=Youngs_modulus, Poisson_ratio=Poisson_ratio)