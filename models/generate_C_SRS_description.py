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


def generate_C_SRS_description(mesh_vertices, mesh_triangles, pullpoint_locations, pulley_locations,ee_vertices_list, density, thickness, Youngs_modulus, Poisson_ratio):
    # identify boundary vertices (edges shared by only one triangle)
    edge_count = defaultdict(int)
    for tri in mesh_triangles:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            edge_count[edge] += 1
    boundary_vertex_set = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertex_set.update(edge)
    interior_indices = np.array([i for i in range(len(mesh_vertices)) if i not in boundary_vertex_set])

    # find the closest interior (non-boundary) vertex to each pullpoint
    pp_idx = []
    for i in range(len(pullpoint_locations)):
        dists = np.linalg.norm(mesh_vertices[interior_indices] - pullpoint_locations[i], axis=1)
        idx = interior_indices[np.argmin(dists)]
        mesh_vertices[idx] = pullpoint_locations[i]
        pp_idx.append(idx)

    ee_idx = []
    for i in range(len(ee_vertices_list)):
        idx = np.argmin(np.linalg.norm(mesh_vertices - ee_vertices_list[i], axis=1))
        mesh_vertices[idx] = ee_vertices_list[i]
        ee_idx.append(idx)
    print("ee_idx:", ee_idx)

    edge_list, weight_list = generate_edge_matrix(mesh_vertices, mesh_triangles)
    neighbour_list, neighbour_edge_list = generate_neighbour_list(mesh_triangles, len(mesh_vertices), edge_list)
    area_list = cal_area_list(mesh_vertices, mesh_triangles)
    initial_SK_list = get_ARAP_initial_SK_list(mesh_vertices, mesh_triangles, edge_list, weight_list, neighbour_list, neighbour_edge_list)
    stiffness_matrices = []
    for tri in mesh_RF_triangles:
        X = mesh_vertices[tri]
        K = ebst_stiffness(tri, X, Youngs_modulus, Poisson_ratio, thickness)
        stiffness_matrices.append(K)
    mass_mat = cal_mass_matrix(mesh_vertices, mesh_triangles, density, thickness)
    visualize_3d_mesh(mesh_vertices, mesh_triangles, pp_idx, pulley_locations, ee_idx)
    with open(folder + "C_SRS_description.pkl", "wb") as f:
        pickle.dump({
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
            "mesh_RF_triangles": mesh_RF_triangles,
            "edge_list": edge_list,
            "weight_list": weight_list,
            "neighbour_list": neighbour_list,
            "neighbour_edge_list": neighbour_edge_list,
            "area_list": area_list,
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