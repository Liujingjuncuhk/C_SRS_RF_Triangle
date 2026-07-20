import pickle
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
import torch
import torch.nn as nn
import joblib


def get_normal(tri):
    """Return the unit normal of a triangle."""
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


class Flying_carpet_fixedEnd:
    def __init__(self, description_file):
        with open(description_file, 'rb') as f:
            self.description = pickle.load(f)
        self.vertices = self.description['mesh_vertices']
        
        self.mesh_triangles = self.description['mesh_triangles']
        roz_z = 90 # rotate the initial vertices around z axis by 90 degrees
        Rz = np.array([[np.cos(np.radians(roz_z)), -np.sin(np.radians(roz_z)), 0],
                       [np.sin(np.radians(roz_z)), np.cos(np.radians(roz_z)), 0],
                       [0, 0, 1]])
        self.vertices = (Rz @ self.vertices.T).T
        initial_translation = np.array([100, 65, 0]) * 1e-3
        self.vertices[:, 0] += initial_translation[0]
        self.vertices[:, 1] += initial_translation[1]
        self.vertices[:, 2] += initial_translation[2]
        # check if there are <0 element in mesh_RF_triangles
        # self.pp_idx = self.description['pp_idx']
        self.pp_bary_tri_idx = self.description['pp_bary_tri_idx']
        self.pp_bary_coords = self.description['pp_bary_coords']
        self.pp_bary_offsets = self.description['pp_bary_offsets']
        self.pulley_location = self.description['pulley_locations']
        self.mesh_RF_triangles = self.description['mesh_RF_triangles']
        self.ee_idx = self.description['ee_idx']
        self.stiffness_matrices = self.description['stiffness_matrices']
        self.mass_matrix = self.description['mass_matrix']
        self.num_vertices = self.vertices.shape[0]
        self.num_triangles = self.mesh_triangles.shape[0]
        self.num_RF_triangles = self.mesh_RF_triangles.shape[0]
        self.initial_ARAP_SK_list = self.description['initial_ARAP_SK_list']
        self.area_list = self.description['area_list']
        self.bending_ele_idx = self.description['bending_ele_idx']
        self.bending_ele_param = self.description['bending_ele_param']
        self.bending_weight_list = self.description['bending_weight_list']
        self.mem_weight_list = self.description['mem_weight_list']
        
        print("max bending weight: ", np.max(self.bending_weight_list))
        print("ave bending weight: ", np.mean(self.bending_weight_list))
        print("max mem weight: ", np.max(self.mem_weight_list))
        print("ave mem weight: ", np.mean(self.mem_weight_list))
        self.n_bending_ele = len(self.bending_ele_idx)
        self.thickness = self.description['thickness']
        self.Youngs_modulus = self.description['Youngs_modulus']
        self.Poisson_ratio = self.description['Poisson_ratio']
        self.density = self.description['density']
        self.density = 619.230769230769
        self.ARAP_weight_list = self.description['weight_list']
        self.edge_list = self.description['edge_list']
        self.neighbour_list = self.description['neighbour_list']
        self.neighbour_edge_list = self.description['neighbour_edge_list']        
        self.neighbour_edge_weight_list = []
        for i in range(self.num_vertices):
            neighbour_edges = self.neighbour_edge_list[i]
            neighbour_weights = []
            for edge in neighbour_edges:
                neighbour_weights.append(self.ARAP_weight_list[edge])
            self.neighbour_edge_weight_list.append(neighbour_weights)
        self.initial_patch_list = self.get_patch_list(self.vertices)
        self.N33 = np.eye(3) - 1/3*np.ones((3,3))
        self.initial_tri_SK_list = self.get_tri_SK_list(self.vertices)
        self.nCable = len(self.pulley_location)
        self.initial_cable_length = self.get_cable_length_bary(self.vertices)
        self.gravity_dir = np.array([0, 0, -1])
        self.gravity = 9.81
        self.gravity_vec = np.zeros(self.num_vertices * 3)
        for i in range(self.num_vertices):
            self.gravity_vec[3*i:3*i+3] = self.mass_matrix[3*i:3*i+3, 3*i:3*i+3] @ self.gravity_dir * self.gravity
        self.qe0_list = [np.zeros(18) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            tri = self.mesh_RF_triangles[i]
            for j in range(6):
                if tri[j] == -1:
                    continue
                self.qe0_list[i][3*j:3*j+3] = self.vertices[tri[j]]
        self.N1818 = np.zeros((18, 18))
        for i in range(6):
            for j in range(3):
                for k in range(6):
                    if k==i:
                        self.N1818[3*i+j, 3*k+j] = 5.0/6.0
                    else:
                        self.N1818[3*i+j, 3*k+j] = -1.0/6.0
        self.N1212 = np.zeros((12, 12))
        for i in range(4):
            for j in range(3):
                for k in range(4):
                    if k==i:
                        self.N1212[3*i+j, 3*k+j] = 3.0/4.0
                    else:
                        self.N1212[3*i+j, 3*k+j] = -1.0/4.0
        self.N44 = np.eye(4) - 1/4*np.ones((4,4))
        self.N66 = np.eye(6) - 1/6*np.ones((6,6))
        self.fixed_region = [[-0.1, 0.02], [-0.1, 1]]
        self.get_fixed_idx(self.vertices, self.fixed_region)
        self.nFixed = len(self.fixed_idx)
        self.nMoving = self.num_vertices - self.nFixed
        self.W_mat = np.zeros((self.num_vertices * 3, self.num_vertices * 3))
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] == -1:
                self.W_mat[3*i:3*i+3, 3*i:3*i+3] = np.zeros((3,3))
            else:
                for j in range(3):
                    self.W_mat[3*i+j, 3*i+j] = 1 / self.mass_matrix[3*i+j, 3*i+j]


    def get_fixed_idx(self, vertices, fixed_region):
        self.fixed_idx = []
        self.idxFixed_2_idxAll = []
        self.idxMoving_2_idxAll = []
        self.idxAll_2_idxMoving = [-1 for _ in range(self.num_vertices)]
        idx_moving = 0
        for i in range(vertices.shape[0]):
            v = vertices[i]
            if fixed_region[0][0] <= v[0] <= fixed_region[0][1] and fixed_region[1][0] <= v[1] <= fixed_region[1][1]:
                self.fixed_idx.append(i)
                self.idxFixed_2_idxAll.append(i)
            else:
                self.idxMoving_2_idxAll.append(i)
                self.idxAll_2_idxMoving[i] = idx_moving
                idx_moving += 1

    def get_tri_SK_list(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        tri_SK_list = [np.zeros((3,3)) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            tri = self.mesh_RF_triangles[i]
            X = vertices[tri[:3]]
            SK = self.N33 @ X
            tri_SK_list[i] = SK
        return tri_SK_list

    def get_patch_list(self, vertices):
        # check size of vertices
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        patch_list = [np.zeros((6,3)) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            tri = self.mesh_RF_triangles[i]
            X = vertices[tri]
            patch_list[i] = X
        return patch_list

    def vertices_to_q(self, vertices):
        return vertices.flatten()

    def q_to_vertices(self, q):
        return q.reshape(-1, 3)

    def q_to_q_moving(self, q):
        q_moving = np.zeros((self.nMoving * 3, ))
        for i in range(self.num_vertices):
            idx_moving = self.idxAll_2_idxMoving[i]
            if idx_moving != -1:
                q_moving[3*idx_moving:3*idx_moving+3] = q[3*i:3*i+3]
        return q_moving

    def q_moving_to_q(self, q_moving):
        q = self.vertices_to_q(self.vertices)
        for i in range(self.nMoving):
            idx_all = self.idxMoving_2_idxAll[i]
            q[3*idx_all:3*idx_all+3] = q_moving[3*i:3*i+3]
        return q

    def get_R_list(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        tri_SK_list = self.get_tri_SK_list(vertices)
        R_list = [np.eye(3) for _ in range(self.num_RF_triangles)]
        R_list_1818 = [np.eye(18) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            initial_tri_SK = self.initial_tri_SK_list[i]
            cur_tri_SK = tri_SK_list[i]
            u, _, vh = np.linalg.svd(cur_tri_SK.T @ initial_tri_SK)
            R_list[i] = u @ vh
            for j in range(6):
                R_list_1818[i][3*j:3*j+3, 3*j:3*j+3] = R_list[i]
        return R_list, R_list_1818

    def assemble_K(self, R_list_1818):
        Ke_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        Ke0_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        K_mat = np.zeros((self.num_vertices * 3, self.num_vertices * 3))
        f0 = np.zeros(self.num_vertices * 3)
        for i in range(self.num_RF_triangles):
            Ke_list[i] = R_list_1818[i] @ self.stiffness_matrices[i] @ R_list_1818[i].T
            Ke0_list[i] = R_list_1818[i] @ self.stiffness_matrices[i]
            tri = self.mesh_RF_triangles[i]
            qe0 = self.qe0_list[i]
            for j in range(6):
                if tri[j] == -1:
                    continue
                for k in range(6):
                    if tri[k] == -1:
                        continue
                    K_mat[3*tri[j]:3*tri[j]+3, 3*tri[k]:3*tri[k]+3] += Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
            f0e = Ke0_list[i] @ qe0
            for j in range(6):
                if tri[j] == -1:
                    continue
                f0[3*tri[j]:3*tri[j]+3] += f0e[3*j:3*j+3]
        return K_mat, f0

    def assemble_K_tilde(self, R_list_1818):
        """Assemble K_mm, f0_m, and the fixed-DOF term -K_mf q_f."""
        Ke_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        Ke0_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        K_tilde = np.zeros((self.nMoving * 3, self.nMoving * 3))
        f0_tilde = np.zeros(self.nMoving * 3)
        K_tilde_vec2add = np.zeros(3*self.nMoving,)
        for i in range(self.num_RF_triangles):
            Ke_list[i] = R_list_1818[i] @ self.stiffness_matrices[i] @ R_list_1818[i].T
            Ke0_list[i] = R_list_1818[i] @ self.stiffness_matrices[i]
            tri = self.mesh_RF_triangles[i]
            qe0 = self.qe0_list[i]
            f0e = Ke0_list[i] @ qe0
            for j in range(6):
                if tri[j] == -1:
                    continue
                idx_j_moving = self.idxAll_2_idxMoving[tri[j]]
                if idx_j_moving == -1:
                    continue

                row = slice(3*idx_j_moving, 3*idx_j_moving+3)
                f0_tilde[row] += f0e[3*j:3*j+3]
                for k in range(6):
                    if tri[k] == -1:
                        continue
                    Ke_jk = Ke_list[i][3*j:3*j+3, 3*k:3*k+3]
                    idx_k_moving = self.idxAll_2_idxMoving[tri[k]]
                    if idx_k_moving == -1:
                        K_tilde_vec2add[row] -= Ke_jk @ self.vertices[tri[k]]
                    else:
                        col = slice(3*idx_k_moving, 3*idx_k_moving+3)
                        K_tilde[row, col] += Ke_jk

        return K_tilde, f0_tilde, K_tilde_vec2add

    def get_pp_location_bary(self, vertices):
        """Compute pull-point world positions from barycentric coords + normal offset."""
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_location = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            idx_tri = self.description['pp_bary_tri_idx'][i]
            bary = self.description['pp_bary_coords'][i]
            offset = self.description['pp_bary_offsets'][i]
            tri = self.mesh_triangles[idx_tri]
            pp_on_surface = bary @ vertices[tri]
            n = get_normal(vertices[tri])
            pp_location[i] = pp_on_surface + offset * n
        return pp_location

    def get_cable_length_bary(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_locations = self.get_pp_location_bary(vertices)
        return [np.linalg.norm(self.pulley_location[i] - pp_locations[i]) for i in range(self.nCable)]

    def get_cable_vec_bary(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_locations = self.get_pp_location_bary(vertices)
        cable_vec = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            vec = pp_locations[i] - self.pulley_location[i]
            cable_vec[i] = vec / np.linalg.norm(vec)
        return cable_vec

    def get_ee_pos(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        ee_id = self.ee_idx[0]
        return vertices[ee_id]

    def get_fb_surface(self, vertices):
        """Return vertices shifted by half-thickness along vertex normals."""
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        vertex_normals = np.zeros_like(vertices, dtype=float)
        tol = 1e-12
        for tri in self.mesh_triangles:
            tri_vertices = vertices[tri]
            tri_normal = np.cross(tri_vertices[1] - tri_vertices[0], tri_vertices[2] - tri_vertices[0])
            if np.linalg.norm(tri_normal) < tol:
                continue
            vertex_normals[tri] += tri_normal

        normal_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        valid_mask = normal_norms[:, 0] > tol
        vertex_normals[valid_mask] /= normal_norms[valid_mask]
        if np.any(~valid_mask):
            vertex_normals[~valid_mask] = np.array([0.0, 0.0, 1.0])

        return vertices + 0.5 * self.thickness * vertex_normals

    def FKD_free_static(self, show_info = False):
        """Solve the gravity-only corotational equilibrium on moving vertices."""
        Q_a = self.vertices_to_q(self.vertices).reshape((3*self.num_vertices, ))
        Q_list = [Q_a.copy()]
        gravity_tilde = self.q_to_q_moving(self.gravity_vec)
        max_iter = 500
        tol = 1e-6

        for iteration in range(max_iter):
            _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            K_tilde, f0_tilde, K_tilde_vec2add = self.assemble_K_tilde(R_list_1818)

            # Moving part of K q = f0 + f_gravity, after moving the
            # fixed-vertex contribution K_mf q_f to the right-hand side.
            rhs = f0_tilde + gravity_tilde + K_tilde_vec2add
            try:
                Q_moving = np.linalg.solve(K_tilde, rhs)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "The reduced stiffness matrix is singular; check that the "
                    "fixed region removes all rigid-body modes."
                ) from exc

            Q_next = self.q_moving_to_q(Q_moving)
            diff = np.linalg.norm(Q_next - Q_a) / (3*self.nMoving)
            Q_a = Q_next
            Q_list.append(Q_a.copy())

            if show_info:
                residual = np.linalg.norm(K_tilde @ Q_moving - rhs)
                print(
                    f"static iteration {iteration + 1}: diff = {diff:.7e}, "
                    f"linear residual = {residual:.7e}"
                )
            if diff < tol:
                if show_info:
                    print(f"Static solve converged in {iteration + 1} iterations.")
                break
        else:
            if show_info:
                print(
                    f"Static solve reached {max_iter} iterations without "
                    f"meeting the tolerance (last diff = {diff:.7e})."
                )

        return Q_list

    def reassemble_stiffness_matrices(self, Youngs_modulus, Poisson_ratio):
        """Rebuild all EBST element matrices for new elastic parameters."""
        E = float(Youngs_modulus)
        nu = float(Poisson_ratio)
        if not np.isfinite(E) or E <= 0:
            raise ValueError("Youngs_modulus must be a finite positive value.")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("Poisson_ratio must satisfy -1 < nu < 0.5.")
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("thickness must be a finite positive value.")

        t = self.thickness
        tol = 1e-12
        C = np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
        Dm = E * t / (1.0 - nu**2) * C
        Db = E * t**3 / (12.0 * (1.0 - nu**2)) * C

        stiffness_matrices = []
        for element_idx, indices in enumerate(self.mesh_RF_triangles):
            indices = np.asarray(indices, dtype=int)
            if np.any(indices[:3] == -1):
                raise ValueError(
                    f"Element {element_idx} has a ghost node in its central triangle."
                )
            X = self.vertices[indices]

            x0, x1, x2 = X[:3]
            v1, v2 = x1 - x0, x2 - x0
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < tol:
                raise ValueError(f"Element {element_idx} has a degenerate central triangle.")

            area = 0.5 * normal_norm
            t3 = normal / normal_norm
            v1_norm = np.linalg.norm(v1)
            if v1_norm < tol:
                raise ValueError(f"Element {element_idx} has a zero-length first edge.")
            e1 = v1 / v1_norm
            e2 = np.cross(t3, e1)

            def cst_grad(points):
                """Shape-function gradients expressed in the central local frame."""
                u1, u2 = points[1] - points[0], points[2] - points[0]
                tri_normal = np.cross(u1, u2)
                twice_area = np.linalg.norm(tri_normal)
                if twice_area < tol:
                    return None
                u1_norm = np.linalg.norm(u1)
                if u1_norm < tol:
                    return None
                f1 = u1 / u1_norm
                f3 = tri_normal / twice_area
                f2 = np.cross(f3, f1)
                local_points = (points - points[0]) @ np.vstack([f1, f2]).T
                (xa, ya), (xb, yb), (xc, yc) = local_points
                determinant = (xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)
                if abs(determinant) < tol:
                    return None
                b = np.array([yb - yc, yc - ya, ya - yb]) / determinant
                c = np.array([xc - xb, xa - xc, xb - xa]) / determinant
                frame_transform = np.array([
                    [f1 @ e1, f1 @ e2],
                    [f2 @ e1, f2 @ e2],
                ])
                return np.column_stack([b, c]) @ frame_transform

            dNm = cst_grad(X[:3])
            if dNm is None:
                raise ValueError(f"Element {element_idx} has a degenerate central triangle.")

            Bm = np.zeros((3, 18))
            for local_node in range(3):
                Na, Nb = dNm[local_node]
                col = slice(3*local_node, 3*local_node+3)
                Bm[0, col] = Na * e1
                Bm[1, col] = Nb * e2
                Bm[2, col] = Na * e2 + Nb * e1

            Bb = np.zeros((3, 18))
            side_nodes = ((1, 2, 3), (2, 0, 4), (0, 1, 5))
            for side, nodes in enumerate(side_nodes):
                if indices[nodes[2]] == -1:
                    continue
                dNi = cst_grad(X[list(nodes)])
                if dNi is None:
                    continue
                NMa, NMb = dNm[side]
                for local_node, patch_node in enumerate(nodes):
                    Ta, Tb = dNi[local_node]
                    coefficients = np.array([
                        NMa * Ta,
                        NMb * Tb,
                        NMa * Tb + NMb * Ta,
                    ])
                    col = slice(3*patch_node, 3*patch_node+3)
                    Bb[:, col] += np.outer(coefficients, t3)

            K = area * (Bm.T @ Dm @ Bm) + area * (Bb.T @ Db @ Bb)
            for slot, vertex_idx in enumerate(indices):
                if vertex_idx == -1:
                    dofs = slice(3*slot, 3*slot+3)
                    K[dofs, :] = 0.0
                    K[:, dofs] = 0.0
            stiffness_matrices.append(K)

        self.stiffness_matrices = stiffness_matrices
        self.Youngs_modulus = E
        self.Poisson_ratio = nu
        self.description['stiffness_matrices'] = stiffness_matrices
        self.description['Youngs_modulus'] = E
        self.description['Poisson_ratio'] = nu
        return self.stiffness_matrices


    def visualize_vert(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        

        # make fixed idx black
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        # add grid
        xmin = -0.02 
        xmax = 0.22
        ymin = 0
        ymax = 0.13
        zmin = -0.1
        zmax = 0.1
        plotter.view_vector((2, 2, 2))
        plotter.camera.zoom(0.7)
        plotter.add_points(np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]]), color='white', point_size=0.1, label='Axis Limits')
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_fb_surface(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        fb_vertices = self.get_fb_surface(vertices)
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))
        mesh_surface = pv.PolyData(vertices, faces)
        mesh_fb = pv.PolyData(fb_vertices, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh_surface, color='lightgray', show_edges=True, opacity=0.35, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightblue', show_edges=True, opacity=0.95, label='FB Mid-Surface')
        plotter.add_points(fb_vertices[self.ee_idx], color='magenta', point_size=10, label='FB EE Vertices')
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_fb_surface_w_gt(self, vertices, gt_pts):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        plotter = pv.Plotter(off_screen=True)
        fb_vertices = self.get_fb_surface(vertices)
        vertices_mm = vertices * 1e3
        fb_vertices_mm = fb_vertices * 1e3
        gt_pts_mm = np.asarray(gt_pts) * 1e3
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))
        mesh_surface = pv.PolyData(vertices_mm, faces)
        mesh_fb = pv.PolyData(fb_vertices_mm, faces)
        plotter.add_mesh(mesh_surface, color='lightblue', show_edges=True, opacity=0.95, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, opacity=0.35, label='FB Mid-Surface')
        plotter.add_points(gt_pts_mm, color='green', point_size=10, opacity=0.55, label='Ground Truth Points')
        plotter.show_grid()
        plotter.show_axes()
        plotter.view_vector((2, 2, 2))
        plotter.camera.zoom(0.7)

        xmin = 1e3 * -0.02
        xmax = 1e3 * 0.3
        ymin = 1e3 * 0
        ymax = 1e3 * 0.16
        zmin = 1e3 * -0.1
        zmax = 1e3 * 0.1
        plotter.set_scale(xscale=1, yscale=1, zscale=1)
        plotter.show_bounds(
            bounds=(xmin, xmax, ymin, ymax, zmin, zmax),
            grid='back',
            location='outer',
            all_edges=True,
        )
        return plotter

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet_fixedEnd(description_file)
    Q_list = flying_carpet.FKD_free_static(show_info=True)

    flying_carpet.visualize_vert(Q_list[-1])
