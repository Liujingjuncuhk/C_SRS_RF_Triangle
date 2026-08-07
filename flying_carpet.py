import pickle
from utilities import *
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize
import torch
import torch.nn as nn
import joblib
import time

class Flying_carpet:
    @staticmethod
    def _build_canonical_rf_triangles(mesh_triangles):
        """Build EBST patches with slots 3:6 opposite edges 12, 20, and 01."""
        triangles = np.asarray(mesh_triangles, dtype=int)
        edge_map = {}
        for triangle_idx, (v0, v1, v2) in enumerate(triangles):
            for a, b, apex in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
                edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(
                    (triangle_idx, int(apex))
                )

        rf_triangles = np.full((len(triangles), 6), -1, dtype=triangles.dtype)
        rf_triangles[:, :3] = triangles
        for triangle_idx, triangle in enumerate(triangles):
            # EBST side i is the edge opposite central node i.
            for slot, (a, b) in enumerate(((1, 2), (2, 0), (0, 1)), start=3):
                incidents = edge_map[tuple(sorted((int(triangle[a]), int(triangle[b]))))]
                if len(incidents) > 2:
                    raise ValueError(
                        f"Non-manifold edge ({triangle[a]}, {triangle[b]}) is not supported."
                    )
                for neighbour_idx, apex in incidents:
                    if neighbour_idx != triangle_idx:
                        rf_triangles[triangle_idx, slot] = apex
                        break
        return rf_triangles

    @staticmethod
    def _best_fit_rotation(current_shape, reference_shape):
        """Return the proper rotation aligning reference_shape to current_shape."""
        u, _, vh = np.linalg.svd(current_shape.T @ reference_shape)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vh)
        return u @ correction @ vh

    def __init__(self, description_file):
        with open(description_file, 'rb') as f:
            self.description = pickle.load(f)
        self.vertices = self.description['mesh_vertices']
        
        self.mesh_triangles = self.description['mesh_triangles']
        initial_translation = np.array([280, 380, 300]) * 1e-3
        self.vertices[:, 0] += initial_translation[0]
        self.vertices[:, 1] += initial_translation[1]
        self.vertices[:, 2] += initial_translation[2]
        # check if there are <0 element in mesh_RF_triangles
        # self.pp_idx = self.description['pp_idx']
        self.pp_bary_tri_idx = self.description['pp_bary_tri_idx']
        self.pp_bary_coords = self.description['pp_bary_coords']
        self.pp_bary_offsets = self.description['pp_bary_offsets']
        self.pulley_location = self.description['pulley_locations']
        # Reconstruct the RF patches from connectivity. Older description files
        # stored the three neighbour slots in a different order from the EBST
        # stiffness routine.
        self.mesh_RF_triangles = self._build_canonical_rf_triangles(self.mesh_triangles)
        self.description['mesh_RF_triangles'] = self.mesh_RF_triangles
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
        self.Youngs_modulus = 3.0e7
        # self.Youngs_modulus = 4.2e6
        self.Poisson_ratio = self.description['Poisson_ratio']
        self.Poisson_ratio = 0.39
        stored_density = float(self.description['density'])
        self.density = stored_density
        self.density = 619.230769230769
        if not np.isclose(self.density, stored_density):
            self.mass_matrix = (
                np.asarray(self.mass_matrix, dtype=float).copy()
                * (self.density / stored_density)
            )
            self.description['mass_matrix'] = self.mass_matrix
            self.description['density'] = self.density
        self.reassemble_stiffness_matrices(self.Youngs_modulus, self.Poisson_ratio)
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
        self.get_fixed_idx()
        self.initial_ARAP_shape_list = []
        for i in range(self.num_vertices):
            neighbour_list = self.neighbour_list[i]
            nNeighbour = len(neighbour_list)
            ARAP_shape = np.zeros((nNeighbour, 3))
            for j in range(nNeighbour):
                neighbour_idx = neighbour_list[j]
                ARAP_shape[j] = self.vertices[neighbour_idx] - self.vertices[i]
            self.initial_ARAP_shape_list.append(ARAP_shape)
        print("initial_ARAP_shape_list length: ", len(self.initial_ARAP_shape_list))
        self.initial_cable_vec = self.get_cable_vec_bary(self.vertices)
        self.initial_ghost_vertices = self.get_pp_location_bary(self.vertices)
        self.initial_ghost_shape_list = self.get_ghost_shape(self.vertices, self.get_pp_location_bary(self.vertices))
        self.W_mat = np.zeros((self.num_vertices * 3, self.num_vertices * 3))
        for i in range(self.num_vertices):
            for j in range(3):
                self.W_mat[3*i+j, 3*i+j] = 1 / self.mass_matrix[3*i+j, 3*i+j]
        self.assemble_CG_matrices()

    def get_fixed_idx(self):
        self.fixed_idx = []
        self.idxFixed_2_idxAll = []
        self.idxMoving_2_idxAll = []
        self.idxAll_2_idxMoving = [-1 for _ in range(self.num_vertices)]
        idx_moving = 0
        fixed_idx = self.ee_idx.copy()
        for i in range(self.num_vertices):
            if i in fixed_idx:
                self.fixed_idx.append(i)
                self.idxFixed_2_idxAll.append(i)
            else:
                self.idxMoving_2_idxAll.append(i)
                self.idxAll_2_idxMoving[i] = idx_moving
                idx_moving += 1
        self.nMoving = self.num_vertices - len(self.fixed_idx)


    def reassemble_stiffness_matrices(self, Youngs_modulus, Poisson_ratio):
        """Rebuild all EBST element stiffness matrices for new material values."""
        E = float(Youngs_modulus)
        nu = float(Poisson_ratio)
        t = float(self.thickness)

        if not np.isfinite(E) or E <= 0.0:
            raise ValueError("Youngs_modulus must be a finite positive value.")
        if not np.isfinite(nu) or not -1.0 < nu < 0.5:
            raise ValueError("Poisson_ratio must satisfy -1 < nu < 0.5.")
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError("thickness must be a finite positive value.")

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

            # Indexing with -1 is harmless for a ghost slot because that slot is
            # skipped below and its rows and columns are explicitly zeroed.
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
                """Return CST shape gradients in the central triangle frame."""
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
                col = slice(3 * local_node, 3 * local_node + 3)
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
                    col = slice(3 * patch_node, 3 * patch_node + 3)
                    Bb[:, col] += np.outer(coefficients, t3)

            K = area * (Bm.T @ Dm @ Bm) + area * (Bb.T @ Db @ Bb)
            for slot, vertex_idx in enumerate(indices):
                if vertex_idx == -1:
                    dofs = slice(3 * slot, 3 * slot + 3)
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


    def assemble_CG_matrices(self):
        # Values in the stored weight lists are quadratic-energy
        # coefficients. Least-squares rows therefore use sqrt(weight), since
        # ||sqrt(w) (Aq-b)||^2 = w ||Aq-b||^2.
        mem_block   = 9  * self.num_triangles
        bend_block  = 3 * len(self.bending_ele_idx)
        cable_block = 3  * self.nCable
        ghost_block = 12 * self.nCable
        matA_size = mem_block + bend_block + cable_block + ghost_block
        max_weight = np.max((np.max(self.bending_weight_list), np.max(self.mem_weight_list)))
        self.weight_cable = 0.5 * max_weight
        self.weight_ghost = 0.5 * max_weight
        self.nNeighbour_list = []
        for i in range(self.num_vertices):
            self.nNeighbour_list.append(len(self.neighbour_list[i]))
        self.matA_all = np.zeros((matA_size, 3*self.num_vertices + 3*self.nCable))
        print("matA_size: ", matA_size)
        for i in range(self.num_triangles):
            mem_weight = np.sqrt(self.mem_weight_list[i])
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    for jp in range(3): # iterate over all triangle vertices (columns)
                        v_jp = self.mesh_triangles[i][jp]
                        coeff = (2.0/3.0) if jp == j else (-1.0/3.0)
                        self.matA_all[idx_row_start+k, 3*v_jp+k] = mem_weight * coeff

        for i in range(len(self.bending_ele_idx)):
            bending_weight = np.sqrt(self.bending_weight_list[i])
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            for j in range(4):
                v_idx = self.bending_ele_idx[i,j]
                c = self.bending_ele_param[i,j]
                for k in range(3):
                    self.matA_all[mem_block + 3*i + k, 3*v_idx+k] = bending_weight * c

        for i in range(self.nCable):
            for k in range(3):
                self.matA_all[mem_block + bend_block + 3*i + k, 3*self.num_vertices+3*i+k] = np.sqrt(self.weight_cable)

        for i in range(self.nCable):
            row_start = mem_block + bend_block + cable_block + 12*i
            tri_idx = self.pp_bary_tri_idx[i]
            tri = self.mesh_triangles[tri_idx]
            idx_all = [tri[0], tri[1], tri[2], self.num_vertices + i]
            for j in range(4):
                idxj = idx_all[j]
                for k in range(4):
                    idxk = idx_all[k]
                    self.matA_all[row_start + 3*j:row_start+3*j+3, 3*idxk:3*idxk+3] += np.sqrt(self.weight_ghost) * self.N1212[3*j:3*j+3, 3*k:3*k+3]

        print("matA_all shape: ", self.matA_all.shape)
        print("matA_all rank: ", np.linalg.matrix_rank(self.matA_all))
        self.matAT = self.matA_all.T
        self.matATA = self.matA_all.T @ self.matA_all
        self.matATA_inv_AT = np.linalg.inv(self.matATA) @ self.matAT
        self.matATA_inv = np.linalg.inv(self.matATA)
        self.K_CG = self.matATA_inv_AT[:, -15*self.nCable:-12*self.nCable]

    def reassemble_CG_matrices(self, ratio_weight_bending, ratio_weight_cable=10):
        mem_block   = 9  * self.num_triangles
        bend_block  = 3 * len(self.bending_ele_idx)
        cable_block = 3  * self.nCable
        ghost_block = 12 * self.nCable
        matA_size = mem_block + bend_block + cable_block + ghost_block
        for i in range(len(self.bending_weight_list)):
            self.bending_weight_list[i] = ratio_weight_bending * self.bending_weight_list[i]
        max_weight = np.max((np.max(self.mem_weight_list), np.max(self.bending_weight_list)))

        self.weight_cable = ratio_weight_cable * max_weight
        self.weight_ghost = ratio_weight_cable * max_weight
        self.nNeighbour_list = []
        for i in range(self.num_vertices):
            self.nNeighbour_list.append(len(self.neighbour_list[i]))
        self.matA_all = np.zeros((matA_size, 3*self.num_vertices + 3*self.nCable))
        print("matA_size: ", matA_size)
        for i in range(self.num_triangles):
            mem_weight = np.sqrt(self.mem_weight_list[i])
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    for jp in range(3): # iterate over all triangle vertices (columns)
                        v_jp = self.mesh_triangles[i][jp]
                        coeff = (2.0/3.0) if jp == j else (-1.0/3.0)
                        self.matA_all[idx_row_start+k, 3*v_jp+k] = mem_weight * coeff

        for i in range(len(self.bending_ele_idx)):
            bending_weight = np.sqrt(self.bending_weight_list[i])
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            for j in range(4):
                v_idx = self.bending_ele_idx[i,j]
                c = self.bending_ele_param[i,j]
                for k in range(3):
                    self.matA_all[mem_block + 3*i + k, 3*v_idx+k] = bending_weight * c

        for i in range(self.nCable):
            for k in range(3):
                self.matA_all[mem_block + bend_block + 3*i + k, 3*self.num_vertices+3*i+k] = np.sqrt(self.weight_cable)

        for i in range(self.nCable):
            row_start = mem_block + bend_block + cable_block + 12*i
            tri_idx = self.pp_bary_tri_idx[i]
            tri = self.mesh_triangles[tri_idx]
            idx_all = [tri[0], tri[1], tri[2], self.num_vertices + i]
            for j in range(4):
                idxj = idx_all[j]
                for k in range(4):
                    idxk = idx_all[k]
                    self.matA_all[row_start + 3*j:row_start+3*j+3, 3*idxk:3*idxk+3] += self.weight_ghost * self.N1212[3*j:3*j+3, 3*k:3*k+3]

        print("matA_all shape: ", self.matA_all.shape)
        print("matA_all rank: ", np.linalg.matrix_rank(self.matA_all))
        self.matAT = self.matA_all.T
        self.matATA = self.matA_all.T @ self.matA_all
        self.matATA_inv_AT = np.linalg.inv(self.matATA) @ self.matAT
        self.matATA_inv = np.linalg.inv(self.matATA)
        self.K_CG = self.matATA_inv_AT[:, -15*self.nCable:-12*self.nCable]


    def get_CG_Jacobian(self, vertices, ghost_vertices=None):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        if ghost_vertices is None:
            ghost_vertices = self.get_pp_location_bary(vertices)
        R_cable_list = self.get_rotation_cable_ghost(ghost_vertices)
        Bmat = np.zeros((3*self.nCable, self.nCable))
        cable_row_weight = np.sqrt(self.weight_cable)
        for i in range(self.nCable):
            cable_vec_rotated = R_cable_list[i] @ self.initial_cable_vec[i]
            for k in range(3):
                Bmat[3*i+k, i] = cable_row_weight * cable_vec_rotated[k]
        J = self.K_CG @ Bmat
        return J

    def get_CG_Jacobian_FD_EE(self, vertices, ghost_vertices=None,
                              eps=1e-3, iter=10):
        """Finite-difference the converged Shape-Up EE response.

        Each cable target length is perturbed in both directions and each
        perturbed configuration is solved with at most ``iter`` local/global
        Shape-Up iterations.  The returned matrix maps cable-length changes
        to the stacked xyz coordinates of ``self.ee_idx``.
        """
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape == (self.num_vertices, 3):
            vertices = vertices.copy()
        elif vertices.size == 3 * self.num_vertices:
            vertices = self.q_to_vertices(
                vertices.reshape(3 * self.num_vertices)
            )
        else:
            raise ValueError(
                "vertices should be either a 3n vector or an n by 3 array."
            )
        if np.any(~np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values.")

        eps = float(eps)
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive value.")
        if not isinstance(iter, (int, np.integer)) or iter <= 0:
            raise ValueError("iter must be a positive integer.")

        if ghost_vertices is None:
            current_lengths = np.asarray(
                self.get_cable_length_bary(vertices), dtype=float
            )
        else:
            ghost_vertices = np.asarray(ghost_vertices, dtype=float)
            if ghost_vertices.shape != (self.nCable, 3):
                raise ValueError(
                    "ghost_vertices must have shape (nCable, 3)."
                )
            if np.any(~np.isfinite(ghost_vertices)):
                raise ValueError(
                    "ghost_vertices must contain only finite values."
                )
            current_lengths = np.linalg.norm(
                ghost_vertices - self.pulley_location, axis=1
            )

        if np.any(current_lengths - eps <= 0.0):
            raise ValueError(
                "eps is too large: perturbed cable lengths must stay positive."
            )

        J = np.zeros((3 * len(self.ee_idx), self.nCable))
        for cable_idx in range(self.nCable):
            lengths_plus = current_lengths.copy()
            lengths_minus = current_lengths.copy()
            lengths_plus[cable_idx] += eps
            lengths_minus[cable_idx] -= eps

            vertices_plus = self.deform_CG(
                lengths_plus, vertices.copy(), max_iter=iter, tol=1e-8
            )
            vertices_minus = self.deform_CG(
                lengths_minus, vertices.copy(), max_iter=iter, tol=1e-8
            )
            ee_plus = self.get_ee_poses(vertices_plus).reshape(-1)
            ee_minus = self.get_ee_poses(vertices_minus).reshape(-1)
            J[:, cable_idx] = (ee_plus - ee_minus) / (2.0 * eps)

        return J

        
    def get_ARAP_shape_list(self, vertices):
        ARAP_shape_list = self.initial_ARAP_shape_list.copy()
        for i in range(self.num_vertices):
            neighbour_list = self.neighbour_list[i]
            nNeighbour = len(neighbour_list)
            ARAP_shape = np.zeros((nNeighbour, 3))
            for j in range(nNeighbour):
                neighbour_idx = neighbour_list[j]
                ARAP_shape[j] = vertices[neighbour_idx] - vertices[i]
            ARAP_shape_list[i] = ARAP_shape
        return ARAP_shape_list
    
    def get_rotation_tri(self, vertices):
        R_list = [np.eye(3) for _ in range(self.num_triangles)]
        for i in range(self.num_triangles):
            tri = self.mesh_triangles[i]
            v0, v1, v2 = tri
            initial_tri_sk = self.initial_tri_SK_list[i]
            cur_tri = vertices[tri]
            cur_tri_sk = self.N33 @ cur_tri
            R_list[i] = self._best_fit_rotation(cur_tri_sk, initial_tri_sk)
        return R_list

    def get_rotation_ARAP(self, ARAP_shape_list):
        R_list = [np.eye(3) for _ in range(self.num_vertices)]
        for i in range(self.num_vertices):
            ARAP_shape = ARAP_shape_list[i]
            ARAP_initial_shape = self.initial_ARAP_shape_list[i]
            R_list[i] = self._best_fit_rotation(ARAP_shape, ARAP_initial_shape)
        return R_list

    def get_rotation_cable(self, vertices):
        cur_cable_vec = self.get_cable_vec(vertices)
        R_list = [np.eye(3) for _ in range(self.nCable)]
        # find the R matrix that rotates the initial cable vec to the current cable vec, with rotation axis perpendicular to both vecs
        for i in range(self.nCable):
            initial_cable_vec = self.initial_cable_vec[i]
            cur_cable_vec_i = cur_cable_vec[i]
            if np.linalg.norm(cur_cable_vec_i) < 1e-6 or np.linalg.norm(initial_cable_vec) < 1e-6:
                R_list[i] = np.eye(3)
                continue
            rotation_axis = np.cross(initial_cable_vec, cur_cable_vec_i)
            if np.linalg.norm(rotation_axis) < 1e-6:
                R_list[i] = np.eye(3)
                continue
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(initial_cable_vec, cur_cable_vec_i), -1.0, 1.0))
            K = np.array([[ 0,                   -rotation_axis[2],  rotation_axis[1]],
                          [ rotation_axis[2],     0,                 -rotation_axis[0]],
                          [-rotation_axis[1],     rotation_axis[0],   0              ]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R_list[i] = R
        return R_list

    def get_rotation_ghost(self, vertices, ghost_vertices):
        R_list = [np.eye(3) for _ in range(self.nCable)]
        cur_ghost_shapes = self.get_ghost_shape(vertices, ghost_vertices)
        for i in range(self.nCable):
            initial_ghost_shape = self.initial_ghost_shape_list[i]
            cur_ghost_shape = cur_ghost_shapes[i]
            R_list[i] = self._best_fit_rotation(
                cur_ghost_shape, initial_ghost_shape
            )
        return R_list

    def get_rotation_cable_ghost(self, ghost_vertices):
        cur_cable_vec = []
        R_list = []
        for i in range(self.nCable): # cable vec pointing from pulley location to ghost vertex
            cur_cable_vec.append(ghost_vertices[i] - self.pulley_location[i])
            cur_cable_vec[i] /= np.linalg.norm(cur_cable_vec[i])
            initial_cable_vec = self.initial_cable_vec[i]
            if np.linalg.norm(cur_cable_vec[i]) < 1e-6 or np.linalg.norm(initial_cable_vec) < 1e-6:
                R_list.append(np.eye(3))
                continue
            rotation_axis = np.cross(initial_cable_vec, cur_cable_vec[i])
            if np.linalg.norm(rotation_axis) < 1e-6:
                R_list.append(np.eye(3))
                continue
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(initial_cable_vec, cur_cable_vec[i]), -1.0, 1.0))
            K = np.array([[ 0,                   -rotation_axis[2],  rotation_axis[1]],
                          [ rotation_axis[2],     0,                 -rotation_axis[0]],
                          [-rotation_axis[1],     rotation_axis[0],   0              ]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R_list.append(R)
        return R_list

    def get_Bvec_CG(self,vertices, ghost_vertices, R_list_tri, R_list_cable, tar_cable_length):
        matA_shape = self.matA_all.shape[0]
        bVec = np.zeros((matA_shape, ))
        ghost_R_list = self.get_rotation_ghost(vertices, ghost_vertices)
        for i in range(self.num_triangles):
            initial_tri_sk = self.initial_tri_SK_list[i]
            R_tri = R_list_tri[i]
            mem_row_weight = self.mem_weight_list[i]
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    bVec[idx_row_start+k] += mem_row_weight * (R_tri @ initial_tri_sk.T).T[j, k]
        # print("bvec shape: ", bVec.shape)
        cable_row_weight = self.weight_cable
        for i in range(self.nCable):
            R_cable = R_list_cable[i]
            initial_cable_vec = self.initial_cable_vec[i]
            vec_rotated = R_cable @ initial_cable_vec
            for k in range(3):
                bVec[self.matA_all.shape[0] - 15*self.nCable + 3*i+k] += cable_row_weight * (vec_rotated[k] * tar_cable_length[i] + self.pulley_location[i, k])
        ghost_row_weight = self.weight_ghost
        for i in range(self.nCable):
            R_ghost = ghost_R_list[i]
            initial_ghost_shape = self.initial_ghost_shape_list[i]
            rotated_ghost_shape = (R_ghost @ initial_ghost_shape.T).T
            for j in range(4):
                idx_row_start = matA_shape - 12*self.nCable + 12*i + 3*j
                for k in range(3):
                    bVec[idx_row_start+k] += ghost_row_weight * rotated_ghost_shape[j, k]
        return bVec
    
    def FKD_time(self, target_cable_length, total_time, starting_vertices, tol = 1e-4, show_info = False, h = 0.1):
        target_cable_length = np.asarray(target_cable_length, dtype=float).reshape(-1)
        if target_cable_length.size != self.nCable:
            raise ValueError(f"target_cable_length must contain {self.nCable} values.")
        if np.any(~np.isfinite(target_cable_length)) or np.any(target_cable_length <= 0.0):
            raise ValueError("target_cable_length must contain finite positive values.")
        if not np.isfinite(total_time) or total_time < 0.0:
            raise ValueError("total_time must be a finite non-negative value.")
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("h must be a finite positive value.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")

        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError("starting_vertices should be either a 3n vector or an n by 3 array.")
        Q_a = Q_a.reshape((3*self.num_vertices, ))
        Q_ad = np.zeros((3*self.num_vertices, ))
        t_a = 0.0
        # h = 0.1
        Q_list = [Q_a.copy()]
        diff_count = 0
        cable_tension = np.zeros(self.nCable)
        t_start = time.time()
        diff_list = []
        time_list = [0.0]
        while t_a < total_time:
            dt = min(h, total_time - t_a)
            Q_a_last = Q_a.copy()
            t0 = time.time()
            R_list, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            t1 = time.time()
            K_mat, f0 = self.assemble_K(R_list_1818)
            t2 = time.time()
            disp = Q_a - self.vertices_to_q(self.vertices)
            denom = disp @ self.mass_matrix @ disp
            stiffness_energy = max(float(disp @ K_mat @ disp), 0.0)
            damping_coeff = np.sqrt(stiffness_energy / denom) if denom > 1e-30 else 0.0
            C_mat = 2 * damping_coeff * self.mass_matrix

            A_mat = (1.0/dt)*np.eye(3*self.num_vertices) + dt * self.W_mat @ K_mat + self.W_mat @ C_mat
            lu, piv = lu_factor(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + dt * Q_ad) + f0 + self.gravity_vec - C_mat @ Q_ad)
            # dv_free = A_inv @ b_vec
            dv_free = lu_solve((lu, piv), b_vec)
            Q_free = Q_a + dt * Q_ad + dt * dv_free
            # Solve the nonlinear length constraints with a few updated
            # linearizations. Each increment uses
            # q_correction = dt * A^-1 M^-1 H^T lambda, whose Delassus
            # operator is dt * H A^-1 M^-1 H^T.
            dv_cor = np.zeros_like(dv_free)
            cable_tension = np.zeros(self.nCable)
            constraint_solve_tol = min(tol, 1e-8)
            for _ in range(5):
                Q_constrained = Q_free + dt * dv_cor
                current_lengths = np.asarray(
                    self.get_cable_length_bary(Q_constrained)
                )
                phi = target_cable_length - current_lengths
                if np.max(np.maximum(-phi, 0.0)) <= constraint_solve_tol:
                    break

                H = -self.get_cable_Jacobian_bary(Q_constrained)
                Z = lu_solve((lu, piv), self.W_mat @ H.T)
                lcp_Mmat = dt * H @ Z
                tension_increment = projected_gauss_seidel_lcp(lcp_Mmat, phi)
                if np.linalg.norm(tension_increment, np.inf) <= 1e-14:
                    break
                dv_cor += Z @ tension_increment
                cable_tension += tension_increment

            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + dt * Q_ad
            t_a += dt
            Q_list.append(Q_a.copy())
            diff = np.linalg.norm(Q_a - Q_a_last) / self.num_vertices
            current_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            constraint_error = float(
                np.max(np.maximum(current_lengths - target_cable_length, 0.0))
            )
            # if diff < 1e-5:
            #     h *= 0.1
            # if diff < tol and min(phi_Qfree.flatten()) > -1e-3:
            diff_list.append(diff)
            time_list.append(t_a)
            if diff < tol and constraint_error < tol:
                diff_count += 1
                if diff_count >= 10:
                    # print(f"Converged at time {t_a:.2f} with diff {diff:.6f} for 10 consecutive steps, stopping simulation.")
                    break
            else:
                diff_count = 0
            t3 = time.time()
            if show_info:
                print(f"t_a: {t_a:.3f}, diff: {diff:.7f}, constraint error: {constraint_error:.7f}, time for R_list: {t1-t0:.4f}s, time for K_mat: {t2-t1:.4f}s, total time for this step: {t3-t0:.4f}s")
        t_end = time.time()
        # print(f"Total simulation time: {t_end - t_start:.2f}s")
        vert_length = self.q_to_vertices(Q_a)
        return Q_list, vert_length, cable_tension
        # return Q_list, cable_tension, diff_list, time_list

    def FKD_static(self, target_cable_length, starting_vertices, tol=1e-4,
                   max_iter=300, show_info=False):
        target_cable_length = np.asarray(target_cable_length, dtype=float).reshape(-1)
        if target_cable_length.size != self.nCable:
            raise ValueError(f"target_cable_length must contain {self.nCable} values.")
        if np.any(~np.isfinite(target_cable_length)) or np.any(target_cable_length <= 0.0):
            raise ValueError("target_cable_length must contain finite positive values.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        starting_vertices = np.asarray(starting_vertices, dtype=float)
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.reshape(3*self.num_vertices).copy()
        elif starting_vertices.shape == (self.num_vertices, 3):
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError("starting_vertices should be either a 3n vector or an n by 3 array.")
        Q_list = [Q_a.copy()]
        cable_tension = np.zeros(self.nCable)
        ndof = 3 * self.num_vertices
        constraint_tol = min(tol, 1e-8)
        tension_tol = 1e-10

        for n_iter in range(max_iter):
            Q_a_last = Q_a.copy()
            _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            K_mat, f0 = self.assemble_K(R_list_1818)
            load = f0 + self.gravity_vec

            # K_mat is singular for a free carpet because rigid-body motions
            # have no elastic energy.  Solve it together with the active cable
            # constraints; those constraints remove the relevant null modes.
            current_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            phi = target_cable_length - current_lengths
            print("phi: ", phi)
            H_all = -self.get_cable_Jacobian_bary(Q_a)

            active = set(np.flatnonzero(
                (phi <= constraint_tol) | (cable_tension > tension_tol)
            ).tolist())
            print("active: ", active)
            blocked = set()
            Q_candidate = Q_a.copy()
            tension_candidate = np.zeros(self.nCable)

            for _ in range(2 * self.nCable + 2):
                active_idx = np.array(sorted(active), dtype=int)
                print("active_idx: ", active_idx)
                if active_idx.size == 0:
                    # Without an active cable set the free structure has no
                    # unique static pose under gravity.
                    raise np.linalg.LinAlgError(
                        "Static equilibrium has no active cable constraints to "
                        "remove the rigid-body modes."
                    )

                H = H_all[active_idx]
                kkt = np.block([
                    [K_mat, -H.T],
                    [H, np.zeros((active_idx.size, active_idx.size))],
                ])
                constraint_rhs = H @ Q_a - phi[active_idx]
                rhs = np.concatenate((load, constraint_rhs))
                solution, _, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=1e-10)
                Q_candidate = solution[:ndof]
                lambda_active = solution[ndof:]

                # A negative multiplier would require a cable to push. Remove
                # the most negative cable and resolve the active set.
                negative = np.flatnonzero(lambda_active < -tension_tol)
                if negative.size:
                    local_idx = negative[np.argmin(lambda_active[negative])]
                    cable_idx = int(active_idx[local_idx])
                    active.remove(cable_idx)
                    blocked.add(cable_idx)
                    continue

                candidate_lengths = np.asarray(
                    self.get_cable_length_bary(Q_candidate)
                )
                violation = candidate_lengths - target_cable_length
                inactive = [
                    i for i in range(self.nCable)
                    if i not in active and i not in blocked
                ]
                if inactive:
                    worst = max(inactive, key=lambda i: violation[i])
                    if violation[worst] > constraint_tol:
                        active.add(worst)
                        continue

                tension_candidate[active_idx] = np.maximum(lambda_active, 0.0)
                break
            else:
                raise RuntimeError("Cable active-set solve did not converge.")

            Q_a = Q_candidate
            cable_tension = tension_candidate
            Q_list.append(Q_a.copy())

            final_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            constraint_error = float(np.max(np.maximum(
                final_lengths - target_cable_length, 0.0
            )))
            H_final = -self.get_cable_Jacobian_bary(Q_a)
            equilibrium_residual = (
                K_mat @ Q_a - load - H_final.T @ cable_tension
            )
            relative_residual = np.linalg.norm(equilibrium_residual) / max(
                np.linalg.norm(load), 1.0
            )
            diff = np.linalg.norm(Q_a - Q_a_last) / np.sqrt(ndof)

            if show_info:
                print(
                    f"static iteration {n_iter + 1}: diff = {diff:.7e}, "
                    f"constraint error = {constraint_error:.7e}, "
                    f"relative residual = {relative_residual:.7e}, "
                    f"active cables = {sorted(active)}"
                )
            if diff < tol and constraint_error < tol and relative_residual < tol:
                if show_info:
                    print(f"Static solve converged in {n_iter + 1} iterations.")
                break
        else:
            if show_info:
                print(
                    f"Static solve reached {max_iter} iterations without "
                    "meeting all convergence criteria."
                )

        vert_length = self.q_to_vertices(Q_a)
        return Q_list, vert_length, cable_tension

    def FKD_static_fixedRotation(self, target_cable_length, starting_vertices, tol=1e-4,
                   max_iter=300, show_info=False):
        """Solve static equilibrium while keeping the initial rotations fixed."""
        target_cable_length = np.asarray(
            target_cable_length, dtype=float
        ).reshape(-1)
        if target_cable_length.size != self.nCable:
            raise ValueError(
                f"target_cable_length must contain {self.nCable} values."
            )
        if np.any(~np.isfinite(target_cable_length)) or np.any(
            target_cable_length <= 0.0
        ):
            raise ValueError(
                "target_cable_length must contain finite positive values."
            )
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        starting_vertices = np.asarray(starting_vertices, dtype=float)
        if starting_vertices.shape == (self.num_vertices, 3):
            Q_a = self.vertices_to_q(starting_vertices)
        elif starting_vertices.size == 3 * self.num_vertices:
            Q_a = starting_vertices.reshape(3 * self.num_vertices).copy()
        else:
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 array."
            )
        if np.any(~np.isfinite(Q_a)):
            raise ValueError("starting_vertices must contain only finite values.")

        Q_list = [Q_a.copy()]
        cable_tension = np.zeros(self.nCable)
        ndof = 3 * self.num_vertices
        constraint_tol = min(tol, 1e-8)
        tension_tol = 1e-10

        # Freeze the local element rotations at the configuration supplied to
        # this solve.  In contrast, FKD_static recomputes them every iteration.
        _, initial_rotations = self.get_R_list(self.q_to_vertices(Q_a))
        K_mat, f0 = self.assemble_K(initial_rotations)
        load = f0 + self.gravity_vec

        for n_iter in range(max_iter):
            Q_a_last = Q_a.copy()
            current_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            phi = target_cable_length - current_lengths
            H_all = -self.get_cable_Jacobian_bary(Q_a)

            active = set(np.flatnonzero(
                (phi <= constraint_tol) | (cable_tension > tension_tol)
            ).tolist())
            blocked = set()
            Q_candidate = Q_a.copy()
            tension_candidate = np.zeros(self.nCable)

            for _ in range(2 * self.nCable + 2):
                active_idx = np.array(sorted(active), dtype=int)
                if active_idx.size == 0:
                    raise np.linalg.LinAlgError(
                        "Static equilibrium has no active cable constraints to "
                        "remove the rigid-body modes."
                    )

                H = H_all[active_idx]
                kkt = np.block([
                    [K_mat, -H.T],
                    [H, np.zeros((active_idx.size, active_idx.size))],
                ])
                constraint_rhs = H @ Q_a - phi[active_idx]
                rhs = np.concatenate((load, constraint_rhs))
                solution, _, _, _ = np.linalg.lstsq(kkt, rhs, rcond=1e-10)
                Q_candidate = solution[:ndof]
                lambda_active = solution[ndof:]

                negative = np.flatnonzero(lambda_active < -tension_tol)
                if negative.size:
                    local_idx = negative[np.argmin(lambda_active[negative])]
                    cable_idx = int(active_idx[local_idx])
                    active.remove(cable_idx)
                    blocked.add(cable_idx)
                    continue

                candidate_lengths = np.asarray(
                    self.get_cable_length_bary(Q_candidate)
                )
                violation = candidate_lengths - target_cable_length
                inactive = [
                    i for i in range(self.nCable)
                    if i not in active and i not in blocked
                ]
                if inactive:
                    worst = max(inactive, key=lambda i: violation[i])
                    if violation[worst] > constraint_tol:
                        active.add(worst)
                        continue

                tension_candidate[active_idx] = np.maximum(
                    lambda_active, 0.0
                )
                break
            else:
                raise RuntimeError("Cable active-set solve did not converge.")

            Q_a = Q_candidate
            cable_tension = tension_candidate
            Q_list.append(Q_a.copy())

            final_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            constraint_error = float(np.max(np.maximum(
                final_lengths - target_cable_length, 0.0
            )))
            H_final = -self.get_cable_Jacobian_bary(Q_a)
            equilibrium_residual = (
                K_mat @ Q_a - load - H_final.T @ cable_tension
            )
            relative_residual = np.linalg.norm(equilibrium_residual) / max(
                np.linalg.norm(load), 1.0
            )
            diff = np.linalg.norm(Q_a - Q_a_last) / np.sqrt(ndof)

            if show_info:
                print(
                    f"fixed-rotation static iteration {n_iter + 1}: "
                    f"diff = {diff:.7e}, "
                    f"constraint error = {constraint_error:.7e}, "
                    f"relative residual = {relative_residual:.7e}, "
                    f"active cables = {sorted(active)}"
                )
            if diff < tol and constraint_error < tol and relative_residual < tol:
                if show_info:
                    print(
                        "Fixed-rotation static solve converged in "
                        f"{n_iter + 1} iterations."
                    )
                break
        else:
            if show_info:
                print(
                    f"Fixed-rotation static solve reached {max_iter} "
                    "iterations without meeting all convergence criteria."
                )

        final_vertices = self.q_to_vertices(Q_a)
        return Q_list, final_vertices, cable_tension

    def get_FD_Jacobian_EE(self, Q, delta = 1e-3):
        Q = np.asarray(Q, dtype=float)
        if Q.shape == (self.num_vertices, 3):
            Q = self.vertices_to_q(Q)
        elif Q.size == 3 * self.num_vertices:
            Q = Q.reshape(3 * self.num_vertices)
        else:
            raise ValueError(
                "Q should be either a 3n vector or an n by 3 array."
            )
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a finite positive value.")

        vertices = self.q_to_vertices(Q)
        current_lengths = np.asarray(
            self.get_cable_length_bary(vertices), dtype=float
        )
        ee_current = self.get_ee_poses(vertices).reshape(-1)
        n_ee = len(self.ee_idx)
        Jacobian = np.zeros((3 * n_ee, self.nCable))

        for cable_idx in range(self.nCable):
            lengths_plus = current_lengths.copy()
            lengths_minus = current_lengths.copy()
            lengths_plus[cable_idx] += delta
            lengths_minus[cable_idx] -= delta

            _, vertices_plus, _ = self.FKD_time(
                lengths_plus,5, vertices, tol=1e-5, show_info=False
            )
            _, vertices_minus, _ = self.FKD_time(
                lengths_minus,5, vertices, tol=1e-5, show_info=False
            )
            length_plus_result = np.asarray(
                self.get_cable_length_bary(vertices_plus), dtype=float
            )
            length_minus_result = np.asarray(
                self.get_cable_length_bary(vertices_minus), dtype=float
            )

            d_plus = length_plus_result[cable_idx] - current_lengths[cable_idx]
            d_minus = current_lengths[cable_idx] - length_minus_result[cable_idx]

            print(f"cable_idx: {cable_idx}, d_plus: {d_plus}, d_minus: {d_minus}")
            ee_plus = self.get_ee_poses(vertices_plus).reshape(-1)
            ee_minus = self.get_ee_poses(vertices_minus).reshape(-1)

            Jac_plus = (ee_plus - ee_current) / d_plus if abs(d_plus) >= 1e-8 else np.zeros_like(ee_current)
            Jac_minus = (ee_current - ee_minus) / d_minus if abs(d_minus) >= 1e-8 else np.zeros_like(ee_current)
            Jacobian[:, cable_idx] = 0.5 * (Jac_plus + Jac_minus)

        return Jacobian

    def get_FD_Jacobian_EE_fixedRotation(self, Q, delta = 1e-3):
        """Finite-difference EE Jacobian using frozen initial rotations."""
        Q = np.asarray(Q, dtype=float)
        if Q.shape == (self.num_vertices, 3):
            Q = self.vertices_to_q(Q)
        elif Q.size == 3 * self.num_vertices:
            Q = Q.reshape(3 * self.num_vertices)
        else:
            raise ValueError(
                "Q should be either a 3n vector or an n by 3 array."
            )
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a finite positive value.")

        vertices = self.q_to_vertices(Q)
        current_lengths = np.asarray(
            self.get_cable_length_bary(vertices), dtype=float
        )
        ee_current = self.get_ee_poses(vertices).reshape(-1)
        Jacobian = np.zeros((3 * len(self.ee_idx), self.nCable))

        for cable_idx in range(self.nCable):
            lengths_plus = current_lengths.copy()
            lengths_minus = current_lengths.copy()
            lengths_plus[cable_idx] += delta
            lengths_minus[cable_idx] -= delta

            _, vertices_plus, _ = self.FKD_static_fixedRotation(
                lengths_plus, vertices, tol=1e-5, show_info=False
            )
            _, vertices_minus, _ = self.FKD_static_fixedRotation(
                lengths_minus, vertices, tol=1e-5, show_info=False
            )

            final_plus = np.asarray(
                self.get_cable_length_bary(vertices_plus), dtype=float
            )
            final_minus = np.asarray(
                self.get_cable_length_bary(vertices_minus), dtype=float
            )
            d_plus = final_plus[cable_idx] - current_lengths[cable_idx]
            d_minus = current_lengths[cable_idx] - final_minus[cable_idx]

            ee_plus = self.get_ee_poses(vertices_plus).reshape(-1)
            ee_minus = self.get_ee_poses(vertices_minus).reshape(-1)
            valid_plus = abs(d_plus) >= 1e-8
            valid_minus = abs(d_minus) >= 1e-8
            if valid_plus and valid_minus:
                jac_plus = (ee_plus - ee_current) / d_plus
                jac_minus = (ee_current - ee_minus) / d_minus
                Jacobian[:, cable_idx] = 0.5 * (jac_plus + jac_minus)
            elif valid_plus:
                Jacobian[:, cable_idx] = (ee_plus - ee_current) / d_plus
            elif valid_minus:
                Jacobian[:, cable_idx] = (ee_current - ee_minus) / d_minus

        return Jacobian

    def get_CG_Jacobian_EE_FD(self, Q, delta=1e-3, max_iter=300):
        Q = np.asarray(Q, dtype=float)
        if Q.shape == (self.num_vertices, 3):
            Q = self.vertices_to_q(Q)
        elif Q.size == 3 * self.num_vertices:
            Q = Q.reshape(3 * self.num_vertices)
        else:
            raise ValueError(
                "Q should be either a 3n vector or an n by 3 array."
            )
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a finite positive value.")

        return self.get_CG_Jacobian_FD_EE(
            self.q_to_vertices(Q), eps=delta, iter=max_iter
        )

    def get_CG_Jacobian_EE(self, Q):
        Jac_all = self.get_CG_Jacobian(Q)
        # The Shape-Up solution stores the physical vertex DOFs first,
        # followed by the auxiliary ghost vertices.  Preserve self.ee_idx
        # order and stack x/y/z rows for each end effector.
        Jac_vertices = Jac_all[:3 * self.num_vertices, :].reshape(
            self.num_vertices, 3, self.nCable
        )
        return Jac_vertices[np.asarray(self.ee_idx, dtype=int)].reshape(
            3 * len(self.ee_idx), self.nCable
        )

    def get_jacobian_IK_FD(self, target_EE_pos, Q, delta = 1e-4):
        Jacobian_FD = self.get_FD_Jacobian_EE(Q, delta)
        ee_poses = self.get_ee_poses(Q)
        Jacobian = np.zeros((self.nCable, ))
        for i in range(self.nCable):
            for j in range(len(self.ee_idx)):
                for k in range(3):
                    Jacobian[i] += (ee_poses[j, k] - target_EE_pos[j, k]) * Jacobian_FD[3*j+k, i]
        return Jacobian
    
    def get_jacobian_IK_CG(self, target_EE_pos, Q):
        Jacobian_CG = self.get_CG_Jacobian_EE(Q)
        ee_poses = self.get_ee_poses(Q)
        Jacobian = np.zeros((self.nCable, ))
        for i in range(self.nCable):
            for j in range(len(self.ee_idx)):
                ee_idx_j = self.ee_idx[j]
                for k in range(3):
                    Jacobian[i] += (ee_poses[j, k] - target_EE_pos[j, k]) * Jacobian_CG[3*j+k, i]
        return Jacobian

    def get_Jacobian_pertubate_EE(self, Q, delta = 1e-3):
        """Return the EE-position Jacobian with respect to cable lengths.

        Each EE Cartesian degree of freedom is perturbed in both directions.
        IK supplies the corresponding cable-length changes.  The achieved EE
        changes are used (rather than the requested ``delta`` alone), which
        makes the estimate tolerant of finite IK convergence error.

        Returns
        -------
        numpy.ndarray
            Matrix with shape ``(3 * len(self.ee_idx), self.nCable)`` mapping
            a cable-length increment to the stacked EE-position increment.
        """
        delta = float(delta)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be a finite positive value.")

        Q = np.asarray(Q, dtype=float)
        if Q.shape == (self.num_vertices, 3):
            vertices = Q.copy()
        elif Q.size == 3 * self.num_vertices:
            vertices = self.q_to_vertices(Q.reshape(3 * self.num_vertices))
        else:
            raise ValueError(
                "Q should be either a 3n vector or an n by 3 array."
            )

        current_ee = np.asarray(self.get_ee_poses(vertices), dtype=float)
        n_ee_dofs = 3 * len(self.ee_idx)
        cable_changes = np.zeros((self.nCable, n_ee_dofs))
        ee_changes = np.zeros((n_ee_dofs, n_ee_dofs))
        ik_tol = 1e-4

        for dof in range(n_ee_dofs):
            ee_idx, coordinate = divmod(dof, 3)
            target_plus = current_ee.copy()
            target_minus = current_ee.copy()
            target_plus[ee_idx, coordinate] += delta
            target_minus[ee_idx, coordinate] -= delta

            lengths_plus, vertices_plus, _ = self.IKD_single(
                target_plus, vertices.copy(), tol=ik_tol, max_iter=100,
                show_info=False, initial_guess=True
            )
            lengths_minus, vertices_minus, _ = self.IKD_single(
                target_minus, vertices.copy(), tol=ik_tol, max_iter=100,
                show_info=False, initial_guess=True
            )

            cable_changes[:, dof] = 0.5 * (
                np.asarray(lengths_plus) - np.asarray(lengths_minus)
            )
            ee_changes[:, dof] = 0.5 * (
                self.get_ee_poses(vertices_plus).reshape(-1)
                - self.get_ee_poses(vertices_minus).reshape(-1)
            )

        return ee_changes @ np.linalg.pinv(cable_changes, rcond=1e-8)

    def IKD_single(self, target_EE_pos, starting_vertices, tol = 1e-3, max_iter = 500, show_info = False, initial_guess = True):
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q = self.vertices_to_q(starting_vertices)
        def get_diff(Q):
            ee_poses = self.get_ee_poses(Q)
            diff = 1/2*np.linalg.norm(ee_poses - target_EE_pos)**2
            return diff

        def get_diff_cartesian(Q):
            ee_poses = self.get_ee_poses(Q)
            diff = 0
            for i in range(len(self.ee_idx)):
                diff += np.linalg.norm(ee_poses[i] - target_EE_pos[i])
            diff = diff / len(self.ee_idx)
            return diff

        def get_jacobian(Q):
            ee_poses = self.get_ee_poses(Q)
            Jac_CG = self.get_CG_Jacobian(Q)
            # print("Jac_CG shape: ", Jac_CG.shape)
            Jacobian = np.zeros((self.nCable, ))
            for i in range(self.nCable):
                for j in range(len(self.ee_idx)):
                    ee_idx_j = self.ee_idx[j]
                    for k in range(3):
                        Jacobian[i] += (ee_poses[j, k] - target_EE_pos[j, k]) * Jac_CG[3*ee_idx_j+k, i]
            return Jacobian
        
        if initial_guess:
            starting_vertices = self.get_fixedEE_guess_vertices(target_EE_pos)
            cur_length = self.get_cable_length_bary(starting_vertices)
            # Q_list, starting_vertices, cable_tension = self.FKD_static(cur_length, starting_vertices, tol = 5e-5, show_info=False)
            Q_list, starting_vertices, cable_tension = self.FKD_time(cur_length, 10, starting_vertices, tol = 5e-5, h = 0.01, show_info=False)
            # self.visualize_vert(starting_vertices)
        Q = self.vertices_to_q(starting_vertices)
        final_Q_list = [Q.copy()]
        for i in range(max_iter):
            dl = 0.1
            jac = get_jacobian(Q)
            diff = get_diff(Q)
            cur_length = self.get_cable_length_bary(Q)
            cmd_diff = [0 for _ in range(self.nCable)]
            cmd_length = cur_length.copy()
            # alpha = 1
            # dl = alpha*diff/(np.max(np.abs(jac)))
            if i == 0:
                for k in range(self.nCable):
                    cmd_diff[k] = -dl * jac[k]
            else:
                for k in range(self.nCable):
                    if cable_tension[k] < 1e-5 and jac[k]< 0:
                        cmd_diff[k] = 0
                    else:
                        cmd_diff[k] = -dl * jac[k]
            # for k in range(self.nCable):
            #     cmd_diff[k] = -dl * jac[k]
            # cmd_diff = clamp_diff(cmd_diff, min_bound = 1e-3, max_bound = 0.05)
            for k in range(self.nCable):
                cmd_length[k] += cmd_diff[k]
            Q_list, starting_vertices, cable_tension = self.FKD_time(cmd_length, 10, Q, tol = 1e-4, h = 0.01, show_info=False)
            
            # self.visualize_IKD_result(target_EE_pos, starting_vertices)
            # input("Press Enter to continue...")
            Q = self.vertices_to_q(starting_vertices)
            final_Q_list.append(Q.copy())
            diff_cart = get_diff_cartesian(Q)
            diff = get_diff(Q)
            if show_info:
                print("Iteration {}: diff = {}, diff_cart = {}, dl = {}, Jacobian: {}, cable_tension: {}, cmd_diff: {}".format(i, diff, diff_cart, dl, np.round(jac, 5), np.round(cable_tension, 5), np.round(cmd_diff, 5)))
            if diff < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break
            cur_length = self.get_cable_length_bary(Q)
        return cur_length, starting_vertices, final_Q_list

    def IKD_single_returnMore(self, target_EE_pos, starting_vertices, tol = 1e-3, max_iter = 500, show_info = False, initial_guess = True):
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q = self.vertices_to_q(starting_vertices)
        def get_diff(Q):
            ee_poses = self.get_ee_poses(Q)
            # diff = 1/2*np.linalg.norm(ee_poses - target_EE_pos)**2
            diff = 0
            for i in range(len(self.ee_idx)):
                diff += 1/2 * np.linalg.norm(ee_poses[i] - target_EE_pos[i])**2
            return diff

        def get_diff_cartesian(Q):
            ee_poses = self.get_ee_poses(Q)
            diff = 0
            for i in range(len(self.ee_idx)):
                diff += np.linalg.norm(ee_poses[i] - target_EE_pos[i])
            diff = diff / len(self.ee_idx)
            return diff

        def get_jacobian(Q):
            ee_poses = self.get_ee_poses(Q)
            Jac_CG = self.get_CG_Jacobian(Q)
            # print("Jac_CG shape: ", Jac_CG.shape)
            Jacobian = np.zeros((self.nCable, ))
            for i in range(self.nCable):
                for j in range(len(self.ee_idx)):
                    ee_idx_j = self.ee_idx[j]
                    for k in range(3):
                        Jacobian[i] += (ee_poses[j, k] - target_EE_pos[j, k]) * Jac_CG[3*ee_idx_j+k, i]
            return Jacobian
        
        if initial_guess:
            starting_vertices = self.get_fixedEE_guess_vertices(target_EE_pos)
            cur_length = self.get_cable_length_bary(starting_vertices)
            # Q_list, starting_vertices, cable_tension = self.FKD_static(cur_length, starting_vertices, tol = 5e-5, show_info=False)
            Q_list, starting_vertices, cable_tension = self.FKD_time(cur_length, 10, starting_vertices, tol = 5e-5, h = 0.01, show_info=False)
            # self.visualize_vert(starting_vertices)
        Q = self.vertices_to_q(starting_vertices)
        final_Q_list = [Q.copy()]
        diff_list = [get_diff(Q)]
        Jac_list = []
        dl_list = []
        for i in range(max_iter):
            dl = 1
            
            jac = get_jacobian(Q)
            diff = get_diff(Q)
            cur_length = self.get_cable_length_bary(Q)
            cmd_diff = [0 for _ in range(self.nCable)]
            cmd_length = cur_length.copy()
            alpha = 1
            dl = alpha*diff/(np.max(np.abs(jac)))
            if i == 0:
                for k in range(self.nCable):
                    cmd_diff[k] = -dl * jac[k]
            else:
                for k in range(self.nCable):
                    if cable_tension[k] < 1e-5 and jac[k]< 0:
                        cmd_diff[k] = 0
                    else:
                        cmd_diff[k] = -dl * jac[k]
            dl_list.append(cmd_diff)
            # for k in range(self.nCable):
            #     cmd_diff[k] = -dl * jac[k]
            # cmd_diff = clamp_diff(cmd_diff, min_bound = 1e-3, max_bound = 0.05)
            for k in range(self.nCable):
                cmd_length[k] += cmd_diff[k]
            Q_list, starting_vertices, cable_tension = self.FKD_time(cmd_length, 10, Q, tol = 1e-4, h = 0.01, show_info=False)
            
            # self.visualize_IKD_result(target_EE_pos, starting_vertices)
            # input("Press Enter to continue...")
            Q = self.vertices_to_q(starting_vertices)
            final_Q_list.append(Q.copy())
            diff_cart = get_diff_cartesian(Q)
            diff = get_diff(Q)
            diff_list.append(diff)
            Jac_list.append(jac)
            if show_info:
                print("Iteration {}: diff = {}, diff_cart = {}, dl = {}, Jacobian: {}, cable_tension: {}, cmd_diff: {}".format(i, diff, diff_cart, dl, np.round(jac, 5), np.round(cable_tension, 5), np.round(cmd_diff, 5)))
            if diff < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break
            cur_length = self.get_cable_length_bary(Q)
        return Jac_list, dl_list, diff_list, cur_length, starting_vertices, final_Q_list


    def IKD_single_FD(self, target_EE_pos, starting_vertices, tol = 1e-3, max_iter = 500, show_info = False, initial_guess = True):
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q = self.vertices_to_q(starting_vertices)
        def get_diff(Q):
            ee_poses = self.get_ee_poses(Q)
            diff = 1/2*np.linalg.norm(ee_poses - target_EE_pos)**2
            return diff

        def get_diff_cartesian(Q):
            ee_poses = self.get_ee_poses(Q)
            diff = 0
            for i in range(len(self.ee_idx)):
                diff += np.linalg.norm(ee_poses[i] - target_EE_pos[i])
            diff = diff / len(self.ee_idx)
            return diff

        def get_jacobian_FD(Q, cur_length, diff, delta=1e-3):
            """Finite-difference the IK objective with respect to cable commands."""
            jacobian_FD = np.zeros((self.nCable,))
            for cable_idx in range(self.nCable):
                perturbed_length = cur_length.copy()
                # Shortening the cable keeps the unilateral constraint active.
                cable_delta = min(delta, 0.5 * cur_length[cable_idx])
                perturbed_length[cable_idx] -= cable_delta
                _, perturbed_vertices, _ = self.FKD_time(
                    perturbed_length, 10, Q, tol=1e-4, h=0.01,
                    show_info=False
                )
                perturbed_Q = self.vertices_to_q(perturbed_vertices)
                jacobian_FD[cable_idx] = (
                    diff - get_diff(perturbed_Q)
                ) / cable_delta
            return jacobian_FD

        if initial_guess:
            starting_vertices = self.get_fixedEE_guess_vertices(target_EE_pos)
            cur_length = self.get_cable_length_bary(starting_vertices)
            Q_list, starting_vertices, cable_tension = self.FKD_time(cur_length, 10, starting_vertices, tol = 5e-5, h = 0.01, show_info=False)
        Q = self.vertices_to_q(starting_vertices)
        final_Q_list = [Q.copy()]
        for i in range(max_iter):
            dl = 0.1
            diff = get_diff(Q)
            cur_length = self.get_cable_length_bary(Q)
            jac_FD = get_jacobian_FD(Q, cur_length, diff)
            cmd_diff = np.zeros((self.nCable,))
            cmd_length = cur_length.copy()
            for k in range(self.nCable):
                if i > 0 and cable_tension[k] < 1e-5 and jac_FD[k] < 0:
                    continue
                cmd_diff[k] = -dl * jac_FD[k]

            cmd_diff = np.asarray(
                clamp_diff(cmd_diff, min_bound=1e-3, max_bound=0.05)
            )
            cmd_length += cmd_diff
            _, starting_vertices, cable_tension = self.FKD_time(
                cmd_length, 10, Q, tol=1e-4, h=0.01
            )

            Q = self.vertices_to_q(starting_vertices)
            final_Q_list.append(Q.copy())
            diff_cart = get_diff_cartesian(Q)
            if show_info:
                print(
                    "Iteration {}: diff = {}, diff_cart = {}, dl = {}, "
                    "FD gradient: {}, cable_tension: {}, cmd_diff: {}".format(
                        i, diff, diff_cart, dl, np.round(jac_FD, 5),
                        np.round(cable_tension, 5), np.round(cmd_diff, 5)
                    )
                )
            if diff_cart < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break

        cur_length = self.get_cable_length_bary(Q)
        return cur_length, starting_vertices, final_Q_list

    def IKD_single_minimize(self, target_EE_pos, starting_vertices, tol = 1e-3, max_iter = 500, show_info = False, initial_guess = True):
        """Solve IK by minimizing EE error over the commanded cable lengths.

        Each objective evaluation settles the model with ``FKD_time``.  SciPy
        estimates the optimization gradient using three-point finite
        differences.
        """
        target_EE_pos = np.asarray(target_EE_pos, dtype=float)
        expected_target_shape = (len(self.ee_idx), 3)
        if target_EE_pos.shape != expected_target_shape:
            raise ValueError(
                f"target_EE_pos must have shape {expected_target_shape}."
            )
        if np.any(~np.isfinite(target_EE_pos)):
            raise ValueError("target_EE_pos must contain finite values.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")
        if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        starting_vertices = np.asarray(starting_vertices, dtype=float)
        if (
            starting_vertices.ndim in (1, 2)
            and starting_vertices.shape[0] == 3*self.num_vertices
            and starting_vertices.size == 3*self.num_vertices
        ):
            starting_vertices = self.q_to_vertices(
                starting_vertices.reshape(3*self.num_vertices)
            )
        elif starting_vertices.shape != (self.num_vertices, 3):
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 "
                "array."
            )
        if np.any(~np.isfinite(starting_vertices)):
            raise ValueError("starting_vertices must contain finite values.")
        starting_vertices = starting_vertices.copy()

        if initial_guess:
            starting_vertices = self.get_fixedEE_guess_vertices(target_EE_pos)
            initial_length = np.asarray(
                self.get_cable_length_bary(starting_vertices), dtype=float
            )
            _, starting_vertices, _ = self.FKD_time(
                initial_length, 10, starting_vertices, tol=5e-5, h=0.01,
                show_info=False
            )

        initial_length = np.asarray(
            self.get_cable_length_bary(starting_vertices), dtype=float
        )
        evaluation_count = 0
        cached_length = None
        cached_vertices = None
        cached_ee_poses = None
        final_Q_list = [self.vertices_to_q(starting_vertices).copy()]

        def forward_kinematics(cable_length):
            nonlocal evaluation_count
            nonlocal cached_length, cached_vertices, cached_ee_poses
            cable_length = np.asarray(cable_length, dtype=float)
            if (
                cached_length is not None
                and np.array_equal(cable_length, cached_length)
            ):
                return cached_vertices, cached_ee_poses

            _, vertices, _ = self.FKD_time(
                cable_length, 5, starting_vertices, tol=1e-4, h=0.01,
                show_info=False
            )
            ee_poses = np.asarray(self.get_ee_poses(vertices), dtype=float)
            evaluation_count += 1
            cached_length = cable_length.copy()
            cached_vertices = vertices.copy()
            cached_ee_poses = ee_poses
            if show_info:
                print(
                    f"Minimize FK evaluation {evaluation_count}, "
                    f"objective={0.5 * np.sum((ee_poses-target_EE_pos)**2):.6e}",
                    flush=True,
                )
            return cached_vertices, cached_ee_poses

        def objective_function(cable_length):
            _, ee_poses = forward_kinematics(cable_length)
            residual = (ee_poses - target_EE_pos).reshape(-1)
            return 0.5 * residual @ residual

        def save_iteration(cable_length):
            vertices, _ = forward_kinematics(cable_length)
            final_Q_list.append(self.vertices_to_q(vertices).copy())

        result = minimize(
            objective_function,
            initial_length,
            method="SLSQP",
            bounds=[(np.finfo(float).eps, None)] * self.nCable,
            jac="3-point",
            callback=save_iteration,
            tol=tol,
            options={
                "ftol": max(0.5 * tol**2, np.finfo(float).eps),
                "finite_diff_rel_step": 1e-3,
                "maxiter": max_iter,
                "disp": show_info,
            },
        )

        final_vertices, final_ee_poses = forward_kinematics(result.x)
        final_Q = self.vertices_to_q(final_vertices)
        if not np.array_equal(final_Q_list[-1], final_Q):
            final_Q_list.append(final_Q.copy())
        final_length = np.asarray(
            self.get_cable_length_bary(final_vertices), dtype=float
        )
        final_error = np.mean(
            np.linalg.norm(final_ee_poses - target_EE_pos, axis=1)
        )

        if show_info:
            print(
                f"IK minimize evaluations: {evaluation_count}, "
                f"success: {result.success}, final error: {final_error:.7e}"
            )
            if not result.success:
                print(f"Optimizer message: {result.message}")

        return final_length, final_vertices, final_Q_list

    def get_fixedEE_guess_vertices(self, target_EE_pos, show_info = False):
        def assemble_K_tilde(Q):
            R_list, R_list_1818 = self.get_R_list(self.q_to_vertices(Q))
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
                        # find the idx of tri[k] in the ee_idx list
                        idx_ee = -1
                        for l in range(len(self.ee_idx)):
                            if tri[k] == self.ee_idx[l]:
                                idx_ee = l
                                break
                        if idx_k_moving == -1:
                            K_tilde_vec2add[row] -= Ke_jk @ target_EE_pos[idx_ee]
                        else:
                            col = slice(3*idx_k_moving, 3*idx_k_moving+3)
                            K_tilde[row, col] += Ke_jk

            return K_tilde, f0_tilde, K_tilde_vec2add

        Q_a = self.vertices_to_q(self.vertices).reshape((3*self.num_vertices, ))
        Q_list = [Q_a.copy()]
        gravity_tilde = self.q_to_q_moving(self.gravity_vec)
        max_iter = 500
        tol = 1e-6
        for iteration in range(max_iter):
            K_tilde, f0_tilde, K_tilde_vec2add = assemble_K_tilde(Q_a)
            rhs = f0_tilde + gravity_tilde + K_tilde_vec2add
            try:
                Q_moving = np.linalg.solve(K_tilde, rhs)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "The reduced stiffness matrix is singular; check that the "
                    "fixed region removes all rigid-body modes."
                ) from exc

            Q_next = self.q_moving_to_q(Q_moving, target_EE_pos)
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
        return self.q_to_vertices(Q_a)

    def deform_CG(self, tar_cable_length, starting_vertices, max_iter = 300, tol = 1e-8):
        cur_vertices = starting_vertices.copy()
        cur_q = self.vertices_to_q(starting_vertices)
        q_last = cur_q.copy()
        ghost_vertices = self.get_pp_location_bary(cur_vertices)
        time_total_start = time.time()
        time_R_total = 0
        for i in range(max_iter):
            time_R_start = time.time()
            R_list_cable = self.get_rotation_cable_ghost(ghost_vertices)
            R_list_tri = self.get_rotation_tri(cur_vertices)
            bVec = self.get_Bvec_CG(cur_vertices, ghost_vertices, R_list_tri, R_list_cable, tar_cable_length)
            time_R_total += time.time() - time_R_start
            cur_q_all = self.matATA_inv_AT @ bVec

            cur_q = cur_q_all[:3*self.num_vertices]
            cur_ghost_q = cur_q_all[3*self.num_vertices:]
            ghost_vertices = cur_ghost_q.reshape((self.nCable, 3))
            cur_vertices = self.q_to_vertices(cur_q)
            diff = np.linalg.norm(cur_q - q_last)/(3*self.num_vertices)
            q_last = cur_q.copy()
            # print("ARAP iteration {}: diff = {}".format(i, diff))
            if diff < tol:
                break
        time_total = time.time() - time_total_start
        print("time of R in total time: ", time_R_total, "total time: ", time_total, "time of R ratio: ", time_R_total/time_total)
        return cur_vertices

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


    def assemble_K(self, R_list_1818):
        Ke_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        Ke0_list = [np.zeros((18,18)) for _ in range(self.num_RF_triangles)]
        K_mat = np.zeros((self.num_vertices * 3, self.num_vertices * 3))
        f0 = np.zeros(self.num_vertices * 3)
        for i in range(self.num_RF_triangles):
            Ke_list[i] = R_list_1818[i] @ self.stiffness_matrices[i] @ R_list_1818[i].T 
            Ke0_list[i] = R_list_1818[i] @ self.stiffness_matrices[i]
            tri = self.mesh_RF_triangles[i]
            # print(f"tri: {tri}")
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
      

    def get_R_list(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        tri_SK_list = self.get_tri_SK_list(vertices)
        R_list = [np.eye(3) for _ in range(self.num_RF_triangles)]
        R_list_1818 = [np.eye(18) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            initial_tri_SK = self.initial_tri_SK_list[i]
            cur_tri_SK = tri_SK_list[i]
            R_list[i] = self._best_fit_rotation(cur_tri_SK, initial_tri_SK)
            for j in range(6):
                R_list_1818[i][3*j:3*j+3, 3*j:3*j+3] = R_list[i]
        return R_list, R_list_1818
    
    def vertices_to_q(self, vertices):
        # map the n by 3 vertices to a 3n vector q
        q = vertices.flatten()
        return q
    
    def q_to_vertices(self, q):
        # map the 3n vector q to n by 3 vertices
        vertices = q.reshape(-1, 3)
        return vertices

    def get_ee_poses(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        ee_poses = vertices[self.ee_idx]
        return ee_poses

    def get_ghost_shape(self, vertices, ghost_vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        ghost_shape_list = []
        for i in range(self.nCable):
            pp_tri = self.pp_bary_tri_idx[i]
            tri = self.mesh_triangles[pp_tri]
            ghost_shape = np.zeros((4,3))
            for j in range(3):
                v_idx = tri[j]
                ghost_shape[j] = vertices[v_idx]
            ghost_shape[3] = ghost_vertices[i]
            ghost_shape_list.append(self.N44 @ ghost_shape)
        return ghost_shape_list

    def get_cable_length(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        cable_length = [0 for _ in range(self.nCable)]
        for i in range(self.nCable):
            pulley_location = self.pulley_location[i]
            pp_vertex = vertices[self.pp_idx[i]]
            cable_length[i] = np.linalg.norm(pulley_location - pp_vertex)
        return cable_length

    def get_cable_length_bary(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_locations = self.get_pp_location_bary(vertices)
        return [np.linalg.norm(self.pulley_location[i] - pp_locations[i]) for i in range(self.nCable)]

    def get_cable_vec(self, vertices): # cable vec point from the pulley to the pull point, normalized
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        cable_vec = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            pulley_location = self.pulley_location[i]
            pp_vertex = vertices[self.pp_idx[i]]
            vec = pp_vertex - pulley_location
            cable_vec[i] = vec / np.linalg.norm(vec)
        return cable_vec

    def get_cable_vec_bary(self, vertices): # cable vec point from the pulley to the pull point, normalized
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_locations = self.get_pp_location_bary(vertices)
        cable_vec = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            pulley_location = self.pulley_location[i]
            vec = pp_locations[i] - pulley_location
            cable_vec[i] = vec / np.linalg.norm(vec)
        return cable_vec

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

    def get_cable_Jacobian_bary(self, vertices):
        """
        (nCable, nVertices*3) Jacobian of cable lengths w.r.t. all vertex DOFs,
        for the barycentric+offset pull-point representation.

        Each column block (3 cols for vertex k) is:
            u_hat @ (bary_k * I  +  offset * d(t3)/d(v_k))
        where d(t3)/d(v_k) = P @ Gn_k / (2*area),  P = I - t3 t3^T.
        """
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        cable_Jacobian = np.zeros((self.nCable, self.num_vertices * 3))
        pp_locations = self.get_pp_location_bary(vertices)
        for i in range(self.nCable):
            idx_tri = self.description['pp_bary_tri_idx'][i]
            bary    = self.description['pp_bary_coords'][i]
            offset  = self.description['pp_bary_offsets'][i]
            tri     = self.mesh_triangles[idx_tri]
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1, e2 = v1 - v0, v2 - v0
            n  = np.cross(e1, e2)
            A2 = np.linalg.norm(n)          # 2 * triangle area
            t3 = n / A2
            P  = np.eye(3) - np.outer(t3, t3)
            # dn/dv_k skew matrices: Gn = (skew(e2-e1), -skew(e2), skew(e1))
            Gn = (skew(e2 - e1), -skew(e2), skew(e1))
            vec   = pp_locations[i] - self.pulley_location[i]
            u_hat = vec / np.linalg.norm(vec)
            for k in range(3):
                Gt = (P @ Gn[k]) / A2          # d(t3)/d(v_k), shape (3,3)
                dpp_dvk = bary[k] * np.eye(3) + offset * Gt
                col = 3 * tri[k]
                cable_Jacobian[i, col:col + 3] = u_hat @ dpp_dvk
        return cable_Jacobian

    def visualize_vert(self, vertices):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        plotter.add_points(pp_locations, color='blue', point_size=10, label='Pullpoints')
        plotter.add_points(self.pulley_location, color='blue', point_size=10, label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')

        # add grid
        plotter.show_grid()
        plotter.show_axes()
        # plotter.add_legend()
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
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_vert_paper(self, vertices):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))
        mesh_original = pv.PolyData(self.vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_original, color='lightgray', show_edges=False, opacity = 0.5)
        plotter.add_mesh(mesh, color='blue', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        
        plotter.add_points(self.pulley_location, color='blue', point_size=10, label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_points(pp_locations[i], color='blue', point_size=10, label='Pullpoints')
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')

        # add grid
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()


    def visualize_IKD_result(self, target_ee_pos, vertices):
        ee_poses = self.get_ee_poses(vertices)
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        plotter.add_points(ee_poses, color='red', point_size=10, label='End Effectors')
        # add cables
        pp_locations = self.get_pp_location_bary(vertices)
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        plotter.add_points(target_ee_pos, color='green', point_size=10, label='Target EE Pos')
        # add grid
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def get_pp_location_bary(self, vertices):
        """Compute pull-point world positions from barycentric coords + normal offset."""
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        pp_location = np.zeros((self.nCable, 3))
        for i in range(self.nCable):
            idx_tri = self.description['pp_bary_tri_idx'][i]
            bary    = self.description['pp_bary_coords'][i]
            offset  = self.description['pp_bary_offsets'][i]
            tri     = self.mesh_triangles[idx_tri]
            pp_on_surface = bary @ vertices[tri]
            n = get_normal(vertices[tri])
            pp_location[i] = pp_on_surface + offset * n
        return pp_location

    def check_bending_params(self):
        val_list = []
        for i in range(len(self.bending_ele_idx)):
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            val = c1 * self.vertices[v0] + c2 * self.vertices[v1] + c3 * self.vertices[v2] + c4 * self.vertices[v3]
            val_list.append(val)
        val_array = np.array(val_list)
        print("Bending param values: ", val_array)
        # check if val_array is close to zero
        if np.all(np.linalg.norm(val_array, axis=1) < 1e-6):
            print("Bending params are valid.")

    def get_bending_energy(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        energy = 0
        for i in range(len(self.bending_ele_idx)):
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            val = c1 * vertices[v0] + c2 * vertices[v1] + c3 * vertices[v2] + c4 * vertices[v3]
            energy += self.bending_weight_list[i] * np.linalg.norm(val)**2
        return energy
            
    def q_to_q_moving(self, q):
        q_moving = np.zeros((self.nMoving * 3, ))
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] != -1:
                idx_moving = self.idxAll_2_idxMoving[i]
                q_moving[3*idx_moving:3*idx_moving+3] = q[3*i:3*i+3]
        return q_moving
    
    def q_moving_to_q(self, q_moving, target_EE_pos = None):
        q = self.vertices_to_q(self.vertices)
        for i in range(self.nMoving):
            idx_all = self.idxMoving_2_idxAll[i]
            q[3*idx_all:3*idx_all+3] = q_moving[3*i:3*i+3]
        for i in range(len(self.ee_idx)):
            idx_all = self.ee_idx[i]
            if target_EE_pos is not None:
                q[3*idx_all:3*idx_all+3] = target_EE_pos[i]
            else:
                q[3*idx_all:3*idx_all+3] = self.vertices[idx_all]
        return q

    def visualize_vert_w_fb(self, vertices, fb_pts):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        fb_vertices = self.get_fb_surface(vertices)
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))
        mesh_surface = pv.PolyData(vertices, faces)
        mesh_fb = pv.PolyData(fb_vertices, faces)

        plotter = pv.Plotter()
        # plotter.add_mesh(mesh_surface, color='lightgray', show_edges=True, opacity=0.35, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightblue', show_edges=True, opacity=0.95, label='FB Mid-Surface')
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='FB EE Vertices')
        plotter.add_points(fb_pts, color='yellow', point_size=8, render_points_as_spheres=True)

        
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()
        

    def replay_Q_list(self, Q_list, filePath="./flying_carpet_FKD.mp4", framerate=10,
                       window_size=(1024, 768)):
        def _to_vertices(Q):
            if Q.shape[0] == 3 * self.num_vertices:
                return self.q_to_vertices(Q)
            return Q

        plotter = pv.Plotter(off_screen=True, window_size=window_size)

        vertices0 = _to_vertices(Q_list[0])
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))

        # Build all actors once; update .points in-place each frame
        surf = pv.PolyData(vertices0.copy(), faces)
        plotter.add_mesh(surf, color='lightgray', show_edges=True)

        pp_cloud = pv.PolyData(vertices0[self.pp_idx].copy())
        plotter.add_mesh(pp_cloud, color='blue', point_size=10,
                         render_points_as_spheres=True, label='Pull points')

        pulley_cloud = pv.PolyData(self.pulley_location.copy())
        plotter.add_mesh(pulley_cloud, color='cyan', point_size=10,
                         render_points_as_spheres=True, label='Pulleys')

        ee_cloud = pv.PolyData(vertices0[self.ee_idx].copy())
        plotter.add_mesh(ee_cloud, color='red', point_size=10,
                         render_points_as_spheres=True, label='End Effectors')

        # All cable segments in a single PolyData so points update in-place
        cable_pts = np.empty((2 * self.nCable, 3))
        for i in range(self.nCable):
            cable_pts[2 * i]     = vertices0[self.pp_idx[i]]
            cable_pts[2 * i + 1] = self.pulley_location[i]
        cable_lines = np.array([[2, 2 * i, 2 * i + 1]
                                 for i in range(self.nCable)]).flatten()
        cables = pv.PolyData()
        cables.points = cable_pts.copy()
        cables.lines = cable_lines
        plotter.add_mesh(cables, color='blue', line_width=2)

        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.open_movie(filePath, framerate=framerate)

        for Q in Q_list:
            vertices = _to_vertices(Q)
            surf.points = vertices.copy()
            pp_cloud.points = vertices[self.pp_idx].copy()
            ee_cloud.points = vertices[self.ee_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = vertices[self.pp_idx[i]]
            cables.points = cable_pts.copy()
            plotter.write_frame()

        plotter.close()


    def replay_IKD_Q_list(self, target_EE_pos, Q_list, filePath="./flying_carpet_IKD.mp4", window_size=(1024, 768), framerate=1):
        def _to_vertices(Q):
            if Q.shape[0] == 3 * self.num_vertices:
                return self.q_to_vertices(Q)
            return Q
        plotter = pv.Plotter(off_screen=True, window_size=window_size)

        vertices0 = _to_vertices(Q_list[0])
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))

        # Build all actors once; update .points in-place each frame
        surf = pv.PolyData(vertices0.copy(), faces)
        plotter.add_mesh(surf, color='lightgray', show_edges=True)

        pp_locations = self.get_pp_location_bary(vertices0)
        pp_cloud = pv.PolyData(pp_locations)
        plotter.add_mesh(pp_cloud, color='blue', point_size=10,
                         render_points_as_spheres=True, label='Pull points')

        pulley_cloud = pv.PolyData(self.pulley_location.copy())
        plotter.add_mesh(pulley_cloud, color='cyan', point_size=10,
                         render_points_as_spheres=True, label='Pulleys')

        ee_cloud = pv.PolyData(vertices0[self.ee_idx].copy())
        plotter.add_mesh(ee_cloud, color='red', point_size=10,
                         render_points_as_spheres=True, label='End Effectors')


        # add target ee pos as a green point
        for i in range(len(self.ee_idx)):
            plotter.add_points(target_EE_pos[i].reshape((1,3)), color='green', point_size=10,
                                render_points_as_spheres=True, label='Target EE Pos' if i==0 else None)
        # All cable segments in a single PolyData so points update in-place
        cable_pts = np.empty((2 * self.nCable, 3))
        for i in range(self.nCable):
            cable_pts[2 * i]     = pp_locations[i]
            cable_pts[2 * i + 1] = self.pulley_location[i]
        cable_lines = np.array([[2, 2 * i, 2 * i + 1]
                                 for i in range(self.nCable)]).flatten()
        cables = pv.PolyData()
        cables.points = cable_pts.copy()
        cables.lines = cable_lines
        plotter.add_mesh(cables, color='blue', line_width=2)

        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.open_movie(filePath, framerate=framerate)
        for Q in Q_list:
            vertices = _to_vertices(Q)
            surf.points = vertices.copy()
            
            pp_locations = self.get_pp_location_bary(vertices)
            pp_cloud.points = pp_locations
            ee_cloud.points = vertices[self.ee_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = pp_locations[i]
            cables.points = cable_pts.copy()
            plotter.write_frame()

        plotter.close()

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    icl = flying_carpet.initial_cable_length
    shortened_length = 0.04
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    start_time = time.time()
    flying_carpet.reassemble_CG_matrices()
    # Q_list, vert_length, cable_tension = flying_carpet.FKD_time(tcl,5, flying_carpet.vertices, tol=1e-4, show_info=False)
    print("Time taken for FKD_time: ", time.time() - start_time)
    exit(0)
    # flying_carpet.visualize_vert(flying_carpet.vertices)
    # exit(0)
    icl = flying_carpet.initial_cable_length
    shortened_length = 0.04
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    print("Target cable length shortened for " , shortened_length,  ", tcl=", tcl)
    tcl = np.array([372,550,388,570,363,518,379,525])*1e-3
    tcl = tcl.tolist()
    Q_list, vert_length, cable_tension = flying_carpet.FKD_static(tcl, flying_carpet.vertices, max_iter=1000, tol=1e-4, show_info=True)
    flying_carpet.visualize_vert(vert_length)
    exit(0)

    shortened_length = 0.06
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    print("Target cable length shortened for " , shortened_length,  ", tcl=", tcl)
    shortened_length = 0.08
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    print("Target cable length shortened for " , shortened_length,  ", tcl=", tcl)
    exit(0)
    Q_list, vert_length, cable_tension = flying_carpet.FKD_time(tcl, 1, flying_carpet.vertices, tol = 1e-5, show_info=True)
    # flying_carpet.visualize_fb_surface(vert_length)
    flying_carpet.visualize_vert_paper(vert_length)
    # flying_carpet.replay_Q_list(Q_list, filePath="./flying_carpet_FKD.mp4", framerate=10)
    # fcl = flying_carpet.get_cable_length(vert_length)
    # diff_cl = [fcl[i] - tcl[i] for i in range(flying_carpet.nCable)]
    # print("Final cable length: ", fcl)

    # vert_cg = flying_carpet.deform_CG(tcl, flying_carpet.vertices, max_iter=1000, tol=1e-9)
    # flying_carpet.visualize_vert(vert_cg)
    # flying_carpet.visualize_vert(vert_cg)
