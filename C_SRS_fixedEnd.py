import pickle
from utilities import *
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
import torch
import torch.nn as nn
import joblib
import time
from scipy.optimize import minimize
class C_SRS_fixedEnd:
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
            for slot, (a, b) in enumerate(((1, 2), (2, 0), (0, 1)), start=3):
                edge = tuple(sorted((int(triangle[a]), int(triangle[b]))))
                incidents = edge_map[edge]
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
        # check if there are <0 element in mesh_RF_triangles
        # self.pp_idx = self.description['pp_idx']
        self.pp_bary_tri_idx = self.description['pp_bary_tri_idx']
        self.pp_bary_coords = self.description['pp_bary_coords']
        self.pp_bary_offsets = self.description['pp_bary_offsets']
        self.pulley_location = self.description['pulley_locations']
        # Older description files store the three EBST neighbour slots in a
        # different order. Rebuilding from connectivity also fixes those files.
        self.mesh_RF_triangles = self._build_canonical_rf_triangles(
            self.mesh_triangles
        )
        self.description['mesh_RF_triangles'] = self.mesh_RF_triangles
        self.ee_idx = self.description['ee_idx']
        print("EE idxs: ", self.ee_idx)
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
        self.n_bending_ele = len(self.bending_ele_idx)
        self.thickness = self.description['thickness']
        self.Youngs_modulus = self.description['Youngs_modulus']
        self.Youngs_modulus = 3.58e7
        self.Poisson_ratio = self.description['Poisson_ratio']
        self.Poisson_ratio = 0.39
        self.reassemble_stiffness_matrices(
            self.Youngs_modulus, self.Poisson_ratio
        )
        self.density = self.description['density']
        self.ARAP_weight_list = self.description['weight_list']
        self.edge_list = self.description['edge_list']
        self.neighbour_list = self.description['neighbour_list']
        self.neighbour_edge_list = self.description['neighbour_edge_list']        
        self.neighbour_edge_weight_list = []
        self.tracker_r = 0.0075
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
        self.fixed_region = [[-0.1, 0.02], [-0.1, 1]]
        self.get_fixed_idx(self.vertices, self.fixed_region)
        self.nFixed = len(self.fixed_idx)
        self.W_mat = np.zeros((self.num_vertices * 3, self.num_vertices * 3))
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] == -1:
                self.W_mat[3*i:3*i+3, 3*i:3*i+3] = np.zeros((3,3))
            else:
                for j in range(3):
                    self.W_mat[3*i+j, 3*i+j] = 1 / self.mass_matrix[3*i+j, 3*i+j]
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
        self.N44 = np.eye(4) - 1/4*np.ones((4,4))
        self.N1818 = np.zeros((18, 18))
        for i in range(6):
            for j in range(3):
                for k in range(6):
                    if k==i:
                        self.N1818[3*i+j, 3*k+j] = 5.0/6.0
                    else:
                        self.N1818[3*i+j, 3*k+j] = -1.0/6.0
        self.N99 =  np.zeros((9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if k==i:
                        self.N99[3*i+j, 3*k+j] = 2.0/3.0
                    else:
                        self.N99[3*i+j, 3*k+j] = -1.0/3.0
        self.N1212 = np.zeros((12, 12))
        for i in range(4):
            for j in range(3):
                for k in range(4):
                    if k==i:
                        self.N1212[3*i+j, 3*k+j] = 3.0/4.0
                    else:
                        self.N1212[3*i+j, 3*k+j] = -1.0/4.0
        self.nMoving = self.num_vertices - self.nFixed
        self.moving_dof_idx = np.asarray(
            [
                3 * vertex_idx + component
                for vertex_idx in self.idxMoving_2_idxAll
                for component in range(3)
            ],
            dtype=int,
        )
        self.initial_ARAP_shape_list = []
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] == -1:
                continue
            neighbour_list = self.neighbour_list[i]
            nNeighbour = len(neighbour_list)
            ARAP_shape = np.zeros((nNeighbour, 3))
            for j in range(nNeighbour):
                neighbour_idx = neighbour_list[j]
                ARAP_shape[j] = self.vertices[neighbour_idx] - self.vertices[i]
            self.initial_ARAP_shape_list.append(ARAP_shape)
        self.initial_ghost_shape_list = self.get_ghost_shape(self.vertices, self.get_pp_location_bary(self.vertices))
        print("initial_ARAP_shape_list length: ", len(self.initial_ARAP_shape_list))
        print("number of moving vertices: ", self.nMoving)
        self.initial_cable_vec = self.get_cable_vec_bary(self.vertices)
        self.assemble_CG_matrices()
        self.load_ws()
        self.ikModel = IK_MLP()
        # exit(0)

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

    def reassemble_CG_matrices(self, ratio_weight_cable):
        mem_block   = 9  * self.num_triangles
        bend_block  = 3 * len(self.bending_ele_idx)
        cable_block = 3  * self.nCable
        ghost_block = 12 * self.nCable
        matA_size = mem_block + bend_block + cable_block + ghost_block
        max_weight = np.max((np.max(self.mem_weight_list), np.max(self.bending_weight_list)))
        self.weight_cable = ratio_weight_cable * max_weight
        self.weight_ghost = ratio_weight_cable * max_weight
        self.nNeighbour_list = []
        for i in range(self.num_vertices):
            self.nNeighbour_list.append(len(self.neighbour_list[i]))
        self.matA_initial = np.zeros((matA_size, 3*self.num_vertices + 3*self.nCable))
        print("matA_size: ", matA_size)
        self.vecB_2_add = np.zeros((matA_size, ))

        # Membrane blocks: w * (N33 ⊗ I_3) per triangle
        # Row (9i + 3j + k): centroid-subtracted position of local vertex j, coord k
        #   col 3*v_j  + k : +2/3   (same vertex, same coord)
        #   col 3*v_j' + k : -1/3   (other triangle vertices, same coord k)
        for i in range(self.num_triangles):
            mem_weight = self.mem_weight_list[i]
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    for jp in range(3): # iterate over all triangle vertices (columns)
                        v_jp = self.mesh_triangles[i][jp]
                        coeff = (2.0/3.0) if jp == j else (-1.0/3.0)
                        self.matA_initial[idx_row_start+k, 3*v_jp+k] = mem_weight * coeff

        for i in range(len(self.bending_ele_idx)):
            bending_weight = self.bending_weight_list[i]
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            for j in range(4):
                v_idx = self.bending_ele_idx[i,j]
                c = self.bending_ele_param[i,j]
                for k in range(3):
                    self.matA_initial[mem_block + 3*i + k, 3*v_idx+k] = bending_weight * c

        for i in range(self.nCable):
            for k in range(3):
                self.matA_initial[mem_block + bend_block + 3*i + k, 3*self.num_vertices+3*i+k] = self.weight_cable

        for i in range(self.nCable):
            row_start = mem_block + bend_block + cable_block + 12*i
            tri_idx = self.pp_bary_tri_idx[i]
            tri = self.mesh_triangles[tri_idx]
            idx_all = [tri[0], tri[1], tri[2], self.num_vertices + i]
            for j in range(4):
                idxj = idx_all[j]
                for k in range(4):
                    idxk = idx_all[k]
                    self.matA_initial[row_start + 3*j:row_start+3*j+3, 3*idxk:3*idxk+3] += self.weight_ghost * self.N1212[3*j:3*j+3, 3*k:3*k+3]

        self.matA_all = np.zeros((matA_size, 3*self.nMoving + 3*self.nCable))
        for i in range(self.num_vertices):
            idx_moving = self.idxAll_2_idxMoving[i]
            if idx_moving != -1:
                self.matA_all[:, 3*idx_moving:3*idx_moving+3] = self.matA_initial[:, 3*i:3*i+3]
            else:
                self.vecB_2_add -= self.matA_initial[:, 3*i:3*i+3] @ self.vertices[i]

        self.matA_all[:, 3*self.nMoving:] = self.matA_initial[:, 3*self.num_vertices:]

        print("matA_all shape: ", self.matA_all.shape)
        print("matA_all rank: ", np.linalg.matrix_rank(self.matA_all))
        self.matAT = self.matA_all.T
        self.matATA = self.matA_all.T @ self.matA_all
        self.matATA_inv_AT = np.linalg.inv(self.matATA) @ self.matAT
        self.matATA_inv = np.linalg.inv(self.matATA)
        self.K_CG = self.matATA_inv_AT[:, -15*self.nCable:-12*self.nCable]

    def assemble_CG_matrices(self):
        mem_block   = 9  * self.num_triangles
        bend_block  = 3 * len(self.bending_ele_idx)
        cable_block = 3  * self.nCable
        ghost_block = 12 * self.nCable
        matA_size = mem_block + bend_block + cable_block + ghost_block
        max_weight = np.max((np.max(self.mem_weight_list), np.max(self.bending_weight_list)))
        self.weight_cable = 50 * max_weight
        self.weight_ghost = 50 * max_weight
        self.nNeighbour_list = []
        for i in range(self.num_vertices):
            self.nNeighbour_list.append(len(self.neighbour_list[i]))
        self.matA_initial = np.zeros((matA_size, 3*self.num_vertices + 3*self.nCable))
        print("matA_size: ", matA_size)
        self.vecB_2_add = np.zeros((matA_size, ))

        # Membrane blocks: w * (N33 ⊗ I_3) per triangle
        # Row (9i + 3j + k): centroid-subtracted position of local vertex j, coord k
        #   col 3*v_j  + k : +2/3   (same vertex, same coord)
        #   col 3*v_j' + k : -1/3   (other triangle vertices, same coord k)
        for i in range(self.num_triangles):
            mem_weight = self.mem_weight_list[i]
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    for jp in range(3): # iterate over all triangle vertices (columns)
                        v_jp = self.mesh_triangles[i][jp]
                        coeff = (2.0/3.0) if jp == j else (-1.0/3.0)
                        self.matA_initial[idx_row_start+k, 3*v_jp+k] = mem_weight * coeff

        for i in range(len(self.bending_ele_idx)):
            bending_weight = self.bending_weight_list[i]
            v0, v1, v2, v3 = self.bending_ele_idx[i]
            c1, c2, c3, c4 = self.bending_ele_param[i]
            for j in range(4):
                v_idx = self.bending_ele_idx[i,j]
                c = self.bending_ele_param[i,j]
                for k in range(3):
                    self.matA_initial[mem_block + 3*i + k, 3*v_idx+k] = bending_weight * c

        for i in range(self.nCable):
            for k in range(3):
                self.matA_initial[mem_block + bend_block + 3*i + k, 3*self.num_vertices+3*i+k] = self.weight_cable

        for i in range(self.nCable):
            row_start = mem_block + bend_block + cable_block + 12*i
            tri_idx = self.pp_bary_tri_idx[i]
            tri = self.mesh_triangles[tri_idx]
            idx_all = [tri[0], tri[1], tri[2], self.num_vertices + i]
            for j in range(4):
                idxj = idx_all[j]
                for k in range(4):
                    idxk = idx_all[k]
                    self.matA_initial[row_start + 3*j:row_start+3*j+3, 3*idxk:3*idxk+3] += self.weight_ghost * self.N1212[3*j:3*j+3, 3*k:3*k+3]

        self.matA_all = np.zeros((matA_size, 3*self.nMoving + 3*self.nCable))
        for i in range(self.num_vertices):
            idx_moving = self.idxAll_2_idxMoving[i]
            if idx_moving != -1:
                self.matA_all[:, 3*idx_moving:3*idx_moving+3] = self.matA_initial[:, 3*i:3*i+3]
            else:
                self.vecB_2_add -= self.matA_initial[:, 3*i:3*i+3] @ self.vertices[i]

        self.matA_all[:, 3*self.nMoving:] = self.matA_initial[:, 3*self.num_vertices:]

        print("matA_all shape: ", self.matA_all.shape)
        print("matA_all rank: ", np.linalg.matrix_rank(self.matA_all))
        self.matAT = self.matA_all.T
        self.matATA = self.matA_all.T @ self.matA_all
        self.matATA_inv_AT = np.linalg.inv(self.matATA) @ self.matAT
        self.matATA_inv = np.linalg.inv(self.matATA)
        self.K_CG = self.matATA_inv_AT[:, -15*self.nCable:-12*self.nCable]

    def get_ARAP_shape_list(self, vertices):
        ARAP_shape_list = self.initial_ARAP_shape_list.copy()
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] == -1:
                continue
            neighbour_list = self.neighbour_list[i]
            nNeighbour = len(neighbour_list)
            ARAP_shape = np.zeros((nNeighbour, 3))
            for j in range(nNeighbour):
                neighbour_idx = neighbour_list[j]
                ARAP_shape[j] = vertices[neighbour_idx] - vertices[i]
            ARAP_shape_list[self.idxAll_2_idxMoving[i]] = ARAP_shape
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
        R_list = [np.eye(3) for _ in range(self.nMoving)]
        for i in range(self.nMoving):
            ARAP_shape = ARAP_shape_list[i]
            ARAP_initial_shape = self.initial_ARAP_shape_list[i]
            R_list[i] = self._best_fit_rotation(
                ARAP_shape, ARAP_initial_shape
            )
        return R_list

    def get_rotation_cable(self, vertices):
        cur_cable_vec = self.get_cable_vec_bary(vertices)
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
        # check if ghost_R_list is identity
        # for i in range(self.nCable):
        #     if not np.allclose(ghost_R_list[i], np.eye(3)):
        #         print("ghost_R_list[{}] is not identity: \n{}".format(i, ghost_R_list[i]))
        for i in range(self.num_triangles):
            initial_tri_sk = self.initial_tri_SK_list[i]
            R_tri = R_list_tri[i]
            for j in range(3):          # local vertex (row block)
                idx_row_start = 9*i + 3*j
                for k in range(3):      # coordinate direction
                    bVec[idx_row_start+k] += self.mem_weight_list[i] * (R_tri @ initial_tri_sk.T).T[j, k]

        # print("bvec shape: ", bVec.shape)
        for i in range(self.nCable):
            R_cable = R_list_cable[i]
            initial_cable_vec = self.initial_cable_vec[i]
            vec_rotated = R_cable @ initial_cable_vec
            for k in range(3):
                bVec[matA_shape - 15*self.nCable + 3*i+k] += self.weight_cable * (vec_rotated[k] * tar_cable_length[i] + self.pulley_location[i, k])
        
        for i in range(self.nCable):
            R_ghost = ghost_R_list[i]
            initial_ghost_shape = self.initial_ghost_shape_list[i]
            rotated_ghost_shape = (R_ghost @ initial_ghost_shape.T).T
            for j in range(4):
                idx_row_start = matA_shape - 12*self.nCable + 12*i + 3*j
                for k in range(3):
                    bVec[idx_row_start+k] += self.weight_ghost * rotated_ghost_shape[j, k]
        return bVec
    
    def get_CG_Jacobian(self, vertices, ghost_vertices=None):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        if ghost_vertices is None:
            ghost_vertices = self.get_pp_location_bary(vertices)
        R_cable_list = self.get_rotation_cable_ghost(ghost_vertices)
        Bmat = np.zeros((3*self.nCable, self.nCable))
        for i in range(self.nCable):
            cable_vec_rotated = R_cable_list[i] @ self.initial_cable_vec[i]
            for k in range(3):
                Bmat[3*i+k, i] = self.weight_cable * cable_vec_rotated[k]
        J = self.K_CG @ Bmat
        return J
    
    def get_CG_Jacobian_EE(self, vertices, ghost_vertices=None):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        if ghost_vertices is None:
            ghost_vertices = self.get_pp_location_bary(vertices)
        J = self.get_CG_Jacobian(vertices, ghost_vertices)
        ee_idx = self.ee_idx[0]
        J_ee = J[3*ee_idx:3*ee_idx+3, :]
        return J_ee
    
    def get_FD_Jacobian_EE(self, Q,  delta=1e-3):
        if Q.shape[0] != 3 * self.num_vertices:
            Q = self.vertices_to_q(Q)
        vertices = self.q_to_vertices(Q)
        cur_cl = self.get_cable_length_bary(vertices)
        J = np.zeros((3, self.nCable))
        ee_idx = self.ee_idx[0]
        for i in range(self.nCable):
            cl_plus = cur_cl.copy()
            cl_minus = cur_cl.copy()
            cl_plus[i] += delta
            cl_minus[i] -= delta
            Q_list_plus, tension_plus = self.FKD_static_length(vertices, cl_plus)
            Q_list_minus, tension_minus = self.FKD_static_length(vertices, cl_minus)
            vert_plus = self.q_to_vertices(Q_list_plus[-1])
            vert_minus = self.q_to_vertices(Q_list_minus[-1])
            cl_plus_final = self.get_cable_length_bary(vert_plus)
            cl_minus_final = self.get_cable_length_bary(vert_minus)
            delta_cl = cl_plus_final[i] - cl_minus_final[i] 
            Q_ee_plus = vert_plus[ee_idx]
            Q_ee_minus = vert_minus[ee_idx]
            for j in range(3):
                if abs(delta_cl) < 1e-8:
                    J[j, i] = 0.0
                else:
                    J[j, i] = (Q_ee_plus[j] - Q_ee_minus[j]) / delta_cl
        return J


    def deform_CG(self, tar_cable_length, starting_vertices, max_iter = 300, tol = 1e-8):
        cur_vertices = starting_vertices.copy()
        cur_q = self.vertices_to_q(starting_vertices)
        q_last = cur_q.copy()
        ghost_vertices = self.get_pp_location_bary(cur_vertices)
        for i in range(max_iter):
            R_list_cable = self.get_rotation_cable_ghost(ghost_vertices)
            R_list_tri = self.get_rotation_tri(cur_vertices)
            bVec = self.get_Bvec_CG(cur_vertices, ghost_vertices, R_list_tri, R_list_cable, tar_cable_length)
            cur_q_all = self.matATA_inv_AT @ (bVec + self.vecB_2_add)
            cur_q_moving = cur_q_all[:3*self.nMoving]
            ghost_vertices = cur_q_all[3*self.nMoving:].reshape((self.nCable, 3))
            cur_q = self.q_moving_to_q(cur_q_moving)
            cur_vertices = self.q_to_vertices(cur_q)
            diff = np.linalg.norm(cur_q - q_last)/(3*self.num_vertices)
            q_last = cur_q.copy()
            print("cg iteration {}: diff = {}".format(i, diff))
            if diff < tol:
                break
        return cur_vertices

    def FKD_free(self, show_info = False):
        starting_vertices = self.vertices.copy()
        Q_a = self.vertices_to_q(starting_vertices)
        Q_list = [Q_a.copy()]
        total_time = 5
        Q_a = Q_a.reshape((3*self.num_vertices, ))
        Q_ad = np.zeros((3*self.num_vertices, ))
        t_a = 0.0
        h = 0.01
        tol = 1e-7
        diff_count = 0
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            R_list, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            K_mat, f0 = self.assemble_K(R_list_1818)
            disp = Q_a - self.vertices_to_q(self.vertices)
            denom = disp @ self.mass_matrix @ disp
            stiffness_energy = max(float(disp @ K_mat @ disp), 0.0)
            damping_coeff = np.sqrt(stiffness_energy / denom) if denom > 1e-30 else 0.0
            C_mat = 2 * damping_coeff * self.mass_matrix
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) + h * self.W_mat @ K_mat + self.W_mat @ C_mat
            lu, piv = lu_factor(A_mat)
            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec - C_mat @ Q_ad)
            dv = lu_solve((lu, piv), b_vec)
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_list.append(Q_a.copy())
            diff = np.linalg.norm(Q_a - Q_a_last)/(3*self.num_vertices)
            if show_info:
                print(f"t_a: {t_a:.3f}, diff: {diff:.7f}")
            if diff < tol :
                diff_count += 1
                if diff_count >= 10:
                    print(f"Converged at time {t_a:.2f} with diff {diff:.6f} for 10 consecutive steps, stopping simulation.")
                    break
            else:
                diff_count = 0
        return Q_list

    def FKD_time(
        self,
        target_cable_length,
        total_time,
        starting_vertices,
        tol=2e-5,
        show_info=False,
        h=0.002,
    ):
        target_cable_length = np.asarray(target_cable_length, dtype=float).reshape(-1)
        if target_cable_length.size != self.nCable:
            raise ValueError(f"target_cable_length must contain {self.nCable} values.")
        if np.any(~np.isfinite(target_cable_length)) or np.any(
            target_cable_length <= 0.0
        ):
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
        Q_list = [Q_a.copy()]
        diff_count = 0
        cable_tension = np.zeros(self.nCable)
        t_start = time.time()
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

            A_mat = (
                (1.0/dt) * np.eye(3*self.num_vertices)
                + dt * self.W_mat @ K_mat
                + self.W_mat @ C_mat
            )
            lu, piv = lu_factor(A_mat)

            b_vec = self.W_mat @ (
                -K_mat @ (Q_a + dt * Q_ad)
                + f0
                + self.gravity_vec
                - C_mat @ Q_ad
            )
            # dv_free = A_inv @ b_vec
            dv_free = lu_solve((lu, piv), b_vec)

            Q_free = Q_a + dt * Q_ad + dt * dv_free
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
            diff = np.linalg.norm(Q_a - Q_a_last) / np.sqrt(3*self.num_vertices)
            current_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            constraint_error = float(
                np.max(np.maximum(current_lengths - target_cable_length, 0.0))
            )
            # if diff < 1e-5:
            #     h *= 0.1
            if diff < tol and constraint_error < tol:
                diff_count += 1
                if diff_count >= 10:
                    print(f"Converged at time {t_a:.2f} with diff {diff:.6f} for 10 consecutive steps, stopping simulation.")
                    break
            else:
                diff_count = 0
            t3 = time.time()
            if show_info:
                print(
                    f"t_a: {t_a:.3f}, diff: {diff:.7f}, "
                    f"constraint error: {constraint_error:.7f}, "
                    f"time for R_list: {t1-t0:.4f}s, "
                    f"time for K_mat: {t2-t1:.4f}s, "
                    f"total time for this step: {t3-t0:.4f}s"
                )
        t_end = time.time()
        print(f"Total simulation time: {t_end - t_start:.2f}s")
        vert_length = self.q_to_vertices(Q_a)
        return Q_list, vert_length, cable_tension
            
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

    def find_closest_in_ws(self, ee_target):
        """Find the closest point in the workspace to the given target."""
        ee_target = np.asarray(ee_target, dtype=float).reshape(-1)

        # search the closest in ee_pos_list
        ee_pos_array = np.asarray(self.ee_pos_list, dtype=float)
        distances = np.linalg.norm(ee_pos_array - ee_target, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = ee_pos_array[closest_idx]
        closest_cl = self.cl_list[closest_idx]
        closest_vert = self.vertices_list[closest_idx]
        diff = np.linalg.norm(closest_point - ee_target)
        print(f"Closest point in workspace: {closest_point}, cable length: {closest_cl}, distance to target: {diff:.6f}")

        return closest_point, closest_cl, closest_vert

    def FKD_static(self, starting_vertices, cable_tension, tol = 1e-6, show_info = False):
        """Solve corotational equilibrium for prescribed cable tensions."""
        cable_tension = np.asarray(cable_tension, dtype=float).reshape(-1)
        if cable_tension.size != self.nCable:
            raise ValueError(f"cable_tension must contain {self.nCable} values.")
        if np.any(~np.isfinite(cable_tension)) or np.any(cable_tension < 0.0):
            raise ValueError("cable_tension must contain finite non-negative values.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")

        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 array."
            )
        Q_list = [Q_a.copy()]
        gravity_tilde = self.q_to_q_moving(self.gravity_vec)
        max_iter = 500
        for iteration in range(max_iter):
            _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            K_tilde, f0_tilde, K_tilde_vec2add = self.assemble_K_tilde(R_list_1818)
            H_mat_ori = -self.get_cable_Jacobian_bary(Q_a)
            H_mat = H_mat_ori[:, self.moving_dof_idx]
            # Moving part of K q = f0 + f_gravity, after moving the
            # fixed-vertex contribution K_mf q_f to the right-hand side.
            rhs = f0_tilde + gravity_tilde + K_tilde_vec2add + H_mat.T @ cable_tension
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

    def FKD_static_length(self, starting_vertices, target_cable_length, tol = 1e-6, show_info = False):
        target_cable_length = np.asarray(target_cable_length, dtype=float).reshape(-1)
        if target_cable_length.size != self.nCable:
            raise ValueError(f"target_cable_length must contain {self.nCable} values.")
        if np.any(~np.isfinite(target_cable_length)) or np.any(
            target_cable_length <= 0.0
        ):
            raise ValueError("target_cable_length must contain finite positive values.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")

        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 array."
            )
        Q_list = [Q_a.copy()]
        gravity_tilde = self.q_to_q_moving(self.gravity_vec)
        max_iter = 500
        cable_tension = np.zeros(self.nCable)
        for iteration in range(max_iter):
            _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            K_tilde, f0_tilde, K_tilde_vec2add = self.assemble_K_tilde(R_list_1818)
            lu, piv = lu_factor(K_tilde)
            Q_moving_free = lu_solve((lu, piv), f0_tilde + gravity_tilde + K_tilde_vec2add)
            Q_moving_cor = np.zeros_like(Q_moving_free)
            cable_force_moving = np.zeros_like(Q_moving_free)
            cable_tension = np.zeros(self.nCable)
            constraint_solve_tol = min(tol, 1e-8)
            for _ in range(5):
                Q_constrained = self.q_moving_to_q(
                    Q_moving_free + Q_moving_cor
                )
                current_lengths = np.asarray(
                    self.get_cable_length_bary(Q_constrained)
                )
                phi = target_cable_length - current_lengths
                if np.max(np.maximum(-phi, 0.0)) <= constraint_solve_tol:
                    break

                H_all = -self.get_cable_Jacobian_bary(Q_constrained)
                H = H_all[:, self.moving_dof_idx]
                Z = lu_solve((lu, piv), H.T)
                lcp_Mmat = H @ Z
                tension_increment = projected_gauss_seidel_lcp(
                    lcp_Mmat, phi
                )
                if np.linalg.norm(tension_increment, np.inf) <= 1e-14:
                    break
                Q_moving_cor += Z @ tension_increment
                cable_force_moving += H.T @ tension_increment
                cable_tension += tension_increment

            Q_moving = Q_moving_free + Q_moving_cor
            Q_next = self.q_moving_to_q(Q_moving)
            diff = np.linalg.norm(Q_next - Q_a) / (3*self.nMoving)
            Q_a = Q_next
            Q_list.append(Q_a.copy())
            current_lengths = np.asarray(self.get_cable_length_bary(Q_a))
            constraint_error = float(
                np.max(np.maximum(current_lengths - target_cable_length, 0.0))
            )
            rhs = (
                f0_tilde
                + gravity_tilde
                + K_tilde_vec2add
                + cable_force_moving
            )

            if show_info:
                residual = np.linalg.norm(K_tilde @ Q_moving - rhs)
                print(
                    f"static iteration {iteration + 1}: diff = {diff:.7e}, "
                    f"constraint error = {constraint_error:.7e}, "
                    f"linear residual = {residual:.7e}"
                )
            if diff < tol and constraint_error < tol:
                if show_info:
                    print(f"Static solve converged in {iteration + 1} iterations.")
                break
        else:
            if show_info:
                print(
                    f"Static solve reached {max_iter} iterations without "
                    f"meeting the tolerance (last diff = {diff:.7e})."
                )

        return Q_list, cable_tension


    def FKD_get_residual(self, Q, cable_tension):
        _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q))
        K_tilde, f0_tilde, K_tilde_vec2add = self.assemble_K_tilde(R_list_1818)
        Q_moving = self.q_to_q_moving(Q)
        rhs = f0_tilde + self.q_to_q_moving(self.gravity_vec) + K_tilde_vec2add + (-self.get_cable_Jacobian_bary(Q)[:, self.moving_dof_idx]).T @ cable_tension
        residual = K_tilde @ Q_moving - rhs
        return residual


    def FKD_trajectory(self, cl_list, time_list):
        pass

    def IKD_force(
        self,
        target_ee_pos,
        starting_vertices,
        starting_force,
        tol=1e-3,
        show_info=False,
    ):
        """Find non-negative cable forces that move the end effector to a target.

        ``FKD_static`` is used for every forward evaluation.  A local
        force-to-end-effector Jacobian is obtained by differentiating the
        reduced static equilibrium while holding the current corotational
        linearisation fixed,

            K_tilde dq/df = H.T.

        The objective is ``0.5 * ||x_ee - x_target||**2``. Its force-space
        gradient is ``J_ee.T @ (x_ee - x_target)``. A fixed-gain projected
        gradient step is used so cable tensions remain non-negative.

        Parameters
        ----------
        target_ee_pos : array-like, shape (3,)
            Desired end-effector position.
        starting_vertices : array-like, shape (num_vertices, 3) or (3*num_vertices,)
            Initial configuration supplied to the first static solve.
        starting_force : array-like, shape (nCable,)
            Initial cable tensions. Values must be finite and non-negative.
        tol : float
            Cartesian end-effector error tolerance, in metres.
        show_info : bool
            Print the error and force step at every iteration.

        Returns
        -------
        final_vertices : ndarray, shape (num_vertices, 3)
            Final equilibrium configuration.
        cable_length : ndarray, shape (nCable,)
            Cable lengths measured in the final configuration.
        cable_force : ndarray, shape (nCable,)
            Non-negative cable tensions used to produce the configuration.
        """
        target_ee_pos = np.asarray(target_ee_pos, dtype=float).reshape(-1)
        if target_ee_pos.size != 3 or np.any(~np.isfinite(target_ee_pos)):
            raise ValueError("target_ee_pos must contain three finite values.")

        starting_vertices = np.asarray(starting_vertices, dtype=float)
        if starting_vertices.shape == (self.num_vertices, 3):
            Q_start = self.vertices_to_q(starting_vertices)
        elif starting_vertices.size == 3 * self.num_vertices:
            Q_start = starting_vertices.reshape(3 * self.num_vertices).copy()
        else:
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 array."
            )
        if np.any(~np.isfinite(Q_start)):
            raise ValueError("starting_vertices must contain only finite values.")

        cable_force = np.asarray(starting_force, dtype=float).reshape(-1)
        if cable_force.size != self.nCable:
            raise ValueError(f"starting_force must contain {self.nCable} values.")
        if np.any(~np.isfinite(cable_force)) or np.any(cable_force < 0.0):
            raise ValueError("starting_force must contain finite non-negative values.")
        cable_force = cable_force.copy()

        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive value.")

        idx_ee_moving = self.idxAll_2_idxMoving[self.ee_idx[0]]
        if idx_ee_moving < 0:
            raise ValueError("The end-effector vertex is fixed and cannot be controlled.")

        max_iter = 50
        max_force_step = 0.25
        gradient_gain = 100.0
        forward_tol = min(1e-7, tol * 0.1)

        Q_list = self.FKD_static(
            Q_start, cable_force, tol=forward_tol, show_info=False
        )
        Q = Q_list[-1]
        error = self.get_ee_pos(Q) - target_ee_pos
        error_norm = float(np.linalg.norm(error))

        if show_info:
            print(
                f"force IK iteration 0: error = {error_norm:.7e}, "
                f"force = {np.round(cable_force, 6)}"
            )

        for iteration in range(max_iter):
            if error_norm < tol:
                if show_info:
                    print(
                        f"Force IK converged in {iteration} iterations "
                        f"with error {error_norm:.7e}."
                    )
                break

            # Tangent of the same equilibrium equation used in FKD_static:
            # K_tilde q_moving = ... + H.T @ cable_force.
            _, R_list_1818 = self.get_R_list(self.q_to_vertices(Q))
            K_tilde, _, _ = self.assemble_K_tilde(R_list_1818)
            H_all = -self.get_cable_Jacobian_bary(Q)
            H = H_all[:, self.moving_dof_idx]
            try:
                lu, piv = lu_factor(K_tilde)
                dq_dforce = lu_solve((lu, piv), H.T)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise np.linalg.LinAlgError(
                    "Could not form the force-space Jacobian because the "
                    "reduced stiffness matrix is singular."
                ) from exc

            ee_rows = slice(3 * idx_ee_moving, 3 * idx_ee_moving + 3)
            J_ee = dq_dforce[ee_rows, :]

            objective_gradient = J_ee.T @ error
            force_step = -gradient_gain * objective_gradient

            # Keep a single gradient update from becoming excessively large.
            step_scale = float(np.max(np.abs(force_step)))
            if step_scale > max_force_step:
                force_step *= max_force_step / step_scale

            next_force = np.maximum(cable_force + force_step, 0.0)
            projected_step = next_force - cable_force
            if np.linalg.norm(projected_step, np.inf) < 1e-12:
                if show_info:
                    print(
                        f"Force IK stopped at iteration {iteration + 1}: "
                        "the projected gradient step is zero."
                    )
                break

            # One forward computation per gradient iteration; there is no
            # trial step, damping solve, or line search.
            cable_force = next_force
            Q_list = self.FKD_static(
                Q, cable_force, tol=forward_tol, show_info=False
            )
            Q = Q_list[-1]
            error = self.get_ee_pos(Q) - target_ee_pos
            error_norm = float(np.linalg.norm(error))

            if show_info:
                print(
                    f"force IK iteration {iteration + 1}: "
                    f"error = {error_norm:.7e}, "
                    f"|df|_inf = {np.linalg.norm(projected_step, np.inf):.7e}, "
                    f"force = {np.round(cable_force, 6)}"
                )
        else:
            if show_info:
                print(
                    f"Force IK reached {max_iter} iterations with error "
                    f"{error_norm:.7e}."
                )

        final_vertices = self.q_to_vertices(Q)
        cable_length = self.get_cable_length_bary(final_vertices)
        return final_vertices, cable_length, cable_force

    def IKD_single(self, target_ee_pos, starting_vertices, AA = False, tol = 1e-3, show_info = False):
        idx_ee = self.ee_idx[0]
        idx_ee_moving = self.idxAll_2_idxMoving[idx_ee]
        max_iter = 50
        AA_memory = 5
        aa_cl_list = np.zeros((AA_memory, self.nCable))
        aa_diff_list = np.zeros((AA_memory, ))
        aa_ee_pos_list = np.zeros((AA_memory, 3))
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q = self.vertices_to_q(starting_vertices)
        def get_diff(Q):
            ee_pos = self.get_ee_pos(Q)
            diff = 1/2 * np.linalg.norm(ee_pos - target_ee_pos) ** 2
            return diff
        def get_jacobian(Q_a):
            J_Moving = self.get_CG_Jacobian(self.q_to_vertices(Q_a))
            Jac = np.zeros((self.nCable, ))
            for i in range(self.nCable):
                ee_pos = self.get_ee_pos(Q_a)
                for j in range(3):
                    Jac[i] += J_Moving[3*idx_ee_moving+j, i] * (ee_pos[j] - target_ee_pos[j])
            return Jac
        
        def get_jacobian_fd(Q_a, eps = 1e-4):
            Jac = np.zeros((self.nCable, ))
            cur_cl = self.get_cable_length_bary(Q_a)
            for i in range(self.nCable):
                cl_plus = cur_cl.copy()
                cl_minus = cur_cl.copy()
                cl_plus[i] += eps
                cl_minus[i] -= eps
                Q_plus_list, cable_tension_plus = self.FKD_static_length(Q_a, cl_plus)
                Q_minus_list, cable_tension_minus= self.FKD_static_length(Q_a, cl_minus)
                vert_plus = self.q_to_vertices(Q_plus_list[-1])
                vert_minus = self.q_to_vertices(Q_minus_list[-1])
                cl_plus = self.get_cable_length_bary(vert_plus)
                cl_minus = self.get_cable_length_bary(vert_minus)
                cl_diff = cl_plus[i] - cl_minus[i]
                ee_pos_plus = self.get_ee_pos(vert_plus)
                ee_pos_minus = self.get_ee_pos(vert_minus)
                diff_plus = 1/2 * np.linalg.norm(ee_pos_plus - target_ee_pos) ** 2
                diff_minus = 1/2 * np.linalg.norm(ee_pos_minus - target_ee_pos) ** 2
                if abs(cl_diff) < 1e-8:
                    Jac[i] = 0.0
                else:
                    Jac[i] = (diff_plus - diff_minus) / (2*eps)
            return Jac
        
        def get_jacobian_fd_cg(Q_a, eps = 1e-4):
            Jac = np.zeros((self.nCable, ))
            cur_cl = self.get_cable_length_bary(Q_a)
            for i in range(self.nCable):
                cl_plus = cur_cl.copy()
                cl_minus = cur_cl.copy()
                cl_plus[i] += eps
                cl_minus[i] -= eps
                vert_plus = self.deform_CG(cl_plus, self.q_to_vertices(Q_a))
                vert_minus = self.deform_CG(cl_minus, self.q_to_vertices(Q_a))
                cl_plus = self.get_cable_length_bary(self.vertices_to_q(vert_plus))
                cl_minus = self.get_cable_length_bary(self.vertices_to_q(vert_minus))
                cl_diff = cl_plus[i] - cl_minus[i]
                ee_pos_plus = self.get_ee_pos(self.vertices_to_q(vert_plus))
                ee_pos_minus = self.get_ee_pos(self.vertices_to_q(vert_minus))
                diff_plus = 1/2 * np.linalg.norm(ee_pos_plus - target_ee_pos) ** 2
                diff_minus = 1/2 * np.linalg.norm(ee_pos_minus - target_ee_pos) ** 2
                Jac[i] = (diff_plus - diff_minus) / cl_diff
            return Jac
        
        def flatten_jac(jac, flatten_ratio = 0.5):
            
            max_abs_jac = np.max(np.abs(jac))
            threshold = 0.5 * max_abs_jac
            jac_toreturn = jac.copy()
            # if max_abs_jac > 0:
            for i in range(self.nCable):
                if np.abs(jac[i]) > threshold:
                    jac_toreturn[i] *= flatten_ratio
                
            return jac_toreturn

        tol_fd = 3e-3
        cur_length = self.get_cable_length_bary(Q)
        Q_list, cable_tension = self.FKD_static_length(starting_vertices,cur_length)
        starting_vertices = self.q_to_vertices(Q_list[-1])
        Q = self.vertices_to_q(starting_vertices)
        Q_list_final = [Q.copy()]
        for i in range(max_iter):
            dl = 0.5
            
            diff = get_diff(Q)
            diff_cart = 2*np.sqrt(diff)
            if np.sqrt(2*diff) > tol_fd:
                jac = get_jacobian(Q)
            else:
                jac = get_jacobian_fd(Q)
            # jac = get_jacobian_fd(Q)
            # jac = get_jacobian(Q)
            # jac = flatten_jac(jac)
            jac[1] *= 0.1
            diff_foreach = diff/self.nCable
            cur_length = self.get_cable_length_bary(Q)
            cmd_diff = [0 for _ in range(self.nCable)]
            cmd_length = cur_length.copy()
            # alpha = 10
            # dl = alpha*diff/(np.max(np.abs(jac))+1e-6)
            for k in range(self.nCable):
                if cable_tension[k] < 1e-5 and jac[k] < 0:
                    cmd_diff[k] = 0
                else:
                    # cmd_diff[k] = diff_foreach / jac[k]
                    cmd_diff[k] = -dl * jac[k]
            cmd_diff = clamp_diff(cmd_diff, min_bound = 1e-4, max_bound = 5e-3)
            for k in range(self.nCable):
                cmd_length[k] += cmd_diff[k]
            Q_list, cable_tension = self.FKD_static_length(Q, cmd_length)
            starting_vertices = self.q_to_vertices(Q_list[-1])
            Q = self.vertices_to_q(starting_vertices)
            diff = get_diff(Q)
            if AA:
                aa_cl_list[i%AA_memory] = cmd_length.copy()
                aa_diff_list[i%AA_memory] = diff
                aa_ee_pos_list[i%AA_memory] = self.get_ee_pos(Q)
                if i > 0 and i%AA_memory == 0:
                    print("Performing Anderson Acceleration at iteration {}".format(i))
                    # cl_cmd_next = anderson_step(aa_cl_list, aa_diff_list, beta=1, lam=1e-8, m=AA_memory-1)
                    # cl_cmd_next = anderson_step_vertex(aa_cl_list, aa_ee_pos_list, target_ee_pos, beta=1, lam=1e-8, m=AA_memory-1)
                    # print("aa_cl_list: ", aa_cl_list)
                    # print("aa_diff_list: ", aa_diff_list)
                    # cl_cmd_next = anderson_my_parabola(aa_cl_list, aa_diff_list)
                    cl_cmd_next = anderson_step_vertex(aa_cl_list, aa_ee_pos_list, target_ee_pos, beta=1, lam=1e-8, m=AA_memory-1)
                    # cl_cmd_next = my_aa(aa_cl_list, aa_diff_list)
                    aa_cl_list = np.zeros((AA_memory, self.nCable))
                    aa_diff_list = np.zeros((AA_memory, ))
                    aa_ee_pos_list = np.zeros((AA_memory, 3))
                    Q_list, starting_vertices, cable_tension = self.FKD_time(cl_cmd_next, 1, Q, tol = 1e-5)
                    Q = self.vertices_to_q(starting_vertices)
            diff = np.sqrt(2*diff)
            if show_info:
                # print("Iteration {}: diff = {}".format(i, diff))
                print("Iteration {}: diff = {}, cmd_diff: {}".format(i, diff, np.round(cmd_diff, 5)*1e3))
            # self.visualize_IKD_result(self.q_to_vertices(Q), target_ee_pos)
            Q_list_final.append(Q.copy())
            cur_length = self.get_cable_length_bary(Q)
            if diff < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break
        return cur_length, starting_vertices, Q_list_final

    def IKD_minimize(self, target_ee_pos, starting_vertices, tol = 1e-3, show_info = False):
        """Solve inverse kinematics by optimizing the cable lengths directly.

        The objective is the squared Cartesian distance between the end
        effector and ``target_ee_pos``.  Each objective evaluation runs the
        static forward-kinematics solver from ``starting_vertices``.

        Returns
        -------
        cable_length : ndarray, shape (nCable,)
            Realized cable lengths in the optimized equilibrium.
        final_vertices : ndarray, shape (num_vertices, 3)
            Optimized equilibrium configuration.
        """
        target_ee_pos = np.asarray(target_ee_pos, dtype=float).reshape(-1)
        if target_ee_pos.size != 3 or np.any(~np.isfinite(target_ee_pos)):
            raise ValueError("target_ee_pos must contain three finite values.")

        starting_vertices = np.asarray(starting_vertices, dtype=float)
        if starting_vertices.shape[0] == 3*self.num_vertices:
            starting_vertices = self.q_to_vertices(
                starting_vertices.reshape(3*self.num_vertices)
            )
        elif starting_vertices.shape != (self.num_vertices, 3):
            raise ValueError(
                "starting_vertices should be either a 3n vector or an n by 3 array."
            )
        starting_vertices = starting_vertices.copy()

        initial_length = np.asarray(
            self.get_cable_length_bary(starting_vertices), dtype=float
        )
        evaluation_count = 0
        cached_length = None
        cached_vertices = None
        cached_ee_pos = None

        def forward_kinematics(cable_length):
            nonlocal evaluation_count, cached_length, cached_vertices, cached_ee_pos
            cable_length = np.asarray(cable_length, dtype=float)
            if cached_length is not None and np.array_equal(
                cable_length, cached_length
            ):
                return cached_vertices, cached_ee_pos

            Q_list, _ = self.FKD_static_length(
                starting_vertices, cable_length
            )
            vertices = self.q_to_vertices(Q_list[-1])
            ee_pos = np.asarray(self.get_ee_pos(vertices), dtype=float)
            evaluation_count += 1
            cached_length = cable_length.copy()
            cached_vertices = vertices
            cached_ee_pos = ee_pos
            return vertices, ee_pos

        def objective_function(cable_length):
            _, ee_pos = forward_kinematics(cable_length)
            residual = ee_pos - target_ee_pos
            return 0.5 * residual @ residual

        result = minimize(
            objective_function,
            initial_length,
            method="SLSQP",
            bounds=[(np.finfo(float).eps, None)] * self.nCable,
            jac="3-point",
            tol=tol,
            options={
                "ftol": max(0.5 * tol**2, np.finfo(float).eps),
                "finite_diff_rel_step": 1e-3,
                "maxiter": 50,
                "disp": show_info,
            },
        )

        final_vertices, final_ee_pos = forward_kinematics(result.x)
        final_length = np.asarray(
            self.get_cable_length_bary(final_vertices), dtype=float
        )
        final_error = np.linalg.norm(final_ee_pos - target_ee_pos)

        if show_info:
            print(
                f"IK minimize evaluations: {evaluation_count}, "
                f"success: {result.success}, final error: {final_error:.7e}"
            )
            if not result.success:
                print(f"Optimizer message: {result.message}")

        return final_length, final_vertices

    def generate_ws(self, cable_length_ranges, total_number = 1000, saveFile = 'training_data_1.pkl'):
        def generate_ws_cl_input(cable_length_ranges, total_number):
            # cable length is in [[c1_min, c1_max], ..., [c6_min, c6_max]]
            ranges = np.array(cable_length_ranges)  # (nCable, 2)
            samples = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(total_number, len(ranges)))
            return [list(samples[i]) for i in range(total_number)]
        cl_to_test = generate_ws_cl_input(cable_length_ranges, total_number)
        data_list = []
        for i in range(len(cl_to_test)):
            cl = cl_to_test[i]
            Q_list, cable_tension = self.FKD_static_length(self.vertices, cl)
            vert_length = self.q_to_vertices(Q_list[-1])
            fcl = self.get_cable_length_bary(vert_length)
            ee_pos = self.get_ee_pos(vert_length)
            print("Test {}: cable length {}, ee pos {}".format(i, np.round(fcl, 3), np.round(ee_pos, 3)))
            # save the data for training
            data = {
                'cable_length': fcl,
                'ee_pos': ee_pos,
                'vertices': vert_length,
                'cable_tension': cable_tension
            }
            data_list.append(data)
        with open(saveFile, 'wb') as f:
            pickle.dump(data_list, f)

    def IK_CG(self, target_ee_pos, starting_vertices, tol = 1e-5):
        idx_ee = self.ee_idx[0]
        idx_ee_moving = self.idxAll_2_idxMoving[idx_ee]
        max_iter = 100
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q = self.vertices_to_q(starting_vertices)
        def get_diff(Q):
            ee_pos = self.get_ee_pos(Q)
            diff = 1/2 * np.linalg.norm(ee_pos - target_ee_pos) ** 2
            return diff
        def get_jacobian(Q_a):
            J_Moving = self.get_CG_Jacobian(self.q_to_vertices(Q_a))
            Jac = np.zeros((self.nCable, ))
            for i in range(self.nCable):
                ee_pos = self.get_ee_pos(Q_a)
                for j in range(3):
                    Jac[i] += J_Moving[3*idx_ee_moving+j, i] * (ee_pos[j] - target_ee_pos[j])
            return Jac
        for i in range(max_iter):
            dl = 1
            jac = get_jacobian(Q)
            diff = get_diff(Q)
            cur_length = self.get_cable_length_bary(Q)
            cmd_diff = [0 for _ in range(self.nCable)]
            cmd_length = cur_length.copy()
            for k in range(self.nCable):
                cmd_diff[k] = -dl * jac[k]
                cmd_length[k] += cmd_diff[k]

            starting_vertices = self.deform_CG(cmd_length, self.q_to_vertices(Q))
            diff = get_diff(self.vertices_to_q(starting_vertices))
            diff = np.sqrt(2*diff)

            print("Iteration {}: diff = {}, dl = {}".format(i, diff, dl))
            if diff < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break
            Q = self.vertices_to_q(starting_vertices)
            cur_length = self.get_cable_length(Q)
        return cmd_length, starting_vertices

    def get_config_fixEE(self, target_ee_pos, total_time, starting_vertices):
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError("starting_vertices should be either a 3n vector or an n by 3 array.")
        
        Q_a = Q_a.reshape((3*self.num_vertices, ))
        for i in range(3):
            Q_a[3*self.ee_idx[0]+i] = target_ee_pos[i]
        Q_ad = np.zeros((3*self.num_vertices, ))
        t_a = 0.0
        h = 0.002
        tol = 2e-5
        idx_ee = self.ee_idx[0]
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_list = [Q_a.copy()]
        starting_cable_length = self.get_cable_length(starting_vertices)
        diff_count = 0
        t_start = time.time()
        W_mat_this = self.W_mat.copy()
        for i in range(3):
            W_mat_this[3*idx_ee+i, 3*idx_ee+i] = 0
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            t0 = time.time()
            R_list, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            t1 = time.time()
            K_mat, f0 = self.assemble_K(R_list_1818)
            t2 = time.time()
            A_mat = (1.0/h)*np.eye(3*self.num_vertices) + h * W_mat_this @ K_mat
            lu, piv = lu_factor(A_mat)

            b_vec = W_mat_this @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec)
            dv_free = lu_solve((lu, piv), b_vec)
            Q_a = Q_a + h * Q_ad + h * dv_free
            t_a += h
            Q_list.append(Q_a.copy())
            diff = np.linalg.norm(Q_a - Q_a_last)/(3*self.num_vertices)
            # if diff < 1e-5:
            #     h *= 0.1
            if diff < tol:
                diff_count += 1
                if diff_count >= 10:
                    break
        vert_fixEE = self.q_to_vertices(Q_a)
        return vert_fixEE

    def get_fixedEE_guess_vertices(self, target_EE_pos):
        """Return a static FEM shape with the end effector fixed at ``target_EE_pos``.

        The vertices in ``self.fixed_idx`` remain at their reference positions and
        the first end-effector vertex is prescribed at the requested position.  All
        remaining degrees of freedom are found from a corotational FEM equilibrium
        under gravity.  This is intended as a mechanically plausible initial guess
        for inverse kinematics; cable forces are deliberately not included.

        Parameters
        ----------
        target_EE_pos : array_like, shape (3,)
            Desired world position of the end-effector vertex.

        Returns
        -------
        numpy.ndarray, shape (num_vertices, 3)
            The complete deformed mesh.  The returned EE position is exactly the
            supplied target (up to its conversion to floating point).
        """
        target = np.asarray(target_EE_pos, dtype=float)
        if target.size != 3:
            raise ValueError("target_EE_pos must contain exactly three coordinates.")
        target = target.reshape(3)
        if not np.all(np.isfinite(target)):
            raise ValueError("target_EE_pos must contain only finite values.")

        ee_idx = int(np.asarray(self.ee_idx).reshape(-1)[0])
        if ee_idx in self.fixed_idx:
            raise ValueError(
                "The end-effector vertex is also in fixed_idx, so it cannot be "
                "prescribed independently."
            )

        constrained_vertices = np.asarray(
            list(dict.fromkeys([int(i) for i in self.fixed_idx] + [ee_idx])),
            dtype=int,
        )
        constrained_dofs = (
            3 * constrained_vertices[:, None] + np.arange(3)[None, :]
        ).reshape(-1)
        all_dofs = np.arange(3 * self.num_vertices)
        free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=True)

        q = self.vertices_to_q(self.vertices.copy()).astype(float, copy=False)
        q[3 * ee_idx:3 * ee_idx + 3] = target
        prescribed_q = q[constrained_dofs].copy()

        # This routine supplies an initial guess, so use a modest iteration cap.
        # Relaxation suppresses the oscillation that a raw corotational fixed-point
        # iteration can exhibit for targets far from the reference configuration.
        max_iter = 20
        tol = 1e-6
        relaxation = 0.7
        for _ in range(max_iter):
            _, rotations = self.get_R_list(self.q_to_vertices(q))
            stiffness, rest_force = self.assemble_K(rotations)
            rhs = (
                rest_force[free_dofs]
                + self.gravity_vec[free_dofs]
                - stiffness[np.ix_(free_dofs, constrained_dofs)] @ prescribed_q
            )
            stiffness_free = sp.csc_matrix(
                stiffness[np.ix_(free_dofs, free_dofs)]
            )
            try:
                q_free = spla.spsolve(stiffness_free, rhs)
            except (RuntimeError, ValueError) as exc:
                raise np.linalg.LinAlgError(
                    "The fixed-EE FEM system is singular; check the fixed region "
                    "and mesh connectivity."
                ) from exc
            if not np.all(np.isfinite(q_free)):
                raise np.linalg.LinAlgError(
                    "The fixed-EE FEM system produced a non-finite solution; "
                    "check the fixed region and mesh connectivity."
                )

            q_next = q.copy()
            q_next[free_dofs] = (
                (1.0 - relaxation) * q[free_dofs] + relaxation * q_free
            )
            q_next[constrained_dofs] = prescribed_q
            diff = np.linalg.norm(q_next - q) / max(1, free_dofs.size)
            q = q_next
            if diff < tol:
                break

        return self.q_to_vertices(q)

    def get_fixedEE_guess_vertives(self, target_EE_pos):
        """Backward-compatible alias for the originally requested misspelling."""
        return self.get_fixedEE_guess_vertices(target_EE_pos)

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
            for local_node, vertex_idx in enumerate(tri):
                if vertex_idx != -1:
                    patch_list[i][local_node] = vertices[vertex_idx]
        return patch_list
    
    def get_ee_pos(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        ee_id = self.ee_idx[0]  
        ee_pos = vertices[ee_id]
        return ee_pos

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
    
    def q_to_q_moving(self, q):
        q_moving = np.zeros((self.nMoving * 3, ))
        for i in range(self.num_vertices):
            if self.idxAll_2_idxMoving[i] != -1:
                idx_moving = self.idxAll_2_idxMoving[i]
                q_moving[3*idx_moving:3*idx_moving+3] = q[3*i:3*i+3]
        return q_moving
    
    def q_moving_to_q(self, q_moving):
        q = self.vertices_to_q(self.vertices)
        for i in range(self.nMoving):
            idx_all = self.idxMoving_2_idxAll[i]
            q[3*idx_all:3*idx_all+3] = q_moving[3*i:3*i+3]
        return q

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

    def get_fb_surface(self, vertices):
        """Return vertices shifted by half-thickness toward the mid-surface along vertex normals."""
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
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)

        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        plotter.add_points(pp_locations, color='blue', point_size=10
                            , label='Pullpoints')
        plotter.add_points(self.pulley_location, color='blue', point_size=10
                            , label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')

        # make fixed idx black
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        # add grid
        plotter.show_grid()
        # plotter.show_axes()
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
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        plotter.show_grid()
        filtered_region = [-0.02, 0.3, 0, 0.16, -0.02, 0.02]
        xmin = -0.02 
        xmax = 0.3
        ymin = 0
        ymax = 0.16
        zmin = -0.1
        zmax = 0.1
        plotter.add_points(np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]]), color='white', point_size=0.1, label='Axis Limits')

        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_fb_surface_w_gt(self, vertices, gt_pts):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        
        # make plotter offscreen
        # plotter = pv.Plotter(off_screen=True)
        plotter = pv.Plotter()
        fb_vertices = self.get_fb_surface(vertices)
        vertices = vertices*1e3
        fb_vertices = fb_vertices*1e3
        gt_pts = gt_pts*1e3
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))
        mesh_surface = pv.PolyData(vertices, faces)
        mesh_fb = pv.PolyData(fb_vertices, faces)
        plotter.add_mesh(mesh_surface, color='lightblue', show_edges=True, opacity=0.95, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, opacity=0.35, label='FB Mid-Surface')
        # plotter.add_points(fb_vertices[self.ee_idx], color='magenta', point_size=10, label='FB EE Vertices')
        # plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        plotter.add_points(gt_pts, color='green', point_size=10, opacity=0.55, label='Ground Truth Points')
        plotter.show_grid()
        plotter.show_axes()
        # plotter.add_legend()
        # set viewpoint
        # plotter.view_vector((2, 2, 2))
        # plotter.camera.zoom(0.7)
        # plotter.camera.zoom(0.5)

        # make axis equal
        filtered_region = [-0.02, 0.3, 0, 0.16, -0.02, 0.02]
        xmin =1e3* -0.02 
        xmax =1e3* 0.3
        ymin =1e3* 0
        ymax =1e3* 0.16
        zmin =1e3* -0.1
        zmax =1e3* 0.1
        # add ghost point at (xmin, ymin, zmin) and (xmax, ymax, zmax) to set the axis limits
        plotter.add_points(np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]]), color='white', point_size=0.1, label='Axis Limits')
        plotter.show()
        return plotter

    def get_ee_normvec(self, vert):
        if vert.shape[0] != self.num_vertices:
            vert = self.q_to_vertices(vert)

        ee_id = int(np.asarray(self.ee_idx).reshape(-1)[0])
        ee_triangles = self.mesh_triangles[
            np.any(self.mesh_triangles == ee_id, axis=1)
        ]
        if ee_triangles.shape[0] == 0:
            raise ValueError(f"EE vertex {ee_id} does not belong to any triangle.")

        triangle_vertices = vert[ee_triangles]
        face_normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        normal_vec_ee = np.sum(face_normals, axis=0)
        normal_norm = np.linalg.norm(normal_vec_ee)
        if normal_norm < 1e-12:
            raise ValueError(f"Cannot determine a normal for EE vertex {ee_id}.")

        return normal_vec_ee / normal_norm

    def get_tracker_pos(self, vert):
        normal_vec_ee = self.get_ee_normvec(vert)
        ee_pos = self.get_ee_pos(vert)
        tracker_pos = ee_pos + normal_vec_ee*self.tracker_r
        return tracker_pos

    def get_tracker_ee(self, vert, tracker_pos):
        normal_vec_ee = self.get_ee_normvec(vert)
        tracker_ee_pos = tracker_pos - normal_vec_ee*self.tracker_r
        return tracker_ee_pos

    def visualize_IKD_result(self, vertices, target_ee_pos):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        plotter.add_points(pp_locations, color='blue', point_size=10
                            , label='Pullpoints')
        plotter.add_points(self.pulley_location, color='blue', point_size=10
                            , label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)

        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')

        # make fixed idx black
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')

        # add target ee pos as a green point
        plotter.add_points(target_ee_pos.reshape((1,3)), color='green', point_size=10, label='Target EE Position')

        # add grid
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_ws(self, vertices, ws_pts):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        plotter.add_points(pp_locations, color='blue', point_size=10
                            , label='Pullpoints')
        plotter.add_points(self.pulley_location, color='blue', point_size=10
                            , label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')
        # add all points in ws_pts as cyan points
        plotter.add_points(ws_pts, color='cyan', point_size=5, label='WS Points')

        # make fixed idx black
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        # add grid
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def visualize_vert_paper(self, vertices):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))
        mesh_original = pv.PolyData(self.vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_original, color='lightgray', show_edges=True, opacity = 0.5, edge_color = 'grey')
        plotter.add_mesh(mesh, color='lightblue', show_edges=True,opacity = 0.9 , edge_color = 'grey')
        pp_locations = self.get_pp_location_bary(vertices)
        
        plotter.add_points(self.pulley_location, color='blue', point_size=10, label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            if i == 0 or i == 2 or i == 4:
                plotter.add_points(pp_locations[i], color='red', point_size=10, label='Pullpoints')
                # plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='green', width=2)
        # add fixed vertices
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
        # annotate ee vertices
        # plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')

        # add grid
        # plotter.show_grid()
        # plotter.show_axes()
        # plotter.add_legend()
        plotter.show()

    def visualize_planned_traj(self, vertices, traj):
        mesh = pv.PolyData(vertices, np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles)))
        ws_pts = np.array(self.ee_pos_list)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = self.get_pp_location_bary(vertices)
        plotter.add_points(pp_locations, color='blue', point_size=10
                            , label='Pullpoints')
        plotter.add_points(self.pulley_location, color='blue', point_size=10
                            , label='Pulleys')
        # add lines between pullpoints and pulleys
        for i in range(self.nCable):
            plotter.add_lines(np.array([pp_locations[i], self.pulley_location[i]]), color='blue', width=2)
        # annotate ee vertices
        plotter.add_points(vertices[self.ee_idx], color='red', point_size=10, label='End Effectors')
        # add all points in ws_pts as cyan points
        plotter.add_points(ws_pts, color='cyan', point_size=5, label='WS Points', opacity=0.5)

        # make fixed idx black
        plotter.add_points(vertices[self.fixed_idx], color='black', point_size=10, label='Fixed Vertices')


        # traj is a nX3 array, add it as a magenta line
        # plotter.add_lines(traj, color='magenta', width=2, label='Planned Trajectory')
        for i in range(traj.shape[0]-1):
            plotter.add_lines(np.array([traj[i], traj[i+1]]), color='magenta', width=5)
        plotter.add_lines(np.array([traj[-1], traj[0]]), color='magenta', width=5)
        # add grid
        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

    def load_ws(self, filePath="./data/training_data_all.pkl"):
        with open(filePath, 'rb') as f:
            data = pickle.load(f)
        self.cl_list = data['cable_length']
        self.ee_pos_list = data['ee_pos']
        self.vertices_list = data['vertices']
        self.cable_tension_list = data['cable_tension']

    def replay_Q_list(self, Q_list, filePath="./c_srs_simulation.mp4", framerate=10,
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

        pp_location = self.get_pp_location_bary(vertices0)
        pp_cloud = pv.PolyData(pp_location.copy())
        plotter.add_mesh(pp_cloud, color='blue', point_size=10,
                         render_points_as_spheres=True, label='Pull points')

        pulley_cloud = pv.PolyData(self.pulley_location.copy())
        plotter.add_mesh(pulley_cloud, color='cyan', point_size=10,
                         render_points_as_spheres=True, label='Pulleys')

        ee_cloud = pv.PolyData(vertices0[self.ee_idx].copy())
        plotter.add_mesh(ee_cloud, color='red', point_size=10,
                         render_points_as_spheres=True, label='End Effectors')

        fixed_cloud = pv.PolyData(vertices0[self.fixed_idx].copy())
        plotter.add_mesh(fixed_cloud, color='black', point_size=10,
                         render_points_as_spheres=True, label='Fixed Vertices')

        # All cable segments in a single PolyData so points update in-place
        cable_pts = np.empty((2 * self.nCable, 3))
        for i in range(self.nCable):
            cable_pts[2 * i]     = pp_location[i]
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
            pp_cloud.points = pp_locations.copy()
            ee_cloud.points = vertices[self.ee_idx].copy()
            fixed_cloud.points = vertices[self.fixed_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = pp_locations[i]
            cables.points = cable_pts.copy()
            plotter.write_frame()

        plotter.close()

    def replay_IKD_Q_list(self, ee_target_pos, Q_list, filePath="./c_srs_simulation.mp4", framerate=10,
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

        # pp_cloud = pv.PolyData(vertices0[self.pp_idx].copy())
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

        fixed_cloud = pv.PolyData(vertices0[self.fixed_idx].copy())
        plotter.add_mesh(fixed_cloud, color='black', point_size=10,
                         render_points_as_spheres=True, label='Fixed Vertices')


        # add target ee pos as a green point
        plotter.add_points(ee_target_pos.reshape((1,3)), color='green', point_size=10,
                         render_points_as_spheres=True, label='Target EE Position')
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
            fixed_cloud.points = vertices[self.fixed_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = pp_locations[i]
            cables.points = cable_pts.copy()
            plotter.write_frame()

        plotter.close()

    def replay_IKD_trajectory(self, ee_target_pos, vert_list, filePath="./fixedEnd_IKD_traj.mp4", framerate = 30):
        window_size=(1024, 768)
        plotter = pv.Plotter(off_screen=True, window_size=window_size)

        vertices0 = vert_list[0]
        faces = np.hstack((np.full((self.mesh_triangles.shape[0], 1), 3), self.mesh_triangles))

        # Build all actors once; update .points in-place each frame
        surf = pv.PolyData(vertices0.copy(), faces)
        plotter.add_mesh(surf, color='lightgray', show_edges=True)

        pp_location = self.get_pp_location_bary(vertices0)
        pp_cloud = pv.PolyData(pp_location.copy())
        plotter.add_mesh(pp_cloud, color='blue', point_size=10,
                         render_points_as_spheres=True, label='Pull points')

        pulley_cloud = pv.PolyData(self.pulley_location.copy())
        plotter.add_mesh(pulley_cloud, color='cyan', point_size=10,
                         render_points_as_spheres=True, label='Pulleys')

        ee_cloud = pv.PolyData(vertices0[self.ee_idx].copy())
        plotter.add_mesh(ee_cloud, color='red', point_size=10,
                         render_points_as_spheres=True, label='End Effectors')
        
        fixed_cloud = pv.PolyData(vertices0[self.fixed_idx].copy())
        plotter.add_mesh(fixed_cloud, color='black', point_size=10,
                         render_points_as_spheres=True, label='Fixed Vertices')

        cable_pts = np.empty((2 * self.nCable, 3))
        for i in range(self.nCable):
            cable_pts[2 * i]     = pp_location[i]
            cable_pts[2 * i + 1] = self.pulley_location[i]
        cable_lines = np.array([[2, 2 * i, 2 * i + 1]
                                 for i in range(self.nCable)]).flatten()
        cables = pv.PolyData()
        cables.points = cable_pts.copy()
        cables.lines = cable_lines
        plotter.add_mesh(cables, color='blue', line_width=2)

        # add ee target pos as line segment
        nTarget = len(ee_target_pos)
        for i in range(nTarget-1):
            plotter.add_lines(np.array([ee_target_pos[i], ee_target_pos[i+1]]), color='green', width=2)
        plotter.add_lines(np.array([ee_target_pos[-1], ee_target_pos[0]]), color='green', width=2)

        plotter.show_grid()
        plotter.show_axes()
        plotter.add_legend()
        plotter.open_movie(filePath, framerate=framerate)

        for vertices in vert_list:
            surf.points = vertices.copy()
            pp_locations = self.get_pp_location_bary(vertices)
            pp_cloud.points = pp_locations.copy()
            ee_cloud.points = vertices[self.ee_idx].copy()
            fixed_cloud.points = vertices[self.fixed_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = pp_locations[i]
            cables.points = cable_pts.copy()
            plotter.write_frame()

        plotter.close()

class IK_MLP(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,   128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,   6),
        )
        self.load_state_dict(torch.load("./learning_model/ik_model_best.pth"))
    def forward(self, x):
        return self.net(x)
    
    def predict_cable_length(self, ee_pos):
        scaler_X = joblib.load("./learning_model/scaler_X.pkl")
        scaler_Y = joblib.load("./learning_model/scaler_Y.pkl")
        """ee_pos: (3,) array in metres. Returns (6,) cable lengths in metres."""
        x = scaler_X.transform(ee_pos.reshape(1, 3))
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_norm = self(x_t).numpy()
        return scaler_Y.inverse_transform(y_norm).flatten()

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    icl = c_srs.initial_cable_length.copy()
    print("initial cable length: ", icl)
    print("number of vertices: ", c_srs.num_vertices)
    print("")
    c_srs.visualize_vert(c_srs.vertices)
    exit(0)
    # tcl = [icl[0]+0.1, icl[1]+0.1, icl[2]-0.04, icl[3]+0.1, icl[4]-0.01, icl[5]+0.1]
    # Q_list = c_srs.FKD_static(c_srs.vertices, [1,1,1,1,1,1],tol = 1e-6, show_info = True)
    # c_srs.visualize_vert(Q_list[-1])
    cl_final = c_srs.get_cable_length_bary(Q_list[-1])
    Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl_final, tol = 1e-6, show_info = True)
    c_srs.visualize_vert(Q_list[-1])
    print("cable tension: ", cable_tension)
    # Q_list = c_srs.FKD_free_static(1)
    # c_srs.visualize_fb_surface(c_srs.q_to_vertices(Q_list[-1]))
    # c_srs.replay_Q_list(Q_list, "./c_srs_free_static.mp4")
    exit(0)
    # cl_range_1 = [[icl[0]-0.08, icl[0]-0.02], 
    #             [icl[1]-0.08, icl[1]-0.02],
    #             [icl[2]-0.08, icl[2]-0.02],
    #             [icl[3], icl[3]+0.05],
    #             [icl[4], icl[4]+0.05],
    #             [icl[5], icl[5]+0.05]]
    
    # cl_range_2 = [[icl[0]-0.03, icl[0]+0.01], 
    #             [icl[1]-0.03, icl[1]+0.01],
    #             [icl[2]-0.03, icl[2]+0.01],
    #             [icl[3]-0.04, icl[3]+0.01],
    #             [icl[4]-0.04, icl[4]+0.01],
    #             [icl[5]-0.04, icl[5]+0.01]]
    # c_srs.generate_ws(cl_range_1, total_number=1000, saveFile='training_data_1.pkl')
    c_srs.generate_ws(cl_range_2, total_number=1000, saveFile='training_data_2.pkl')
    # exit(0)
    tcl = [icl[0]+0.1, icl[1]+0.1, icl[2]-0.04, icl[3]+0.1, icl[4]-0.01, icl[5]+0.1]
    Q_list, vert_length, cable_tension = c_srs.FKD_time(tcl, 1, c_srs.vertices, tol = 1e-4, show_info = True)
    fcl = c_srs.get_cable_length_bary(vert_length)
    # vert_cg = c_srs.deform_CG(fcl, c_srs.vertices, max_iter=1000, tol=1e-9)
    # print("cable tension: ", cable_tension)
    c_srs.visualize_vert_paper(vert_length)
    # c_srs.visualize_vert(vert_cg)
    exit(0)
    # cur_length, starting_vertices = c_srs.IKD_single(ee_target,  c_srs.vertices, AA = False, tol = 1e-3)
    # cur_length, starting_vertices = c_srs.IK_CG(ee_target, c_srs.vertices, tol = 1e-3)
    print("final cable length after IKD: ", cur_length)
    c_srs.visualize_IKD_result(starting_vertices, ee_target)
    exit(0)
    
    fcl = c_srs.get_cable_length(c_srs.q_to_vertices(Q_list[-1]))
    print("ee pos: ", c_srs.get_ee_pos(vert_length))
    print("target cable length: ", tcl)
    print("final cable length: ", fcl)
    c_srs.visualize_vert(vert_length)
    # vert_cg = c_srs.deform_CG(fcl, c_srs.vertices)
    # Jac = c_srs.get_CG_Jacobian(vert_cg)
    # print("CG Jacobian: ", Jac)
    # print("tcl: ", tcl)
    
    
    # c_srs.replay_Q_list(Q_list)
