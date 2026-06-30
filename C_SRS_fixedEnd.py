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
class C_SRS_fixedEnd:
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
        self.mesh_RF_triangles = self.description['mesh_RF_triangles']
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
        self.Poisson_ratio = self.description['Poisson_ratio']
        self.density = self.description['density']
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

    def assemble_CG_matrices(self):
        mem_block   = 9  * self.num_triangles
        bend_block  = 3 * len(self.bending_ele_idx)
        cable_block = 3  * self.nCable
        ghost_block = 12 * self.nCable
        matA_size = mem_block + bend_block + cable_block + ghost_block
        max_weight = np.max((np.max(self.mem_weight_list), np.max(self.bending_weight_list)))
        self.weight_cable = 0.5 * max_weight
        self.weight_ghost = 0.5 * max_weight
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
            u, s, vh = np.linalg.svd(cur_tri_sk.T @ initial_tri_sk)
            if np.linalg.det(u @ vh) < 0:
                u[:, -1] *= -1
            R = u @ vh
            R_list[i] = R
        return R_list

    def get_rotation_ARAP(self, ARAP_shape_list):
        R_list = [np.eye(3) for _ in range(self.nMoving)]
        for i in range(self.nMoving):
            ARAP_shape = ARAP_shape_list[i]
            ARAP_initial_shape = self.initial_ARAP_shape_list[i]
            u, s, vh = np.linalg.svd(ARAP_shape.T @ ARAP_initial_shape)
            R = u @ vh
            R_list[i] = R
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
            u, s, vh = np.linalg.svd(cur_ghost_shape.T @ initial_ghost_shape)
            if np.linalg.det(u @ vh) < 0:
                u[:, -1] *= -1
            R = u @ vh
            R_list[i] = R
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

    def FKD_time(self, target_cable_length, total_time, starting_vertices, tol = 2e-5, show_info = False):
        if starting_vertices.shape[0] == 3*self.num_vertices:
            Q_a = starting_vertices.copy()
        elif starting_vertices.shape[0] == self.num_vertices:
            Q_a = self.vertices_to_q(starting_vertices)
        else:
            raise ValueError("starting_vertices should be either a 3n vector or an n by 3 array.")
        Q_a = Q_a.reshape((3*self.num_vertices, ))
        Q_ad = np.zeros((3*self.num_vertices, ))
        t_a = 0.0
        h = 0.002
        phi_Qfree = np.zeros((self.nCable, 1))
        H_free = np.zeros((self.nCable, 3*self.num_vertices))
        Q_list = [Q_a.copy()]
        diff_count = 0
        t_start = time.time()
        while t_a < total_time:
            Q_a_last = Q_a.copy()
            t0 = time.time()
            R_list, R_list_1818 = self.get_R_list(self.q_to_vertices(Q_a))
            t1 = time.time()
            K_mat, f0 = self.assemble_K(R_list_1818)
            t2 = time.time()
            disp = Q_a - self.vertices_to_q(self.vertices)
            denom = disp @ self.mass_matrix @ disp
            damping_coeff = np.sqrt((disp @ K_mat @ disp) / denom) if denom > 1e-30 else 0.0
            C_mat = 2 * damping_coeff * self.mass_matrix

            A_mat = (1.0/h)*np.eye(3*self.num_vertices) + h * self.W_mat @ K_mat + self.W_mat @ C_mat
            lu, piv = lu_factor(A_mat)

            b_vec = self.W_mat @ (-K_mat @ (Q_a + h * Q_ad) + f0 + self.gravity_vec - C_mat @ Q_ad)
            # dv_free = A_inv @ b_vec
            dv_free = lu_solve((lu, piv), b_vec)

            Q_free = Q_a + h * Q_ad + h * dv_free
            cl_free = self.get_cable_length_bary(Q_free)
            H_free = -self.get_cable_Jacobian_bary(Q_free)
            for i in range(self.nCable):
                phi_Qfree[i] = target_cable_length[i] - cl_free[i]
            Z1 = lu_solve((lu, piv), H_free.T)                   # (3n, nCable)
            # lcp_Mmat = h * H_free @ self.W_mat @ A_inv @ H_free.T
            lcp_Mmat = h * H_free @ self.W_mat @ Z1

            lcp_q = phi_Qfree.reshape((self.nCable,))
            cable_tension = projected_gauss_seidel_lcp(lcp_Mmat, lcp_q)
            Z2 = lu_solve((lu, piv), self.W_mat @ H_free.T)
            # cable_tension = np.zeros((self.nCable,))
            # dv_cor = A_inv @ (self.W_mat @ H_free.T @ cable_tension).reshape((3*self.num_vertices, ))
            dv_cor = Z2 @ cable_tension
            dv = dv_free + dv_cor
            Q_ad = Q_ad + dv
            Q_a = Q_a + h * Q_ad
            t_a += h
            Q_list.append(Q_a.copy())
            diff = np.linalg.norm(Q_a - Q_a_last)/(3*self.num_vertices)
            # if diff < 1e-5:
            #     h *= 0.1
            if diff < tol and min(phi_Qfree.flatten()) > -1e-3:
                diff_count += 1
                if diff_count >= 10:
                    print(f"Converged at time {t_a:.2f} with diff {diff:.6f} for 10 consecutive steps, stopping simulation.")
                    break
            t3 = time.time()
            if show_info:
                print(f"t_a: {t_a:.3f}, diff: {diff:.7f}, time for R_list: {t1-t0:.4f}s, time for K_mat: {t2-t1:.4f}s, total time for this step: {t3-t0:.4f}s")
        t_end = time.time()
        print(f"Total simulation time: {t_end - t_start:.2f}s")
        vert_length = self.q_to_vertices(Q_a)
        return Q_list, vert_length, cable_tension
            

    def IKD_single(self, target_ee_pos, starting_vertices, AA = False, tol = 1e-3):
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
                Q_plus_list, vert_plus, cable_tension_plus = self.FKD_time(cl_plus, 1, Q_a)
                Q_minus_list, vert_minus, cable_tension_minus= self.FKD_time(cl_minus, 1, Q_a)
                ee_pos_plus = self.get_ee_pos(vert_plus)
                ee_pos_minus = self.get_ee_pos(vert_minus)
                diff_plus = 1/2 * np.linalg.norm(ee_pos_plus - target_ee_pos) ** 2
                diff_minus = 1/2 * np.linalg.norm(ee_pos_minus - target_ee_pos) ** 2
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


        cur_length = self.get_cable_length_bary(Q)
        Q_list, starting_vertices, cable_tension = self.FKD_time(cur_length, 1, starting_vertices)
        Q = self.vertices_to_q(starting_vertices)
        Q_list_final = [Q.copy()]
        for i in range(max_iter):
            dl = 0.5
            jac = get_jacobian(Q)
            # jac = get_jacobian_fd(Q)
            diff = get_diff(Q)
            cur_length = self.get_cable_length_bary(Q)
            cmd_diff = [0 for _ in range(self.nCable)]
            cmd_length = cur_length.copy()
            # alpha = 10
            # dl = alpha*diff/(np.max(np.abs(jac))+1e-6)
            for k in range(self.nCable):
                if cable_tension[k] < 1e-5 and jac[k] < 0:
                    cmd_diff[k] = 0
                else:
                    cmd_diff[k] = -dl * jac[k]
            cmd_diff = clamp_diff(cmd_diff, min_bound = 1e-3, max_bound = 5e-3)
            for k in range(self.nCable):
                cmd_length[k] += cmd_diff[k]
            Q_list, starting_vertices, cable_tension = self.FKD_time(cmd_length, 1, Q, tol = 1e-5)
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
            print("Iteration {}: diff = {}, dl = {}, Jacobian: {}, cmd_diff: {}".format(i, diff, dl, np.round(jac, 5), np.round(cmd_diff, 5)))
            # self.visualize_IKD_result(self.q_to_vertices(Q), target_ee_pos)
            Q_list_final.append(Q.copy())
            if diff < tol:
                print("Converged at iteration {} with diff {}".format(i, diff))
                break
            cur_length = self.get_cable_length_bary(Q)
        return cur_length, starting_vertices, Q_list_final

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
            Q_list, vert_length, cable_tension = self.FKD_time(cl, 1, self.vertices)
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
         
    def get_R_list(self, vertices):
        if vertices.shape[0] != self.num_vertices:
            vertices = self.q_to_vertices(vertices)
        tri_SK_list = self.get_tri_SK_list(vertices)
        R_list = [np.eye(3) for _ in range(self.num_RF_triangles)]
        R_list_1818 = [np.eye(18) for _ in range(self.num_RF_triangles)]
        for i in range(self.num_RF_triangles):
            initial_tri_SK = self.initial_tri_SK_list[i]
            cur_tri_SK = tri_SK_list[i]
            u, s, vh = np.linalg.svd(cur_tri_SK.T @ initial_tri_SK)
            R_list[i] = u @ vh
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
                idx_moving = i - self.idxAll_2_idxMoving[i] - 1
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
        plotter.show_axes()
        plotter.add_legend()
        plotter.show()

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

        pp_cloud = pv.PolyData(vertices0[self.pp_idx].copy())
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
            fixed_cloud.points = vertices[self.fixed_idx].copy()
            for i in range(self.nCable):
                cable_pts[2 * i] = vertices[self.pp_idx[i]]
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
    # c_srs.generate_ws(cl_range_2, total_number=1000, saveFile='training_data_2.pkl')
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

