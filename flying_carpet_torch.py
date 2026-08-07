"""Torch implementation of the flying-carpet solvers.

The public API deliberately mirrors :class:`flying_carpet.Flying_carpet` and
accepts/returns numpy arrays.  All expensive per-iteration work stays on the
Torch device; conversion is only performed at the API boundary.
"""

import time
import numpy as np
import torch
import pyvista as pv
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize
from utilities import projected_gauss_seidel_lcp

from flying_carpet import Flying_carpet


class Flying_carpet_torch(Flying_carpet):
    def __init__(self, description_file, device=None, dtype=torch.float64):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = dtype
        # Reuse the well-tested mesh/material preprocessing in the reference
        # implementation, then retain device copies for every iterative solve.
        super().__init__(description_file)
        self._cache_torch_data()

    def _tensor(self, value, dtype=None):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=dtype or self.torch_dtype)
        return torch.as_tensor(value, device=self.device, dtype=dtype or self.torch_dtype)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    def _cache_torch_data(self):
        d, ld = self.torch_dtype, torch.long
        self.vertices_t = self._tensor(np.asarray(self.vertices).copy())
        self.mesh_triangles_t = self._tensor(self.mesh_triangles, ld)
        self.mesh_RF_triangles_t = self._tensor(self.mesh_RF_triangles, ld)
        self.valid_rf_t = self.mesh_RF_triangles_t >= 0
        self.safe_rf_t = self.mesh_RF_triangles_t.clamp_min(0)
        self.stiffness_matrices_t = self._tensor(np.stack(self.stiffness_matrices))
        self.qe0_t = self._tensor(np.stack(self.qe0_list))
        self.initial_tri_SK_t = self._tensor(np.stack(self.initial_tri_SK_list))
        self.initial_cable_vec_t = self._tensor(np.stack(self.initial_cable_vec))
        self.initial_ghost_shape_t = self._tensor(np.stack(self.initial_ghost_shape_list))
        self.pulley_location_t = self._tensor(self.pulley_location)
        self.pp_bary_tri_idx_t = self._tensor(self.pp_bary_tri_idx, ld)
        self.mem_weight_t = self._tensor(self.mem_weight_list)
        self.N33_t, self.N44_t = self._tensor(self.N33), self._tensor(self.N44)
        self.mass_matrix_t, self.W_mat_t = self._tensor(self.mass_matrix), self._tensor(self.W_mat)
        self.gravity_vec_t = self._tensor(self.gravity_vec)
        self.q0_t = self.vertices_t.reshape(-1)
        # Shape-Up's constant normal equations are assembled once and solved on
        # device.  Keeping A is also useful when weights are reassembled.
        if not hasattr(self, "matA_all_t"):
            self.matA_all_t = self._tensor(self.matA_all)
            self.matAT_t = self.matA_all_t.T.contiguous()
            self.matATA_t = self.matAT_t @ self.matA_all_t
            self.K_CG_t = torch.linalg.solve(
                self.matATA_t, self.matAT_t[:, -15*self.nCable:-12*self.nCable]
            )

        # Flattened global DOF indices used by index_add_ during FEM assembly.
        local = torch.arange(3, device=self.device, dtype=ld)
        dofs = 3 * self.safe_rf_t[..., None] + local
        self.rf_dofs_t = dofs.reshape(self.num_RF_triangles, 18)
        self.rf_dof_valid_t = self.valid_rf_t[..., None].expand(-1, -1, 3).reshape(
            self.num_RF_triangles, 18
        )
        self.eye_ndof_t = torch.eye(3 * self.num_vertices, device=self.device, dtype=d)

    def assemble_CG_matrices(self):
        """Assemble Shape-Up's constant least-squares operator on the device."""
        mem_block = 9*self.num_triangles
        bend_block = 3*len(self.bending_ele_idx)
        cable_block = 3*self.nCable
        rows = mem_block+bend_block+cable_block+12*self.nCable
        cols = 3*self.num_vertices+3*self.nCable
        A = torch.zeros((rows, cols), device=self.device, dtype=self.torch_dtype)
        max_weight = max(float(np.max(self.bending_weight_list)),
                         float(np.max(self.mem_weight_list)))
        self.weight_cable = self.weight_ghost = 0.5*max_weight
        eye3 = torch.eye(3, device=self.device, dtype=self.torch_dtype)
        for i, tri in enumerate(np.asarray(self.mesh_triangles)):
            w = float(np.sqrt(self.mem_weight_list[i]))
            for j in range(3):
                for jp, vertex in enumerate(tri):
                    A[9*i+3*j:9*i+3*j+3, 3*vertex:3*vertex+3] = (
                        w*((2/3) if jp == j else (-1/3))*eye3
                    )
        for i, vertices in enumerate(np.asarray(self.bending_ele_idx)):
            w = float(np.sqrt(self.bending_weight_list[i]))
            for vertex, coefficient in zip(vertices, self.bending_ele_param[i]):
                A[mem_block+3*i:mem_block+3*i+3, 3*vertex:3*vertex+3] = (
                    w*float(coefficient)*eye3
                )
        wc, wg = np.sqrt(self.weight_cable), np.sqrt(self.weight_ghost)
        for i in range(self.nCable):
            r = mem_block+bend_block+3*i
            c = 3*self.num_vertices+3*i
            A[r:r+3, c:c+3] = wc*eye3
            r = mem_block+bend_block+cable_block+12*i
            tri = self.mesh_triangles[self.pp_bary_tri_idx[i]]
            ids = list(tri)+[self.num_vertices+i]
            N = self._tensor(self.N1212)
            for j, vj in enumerate(ids):
                for k, vk in enumerate(ids):
                    A[r+3*j:r+3*j+3, 3*vk:3*vk+3] += wg*N[3*j:3*j+3, 3*k:3*k+3]
        self.matA_all_t, self.matAT_t = A, A.T.contiguous()
        self.matATA_t = self.matAT_t@A
        self.K_CG_t = torch.linalg.solve(
            self.matATA_t, self.matAT_t[:, -15*self.nCable:-12*self.nCable]
        )
        # Lightweight compatibility views for inherited visualization/Jacobian code.
        self.matA_all = self._numpy(A)
        self.matAT = self._numpy(self.matAT_t)
        self.matATA = self._numpy(self.matATA_t)
        self.K_CG = self._numpy(self.K_CG_t)
        self.nNeighbour_list = [len(x) for x in self.neighbour_list]

    def reassemble_CG_matrices(self, ratio_bending):
        ratio_bending = float(ratio_bending)
        if not np.isfinite(ratio_bending) or ratio_bending <= 0:
            raise ValueError("ratio_bending must be a finite positive value")
        original = self.bending_weight_list
        self.bending_weight_list = np.asarray(original)*ratio_bending
        try:
            self.assemble_CG_matrices()
        finally:
            self.bending_weight_list = original

    @staticmethod
    def _best_fit_rotation_torch(current, reference):
        """Batched proper rotations aligning ``reference`` with ``current``."""
        u, _, vh = torch.linalg.svd(current.transpose(-1, -2) @ reference)
        sign = torch.linalg.det(u @ vh)
        correction = torch.eye(3, dtype=current.dtype, device=current.device).expand(
            current.shape[:-2] + (3, 3)
        ).clone()
        correction[..., 2, 2] = sign
        return u @ correction @ vh

    def _rotation_tri_torch(self, vertices):
        cur = vertices[self.mesh_triangles_t]
        cur_sk = self.N33_t @ cur
        return self._best_fit_rotation_torch(cur_sk, self.initial_tri_SK_t)

    def _rotation_rf_torch(self, vertices):
        cur = vertices[self.mesh_triangles_t]
        cur_sk = self.N33_t @ cur
        rotations = self._best_fit_rotation_torch(cur_sk, self.initial_tri_SK_t)
        # diag(R,...,R), six blocks per EBST patch
        return torch.block_diag(*([torch.eye(3, device=self.device)])) if False else \
            torch.einsum('eab,jk->ejakb', rotations, torch.eye(6, device=self.device, dtype=self.torch_dtype)).reshape(-1,18,18)

    def _vector_rotations_torch(self, initial, current, eps=1e-12):
        """Stable batched shortest-arc rotation, including antiparallel vectors."""
        a = initial / torch.linalg.vector_norm(initial, dim=1, keepdim=True).clamp_min(eps)
        b = current / torch.linalg.vector_norm(current, dim=1, keepdim=True).clamp_min(eps)
        v = torch.linalg.cross(a, b, dim=1)
        c = (a * b).sum(1).clamp(-1.0, 1.0)
        s = torch.linalg.vector_norm(v, dim=1)
        K = torch.zeros((len(a), 3, 3), device=self.device, dtype=self.torch_dtype)
        K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
        K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
        eye = torch.eye(3, device=self.device, dtype=self.torch_dtype).expand_as(K)
        R = eye + K + (K @ K) / (1.0 + c).clamp_min(eps)[:, None, None]
        # The shortest rotation is ambiguous at 180 degrees. Pick a stable
        # perpendicular axis and construct the exact pi rotation.
        anti = c < (-1.0 + 1e-8)
        if anti.any():
            aa = a[anti]
            basis = torch.eye(3, device=self.device, dtype=self.torch_dtype)[
                aa.abs().argmin(dim=1)
            ]
            axis = torch.linalg.cross(aa, basis, dim=1)
            axis /= torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(eps)
            R[anti] = 2 * axis[:, :, None] * axis[:, None, :] - eye[anti]
        return R

    def get_rotation_tri(self, vertices):
        v = self._tensor(vertices).reshape(self.num_vertices, 3)
        return self._numpy(self._rotation_tri_torch(v))

    def get_R_list(self, vertices):
        v = self._tensor(vertices).reshape(self.num_vertices, 3)
        R = self._rotation_tri_torch(v)
        eye6 = torch.eye(6, device=self.device, dtype=self.torch_dtype)
        R18 = torch.einsum("eab,jk->ejakb", R, eye6).reshape(-1, 18, 18)
        return self._numpy(R), self._numpy(R18)

    def _assemble_K_torch(self, R18):
        """Transform and scatter all element matrices/vectors on the device."""
        Ke = R18 @ self.stiffness_matrices_t @ R18.transpose(1, 2)
        f0e = (R18 @ self.stiffness_matrices_t @ self.qe0_t[..., None]).squeeze(-1)
        ndof = 3 * self.num_vertices
        K = torch.zeros(ndof * ndof, device=self.device, dtype=self.torch_dtype)
        f0 = torch.zeros(ndof, device=self.device, dtype=self.torch_dtype)
        row = self.rf_dofs_t[:, :, None].expand(-1, 18, 18)
        col = self.rf_dofs_t[:, None, :].expand(-1, 18, 18)
        mask2 = self.rf_dof_valid_t[:, :, None] & self.rf_dof_valid_t[:, None, :]
        K.index_add_(0, (row[mask2] * ndof + col[mask2]), Ke[mask2])
        mask1 = self.rf_dof_valid_t
        f0.index_add_(0, self.rf_dofs_t[mask1], f0e[mask1])
        return K.reshape(ndof, ndof), f0

    def assemble_K(self, R_list_1818):
        K, f0 = self._assemble_K_torch(self._tensor(R_list_1818))
        return self._numpy(K), self._numpy(f0)

    def _pp_location_torch(self, vertices):
        tri = self.mesh_triangles_t[self.pp_bary_tri_idx_t]
        # The inherited geometry helper includes the barycentric normal offset;
        # reproduce it in Torch so constraints never leave the device.
        xyz = vertices[tri]
        bary = self._tensor(self.pp_bary_coords)
        pp = (bary[..., None] * xyz).sum(1)
        normal = torch.linalg.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0], dim=1)
        normal /= torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(1e-12)
        return pp + self._tensor(self.pp_bary_offsets)[:, None] * normal

    def _cable_lengths_torch(self, q):
        pp = self._pp_location_torch(q.reshape(self.num_vertices, 3))
        return torch.linalg.vector_norm(pp - self.pulley_location_t, dim=1)

    @staticmethod
    def _skew_torch(v):
        """Batched skew matrices satisfying ``skew(v) @ x == v cross x``."""
        out = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
        out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
        out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
        out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
        return out

    def _cable_jacobian_torch(self, q):
        """Analytic Jacobian of barycentric+normal-offset cable lengths.

        This is the Torch equivalent of ``Flying_carpet.get_cable_Jacobian_bary``.
        It includes the derivative of the triangle unit normal and performs the
        complete formulation and global-DOF assembly on the selected device.
        """
        vertices = q.reshape(self.num_vertices, 3)
        tri_ids = self.mesh_triangles_t[self.pp_bary_tri_idx_t]
        xyz = vertices[tri_ids]
        e1, e2 = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
        normal = torch.linalg.cross(e1, e2, dim=1)
        area2 = torch.linalg.vector_norm(normal, dim=1).clamp_min(1e-12)
        t3 = normal/area2[:, None]
        eye = torch.eye(3, device=self.device, dtype=self.torch_dtype)
        P = eye[None] - t3[:, :, None]*t3[:, None, :]

        # dn/dv_k = [skew(e2-e1), -skew(e2), skew(e1)].
        Gn = torch.stack((self._skew_torch(e2-e1),
                          -self._skew_torch(e2),
                          self._skew_torch(e1)), dim=1)
        Gt = torch.einsum("nij,nkjl->nkil", P, Gn)/area2[:, None, None, None]
        bary = self._tensor(self.pp_bary_coords)
        offset = self._tensor(self.pp_bary_offsets)
        dpp = bary[:, :, None, None]*eye + offset[:, None, None, None]*Gt

        pp = (bary[..., None]*xyz).sum(1) + offset[:, None]*t3
        cable = pp-self.pulley_location_t
        direction = cable/torch.linalg.vector_norm(cable, dim=1, keepdim=True).clamp_min(1e-12)
        blocks = torch.einsum("ni,nkij->nkj", direction, dpp)

        J = torch.zeros((self.nCable, 3*self.num_vertices),
                        device=self.device, dtype=self.torch_dtype)
        local = torch.arange(3, device=self.device)
        columns = (3*tri_ids[:, :, None]+local).reshape(self.nCable, 9)
        rows = torch.arange(self.nCable, device=self.device)[:, None]
        J[rows, columns] = blocks.reshape(self.nCable, 9)
        return J

    def get_cable_Jacobian_bary(self, vertices):
        return self._numpy(self._cable_jacobian_torch(self._tensor(vertices).reshape(-1)))

    def _cable_geometry_cpu(self, q):
        vertices=np.asarray(q,dtype=float).reshape(self.num_vertices,3)
        ids=np.asarray(self.mesh_triangles)[np.asarray(self.pp_bary_tri_idx)]
        xyz=vertices[ids]
        normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        pp=(np.asarray(self.pp_bary_coords)[...,None]*xyz).sum(1)+np.asarray(self.pp_bary_offsets)[:,None]*normal
        return pp,np.linalg.norm(pp-np.asarray(self.pulley_location),axis=1)

    def _cable_jacobian_cpu(self, q):
        vertices=np.asarray(q,dtype=float).reshape(self.num_vertices,3)
        ids=np.asarray(self.mesh_triangles)[np.asarray(self.pp_bary_tri_idx)]
        pp,_=self._cable_geometry_cpu(q)
        J=np.zeros((self.nCable,3*self.num_vertices)); eye=np.eye(3)
        for i,tri in enumerate(ids):
            v0,v1,v2=vertices[tri]; e1,e2=v1-v0,v2-v0
            n=np.cross(e1,e2); area2=max(np.linalg.norm(n),1e-12); t3=n/area2
            P=eye-np.outer(t3,t3)
            def skew(v):
                return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            Gn=(skew(e2-e1),-skew(e2),skew(e1))
            direction=pp[i]-self.pulley_location[i]
            direction/=max(np.linalg.norm(direction),1e-12)
            for k in range(3):
                dpp=self.pp_bary_coords[i][k]*eye+self.pp_bary_offsets[i]*(P@Gn[k])/area2
                J[i,3*tri[k]:3*tri[k]+3]=direction@dpp
        return J

    def _hybrid_global_system(self, q):
        """Perform rotations/element assembly on CUDA and transfer K,f once."""
        q_t=self._tensor(q).reshape(-1)
        R18=self._rotation_rf_torch(q_t.reshape(-1,3))
        K_t,f_t=self._assemble_K_torch(R18)
        return self._numpy(K_t),self._numpy(f_t)

    def _shapeup_bvec_torch(self, vertices, ghosts, target):
        b = torch.zeros(self.matA_all_t.shape[0], device=self.device, dtype=self.torch_dtype)
        Rtri = self._rotation_tri_torch(vertices)
        rotated = Rtri @ self.initial_tri_SK_t.transpose(1, 2)
        b[:9*self.num_triangles] = (
            rotated.transpose(1, 2) * torch.sqrt(self.mem_weight_t)[:, None, None]
        ).reshape(-1)
        Rcable = self._vector_rotations_torch(
            self.initial_cable_vec_t, ghosts - self.pulley_location_t
        )
        cable = (Rcable @ self.initial_cable_vec_t[..., None]).squeeze(-1)
        c0 = b.numel() - 15*self.nCable
        b[c0:c0+3*self.nCable] = torch.sqrt(self._tensor(self.weight_cable)) * (
            cable * target[:, None] + self.pulley_location_t
        ).reshape(-1)
        tri = self.mesh_triangles_t[self.pp_bary_tri_idx_t]
        shape = torch.cat((vertices[tri], ghosts[:, None, :]), dim=1)
        shape = self.N44_t @ shape
        Rghost = self._best_fit_rotation_torch(shape, self.initial_ghost_shape_t)
        rotated_ghost = (Rghost @ self.initial_ghost_shape_t.transpose(1, 2)).transpose(1, 2)
        b[-12*self.nCable:] = torch.sqrt(self._tensor(self.weight_ghost)) * rotated_ghost.reshape(-1)
        return b

    def deform_CG(self, Q, target_cable_length, max_iter=100, tol=1e-5, show_info=False):
        vertices = self._tensor(Q).reshape(self.num_vertices, 3).clone()
        target = self._tensor(target_cable_length).reshape(self.nCable)
        ghosts = self._pp_location_torch(vertices)
        q_last = vertices.reshape(-1).clone()
        # The 500-ish DOF dense normal system is faster in CPU LAPACK. Its
        # factorization is constant, while all local projections/SVDs stay GPU.
        ata=np.asarray(self.matATA,dtype=float)
        lu,pivots=lu_factor(ata,check_finite=False)
        AT=np.asarray(self.matAT,dtype=float)
        for iteration in range(max_iter):
            b = self._shapeup_bvec_torch(vertices, ghosts, target)
            q_all=lu_solve((lu,pivots),AT@self._numpy(b),check_finite=False)
            q_np=q_all[:3*self.num_vertices]
            diff=np.linalg.norm(q_np-self._numpy(q_last))/(3*self.num_vertices)
            vertices=self._tensor(q_np).reshape(-1,3)
            ghosts=self._tensor(q_all[3*self.num_vertices:]).reshape(-1,3)
            q_last=self._tensor(q_np)
            if show_info:
                print(f"Shape-Up iteration {iteration+1}: diff={diff:.7e}")
            if diff < tol:
                break
        return self._numpy(vertices)

    @staticmethod
    def _pgs_lcp_torch(M, q, iterations=100, tol=1e-10):
        x = torch.zeros_like(q)
        diag = torch.diagonal(M).clamp_min(torch.finfo(M.dtype).eps)
        for _ in range(iterations):
            old = x.clone()
            for i in range(q.numel()):
                x[i] = torch.clamp(x[i] - (M[i] @ x + q[i]) / diag[i], min=0.0)
            if torch.max(torch.abs(x-old)).item() < tol:
                break
        return x

    def FKD_time(self, cable_length, total_time, starting_vertices, tol=1e-5,
                 h=0.01, show_info=False):
        target=np.asarray(self._numpy(cable_length),dtype=float).reshape(-1)
        if target.size != self.nCable or np.any(~np.isfinite(target)) or np.any(target <= 0):
            raise ValueError(f"cable_length must contain {self.nCable} positive values")
        q=np.asarray(self._numpy(starting_vertices),dtype=float).reshape(-1).copy()
        qd=np.zeros_like(q); history=[q.copy()]; tension=np.zeros(self.nCable)
        mass_diag=np.diag(np.asarray(self.mass_matrix)); W_diag=np.diag(np.asarray(self.W_mat))
        gravity=np.asarray(self.gravity_vec); q0=np.asarray(self.vertices).reshape(-1)
        t, settled = 0.0, 0
        while t < total_time:
            dt = min(h, total_time-t)
            q_old=q.copy(); K,f0=self._hybrid_global_system(q)
            disp=q-q0; denom=np.dot(mass_diag*disp,disp)
            energy=max(float(disp@K@disp),0.0)
            damp=np.sqrt(energy/denom) if denom>1e-30 else 0.0
            C_diag=2*damp*mass_diag
            # W and C are diagonal: avoid two dense matrix multiplications.
            A=np.eye(q.size)/dt+dt*(W_diag[:,None]*K)
            A[np.diag_indices_from(A)]+=W_diag*C_diag
            rhs=W_diag*(-K@(q+dt*qd)+f0+gravity-C_diag*qd)
            lu,pivots=lu_factor(A,check_finite=False)
            dv_free=lu_solve((lu,pivots),rhs,check_finite=False)
            q_free,dv_cor=q+dt*qd+dt*dv_free,np.zeros_like(q); tension.fill(0)
            for _ in range(5):
                qc=q_free+dt*dv_cor; _,lengths=self._cable_geometry_cpu(qc)
                phi=target-lengths
                if np.max(np.maximum(-phi,0)) <= min(tol,1e-8):
                    break
                H=-self._cable_jacobian_cpu(qc)
                Z=lu_solve((lu,pivots),W_diag[:,None]*H.T,check_finite=False)
                inc=projected_gauss_seidel_lcp(dt*H@Z,phi)
                dv_cor+=Z@inc; tension+=inc
            qd+=dv_free+dv_cor; q+=dt*qd
            t += dt
            history.append(q.copy()); diff=np.linalg.norm(q-q_old)/self.num_vertices
            _,lengths=self._cable_geometry_cpu(q); error=np.max(np.maximum(lengths-target,0))
            settled=settled+1 if diff<tol and error<tol else 0
            if show_info:
                print(f"t={t:.3f}, diff={diff:.7e}, constraint={error:.7e}")
            if settled >= 10:
                break
        return history,q.reshape(-1,3),tension

    def FKD_static(self, target_cable_length, starting_vertices, tol=1e-4,
                   max_iter=300, show_info=False):
        """Hybrid free-carpet static solve with the reference active-set KKT logic."""
        target=np.asarray(self._numpy(target_cable_length),dtype=float).reshape(-1)
        if target.size!=self.nCable or np.any(~np.isfinite(target)) or np.any(target<=0):
            raise ValueError(f"target_cable_length must contain {self.nCable} finite positive values")
        q=np.asarray(self._numpy(starting_vertices),dtype=float).reshape(-1).copy()
        history=[q.copy()]; tension=np.zeros(self.nCable); ndof=q.size
        constraint_tol=min(tol,1e-8); tension_tol=1e-10
        for iteration in range(max_iter):
            old=q.copy(); K,f0=self._hybrid_global_system(q)
            load=f0+np.asarray(self.gravity_vec)
            _,lengths=self._cable_geometry_cpu(q); phi=target-lengths
            H_all=-self._cable_jacobian_cpu(q)
            active=set(np.flatnonzero((phi<=constraint_tol)|(tension>tension_tol)).tolist())
            blocked=set(); candidate=q.copy(); candidate_tension=np.zeros(self.nCable)
            for _ in range(2*self.nCable+2):
                ids=np.array(sorted(active),dtype=int)
                if ids.size==0:
                    raise np.linalg.LinAlgError(
                        "Static equilibrium needs active cables to remove free-body modes"
                    )
                H=H_all[ids]
                kkt=np.block([[K,-H.T],[H,np.zeros((ids.size,ids.size))]])
                rhs=np.concatenate((load,H@q-phi[ids]))
                solution=np.linalg.lstsq(kkt,rhs,rcond=1e-10)[0]
                candidate,lam=solution[:ndof],solution[ndof:]
                negative=np.flatnonzero(lam < -tension_tol)
                if negative.size:
                    cable=int(ids[negative[np.argmin(lam[negative])]])
                    active.remove(cable); blocked.add(cable); continue
                _,candidate_lengths=self._cable_geometry_cpu(candidate)
                violation=candidate_lengths-target
                inactive=[i for i in range(self.nCable) if i not in active and i not in blocked]
                if inactive:
                    worst=max(inactive,key=lambda i:violation[i])
                    if violation[worst]>constraint_tol:
                        active.add(worst); continue
                candidate_tension[ids]=np.maximum(lam,0); break
            else:
                raise RuntimeError("Cable active-set solve did not converge")
            q,tension=candidate,candidate_tension; history.append(q.copy())
            _,final_lengths=self._cable_geometry_cpu(q)
            error=np.max(np.maximum(final_lengths-target,0))
            H_final=-self._cable_jacobian_cpu(q)
            residual=K@q-load-H_final.T@tension
            relative=np.linalg.norm(residual)/max(np.linalg.norm(load),1.0)
            diff=np.linalg.norm(q-old)/np.sqrt(ndof)
            if show_info:
                print(f"static iteration {iteration+1}: diff={diff:.7e}, "
                      f"constraint={error:.7e}, residual={relative:.7e}")
            if diff<tol and error<tol and relative<tol: break
        return history,q.reshape(-1,3),tension

    def IKD_single(self, target_ee_pos, starting_vertices, tol=1e-3,
                   max_iter=500, show_info=False, initial_guess=True):
        """Inverse kinematics using the reference Shape-Up sensitivity update.

        The Shape-Up cable Jacobian supplies the descent direction, while each
        cable-length command is equilibrated by the hybrid static FEM solver.
        Slack cables are prevented from being lengthened in a direction that
        would require them to push.
        """
        target=np.asarray(self._numpy(target_ee_pos),dtype=float)
        expected=(len(self.ee_idx),3)
        if target.shape==(3,) and len(self.ee_idx)==1:
            target=target.reshape(1,3)
        if target.shape!=expected or np.any(~np.isfinite(target)):
            raise ValueError(f"target_ee_pos must have shape {expected} and be finite")
        q=np.asarray(self._numpy(starting_vertices),dtype=float).reshape(-1).copy()
        if q.size!=3*self.num_vertices:
            raise ValueError("starting_vertices must be an (n,3) array or a 3n vector")
        if not np.isfinite(tol) or tol<=0:
            raise ValueError("tol must be a finite positive value")
        if not isinstance(max_iter,(int,np.integer)) or max_iter<=0:
            raise ValueError("max_iter must be a positive integer")

        ee_dofs=(3*np.asarray(self.ee_idx)[:,None]+np.arange(3)).reshape(-1)

        def errors(q_value):
            ee=q_value.reshape(self.num_vertices,3)[np.asarray(self.ee_idx)]
            residual=ee-target
            objective=0.5*np.dot(residual.reshape(-1),residual.reshape(-1))
            cartesian=np.linalg.norm(residual,axis=1).mean()
            return objective,cartesian,residual

        if initial_guess:
            guess=np.asarray(self.get_fixedEE_guess_vertices(target),dtype=float)
            _,guess_lengths=self._cable_geometry_cpu(guess)
            _,guess,cable_tension=self.FKD_time(
                guess_lengths, 5, guess,tol=1e-4,show_info=False
            )
            q=guess.reshape(-1)
        else:
            cable_tension=np.zeros(self.nCable)

        final_Q_list=[q.copy()]
        step=0.1
        for iteration in range(max_iter):
            objective,_,residual=errors(q)
            # get_CG_Jacobian returns dq/dl for physical and ghost DOFs.
            shape_jac=np.asarray(self.get_CG_Jacobian(q),dtype=float)
            ee_jac=shape_jac[ee_dofs,:]
            gradient=ee_jac.T@residual.reshape(-1)
            _,current_length=self._cable_geometry_cpu(q)
            command_delta=-step*gradient
            if iteration>0:
                slack=(cable_tension<1e-5)&(gradient<0)
                command_delta[slack]=0.0
            command_length=current_length+command_delta
            # Cable lengths are physical positive quantities. A bad IK step
            # should fail locally rather than entering the static active set.
            command_length=np.maximum(command_length,1e-8)
            _,vertices,cable_tension=self.FKD_time(
                command_length, 5, q,tol=1e-4,show_info=False
            )
            q=vertices.reshape(-1)
            final_Q_list.append(q.copy())
            objective,cartesian,_=errors(q)
            if show_info:
                print(
                    f"IK iteration {iteration+1}: objective={objective:.7e}, "
                    f"mean EE error={cartesian:.7e}, step={step:.3g}, "
                    f"gradient={np.round(gradient,5)}, "
                    f"tension={np.round(cable_tension,5)}, "
                    f"cable delta={np.round(command_delta,5)}"
                )
            if objective<tol:
                if show_info:
                    print(f"IK converged in {iteration+1} iterations")
                break

        _,final_length=self._cable_geometry_cpu(q)
        return final_length,q.reshape(self.num_vertices,3),final_Q_list

    def IKD_single_minimize(self, ee_target_pos, starting_vertices, max_iter=50,
                            tol=1e-4, show_info=True, initial_guess=False):
        """Minimize end-effector error over commanded cable lengths.

        SLSQP estimates the gradient with three-point finite differences. Every
        distinct objective evaluation runs ``FKD_time`` from the same settled
        starting configuration, keeping the numerical objective deterministic.
        """
        target=np.asarray(self._numpy(ee_target_pos),dtype=float)
        expected=(len(self.ee_idx),3)
        if target.shape==(3,) and len(self.ee_idx)==1:
            target=target.reshape(1,3)
        if target.shape!=expected:
            raise ValueError(f"ee_target_pos must have shape {expected}")
        if np.any(~np.isfinite(target)):
            raise ValueError("ee_target_pos must contain finite values")
        if not np.isfinite(tol) or tol<=0:
            raise ValueError("tol must be a finite positive value")
        if not isinstance(max_iter,(int,np.integer)) or max_iter<=0:
            raise ValueError("max_iter must be a positive integer")

        vertices=np.asarray(self._numpy(starting_vertices),dtype=float)
        if vertices.size!=3*self.num_vertices:
            raise ValueError("starting_vertices must be an (n,3) array or a 3n vector")
        vertices=vertices.reshape(self.num_vertices,3).copy()
        if np.any(~np.isfinite(vertices)):
            raise ValueError("starting_vertices must contain finite values")

        if initial_guess:
            vertices=np.asarray(self.get_fixedEE_guess_vertices(target),dtype=float)
            _,guess_length=self._cable_geometry_cpu(vertices)
            _,vertices,_=self.FKD_time(
                guess_length,10.0,vertices,tol=5e-5,h=0.01,show_info=False
            )

        # This settled state remains fixed across evaluations. Updating it would
        # make finite-difference samples depend on evaluation order.
        evaluation_start=vertices.copy()
        _,initial_length=self._cable_geometry_cpu(evaluation_start)
        evaluation_count=0
        cached_length=None
        cached_vertices=None
        cached_ee=None
        final_Q_list=[evaluation_start.reshape(-1).copy()]

        def forward_kinematics(cable_length):
            nonlocal evaluation_count,cached_length,cached_vertices,cached_ee
            cable_length=np.asarray(cable_length,dtype=float).reshape(-1)
            if cached_length is not None and np.array_equal(cable_length,cached_length):
                return cached_vertices,cached_ee
            _,deformed,_=self.FKD_time(
                cable_length,5.0,evaluation_start,tol=1e-4,h=0.01,
                show_info=False
            )
            ee=np.asarray(deformed,dtype=float)[np.asarray(self.ee_idx)]
            evaluation_count+=1
            cached_length=cable_length.copy()
            cached_vertices=np.asarray(deformed,dtype=float).copy()
            cached_ee=ee.copy()
            if show_info:
                objective=0.5*np.sum((ee-target)**2)
                # print(f"Minimize FK evaluation {evaluation_count}: "
                #       f"objective={objective:.6e}",flush=True)
            return cached_vertices,cached_ee

        def objective_function(cable_length):
            _,ee=forward_kinematics(cable_length)
            residual=(ee-target).reshape(-1)
            return 0.5*np.dot(residual,residual)

        def save_iteration(cable_length):
            deformed,_=forward_kinematics(cable_length)
            final_Q_list.append(deformed.reshape(-1).copy())

        result=minimize(
            objective_function,
            initial_length,
            method="SLSQP",
            bounds=[(np.finfo(float).eps,None)]*self.nCable,
            jac="3-point",
            callback=save_iteration,
            tol=tol,
            options={
                "ftol":max(0.5*tol**2,np.finfo(float).eps),
                "finite_diff_rel_step":1e-3,
                "maxiter":max_iter,
                "disp":show_info,
            },
        )

        final_vertices,final_ee=forward_kinematics(result.x)
        final_q=final_vertices.reshape(-1)
        if not np.array_equal(final_Q_list[-1],final_q):
            final_Q_list.append(final_q.copy())
        _,final_length=self._cable_geometry_cpu(final_vertices)
        final_error=np.linalg.norm(final_ee-target,axis=1).mean()
        if show_info:
            print(f"IK minimize evaluations: {evaluation_count}, "
                  f"success={result.success}, final error={final_error:.7e}")
            if not result.success:
                print(f"Optimizer message: {result.message}")
        return final_length,final_vertices,final_Q_list

    # Compatibility names used by early versions of this module.
    def get_jacobian_CG(self, Q):
        return self.get_CG_Jacobian(Q)

    def get_jacobian_FD(self, Q, delta=1e-3):
        q = np.asarray(Q, dtype=float).reshape(-1)
        lengths = np.asarray(self.get_cable_length_bary(q))
        jac = np.empty((3*len(self.ee_idx), self.nCable))
        for i in range(self.nCable):
            command = lengths.copy(); command[i] += delta
            _, plus, _ = self.FKD_time(command, 0.1, q, tol=1e-5)
            command[i] -= 2*delta
            _, minus, _ = self.FKD_time(command, 0.1, q, tol=1e-5)
            jac[:, i] = ((plus[self.ee_idx]-minus[self.ee_idx])/(2*delta)).reshape(-1)
        return jac

    def get_jacobian_CG_FD(self, Q, delta=1e-3):
        """Finite-difference Jacobian of Shape-Up's CG solve."""
        q = np.asarray(Q, dtype=float).reshape(-1)
        lengths = np.asarray(self.get_cable_length_bary(q))
        jac = np.empty((3*len(self.ee_idx), self.nCable))
        for i in range(self.nCable):
            command = lengths.copy(); command[i] += delta
            plus = self.deform_CG(q, command, max_iter=10, tol=1e-5)
            command[i] -= 2*delta
            minus = self.deform_CG(q, command, max_iter=10, tol=1e-5)
            jac[:, i] = ((plus[self.ee_idx]-minus[self.ee_idx])/(2*delta)).reshape(-1)
        return jac

    # ------------------------------------------------------------------
    # Visualization (PyVista is CPU based, so tensors cross the boundary here)
    def _visual_vertices(self, vertices):
        vertices = self._numpy(vertices)
        if vertices.size != 3*self.num_vertices:
            raise ValueError(
                f"vertices must contain {self.num_vertices} xyz positions"
            )
        return np.asarray(vertices, dtype=float).reshape(self.num_vertices, 3)

    def _visual_faces(self):
        return np.column_stack((
            np.full(self.num_triangles, 3, dtype=np.int64),
            np.asarray(self.mesh_triangles, dtype=np.int64),
        ))

    def _add_cables(self, plotter, pp_locations, color="blue", width=2):
        pulley = np.asarray(self.pulley_location)
        for pp, anchor in zip(pp_locations, pulley):
            plotter.add_lines(np.vstack((pp, anchor)), color=color, width=width)

    def visualize_vert(self, vertices):
        vertices = self._visual_vertices(vertices)
        pp = self._numpy(self._pp_location_torch(self._tensor(vertices)))
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(vertices, self._visual_faces()),
                         color="lightgray", show_edges=True)
        plotter.add_points(pp, color="blue", point_size=10,
                           render_points_as_spheres=True, label="Pull points")
        plotter.add_points(self.pulley_location, color="cyan", point_size=10,
                           render_points_as_spheres=True, label="Pulleys")
        plotter.add_points(vertices[np.asarray(self.ee_idx)], color="red",
                           point_size=10, render_points_as_spheres=True,
                           label="End Effectors")
        self._add_cables(plotter, pp)
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        return plotter.show()

    def visualize_vert_paper(self, vertices):
        vertices = self._visual_vertices(vertices)
        pp = self._numpy(self._pp_location_torch(self._tensor(vertices)))
        faces = self._visual_faces()
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(np.asarray(self.vertices), faces),
                         color="lightgray", show_edges=False, opacity=0.5,
                         label="Initial surface")
        plotter.add_mesh(pv.PolyData(vertices, faces), color="blue",
                         show_edges=True, label="Deformed surface")
        plotter.add_points(pp, color="blue", point_size=10,
                           render_points_as_spheres=True, label="Pull points")
        plotter.add_points(self.pulley_location, color="cyan", point_size=10,
                           render_points_as_spheres=True, label="Pulleys")
        plotter.add_points(vertices[np.asarray(self.ee_idx)], color="red",
                           point_size=10, render_points_as_spheres=True,
                           label="End Effectors")
        self._add_cables(plotter, pp)
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        return plotter.show()

    def visualize_IKD_result(self, target_ee_pos, vertices):
        vertices = self._visual_vertices(vertices)
        target = np.asarray(self._numpy(target_ee_pos), dtype=float).reshape(-1, 3)
        pp = self._numpy(self._pp_location_torch(self._tensor(vertices)))
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(vertices, self._visual_faces()),
                         color="lightgray", show_edges=True)
        plotter.add_points(vertices[np.asarray(self.ee_idx)], color="red",
                           point_size=10, render_points_as_spheres=True,
                           label="End Effectors")
        plotter.add_points(target, color="green", point_size=10,
                           render_points_as_spheres=True, label="Target EE Pos")
        self._add_cables(plotter, pp)
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        return plotter.show()

    def visualize_fb_surface(self, vertices):
        vertices = self._visual_vertices(vertices)
        fb = np.asarray(self.get_fb_surface(vertices))
        faces = self._visual_faces()
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(vertices, faces), color="lightgray",
                         show_edges=True, opacity=0.35, label="Input Surface")
        plotter.add_mesh(pv.PolyData(fb, faces), color="lightblue",
                         show_edges=True, opacity=0.95, label="FB Mid-Surface")
        plotter.add_points(fb[np.asarray(self.ee_idx)], color="magenta",
                           point_size=10, render_points_as_spheres=True,
                           label="FB EE Vertices")
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        return plotter.show()

    def visualize_vert_w_fb(self, vertices, fb_pts):
        vertices = self._visual_vertices(vertices)
        fb = np.asarray(self.get_fb_surface(vertices))
        points = np.asarray(self._numpy(fb_pts), dtype=float).reshape(-1, 3)
        plotter = pv.Plotter()
        plotter.add_mesh(pv.PolyData(fb, self._visual_faces()), color="lightblue",
                         show_edges=True, opacity=0.95, label="FB Mid-Surface")
        plotter.add_points(vertices[np.asarray(self.ee_idx)], color="red",
                           point_size=10, render_points_as_spheres=True,
                           label="End Effectors")
        plotter.add_points(points, color="yellow", point_size=8,
                           render_points_as_spheres=True, label="FB Points")
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        return plotter.show()

    def _replay(self, Q_list, file_path, framerate, window_size,
                target_ee_pos=None):
        if len(Q_list) == 0:
            raise ValueError("Q_list must contain at least one frame")
        vertices = self._visual_vertices(Q_list[0])
        pp = self._numpy(self._pp_location_torch(self._tensor(vertices)))
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        surface = pv.PolyData(vertices.copy(), self._visual_faces())
        plotter.add_mesh(surface, color="lightgray", show_edges=True)
        pp_cloud = pv.PolyData(pp.copy())
        ee_cloud = pv.PolyData(vertices[np.asarray(self.ee_idx)].copy())
        plotter.add_mesh(pp_cloud, color="blue", point_size=10,
                         render_points_as_spheres=True, label="Pull points")
        plotter.add_points(self.pulley_location, color="cyan", point_size=10,
                           render_points_as_spheres=True, label="Pulleys")
        plotter.add_mesh(ee_cloud, color="red", point_size=10,
                         render_points_as_spheres=True, label="End Effectors")
        if target_ee_pos is not None:
            target = np.asarray(self._numpy(target_ee_pos), dtype=float).reshape(-1, 3)
            plotter.add_points(target, color="green", point_size=10,
                               render_points_as_spheres=True, label="Target EE Pos")
        cable_points = np.empty((2*self.nCable, 3), dtype=float)
        cable_points[0::2], cable_points[1::2] = pp, self.pulley_location
        cables = pv.PolyData()
        cables.points = cable_points.copy()
        cables.lines = np.array([[2, 2*i, 2*i+1] for i in range(self.nCable)],
                                dtype=np.int64).reshape(-1)
        plotter.add_mesh(cables, color="blue", line_width=2)
        plotter.show_grid(); plotter.show_axes(); plotter.add_legend()
        plotter.open_movie(str(file_path), framerate=framerate)
        for frame in Q_list:
            vertices = self._visual_vertices(frame)
            pp = self._numpy(self._pp_location_torch(self._tensor(vertices)))
            surface.points = vertices.copy()
            pp_cloud.points = pp.copy()
            ee_cloud.points = vertices[np.asarray(self.ee_idx)].copy()
            cable_points[0::2] = pp
            cables.points = cable_points.copy()
            plotter.write_frame()
        plotter.close()
        return str(file_path)

    def replay_Q_list(self, Q_list, filePath="./flying_carpet_FKD.mp4",
                      framerate=10, window_size=(1024, 768)):
        return self._replay(Q_list, filePath, framerate, window_size)

    def replay_IKD_Q_list(self, target_EE_pos, Q_list,
                          filePath="./flying_carpet_IKD.mp4",
                          window_size=(1024, 768), framerate=1):
        return self._replay(Q_list, filePath, framerate, window_size,
                            target_ee_pos=target_EE_pos)

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet_torch(description_file)
    print("flying carpet running on device: ", flying_carpet.device)
    icl = flying_carpet.initial_cable_length
    shortened_length = 0.04
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    print("Target cable length shortened for " , shortened_length,  ", tcl=", tcl)
    time_start = time.time()
    # Q_list, vert_length, cable_tension = flying_carpet.FKD_time(tcl,5, flying_carpet.vertices, tol=1e-4, show_info=False)
    vert_length = flying_carpet.deform_CG(flying_carpet.vertices, tcl, max_iter=1000, tol=1e-8, show_info=False)
    print("FKD_time finished in ", time.time()-time_start, " seconds")
    flying_carpet.visualize_vert(vert_length)
