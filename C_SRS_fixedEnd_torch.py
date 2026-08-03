"""GPU accelerated fixed-end C-SRS simulation.

Public methods mirror ``C_SRS_fixedEnd`` and use numpy at their boundaries.
Rotations, Shape-Up, FEM assembly, analytic cable Jacobians, and iterative
linear solves remain Torch operations on ``device``.
"""
import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from utilities import projected_gauss_seidel_lcp
import time
from C_SRS_fixedEnd import C_SRS_fixedEnd


class C_SRS_fixedEnd_torch(C_SRS_fixedEnd):
    def __init__(self, description_file, device=None, dtype=torch.float64):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.torch_dtype = dtype
        super().__init__(description_file)
        self._cache_torch()

    def _t(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=dtype or self.torch_dtype)

    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    def _cache_torch(self):
        ld = torch.long
        self.vertices_t = self._t(np.asarray(self.vertices).copy())
        self.tri_t = self._t(self.mesh_triangles, ld)
        self.rf_t = self._t(self.mesh_RF_triangles, ld)
        self.rf_valid_t = self.rf_t >= 0
        self.rf_safe_t = self.rf_t.clamp_min(0)
        self.stiffness_t = self._t(np.stack(self.stiffness_matrices))
        self.qe0_t = self._t(np.stack(self.qe0_list))
        self.initial_tri_t = self._t(np.stack(self.initial_tri_SK_list))
        self.initial_cable_t = self._t(np.stack(self.initial_cable_vec))
        self.initial_ghost_t = self._t(np.stack(self.initial_ghost_shape_list))
        self.pulley_t = self._t(self.pulley_location)
        self.pp_tri_t = self._t(self.pp_bary_tri_idx, ld)
        self.pp_bary_t = self._t(self.pp_bary_coords)
        self.pp_offset_t = self._t(self.pp_bary_offsets)
        self.N33_t, self.N44_t = self._t(self.N33), self._t(self.N44)
        self.mass_t, self.W_t = self._t(self.mass_matrix), self._t(self.W_mat)
        self.gravity_t = self._t(self.gravity_vec)
        self.moving_dof_t = self._t(self.moving_dof_idx, ld)
        fixed_dof = np.setdiff1d(np.arange(3*self.num_vertices), self.moving_dof_idx)
        self.fixed_dof_t = self._t(fixed_dof, ld)
        self.q0_t = self.vertices_t.reshape(-1)
        local = torch.arange(3, device=self.device)
        self.rf_dofs_t = (3*self.rf_safe_t[..., None]+local).reshape(-1, 18)
        self.rf_dof_valid_t = self.rf_valid_t[..., None].expand(-1, -1, 3).reshape(-1, 18)

    # ---------- batched rotations ----------
    @staticmethod
    def _best_fit_rotation_torch(current, reference):
        u, _, vh = torch.linalg.svd(current.transpose(-1, -2)@reference)
        fix = torch.eye(3, device=current.device, dtype=current.dtype).expand(
            current.shape[:-2]+(3, 3)).clone()
        fix[..., 2, 2] = torch.linalg.det(u@vh)
        return u@fix@vh

    def _rotations_torch(self, q):
        xyz = q.reshape(self.num_vertices, 3)[self.tri_t]
        return self._best_fit_rotation_torch(self.N33_t@xyz, self.initial_tri_t)

    def _R18_torch(self, q):
        R = self._rotations_torch(q)
        eye6 = torch.eye(6, device=self.device, dtype=self.torch_dtype)
        return R, torch.einsum("eab,jk->ejakb", R, eye6).reshape(-1, 18, 18)

    @staticmethod
    def _skew(v):
        out = torch.zeros(v.shape[:-1]+(3, 3), device=v.device, dtype=v.dtype)
        out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
        out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
        out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
        return out

    def _vector_rotations(self, initial, current, eps=1e-12):
        a = initial/torch.linalg.vector_norm(initial, dim=1, keepdim=True).clamp_min(eps)
        b = current/torch.linalg.vector_norm(current, dim=1, keepdim=True).clamp_min(eps)
        v, c = torch.linalg.cross(a, b, dim=1), (a*b).sum(1).clamp(-1, 1)
        K = self._skew(v)
        eye = torch.eye(3, device=self.device, dtype=self.torch_dtype).expand_as(K)
        R = eye+K+(K@K)/(1+c).clamp_min(eps)[:, None, None]
        anti = c < -1+1e-8
        if anti.any():
            aa = a[anti]
            basis = torch.eye(3, device=self.device, dtype=self.torch_dtype)[aa.abs().argmin(1)]
            axis = torch.linalg.cross(aa, basis, dim=1)
            axis /= torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(eps)
            R[anti] = 2*axis[:, :, None]*axis[:, None, :]-eye[anti]
        return R

    def get_rotation_tri(self, vertices):
        return self._np(self._rotations_torch(self._t(vertices).reshape(-1)))

    def get_R_list(self, vertices):
        R, R18 = self._R18_torch(self._t(vertices).reshape(-1))
        return self._np(R), self._np(R18)

    # ---------- global and reduced FEM assembly ----------
    def _assemble_global(self, R18):
        Ke = R18@self.stiffness_t@R18.transpose(1, 2)
        f0e = (R18@self.stiffness_t@self.qe0_t[..., None]).squeeze(-1)
        ndof = 3*self.num_vertices
        K = torch.zeros(ndof*ndof, device=self.device, dtype=self.torch_dtype)
        f = torch.zeros(ndof, device=self.device, dtype=self.torch_dtype)
        row = self.rf_dofs_t[:, :, None].expand(-1, 18, 18)
        col = self.rf_dofs_t[:, None, :].expand(-1, 18, 18)
        mask = self.rf_dof_valid_t[:, :, None]&self.rf_dof_valid_t[:, None, :]
        K.index_add_(0, row[mask]*ndof+col[mask], Ke[mask])
        valid = self.rf_dof_valid_t
        f.index_add_(0, self.rf_dofs_t[valid], f0e[valid])
        return K.reshape(ndof, ndof), f

    def _assemble_reduced(self, R18):
        K, f = self._assemble_global(R18)
        m, fixed = self.moving_dof_t, self.fixed_dof_t
        Kmm = K.index_select(0, m).index_select(1, m)
        f0m = f[m]
        fixed_term = -(K.index_select(0, m).index_select(1, fixed)@self.q0_t[fixed])
        return Kmm, f0m, fixed_term

    def assemble_K(self, R_list_1818):
        K, f = self._assemble_global(self._t(R_list_1818))
        return self._np(K), self._np(f)

    def assemble_K_tilde(self, R_list_1818):
        values = self._assemble_reduced(self._t(R_list_1818))
        return tuple(self._np(x) for x in values)

    # ---------- analytic barycentric cable geometry ----------
    def _pp_torch(self, q):
        xyz = q.reshape(self.num_vertices, 3)[self.tri_t[self.pp_tri_t]]
        n = torch.linalg.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0], dim=1)
        n /= torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp_min(1e-12)
        return (self.pp_bary_t[..., None]*xyz).sum(1)+self.pp_offset_t[:, None]*n

    def _lengths_torch(self, q):
        return torch.linalg.vector_norm(self._pp_torch(q)-self.pulley_t, dim=1)

    def _cable_jacobian_torch(self, q):
        vertices = q.reshape(self.num_vertices, 3)
        ids = self.tri_t[self.pp_tri_t]
        xyz = vertices[ids]
        e1, e2 = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
        n = torch.linalg.cross(e1, e2, dim=1)
        area2 = torch.linalg.vector_norm(n, dim=1).clamp_min(1e-12)
        t3 = n/area2[:, None]
        eye = torch.eye(3, device=self.device, dtype=self.torch_dtype)
        P = eye[None]-t3[:, :, None]*t3[:, None, :]
        Gn = torch.stack((self._skew(e2-e1), -self._skew(e2), self._skew(e1)), 1)
        Gt = torch.einsum("nij,nkjl->nkil", P, Gn)/area2[:, None, None, None]
        dpp = self.pp_bary_t[:, :, None, None]*eye+self.pp_offset_t[:, None, None, None]*Gt
        pp = (self.pp_bary_t[..., None]*xyz).sum(1)+self.pp_offset_t[:, None]*t3
        direction = pp-self.pulley_t
        direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
        blocks = torch.einsum("ni,nkij->nkj", direction, dpp)
        J = torch.zeros((self.nCable, 3*self.num_vertices), device=self.device,
                        dtype=self.torch_dtype)
        cols = (3*ids[:, :, None]+torch.arange(3, device=self.device)).reshape(self.nCable, 9)
        J[torch.arange(self.nCable, device=self.device)[:, None], cols] = blocks.reshape(self.nCable, 9)
        return J

    def get_pp_location_bary(self, vertices):
        if not hasattr(self, "pp_bary_t"):
            return super().get_pp_location_bary(vertices)
        return self._np(self._pp_torch(self._t(vertices).reshape(-1)))

    def get_cable_length_bary(self, vertices):
        if not hasattr(self, "pp_bary_t"):
            return super().get_cable_length_bary(vertices)
        return self._np(self._lengths_torch(self._t(vertices).reshape(-1)))

    def get_cable_Jacobian_bary(self, vertices):
        if not hasattr(self, "pp_bary_t"):
            return super().get_cable_Jacobian_bary(vertices)
        return self._np(self._cable_jacobian_torch(self._t(vertices).reshape(-1)))

    # ---------- GPU iterative solvers ----------
    @staticmethod
    def _cg(A, b, x=None, tol=1e-10, max_iter=None):
        """Preconditioned conjugate gradient for the fixed-end SPD system."""
        x = torch.zeros_like(b) if x is None else x.clone()
        max_iter = max_iter or min(4*b.numel(), 2000)
        r = b-A@x
        inv_diag = torch.diagonal(A).abs().clamp_min(torch.finfo(A.dtype).eps).reciprocal()
        z, p = inv_diag*r, inv_diag*r
        rz = r@z
        target = tol*max(torch.linalg.vector_norm(b).item(), 1.0)
        for _ in range(max_iter):
            Ap = A@p
            alpha = rz/(p@Ap).clamp_min(torch.finfo(A.dtype).eps)
            x, r = x+alpha*p, r-alpha*Ap
            if torch.linalg.vector_norm(r).item() <= target:
                break
            z = inv_diag*r
            rz_new = r@z
            p, rz = z+(rz_new/rz.clamp_min(torch.finfo(A.dtype).eps))*p, rz_new
        return x

    @staticmethod
    def _pgs(M, q, tol=1e-8, max_iter=100):
        x = torch.zeros_like(q)
        diag = torch.diagonal(M).clamp_min(torch.finfo(M.dtype).eps)
        for _ in range(max_iter):
            old = x.clone()
            for i in range(q.numel()):
                x[i] = torch.clamp(x[i]-(M[i]@x+q[i])/diag[i], min=0)
            if torch.max(torch.abs(x-old)).item() < tol:
                break
        return x

    def _moving_to_full(self, qm):
        q = self.q0_t.clone(); q[self.moving_dof_t] = qm
        return q

    def _moving_to_full_numpy(self, qm):
        """CPU counterpart used by the hybrid fixed-end static solvers."""
        q = np.asarray(self.vertices, dtype=float).reshape(-1).copy()
        q[np.asarray(self.moving_dof_idx, dtype=int)] = qm
        return q

    def _hybrid_reduced_system(self, q):
        """Rotate/assemble on GPU, then transfer one reduced system to CPU."""
        q_t = self._t(q).reshape(-1)
        _, R18 = self._R18_torch(q_t)
        K_t, f_t, fixed_t = self._assemble_reduced(R18)
        # Exactly one synchronization/transfer point per nonlinear iteration.
        return self._np(K_t), self._np(f_t), self._np(fixed_t)

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
        J=np.zeros((self.nCable,3*self.num_vertices))
        eye=np.eye(3)
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

    # ---------- Shape-Up ----------
    def assemble_CG_matrices(self):
        mem, bend, cable = 9*self.num_triangles, 3*len(self.bending_ele_idx), 3*self.nCable
        rows, fullcols = mem+bend+cable+12*self.nCable, 3*self.num_vertices+3*self.nCable
        A0 = torch.zeros((rows, fullcols), device=self.device, dtype=self.torch_dtype)
        eye = torch.eye(3, device=self.device, dtype=self.torch_dtype)
        for i, tri in enumerate(np.asarray(self.mesh_triangles)):
            w = float(self.mem_weight_list[i])
            for j in range(3):
                for jp, v in enumerate(tri):
                    A0[9*i+3*j:9*i+3*j+3, 3*v:3*v+3] = w*((2/3) if j==jp else -1/3)*eye
        for i, vertices in enumerate(np.asarray(self.bending_ele_idx)):
            for v, c in zip(vertices, self.bending_ele_param[i]):
                A0[mem+3*i:mem+3*i+3, 3*v:3*v+3] = float(self.bending_weight_list[i])*float(c)*eye
        maxw = max(float(np.max(self.mem_weight_list)), float(np.max(self.bending_weight_list)))
        self.weight_cable = self.weight_ghost = 50*maxw
        N12 = self._t(self.N1212)
        for i in range(self.nCable):
            A0[mem+bend+3*i:mem+bend+3*i+3,
               3*self.num_vertices+3*i:3*self.num_vertices+3*i+3] = self.weight_cable*eye
            r = mem+bend+cable+12*i
            ids = list(self.mesh_triangles[self.pp_bary_tri_idx[i]])+[self.num_vertices+i]
            for j in range(4):
                for k in range(4):
                    A0[r+3*j:r+3*j+3, 3*ids[k]:3*ids[k]+3] += self.weight_ghost*N12[3*j:3*j+3, 3*k:3*k+3]
        move_cols = torch.cat((self._t(self.moving_dof_idx, torch.long),
                               torch.arange(3*self.num_vertices, fullcols, device=self.device)))
        fixed = self._t(np.setdiff1d(np.arange(3*self.num_vertices), self.moving_dof_idx), torch.long)
        self.A_cg_t = A0[:, move_cols]
        self.cg_fixed_rhs_t = -(A0[:, fixed]@self._t(self.vertices.reshape(-1))[fixed])
        self.ATA_cg_t, self.AT_cg_t = self.A_cg_t.T@self.A_cg_t, self.A_cg_t.T
        self.matA_all = self._np(self.A_cg_t)
        self.vecB_2_add = self._np(self.cg_fixed_rhs_t)

    def _shapeup_b(self, vertices, ghosts, target):
        rows = self.A_cg_t.shape[0]
        b = torch.zeros(rows, device=self.device, dtype=self.torch_dtype)
        R = self._rotations_torch(vertices.reshape(-1))
        b[:9*self.num_triangles] = ((R@self.initial_tri_t.transpose(1,2)).transpose(1,2)*
                                     self._t(self.mem_weight_list)[:,None,None]).reshape(-1)
        Rc = self._vector_rotations(self.initial_cable_t, ghosts-self.pulley_t)
        cable = (Rc@self.initial_cable_t[...,None]).squeeze(-1)
        start = rows-15*self.nCable
        b[start:start+3*self.nCable] = self.weight_cable*(cable*target[:,None]+self.pulley_t).reshape(-1)
        ids = self.tri_t[self.pp_tri_t]
        shape = self.N44_t@torch.cat((vertices[ids], ghosts[:,None,:]),1)
        Rg = self._best_fit_rotation_torch(shape, self.initial_ghost_t)
        b[-12*self.nCable:] = self.weight_ghost*((Rg@self.initial_ghost_t.transpose(1,2)).transpose(1,2)).reshape(-1)
        return b

    def deform_CG(self, target_cable_length, vertices, max_iter=100, tol=1e-5, show_info=False):
        q = self._t(vertices).reshape(-1); target = self._t(target_cable_length).reshape(-1)
        ghosts, last = self._pp_torch(q), q.clone()
        for it in range(max_iter):
            b = self._shapeup_b(q.reshape(-1,3), ghosts, target)+self.cg_fixed_rhs_t
            sol = self._cg(self.ATA_cg_t, self.AT_cg_t@b, tol=min(tol,1e-10))
            q = self._moving_to_full(sol[:3*self.nMoving])
            ghosts = sol[3*self.nMoving:].reshape(self.nCable,3)
            diff = torch.linalg.vector_norm(q-last)/(3*self.num_vertices); last=q.clone()
            if show_info: print(f"Shape-Up iteration {it+1}: diff={diff.item():.7e}")
            if diff.item()<tol: break
        return self._np(q.reshape(-1,3))

    # ---------- fixed-end static equilibrium ----------
    def FKD_static(self, starting_vertices, cable_tension, tol=1e-6,
                   show_info=False, max_iter=500, linear_tol=1e-10):
        q = np.asarray(self._np(starting_vertices), dtype=float).reshape(-1).copy()
        tension = np.asarray(self._np(cable_tension), dtype=float).reshape(-1)
        if tension.size!=self.nCable or np.any(~np.isfinite(tension)) or np.any(tension<0):
            raise ValueError(f"cable_tension must contain {self.nCable} non-negative values")
        moving = np.asarray(self.moving_dof_idx, dtype=int)
        history=[q.copy()]; gravity=np.asarray(self.gravity_vec)[moving]
        qm = q[moving]
        for it in range(max_iter):
            K,f,fixed=self._hybrid_reduced_system(q)
            # Six cable rows are too small to benefit from CUDA. Keeping their
            # analytic evaluation on CPU also avoids another device round trip.
            H=-self._cable_jacobian_cpu(q)[:,moving]
            rhs=f+gravity+fixed+H.T@tension
            lu,pivots=lu_factor(K,check_finite=False)
            qm_next=lu_solve((lu,pivots),rhs,check_finite=False)
            q_next=self._moving_to_full_numpy(qm_next)
            diff=np.linalg.norm(q_next-q)/(3*self.nMoving)
            q,qm=q_next,qm_next; history.append(q.copy())
            if show_info:
                residual=np.linalg.norm(K@qm-rhs)
                print(f"static iteration {it+1}: diff={diff:.7e}, residual={residual:.7e}")
            if diff<tol: break
        return history

    def FKD_static_length(self, target_cable_length, starting_vertices, tol=1e-6,
                          show_info=False, max_iter=500, linear_tol=1e-10):
        q=np.asarray(self._np(starting_vertices),dtype=float).reshape(-1).copy()
        target=np.asarray(self._np(target_cable_length),dtype=float).reshape(-1)
        if target.size!=self.nCable or np.any(~np.isfinite(target)) or np.any(target<=0):
            raise ValueError(f"target_cable_length must contain {self.nCable} finite positive values")
        moving=np.asarray(self.moving_dof_idx,dtype=int)
        history=[q.copy()]; gravity=np.asarray(self.gravity_vec)[moving]
        tension=np.zeros(self.nCable); qm=q[moving]
        for it in range(max_iter):
            K,f,fixed=self._hybrid_reduced_system(q)
            lu,pivots=lu_factor(K,check_finite=False)
            free_rhs=f+gravity+fixed
            free=lu_solve((lu,pivots),free_rhs,check_finite=False)
            correction=np.zeros_like(free); tension.fill(0)
            for _ in range(5):
                qc=self._moving_to_full_numpy(free+correction)
                _,lengths=self._cable_geometry_cpu(qc)
                phi=target-lengths
                if np.max(np.maximum(-phi,0))<=min(tol,1e-8): break
                H=-self._cable_jacobian_cpu(qc)[:,moving]
                # LAPACK reuses one factorization and handles all cable RHSs.
                Z=lu_solve((lu,pivots),H.T,check_finite=False)
                inc=projected_gauss_seidel_lcp(H@Z,phi)
                correction+=Z@inc; tension+=inc
            qm_next=free+correction; q_next=self._moving_to_full_numpy(qm_next)
            diff=np.linalg.norm(q_next-q)/(3*self.nMoving)
            q,qm=q_next,qm_next; history.append(q.copy())
            _,lengths=self._cable_geometry_cpu(q)
            error=np.max(np.maximum(lengths-target,0))
            if show_info:
                print(f"static iteration {it+1}: diff={diff:.7e}, constraint={error:.7e}")
            if diff<tol and error<tol: break
        return history,tension

    def FKD_free_static(self, show_info=False, tol=1e-6, max_iter=500):
        return self.FKD_static(self.vertices, np.zeros(self.nCable), tol=tol,
                               show_info=show_info, max_iter=max_iter)

    def FKD_time(self, target_cable_length, total_time, starting_vertices,
                 tol=2e-5, show_info=False, h=0.002):
        target=self._t(target_cable_length).reshape(-1)
        if target.numel()!=self.nCable or torch.any(target<=0):
            raise ValueError(f"target_cable_length must contain {self.nCable} positive values")
        q=self._t(starting_vertices).reshape(-1).clone(); qd=torch.zeros_like(q)
        history=[self._np(q).copy()]; tension=torch.zeros_like(target)
        eye=torch.eye(q.numel(),device=self.device,dtype=self.torch_dtype)
        t,settled=0.0,0
        while t<total_time:
            dt=min(h,total_time-t); old=q.clone()
            _,R18=self._R18_torch(q); K,f=self._assemble_global(R18)
            disp=q-self.q0_t; denom=disp@self.mass_t@disp
            energy=torch.clamp(disp@K@disp,min=0)
            damping=torch.sqrt(energy/denom) if denom.item()>1e-30 else energy.new_zeros(())
            C=2*damping*self.mass_t
            A=eye/dt+dt*(self.W_t@K)+self.W_t@C
            rhs=self.W_t@(-K@(q+dt*qd)+f+self.gravity_t-C@qd)
            dv_free=torch.linalg.solve(A,rhs)
            q_free=q+dt*qd+dt*dv_free; correction=torch.zeros_like(q); tension.zero_()
            for _ in range(5):
                qc=q_free+dt*correction; phi=target-self._lengths_torch(qc)
                if torch.clamp(-phi,min=0).max().item()<=min(tol,1e-8): break
                H=-self._cable_jacobian_torch(qc)
                Z=torch.linalg.solve(A,self.W_t@H.T)
                inc=self._pgs(dt*H@Z,phi)
                correction+=Z@inc; tension+=inc
            qd+=dv_free+correction; q+=dt*qd; t+=dt
            # Fixed vertices are exact, avoiding accumulated roundoff.
            q[self.fixed_dof_t]=self.q0_t[self.fixed_dof_t]
            qd[self.fixed_dof_t]=0
            history.append(self._np(q).copy())
            diff=torch.linalg.vector_norm(q-old)/np.sqrt(3*self.num_vertices)
            error=torch.clamp(self._lengths_torch(q)-target,min=0).max()
            settled=settled+1 if diff.item()<tol and error.item()<tol else 0
            if show_info:
                print(f"t={t:.3f}, diff={diff.item():.7e}, constraint={error.item():.7e}")
            if settled>=10: break
        return history,self._np(q.reshape(-1,3)),self._np(tension)

    def FKD_get_residual(self, Q, cable_tension):
        q=self._t(Q).reshape(-1); _,R18=self._R18_torch(q)
        K,f,fixed=self._assemble_reduced(R18)
        H=-self._cable_jacobian_torch(q)[:,self.moving_dof_t]
        rhs=f+self.gravity_t[self.moving_dof_t]+fixed+H.T@self._t(cable_tension)
        return self._np(K@q[self.moving_dof_t]-rhs)

    # Keep the reference IK logic/API; its expensive FKD, rotations, assembly,
    # cable geometry, and Jacobian calls dispatch to the GPU overrides above.
    def IKD_single(self, target_ee_pos, vertices, max_iter=100, tol=1e-5,
                   show_info=False, **kwargs):
        return super().IKD_single(target_ee_pos, vertices, tol=tol,
                                  show_info=show_info, **kwargs)

    # PyVista consumes host arrays. These wrappers allow CUDA tensors as input.
    def visualize_vert(self, vertices):
        return super().visualize_vert(self._np(vertices))

    def visualize_fb_surface(self, vertices):
        return super().visualize_fb_surface(self._np(vertices))

    def visualize_IKD_result(self, vertices, target_ee_pos):
        return super().visualize_IKD_result(self._np(vertices), self._np(target_ee_pos))


if __name__ == "__main__":
    model = C_SRS_fixedEnd_torch(
        "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    )
    print(f"C-SRS Torch backend: {model.device}")
    icl = model.initial_cable_length.copy()
    tcl = [icl[0]-0.03, icl[1]-0.03, icl[2]-0.03, icl[3], icl[4], icl[5]]
    start_time = time.time()
    Q_list, cable_tension = model.FKD_static_length(tcl, starting_vertices=model.vertices, show_info=False)
    print(f"Static solve time: {time.time()-start_time:.3f} seconds")
    start_time = time.time()
    vert = model.deform_CG(tcl, model.vertices, max_iter=100, tol=1e-6, show_info=False)
    print(f"CG solve time: {time.time()-start_time:.3f} seconds")
    model.visualize_vert(vert)
