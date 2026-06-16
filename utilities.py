import numpy as np
# from lemkelcp import lemketableau
import quantecon as qe
from scipy.interpolate import make_interp_spline,griddata,bisplrep, bisplev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from scipy.optimize import minimize
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.linalg import solve
from scipy.spatial import Delaunay
import random
from scipy.spatial.distance import cdist
import pickle
from itertools import product
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit


def generate_dense_tar_ee(tar_ee_pos_list, n_interp = 5):
    n_waypoint = len(tar_ee_pos_list)
    dense_ee_pos_list = []
    for i in range(n_waypoint - 1):
        start_pos = tar_ee_pos_list[i]
        end_pos = tar_ee_pos_list[i + 1]
        for j in range(n_interp):
            ratio = j / n_interp
            interp_pos = start_pos * (1 - ratio) + end_pos * ratio
            dense_ee_pos_list.append(interp_pos)
    dense_ee_pos_list.append(tar_ee_pos_list[-1])
    return dense_ee_pos_list

def cal_vol_tet(tet):
    """
    Calculate the volume of a tetrahedron given its vertices.

    Parameters:
    - tet: A 4x3 numpy array containing the vertices of the tetrahedron.

    Returns:
    - volume: The volume of the tetrahedron.
    """
    # Extract the vertices
    a, b, c, d = tet
    # Calculate the volume using the determinant formula
    volume = np.abs(np.linalg.det(np.array([a - d, b - d, c - d])) / 6)
    return volume

def R33_2_R1212(R_list_33):
    nR = len(R_list_33)
    R_list_1212 = [np.eye(12) for _ in range(nR)]   
    for i in range(nR):
        R_list_1212[i][0:3, 0:3] = R_list_33[i].copy()
        R_list_1212[i][3:6, 3:6] = R_list_33[i].copy()
        R_list_1212[i][6:9, 6:9] = R_list_33[i].copy()
        R_list_1212[i][9:12, 9:12] = R_list_33[i].copy()
    return R_list_1212

def generate_combinations(min_values, max_values, step_size):
    if len(min_values) != len(max_values):
        raise ValueError("min_values and max_values must have the same length")
    
    # Generate sequences for each min-max pair
    sequences = []
    for min_val, max_val in zip(min_values, max_values):
        # Ensure max_val is included by adjusting the range
        sequence = []
        current = min_val
        while current <= max_val:
            sequence.append(current)
            current += step_size
        sequences.append(sequence)
    
    # Generate all possible combinations using Cartesian product
    combinations = list(product(*sequences))
    
    # Convert tuples to lists for nested list output
    return [list(combo) for combo in combinations]

def add_translation(tar_ee_pos, translation):
    tar_ee_pos_return = tar_ee_pos.copy()
    for i in range(len(tar_ee_pos_return)):
        tar_ee_pos_return[i] += translation
    return tar_ee_pos_return

def compute_transformation_icp(P, Q, max_iterations=100, tol=1e-6):
    """
    Compute the 4x4 transformation matrix between point clouds P and Q using ICP.
    
    Parameters:
    - P: np.array of shape (N, 3) - source point cloud
    - Q: np.array of shape (N, 3) - target point cloud
    - max_iterations: int - maximum number of iterations
    - tol: float - convergence tolerance (mean correspondence distance)
    
    Returns:
    - T: np.array of shape (4, 4) - homogeneous transformation matrix
    """
    # Ensure P and Q have the same number of points
    assert P.shape == Q.shape, "Point clouds must have the same number of points."
    
    # Initialize transformation matrix (4x4 homogeneous)
    T = np.identity(4)
    P_transformed = P.copy()
    
    # Build KD-tree for Q for efficient nearest-neighbor search
    Q_tree = cKDTree(Q)
    
    for iteration in range(max_iterations):
        # Step 1: Find closest points (correspondences)
        distances, indices = Q_tree.query(P_transformed, k=1)
        print(f"Iteration {iteration + 1}, distance: {distances}, indices: {indices}")
        # delete repeated points in P_transformed
        Q_correspond = Q[indices]
        # reduced_indices = []
        # for i in range(len(indices)):
        #     if indices[i] not in reduced_indices:
        #         reduced_indices.append(indices[i])
        # if len(reduced_indices) < 8:
        #     reduced_indices = indices
        # Q_correspond = Q[reduced_indices]
        # P_transformed = P_transformed[reduced_indices]
        
        # Step 2: Compute least-squares rigid transformation
        # Center the point clouds to compute rotation
        P_mean = np.mean(P_transformed, axis=0)
        Q_mean = np.mean(Q_correspond, axis=0)
        P_centered = P_transformed - P_mean
        Q_centered = Q_correspond - Q_mean
        
        # Compute rotation using SVD
        H = np.dot(P_centered.T, Q_centered)  # Shape (3, 3)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = Q_mean - np.dot(R, P_mean)
        
        # Form 4x4 transformation matrix for this iteration
        T_iter = np.identity(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        
        # Step 3: Apply transformation to P
        P_homogeneous = np.hstack((P_transformed, np.ones((P.shape[0], 1))))
        P_transformed = np.dot(P_homogeneous, T_iter.T)[:, :3]
        
        # Update cumulative transformation
        T = np.dot(T_iter, T)
        
        # Step 4: Check convergence
        mean_error = np.mean(distances)
        if mean_error < tol:
            print(f"Converged after {iteration + 1} iterations with error {mean_error}")
            break
    return T



def solve_with_known_values(A, b, known_values, all2unknowns, unknowns2all):
    """
    Solve a linear system Ax = b, incorporating known values in x, and return the complete solution vector x.

    Parameters:
    - A: numpy.ndarray (m x n), the coefficient matrix.
    - b: numpy.ndarray (m,), the right-hand side vector.
    - known_values: dict {int: float}, where keys are 0-based indices of known variables in x,
                    and values are their known values.

    Returns:
    - x: numpy.ndarray (n,), the complete solution vector with known and computed values.
    - unknown_indices: list of int, the 0-based indices of the unknown variables.

    Raises:
    - ValueError: If inputs are invalid or the system is unsolvable.
    """
    # print("mat A shape: ", np.shape(A))
    n_all = len(all2unknowns)
    n_unknowns = len(unknowns2all)
    known_indices = sorted(known_values.keys())
    A_reduced = A[unknowns2all][:, unknowns2all]

    b_reduced = b[unknowns2all]
    b_2add = np.zeros_like(b_reduced)
    for i_known in known_indices:
        for j in range(n_all):
            if all2unknowns[j] != -1:
                b_2add[all2unknowns[j]] -= A[j, i_known] * known_values[i_known]
    b_reduced = b_reduced + b_2add
    x_reduced = np.linalg.inv(A_reduced) @ b_reduced
    x_final = np.zeros((n_all,1))
    for i_known in known_indices:
        x_final[i_known] = known_values[i_known]
    for i in range(n_unknowns):
        x_final[unknowns2all[i]] = x_reduced[i]
    return x_final

def average_transform_matrices(transforms):
    """
    Compute the average of a list of 4x4 rigid transformation matrices.
    
    Parameters:
    transforms (list of np.ndarray or list): List of 4x4 transformation matrices.
    
    Returns:
    np.ndarray: The 4x4 average transformation matrix.
    """
    if not transforms:
        raise ValueError("Input list of transforms is empty.")
    
    # Convert to NumPy arrays if needed
    transforms = [np.array(t) for t in transforms]
    
    n = len(transforms)
    
    # Extract rotations (n, 3, 3) and translations (n, 3)
    rot_matrices = np.array([t[:3, :3] for t in transforms])
    translations = np.array([t[:3, 3] for t in transforms])
    
    # Average translation (arithmetic mean)
    avg_trans = np.mean(translations, axis=0)
    
    # Average rotation using SciPy's Rotation mean
    rotations = R.from_matrix(rot_matrices)
    avg_rot = rotations.mean().as_matrix()
    
    # Recompose the 4x4 matrix
    avg_matrix = np.eye(4)
    avg_matrix[:3, :3] = avg_rot
    avg_matrix[:3, 3] = avg_trans
    
    return avg_matrix

def plot_trackerpos_3d(track_points_sim, markerspos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(track_points_sim)):
        ax.scatter(track_points_sim[i, 0], track_points_sim[i, 1], track_points_sim[i, 2], c='b')
        # add a text near the point
        ax.text(track_points_sim[i, 0], track_points_sim[i, 1], track_points_sim[i, 2], 'Sim '+str(i), color='blue')
    for i in range(len(markerspos)):
        ax.scatter(markerspos[i, 0], markerspos[i, 1], markerspos[i, 2], c='r')
        # add a text near the point
        ax.text(markerspos[i, 0], markerspos[i, 1], markerspos[i, 2], 'Mark '+str(i), color='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def compute_affine_transform_3d(A, B):
    """
    Compute the 4x4 affine transformation matrix that maps points in A to B.
    
    Parameters:
    A (np.array): n x 3 array of source points.
    B (np.array): n x 3 array of target points.
    
    Returns:
    transform_matrix (np.array): 4x4 affine transformation matrix.
    """
    A = np.array(A)
    B = np.array(B)
    n = A.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 points for 3D affine transformation.")
    
    # Construct the design matrix M (3n x 12)
    M = np.zeros((3 * n, 12))
    for i in range(n):
        x, y, z = A[i]
        # Row for x'
        M[3*i] = [x, y, z, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        # Row for y'
        M[3*i + 1] = [0, 0, 0, x, y, z, 0, 0, 0, 0, 1, 0]
        # Row for z'
        M[3*i + 2] = [0, 0, 0, 0, 0, 0, x, y, z, 0, 0, 1]
    
    # Target vector q (flattened B)
    q = B.flatten()
    
    # Solve least squares: M p = q
    p, _, _, _ = np.linalg.lstsq(M, q, rcond=None)
    
    # Reshape parameters into 4x4 matrix
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = p[:9].reshape(3, 3)
    transform_matrix[0:3, 3] = p[9:]
    # transformed_A = np.dot(np.hstack((A, np.ones((A.shape[0], 1)))), transform.T)[:, :3]
    return transform_matrix

def resample_points(matrix, m):
    """
    Resamples an n x 3 matrix to an m x 3 matrix using linear interpolation.
    
    Parameters:
    matrix (np.ndarray): The input n x 3 matrix.
    m (int): The desired number of output rows.
    
    Returns:
    np.ndarray: The resampled m x 3 matrix.
    """
    n = matrix.shape[0]
    
    # Define the 'index' of the original rows
    # For a matrix with 5 rows, this is [0, 1, 2, 3, 4]
    original_indices = np.linspace(0, n - 1, num=n)
    
    # Define the 'index' of the new rows
    # If m=10, this creates 10 points evenly spaced between 0 and 4
    new_indices = np.linspace(0, n - 1, num=m)
    
    # Initialize the output matrix
    resampled_matrix = np.zeros((m, 3))
    
    # Interpolate for each column (X, Y, Z) independently
    for i in range(3):
        resampled_matrix[:, i] = np.interp(new_indices, original_indices, matrix[:, i])
        
    return resampled_matrix


def get_tracking_ball_dense_pos(tracking_ball_pos_sim,tracking_ball_pos_exp):
    n_sim = tracking_ball_pos_sim.shape[0]
    n_exp = tracking_ball_pos_exp.shape[0]
    # resample the sim pos to match exp pos number
    simpos_resampled = resample_points(tracking_ball_pos_sim, n_exp)
    return simpos_resampled

def get_dense_taree(tar_ee_pos, n_between = 10):
    n_target = len(tar_ee_pos)
    n_ee = tar_ee_pos[0].shape[0]
    result_ee_pos = []
    ee_pos_between = tar_ee_pos[0].copy()
    for i in range(n_target-1):
        result_ee_pos.append(tar_ee_pos[i])
        ee_pos_1 = tar_ee_pos[i]
        ee_pos_2 = tar_ee_pos[i+1]
        ee_pos_diff = (ee_pos_2 - ee_pos_1)/n_between
        for j in range(n_between-1):
            ee_pos_between = ee_pos_1 + ee_pos_diff*(j+1)
            result_ee_pos.append(ee_pos_between)
    result_ee_pos.append(tar_ee_pos[-1])
    return result_ee_pos


def resample_tcl(tcls):
    # insert in each tcl a mid point between two tcls
    tcls_resampled = []
    for i in range(len(tcls)-1):
        tcl_start = tcls[i]
        tcl_end = tcls[i+1]
        tcls_resampled.append(tcl_start)
        tcl_mid = [(tcl_start[j] + tcl_end[j])/2 for j in range(len(tcl_start))]
        tcls_resampled.append(tcl_mid)
    tcls_resampled.append(tcls[-1])
    return tcls_resampled

def moving_average(data, window_size):
    """Smooth data using simple moving average."""
    data = np.array(data)  # Convert list to array if needed
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def projected_gauss_seidel_lcp(
    M: np.ndarray,
    q: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
    omega: float = 1.3,        # over-relaxation (1.0 = pure GS, 1.3–1.9 common in robotics)
    warm_start: np.ndarray | None = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Solve LCP:  w = M z + q >= 0,  z >= 0,  z^T w = 0
    using Projected Gauss-Seidel (successive over-relaxation).
    
    Returns the least 2-norm solution (minimum impulse) — the physically correct one in robotics.
    
    Parameters
    ----------
    M : (n,n) np.ndarray
        Usually symmetric positive semi-definite in robotics (e.g., Delassus matrix)
    q : (n,) np.ndarray
    max_iter, tol, omega : convergence controls
    warm_start : optional initial guess (greatly speeds up sequential solves)
    
    Returns
    -------
    z : (n,) solution vector
    """
    n = len(q)
    M = np.asarray(M, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Diagonal dominance check (optional warning)
    diag = np.diag(M)
    if np.any(diag <= 0):
        print("Warning: M has non-positive diagonal entries → may diverge!")
    
    z = np.zeros(n, dtype=np.float64) if warm_start is None else warm_start.copy()
    w = M @ z + q
    
    for iteration in range(max_iter):
        z_old = z.copy()
        
        for i in range(n):
            if M[i,i] == 0:
                z[i] = 0  # avoid division by zero
                continue
                
            # Predicted w_i if z_i were free
            w_pred = q[i] + M[i,:i] @ z[:i] + M[i,i+1:] @ z[i+1:]
            
            # Gauss-Seidel update + projection onto z_i >= 0 and w_i >= 0
            z_new = max(0.0, z[i] - omega * (M[i,i] * z[i] + w_pred) / M[i,i])
            
            # Apply relaxation
            z[i] = z_new
            
            # Optional: update w incrementally (saves one full Mv per sweep)
            # w[i] = M[i,i] * z[i] + w_pred
        
        # Full residual (for convergence check)
        w = M @ z + q
        residual = np.maximum(z, w)  # violation = max(z_i, w_i) when one should be zero
        res_norm = np.linalg.norm(residual, np.inf)
        
        if res_norm < tol:
            if verbose:
                print(f"Converged in {iteration+1} iterations (res = {res_norm:.2e})")
            return z
        
        if np.linalg.norm(z - z_old) < tol:
            if verbose:
                print(f"Stalled at iteration {iteration+1}, res = {res_norm:.2e}")
            break
            
    if verbose:
        print(f"Max iterations reached. Final residual: {res_norm:.2e}")
    
    return z


def my_aa(cl_hist, e_hist):
    e_high = np.max(e_hist)
    e_low = np.min(e_hist)
    idx_high = np.argmax(e_hist)
    idx_low = np.argmin(e_hist)
    nC = cl_hist.shape[1]
    cl_next = [0 for _ in range(nC)]
    
    cl_high = cl_hist[idx_high]
    cl_low = cl_hist[idx_low]
    ratio = -e_low/(e_high-e_low)
    for i in range(nC):
        cl_next[i] = cl_low[i] + ratio*(cl_high[i]-cl_low[i])
    return cl_next

def anderson_step(U_hist, f_hist, beta=1.0, lam=1e-8, m=5):
    """
    Gradient-augmented linear Anderson acceleration for a scalar residual.
 
    Drives f(u) -> 0 where u in R^6, f in R.
 
    Parameters
    ----------
    U_hist : (N, 6) array   stored input vectors, oldest first
    f_hist : (N,)   array   stored scalar outputs, f_hist[k] = f(U_hist[k])
    beta   : float          mixing parameter (1.0 = full Newton-like step)
    lam    : float          Tikhonov regularization for ill-conditioning
    m      : int            max window depth (uses last m+1 iterates)
 
    Returns
    -------
    u_next : (6,) array      next input to try
    """
    U_hist = np.asarray(U_hist, dtype=float)
    f_hist = np.asarray(f_hist, dtype=float).ravel()
    N = U_hist.shape[0]
 
    # use last m+1 iterates -> m differences
    k = min(m + 1, N)
    U = U_hist[-k:]            # (k, 6)
    f = f_hist[-k:]            # (k,)
 
    u_m = U[-1]               # current input  (6,)
    f_m = f[-1]               # current scalar residual
 
    if k == 1:
        # no history to build differences; fall back to a tiny gradient-free nudge
        return u_m.copy()
 
    dU = np.diff(U, axis=0)   # (k-1, 6)  rows: Delta u_i
    df = np.diff(f)           # (k-1,)    Delta f_i
 
    Umat = dU.T               # (6, k-1)  columns Delta u_i  -> "U" in derivation
    Fmat = df.reshape(1, -1)  # (1, k-1)  -> "F" in derivation
 
    # --- Anderson coefficients: min-norm LS for scalar residual (Eq. 2) ---
    denom = float((Fmat @ Fmat.T).item()) + lam  # scalar  sum(df^2)+lam
    gamma = (Fmat.T * (f_m / denom)).ravel()       # (k-1,)
 
    # affine-hull extrapolation:  u_m - U gamma
    u_extrap = u_m - Umat @ gamma                  # (6,)
    f_extrap = f_m - float((Fmat @ gamma).item())  # scalar residual estimate
 
    # --- gradient (Jacobian) estimate by LS:  ghat = F U^+  ---
    # ghat in R^{1x6}:  solves  df ~ dU @ ghat^T
    ghat, *_ = np.linalg.lstsq(dU, df, rcond=None)  # (6,)  = grad f
    gg = float(ghat @ ghat) + lam
    d = ghat / gg                                   # (6,) Newton direction per unit residual
 
    u_next = u_extrap - beta * f_extrap * d
    return u_next

def anderson_step_vertex(U_hist, P_hist, p_ref, beta=1.0, lam=1e-6, m=3):
    """
    Linear Anderson acceleration to drive ONE vertex to a reference position.
 
        r(u) = p(u) - p_ref  in R^3,   u in R^6  (6 actuation inputs)
 
    Because n=3 < 6 the input is under-determined: the step taken is the
    MINIMUM-NORM input change that moves the vertex toward p_ref. There is a
    3-D null space of input moves that do not affect this vertex.
 
    Parameters
    ----------
    U_hist : (N, 6) array   stored input vectors, oldest first
    P_hist : (N, 3) array   measured vertex positions, P_hist[k] = p(U_hist[k])
    p_ref  : (3,)   array   target reference position
    beta   : float          mixing parameter (1.0 = full step)
    lam    : float          Tikhonov regularization (keep small)
    m      : int            window depth (uses last m+1 iterates)
 
    Returns
    -------
    u_next : (6,) array
    """
    U_hist = np.asarray(U_hist, float)
    P_hist = np.asarray(P_hist, float)
    p_ref = np.asarray(p_ref, float).ravel()
 
    R_hist = P_hist - p_ref          # (N, 3) residuals
    N = U_hist.shape[0]
 
    k = min(m + 1, N)
    U = U_hist[-k:]                  # (k, 6)
    R = R_hist[-k:]                  # (k, 3)
    u_m = U[-1]                     # (6,)
    r_m = R[-1]                     # (3,)
 
    if k == 1:
        return u_m.copy()
 
    dU = np.diff(U, axis=0)         # (k-1, 6)
    dR = np.diff(R, axis=0)         # (k-1, 3)
    Umat = dU.T                     # (6, k-1)
    Rmat = dR.T                     # (3, k-1)
 
    # --- Anderson coefficients: min || r_m - Rmat gamma ||, regularized ---
    A = Rmat.T @ Rmat + lam * np.eye(Rmat.shape[1])
    gamma = np.linalg.solve(A, Rmat.T @ r_m)         # (k-1,)
 
    u_extrap = u_m - Umat @ gamma                     # (6,)
    r_extrap = r_m - Rmat @ gamma                     # (3,)
 
    # --- Jacobian estimate Jhat (3 x 6):  dR ~ dU @ Jhat^T ---
    JhatT, *_ = np.linalg.lstsq(dU, dR, rcond=None)   # (6, 3) = Jhat^T
    Jhat = JhatT.T                                     # (3, 6)
 
    # min-norm step:  delta = -beta * Jhat^+ r_extrap
    # right pseudoinverse (n<6):  Jhat^+ = Jhat^T (Jhat Jhat^T + lam I)^-1
    JJT = Jhat @ Jhat.T + lam * np.eye(3)             # (3, 3)
    step = Jhat.T @ np.linalg.solve(JJT, r_extrap)    # (6,)
 
    return u_extrap - beta * step

def anderson_my_parabola(U_hist, e_hist):
    nCl = U_hist.shape[1]
    cl_next = [0 for _ in range(nCl)]
    for i in range(nCl):
        cl_hist = U_hist[:, i]
        a, b, c = fit_parabola_3param(cl_hist, e_hist)
        cl_next[i] = b
    return cl_next

def fit_parabola_3param(cl, e):
    x, y = cl, e
    # Fit y = px^2 + qx + r
    p, q, r = np.polyfit(x, y, 2)
    a = p
    b = -q / (2 * p)
    c = r - q**2 / (4 * p)
    return a, b, c

def fit_parabola(cl, e):
    x, y = cl, e

    f = lambda x, a, b: a * (x - b)**2

    # Initial guess from a full quadratic fit
    p, q, _ = np.polyfit(x, y, 2)
    a0, b0 = p, -q / (2 * p)

    (a, b), _ = curve_fit(f, x, y, p0=[a0, b0])
    return a, b

def generate_rectangular_traj(wayPoints, n_interp):
    traj = []
    for i in range(len(wayPoints)-1):
        start = wayPoints[i]
        end = wayPoints[i+1]
        for j in range(n_interp):
            ratio = j / n_interp
            interp_point = start * (1 - ratio) + end * ratio
            traj.append(interp_point)
    traj.append(wayPoints[-1])
    # add from last to first to make it a loop
    for i in range(len(wayPoints)-1, 0, -1):
        start = wayPoints[i]
        end = wayPoints[i-1]
        for j in range(n_interp):
            ratio = j / n_interp
            interp_point = start * (1 - ratio) + end * ratio
            traj.append(interp_point)
    return traj

def clamp_diff(cmd_diff, min_bound = 5e-4, max_bound = 1e-2):
    # max_diff = np.max(np.abs(cmd_diff))
    min_diff = 0
    ratio = 1.0
    for i in range(len(cmd_diff)):
        if abs(cmd_diff[i]) > 0 and (min_diff == 0 or abs(cmd_diff[i]) < min_diff):
            min_diff = abs(cmd_diff[i])

    if min_diff < min_bound and min_diff > 1e-8:
        ratio = min_bound / min_diff
    cmd_diff_new = [cmd_diff[i] * ratio for i in range(len(cmd_diff))]
    ratio = 1.0
    max_diff = np.max(np.abs(cmd_diff_new))
    if max_diff > max_bound:  # check effective max after any scale-up
        ratio = max_bound / max_diff
    cmd_diff_new = [cmd_diff_new[i] * ratio for i in range(len(cmd_diff_new))]
    return cmd_diff_new

if __name__ == '__main__':
    # Example usage
    min_values = [1, 2]
    max_values = [3, 4]
    step_size = 0.34
    result = generate_combinations(min_values, max_values, step_size)
    print(result)
