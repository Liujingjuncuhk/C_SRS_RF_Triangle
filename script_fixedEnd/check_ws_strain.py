import matplotlib
import torch
import torch.nn as nn
import numpy as np
import joblib
import numpy as np
import pyvista as pv
import os
import sys
import inspect
import time
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle

def evaluate_triangle_strain(
    initial_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    rcond: float = 1e-12,
) -> dict:
    """
    Evaluate membrane deformation of a 3D triangular element.

    Parameters
    ----------
    initial_vertices : array_like, shape (3, 3)
        Initial triangle vertices. Each row is [x, y, z].

    deformed_vertices : array_like, shape (3, 3)
        Deformed triangle vertices, using the same vertex ordering.

    rcond : float, optional
        Relative tolerance used for degeneracy checks and pseudoinverse.

    Returns
    -------
    result : dict
        rotation:
            Proper 3x3 rotation matrix mapping the initial triangle
            orientation toward the deformed triangle orientation.

        corotated_vertices:
            Deformed triangle, centered and rotated back into the
            initial triangle's orientation.

        surface_deformation_gradient:
            Rank-2 embedded 3x3 surface deformation gradient.

        corotated_stretch_tensor:
            Rank-2 symmetric stretch tensor after removing rotation.

        principal_stretches:
            Two positive in-plane principal stretches.

        principal_engineering_strains:
            lambda_i - 1.

        principal_green_lagrange_strains:
            0.5 * (lambda_i**2 - 1).

        principal_log_strains:
            log(lambda_i).

        max_abs_strain_percent:
            100 * max(abs(lambda_i - 1)).

        area_ratio:
            Deformed area / initial area.

        rotation_angle_degrees:
            Magnitude of the extracted rigid rotation.
    """
    X = np.asarray(initial_vertices, dtype=float)
    x = np.asarray(deformed_vertices, dtype=float)

    if X.shape != (3, 3) or x.shape != (3, 3):
        raise ValueError(
            "initial_vertices and deformed_vertices must have shape (3, 3)."
        )

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(x)):
        raise ValueError("Vertex coordinates must be finite.")

    # ------------------------------------------------------------------
    # 1. Remove translation by subtracting each triangle's centroid.
    # ------------------------------------------------------------------
    X_centroid = X.mean(axis=0)
    x_centroid = x.mean(axis=0)

    Xc = X - X_centroid
    xc = x - x_centroid

    # Twice the triangle areas.
    initial_double_area = np.linalg.norm(
        np.cross(X[1] - X[0], X[2] - X[0])
    )
    deformed_double_area = np.linalg.norm(
        np.cross(x[1] - x[0], x[2] - x[0])
    )

    initial_edge_lengths = np.array([
        np.linalg.norm(X[1] - X[0]),
        np.linalg.norm(X[2] - X[1]),
        np.linalg.norm(X[0] - X[2]),
    ])

    deformed_edge_lengths = np.array([
        np.linalg.norm(x[1] - x[0]),
        np.linalg.norm(x[2] - x[1]),
        np.linalg.norm(x[0] - x[2]),
    ])

    initial_scale = max(np.max(initial_edge_lengths), 1.0)
    deformed_scale = max(np.max(deformed_edge_lengths), 1.0)

    if initial_double_area <= rcond * initial_scale**2:
        raise ValueError("The initial triangle is degenerate.")

    if deformed_double_area <= rcond * deformed_scale**2:
        raise ValueError("The deformed triangle is collapsed or degenerate.")

    # Vertices are placed in columns:
    #
    # A = [X1-Xbar, X2-Xbar, X3-Xbar]
    #
    # This matrix is 3x3 but has rank 2.
    A = Xc.T
    a = xc.T

    # ------------------------------------------------------------------
    # 2. Embedded rank-2 surface deformation gradient.
    #
    # a = F_surface @ A
    # ------------------------------------------------------------------
    A_pinv = np.linalg.pinv(A, rcond=rcond)
    F_surface = a @ A_pinv

    # ------------------------------------------------------------------
    # 3. Surface polar decomposition:
    #
    # F_surface = R @ U_surface
    #
    # Since F_surface has rank 2, its third singular value is zero.
    # ------------------------------------------------------------------
    U_svd, singular_values, Vt = np.linalg.svd(
        F_surface,
        full_matrices=True,
    )

    # Construct a proper rotation with determinant +1.
    correction = np.eye(3)

    if np.linalg.det(U_svd @ Vt) < 0.0:
        correction[2, 2] = -1.0

    rotation = U_svd @ correction @ Vt

    # Remove the rigid rotation.
    stretch_surface = rotation.T @ F_surface

    # Remove tiny numerical asymmetry.
    stretch_surface = 0.5 * (
        stretch_surface + stretch_surface.T
    )

    # Rotate the centered deformed triangle back into the reference frame.
    corotated_vertices = (rotation.T @ a).T

    # The first two singular values are the in-plane stretches.
    principal_stretches = singular_values[:2]

    if np.any(principal_stretches <= 0.0):
        raise ValueError(
            "The triangle has a nonpositive principal stretch."
        )

    principal_engineering_strains = principal_stretches - 1.0

    principal_green_lagrange_strains = 0.5 * (
        principal_stretches**2 - 1.0
    )

    principal_log_strains = np.log(principal_stretches)

    max_abs_strain_percent = (
        100.0 * np.max(np.abs(principal_engineering_strains))
    )

    area_ratio_geometric = (
        deformed_double_area / initial_double_area
    )

    # This should equal area_ratio_geometric up to numerical precision.
    area_ratio_from_stretches = np.prod(principal_stretches)

    # Rotation magnitude.
    cosine_angle = np.clip(
        0.5 * (np.trace(rotation) - 1.0),
        -1.0,
        1.0,
    )

    rotation_angle_degrees = np.degrees(
        np.arccos(cosine_angle)
    )

    # Reference tangent-plane projector.
    reference_projector = A @ A_pinv

    # Embedded Green-Lagrange surface strain tensor.
    green_lagrange_surface = 0.5 * (
        F_surface.T @ F_surface - reference_projector
    )

    return {
        "rotation": rotation,
        "initial_centroid": X_centroid,
        "deformed_centroid": x_centroid,
        "centered_initial_vertices": Xc,
        "centered_deformed_vertices": xc,
        "corotated_vertices": corotated_vertices,
        "surface_deformation_gradient": F_surface,
        "corotated_stretch_tensor": stretch_surface,
        "green_lagrange_surface_tensor": green_lagrange_surface,
        "principal_stretches": principal_stretches,
        "principal_engineering_strains":
            principal_engineering_strains,
        "principal_green_lagrange_strains":
            principal_green_lagrange_strains,
        "principal_log_strains": principal_log_strains,
        "max_abs_strain_percent": max_abs_strain_percent,
        "area_ratio": area_ratio_geometric,
        "area_ratio_from_stretches": area_ratio_from_stretches,
        "rotation_angle_degrees": rotation_angle_degrees,
    }


def check_ws_strain(c_srs: C_SRS_fixedEnd):
    initial_tri_list = c_srs.get_tri_SK_list(c_srs.vertices)
    ws_vertices = c_srs.vertices_list
    n_ws = len(ws_vertices)
    ave_strain_list = [0 for _ in range(len(ws_vertices))]
    good_idx = []
    max_strain_list = []
    n_good = 0
    for i in range(n_ws):
        
        
        c_srs.vertices = ws_vertices[i]
        tri_list = c_srs.get_tri_SK_list(c_srs.vertices)
        max_strain = 0
        for j in range(len(initial_tri_list)):
            initial_tri = initial_tri_list[j]
            deformed_tri = tri_list[j]
            strain_info = evaluate_triangle_strain(initial_tri, deformed_tri)
            ave_strain_list[i] += strain_info["max_abs_strain_percent"]
            if strain_info["max_abs_strain_percent"] > max_strain:
                max_strain = strain_info["max_abs_strain_percent"]
        print("ws", i, "max strain:", max_strain)
        if max_strain < 2:
            n_good += 1
            good_idx.append(i)
        max_strain_list.append(max_strain)
    print("number of ws with max strain < 2%:", n_good)
    #     ave_strain_list[i] /= len(initial_tri_list)
    #     print("ws", i, "average strain:", ave_strain_list[i])
    with open("data/ws_strain_check.pkl", "wb") as f:
        pickle.dump({"ave_strain_list": ave_strain_list, "good_idx": good_idx, "max_strain_list": max_strain_list}, f)
    
    return ave_strain_list

def modify_ws(c_srs: C_SRS_fixedEnd):
    with open("data/ws_strain_check.pkl", 'rb') as f:
        data = pickle.load(f)
        ave_strain_list = data["ave_strain_list"]
        good_idx = data["good_idx"]
        max_strain_list = data["max_strain_list"]

    ee_pos_ori = np.array(c_srs.ee_pos_list)
    ee_pos_filtered = ee_pos_ori[good_idx]
    print("number of ws with max strain < 2%:", len(good_idx))
    c_srs.visualize_ws(c_srs.vertices, ee_pos_filtered)
    

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    modify_ws(c_srs)
