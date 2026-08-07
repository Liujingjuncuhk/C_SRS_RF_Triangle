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
from C_SRS_fixedEnd_torch import C_SRS_fixedEnd_torch
import pickle

robot_filenames = ["./models/flat_tri_surface/C_SRS_description_bary_sparse.pkl", 
                   "./models/flat_tri_surface/C_SRS_description_bary.pkl",
                   "./models/flat_tri_surface/C_SRS_description_bary_dense.pkl"]


def mesh_resolution_ablation_generation_data():
    tcl = np.array([416, 430, 436, 302,270,286])*1e-3
    c_srs_sparse = C_SRS_fixedEnd_torch(robot_filenames[0])
    c_srs = C_SRS_fixedEnd_torch(robot_filenames[1])
    c_srs_dense = C_SRS_fixedEnd_torch(robot_filenames[2])
    t1 = time.time()
    Q_list, cable_tension = c_srs_sparse.FKD_static_length(tcl, c_srs_sparse.vertices, tol = 1e-6, show_info = False)
    time_cost_sparse = time.time() - t1
    c_srs_sparse.visualize_vert(Q_list[-1])
    n_iter_sparse = len(Q_list); vert_sparse = Q_list[-1]
    t2 = time.time()
    Q_list, cable_tension = c_srs.FKD_static_length(tcl, c_srs.vertices, tol = 1e-6, show_info = False)
    time_cost = time.time() - t2
    n_iter = len(Q_list); vert = Q_list[-1]
    c_srs.visualize_vert(Q_list[-1])
    t3 = time.time()
    Q_list, cable_tension = c_srs_dense.FKD_static_length(tcl, c_srs_dense.vertices, tol = 1e-6, show_info = False)
    time_cost_dense = time.time() - t3
    c_srs_dense.visualize_vert(Q_list[-1])
    n_iter_dense = len(Q_list); vert_dense = Q_list[-1]
    vert_list = [vert_sparse, vert, vert_dense]
    time_cost_list = [time_cost_sparse, time_cost, time_cost_dense]
    print("time cost list is: ", time_cost_list)
    n_iter_list = [n_iter_sparse, n_iter, n_iter_dense]
    print("n_iter list is: ", n_iter_list)
    data_2save = {}
    data_2save["vert_list"] = vert_list
    data_2save["time_cost_list"] = time_cost_list
    data_2save["n_iter_list"] = n_iter_list
    with open('data/FKD_mesh_resolution_ablation.pkl', 'wb') as f:
        pickle.dump(data_2save, f)


def print_info():
    with open('data/FKD_mesh_resolution_ablation.pkl', 'rb') as f:
        data = pickle.load(f)
    print("vert_list: ", data["vert_list"])
    print("time_cost_list: ", data["time_cost_list"])
    print("n_iter_list: ", data["n_iter_list"])
    print("size of vert_sparse: ", data["vert_list"][0].shape[0]/3)
    print("size of vert: ", data["vert_list"][1].shape[0]/3)
    print("size of vert_dense: ", data["vert_list"][2].shape[0]/3)
    print("time_cost_sparse: ", data["time_cost_list"][0])
    print("time_cost: ", data["time_cost_list"][1])
    print("time_cost_dense: ", data["time_cost_list"][2])
    print("n_iter_sparse: ", data["n_iter_list"][0])
    print("n_iter: ", data["n_iter_list"][1])
    print("n_iter_dense: ", data["n_iter_list"][2])
    print("time per iteration sparse: ", data["time_cost_list"][0]/data["n_iter_list"][0])
    print("time per iteration: ", data["time_cost_list"][1]/data["n_iter_list"][1])
    print("time per iteration dense: ", data["time_cost_list"][2]/data["n_iter_list"][2])


def measure_diff_surface(
    point_cloud: np.ndarray,
    vertices: np.ndarray,
    measure_triangles: np.ndarray,
    batch_size: int = 512,
) -> tuple[np.ndarray, float]:
    """
    Compute point-to-triangle-mesh distances.

    Parameters
    ----------
    point_cloud : array_like, shape (P, 3)
        Captured 3D points.

    vertices : array_like, shape (N, 3)
        Current/deformed mesh vertex positions.

    measure_triangles : array_like, shape (M, 3)
        Triangle vertex indices. Indices must be zero-based.

        For example, [0, 3, 7] means that vertices 0, 3, and 7
        form one triangle.

    batch_size : int, optional
        Number of point-cloud points processed simultaneously.
        Reduce this value if memory usage is too high.

    Returns
    -------
    errors : ndarray, shape (P,)
        Unsigned Euclidean distance from each point-cloud point
        to the closest mesh triangle.

    percentile_95 : float
        The 95th percentile of the point-to-mesh distances.

    Notes
    -----
    The returned distances use the same units as `point_cloud`
    and `vertices`.
    """

    points = np.asarray(point_cloud, dtype=np.float64)
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(measure_triangles)

    # --------------------
    # Validate the inputs
    # --------------------
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point_cloud must have shape (P, 3).")

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("measure_triangles must have shape (M, 3).")

    if len(points) == 0:
        raise ValueError("point_cloud must contain at least one point.")

    if len(verts) == 0:
        raise ValueError("vertices must contain at least one vertex.")

    if len(faces) == 0:
        raise ValueError(
            "measure_triangles must contain at least one triangle."
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("point_cloud contains NaN or infinite values.")

    if not np.all(np.isfinite(verts)):
        raise ValueError("vertices contains NaN or infinite values.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Make sure triangle indices are integers.
    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(faces == np.floor(faces)):
            raise ValueError(
                "measure_triangles must contain integer indices."
            )
        faces = faces.astype(np.int64)
    else:
        faces = faces.astype(np.int64, copy=False)

    if faces.min() < 0 or faces.max() >= len(verts):
        raise IndexError(
            "measure_triangles contains an out-of-range vertex index."
        )

    # --------------------------------
    # Construct the mesh triangles
    # --------------------------------
    triangles = verts[faces]  # Shape: (M, 3, 3)

    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    ab = b - a
    ac = c - a
    bc = c - b

    # Triangle normals.
    normal = np.cross(ab, ac)
    normal_squared = np.einsum(
        "ij,ij->i", normal, normal
    )

    # Values used for barycentric coordinates.
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)

    barycentric_denominator = d00 * d11 - d01 * d01

    # A scale-aware tolerance for degenerate triangles and edges.
    maximum_edge_squared = max(
        float(np.max(np.einsum("ij,ij->i", ab, ab))),
        float(np.max(np.einsum("ij,ij->i", ac, ac))),
        float(np.max(np.einsum("ij,ij->i", bc, bc))),
        1.0,
    )

    epsilon = (
        32.0
        * np.finfo(np.float64).eps
        * maximum_edge_squared
    )

    def point_to_segment_distance_squared(
        query_points: np.ndarray,
        segment_start: np.ndarray,
        segment_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Compute squared distances between batched points and segments.

        query_points has shape (B, 1, 3).
        segment_start and segment_vector have shape (1, M, 3).

        Returns an array of shape (B, M).
        """

        relative = query_points - segment_start

        segment_length_squared = np.sum(
            segment_vector * segment_vector,
            axis=2,
        )

        # Avoid division by zero for zero-length edges.
        safe_length_squared = np.where(
            segment_length_squared > epsilon,
            segment_length_squared,
            1.0,
        )

        parameter = (
            np.sum(relative * segment_vector, axis=2)
            / safe_length_squared
        )

        parameter = np.clip(parameter, 0.0, 1.0)

        closest_points = (
            segment_start
            + parameter[..., None] * segment_vector
        )

        return np.sum(
            (query_points - closest_points) ** 2,
            axis=2,
        )

    # Add a batch dimension to the triangle data.
    A = a[None, :, :]
    B = b[None, :, :]
    C = c[None, :, :]

    AB = ab[None, :, :]
    AC = ac[None, :, :]
    BC = bc[None, :, :]

    NORMAL = normal[None, :, :]

    squared_errors = np.empty(
        len(points),
        dtype=np.float64,
    )

    # --------------------------------------
    # Process point-cloud points in batches
    # --------------------------------------
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))

        # Shape: (current_batch_size, 1, 3)
        p = points[start:stop, None, :]

        ap = p - A

        # Squared distance to each triangle's supporting plane.
        signed_plane_numerator = np.sum(
            ap * NORMAL,
            axis=2,
        )

        safe_normal_squared = np.where(
            normal_squared > epsilon,
            normal_squared,
            1.0,
        )

        plane_distance_squared = (
            signed_plane_numerator**2
            / safe_normal_squared[None, :]
        )

        # Determine whether the orthogonal projection of each point
        # lies inside each triangle using barycentric coordinates.
        d20 = np.sum(ap * AB, axis=2)
        d21 = np.sum(ap * AC, axis=2)

        safe_barycentric_denominator = np.where(
            np.abs(barycentric_denominator) > epsilon,
            barycentric_denominator,
            1.0,
        )

        barycentric_v = (
            d11[None, :] * d20
            - d01[None, :] * d21
        ) / safe_barycentric_denominator[None, :]

        barycentric_w = (
            d00[None, :] * d21
            - d01[None, :] * d20
        ) / safe_barycentric_denominator[None, :]

        barycentric_u = (
            1.0 - barycentric_v - barycentric_w
        )

        projection_is_inside = (
            (normal_squared[None, :] > epsilon)
            & (
                np.abs(barycentric_denominator)[None, :]
                > epsilon
            )
            & (barycentric_u >= 0.0)
            & (barycentric_v >= 0.0)
            & (barycentric_w >= 0.0)
        )

        # Distances to the three triangle edges.
        distance_ab_squared = (
            point_to_segment_distance_squared(p, A, AB)
        )

        distance_bc_squared = (
            point_to_segment_distance_squared(p, B, BC)
        )

        distance_ca_squared = (
            point_to_segment_distance_squared(
                p,
                C,
                A - C,
            )
        )

        # If the projection is outside the triangle, the nearest
        # location is on one of its edges. If it is inside, the
        # perpendicular plane distance is the triangle distance.
        triangle_distance_squared = np.minimum(
            np.minimum(
                distance_ab_squared,
                distance_bc_squared,
            ),
            distance_ca_squared,
        )

        triangle_distance_squared = np.where(
            projection_is_inside,
            np.minimum(
                triangle_distance_squared,
                plane_distance_squared,
            ),
            triangle_distance_squared,
        )

        # Closest triangle for each point.
        squared_errors[start:stop] = np.min(
            triangle_distance_squared,
            axis=1,
        )

    errors = np.sqrt(
        np.maximum(squared_errors, 0.0)
    )

    # percentile_95 = float(
    #     np.percentile(errors, 95)
    # )

    error_sorted = np.sort(errors)
    percentile_95 = error_sorted[:int(0.95 * len(error_sorted))]

    return errors, percentile_95

def accuracy_check():
    with open('data/FKD_mesh_resolution_ablation.pkl', 'rb') as f:
        data = pickle.load(f)
    c_srs_sparse = C_SRS_fixedEnd(robot_filenames[0])
    c_srs = C_SRS_fixedEnd(robot_filenames[1])
    c_srs_dense = C_SRS_fixedEnd(robot_filenames[2])
    
    vert_list = data["vert_list"]
    vert_sparse = vert_list[0]; vert = vert_list[1]; vert_dense = vert_list[2]
    fb_sparse = c_srs_sparse.get_fb_surface(vert_sparse)
    fb = c_srs.get_fb_surface(vert)
    fb_dense = c_srs_dense.get_fb_surface(vert_dense)
    
    
    
    file_flat = "data/fixedEnd_FKD_exp_data_forpaper_final.pkl"
    with open(file_flat, "rb") as f:
        data = pickle.load(f)
    pts_list_123 = data["pts_list"]
    errors_sparse, percentile_95_sparse = measure_diff_surface(pts_list_123[1], fb_sparse, c_srs_sparse.mesh_triangles)
    errors, percentile_95 = measure_diff_surface(pts_list_123[1], fb, c_srs.mesh_triangles)
    errors_dense, percentile_95_dense = measure_diff_surface(pts_list_123[1], fb_dense, c_srs_dense.mesh_triangles)

    print("errors_sparse: ", np.mean(errors_sparse), "percentile_95_sparse: ", np.mean(percentile_95_sparse))
    print("errors: ", np.mean(errors), "percentile_95: ", np.mean(percentile_95))
    print("errors_dense: ", np.mean(errors_dense), "percentile_95_dense: ", np.mean(percentile_95_dense))
    

if __name__ == "__main__":
    # accuracy_check()
    print_info()


    # description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    # c_srs = C_SRS_fixedEnd_torch(description_file)
    # ee_target = np.array([0.26, 0.08, 0.03])
    # print("initial cable length is: ", c_srs.initial_cable_length)
    # tcl = np.array([440.7, 426.5, 440.7, 295.0,278.8,295.0])*1e-3
    # Q_list, vert_length, cable_tension = c_srs.FKD_time(tcl, 1, c_srs.vertices, tol = 1e-4, show_info = True)
    # c_srs.visualize_vert(vert_length)


