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
from flying_carpet import Flying_carpet
import pickle
import pickle
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt

cl_1 = (np.array([495, 483, 506, 470, 450, 437, 457, 426])*1e-3).tolist()
cl_2 = (np.array([372, 550, 388, 570, 363, 518, 379, 525])*1e-3).tolist()
cl_3 = (np.array([575, 470, 580, 445, 510, 348, 498, 309])*1e-3).tolist()
cl_list = [cl_1, cl_2, cl_3]



def _as_point_array(points, name: str) -> np.ndarray:
    if isinstance(points, o3d.geometry.PointCloud):
        points = np.asarray(points.points)
    elif hasattr(points, "points") and not isinstance(points, np.ndarray):
        points = np.asarray(points.points)
    else:
        points = np.asarray(points)

    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{name} must contain points with shape (N, 3).")

    points = points[:, :3].astype(np.float64, copy=False)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) == 0:
        raise ValueError(f"{name} contains no finite 3D points.")
    return points



def get_FKD_flat(flying_carpet:Flying_carpet, cable_lengths):
    """
    Get the flat FKD (Forward Kinematics) for a given set of cable lengths.

    Parameters
    ----------
    flying_carpet : Flying_carpet
        The flying carpet object.
    cable_lengths : list of list/array, shape (N, nCable)
        Cable length targets for each waypoint.

    Returns
    -------
    flat_FKD : list of list/array, shape (N, nVertices)
        Flattened FKD results for each waypoint.
    """
    flat_FKD = []
    start_vert_3 = flying_carpet.vertices.copy()
    start_vert_3[:,1] += 0.05
    start_vert_3[:,2] += 0.05

    for i in range(len(cable_lengths)):
        # if i == 2:
        #     start_vert = start_vert_3
        # else:
        start_vert = flying_carpet.vertices
        Q_list, vertices, cable_tension = flying_carpet.FKD_time(cable_lengths[i], 10, start_vert,h = 0.01,tol = 1e-6, show_info=1)
        flat_FKD.append(vertices)
        flying_carpet.visualize_vert(vertices)
    with open("data_flying_carpet/flat_FKD.pkl", "wb") as f:
        pickle.dump(flat_FKD, f)
    return flat_FKD

def get_FKD_3(flying_carpet:Flying_carpet):
    tcl = cl_3
    start_vert = flying_carpet.vertices
    start_vert[:, 1] += 0.1
    start_vert[:,2] += 0.05
    Q_list, vertices, cable_tension = flying_carpet.FKD_time(tcl, 10, start_vert,h = 0.01,tol = 1e-6, show_info=1)
    flying_carpet.visualize_vert(vertices)


def measure_diff_surface(
    point_cloud: np.ndarray,
    vertices: np.ndarray,
    measure_triangles: np.ndarray,
    tol = 0.005,
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
    error_tol_idx = np.where(errors < tol)[0]
    errors_tol = errors[error_tol_idx]
    error_sorted = np.sort(errors)
    percentile_95 = error_sorted[:int(0.9 * len(error_sorted))]

    return errors, percentile_95, errors_tol

def plot_error_histogram(errors, title="Error Histogram"):
    error_sorted = np.sort(errors)
    # plot error sorted in order
    plt.figure(figsize=(8, 6))
    plt.plot(error_sorted, marker='o', linestyle='-', markersize=3)
    plt.title(title)
    plt.xlabel("Point Index (sorted)")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.show()


def construct_fb_triangle(tetrahedra, feedback_idx):
    fb_triangle = []
    for tet in tetrahedra:
        tri_list = []
        for j in range(4):
            if tet[j] in feedback_idx:
                tri_list.append(tet[j])
        if len(tri_list) == 3:
            fb_triangle.append(tri_list)
    fb_triangle = np.array(fb_triangle)
    return fb_triangle


def compare_SOFA_FKD(flying_carpet:Flying_carpet):
    def get_fb_vert(vertices, feedback_idx):
        return vertices[feedback_idx, :]

    feedback_pts_list = []
    for i in range(3):
        feedback_filename = "data_flying_carpet/feedback_points_"+str(i+1)+".pickle"
        with open(feedback_filename, "rb") as f:
            feedback_pts = pickle.load(f)
            feedback_pts_list.append(feedback_pts)
    
    with open("data_flying_carpet/vert_tets.pickle", "rb") as f:
        vert_tets = pickle.load(f)

    with open("data_flying_carpet/other_info.pickle", "rb") as f:
        other_info = pickle.load(f)
        tetrahedra = other_info["tetrahedra"]
        feedback_idx = other_info["fd_idx"]

    with open("data_flying_carpet/flat_FKD.pkl", "rb") as f:
        vert_flat_list = pickle.load(f)

    for i in range(3):
        if i == 0 or i == 1:
            tol = 0.005
        else:
            tol = 0.008
        plotter = pv.Plotter()
        vert_flat = vert_flat_list[i]
        fb_vert_flat = flying_carpet.get_fb_surface(vert_flat)
        vert_tet = vert_tets[i]
        gt_pts = feedback_pts_list[i]
        vert_tet_fb = vert_tet[feedback_idx, :]
        faces = np.hstack((np.full((flying_carpet.mesh_triangles.shape[0], 1), 3), flying_carpet.mesh_triangles))
        mesh_surface = pv.PolyData(vert_flat, faces)
        mesh_SOFA = pv.PolyData(vert_tet)
        mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra])
        mesh_fb = pv.PolyData(fb_vert_flat, faces)
        vert_tet_triangles = construct_fb_triangle(tetrahedra, feedback_idx)
        errors, percentile_95, errors_tol = measure_diff_surface(gt_pts, vert_tet, vert_tet_triangles, tol=0.01)
        
        # plot_error_histogram(errors, title="Error Histogram between GT and SOFA FB points")
        print("Difference between GT and SOFA FB points:", np.mean(errors), " close: ", np.mean(errors_tol))
        errors, percentile_95, errors_tol = measure_diff_surface(gt_pts, fb_vert_flat, flying_carpet.mesh_triangles, tol)
        idx_keep = np.where(errors < tol)[0]
        gt_pts = gt_pts[idx_keep, :]
        # plot the errors
        # plot_error_histogram(errors, title="Error Histogram between GT and FB points")
        print("Difference between GT and FB points:", np.mean(errors), " close ", np.mean(errors_tol))
        # plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True,opacity=0.85, label='FB Mid-Surface')
        plotter.add_mesh(mesh_SOFA, color='lightgrey', show_edges=True, opacity=0.85, label='SOFA Surface')
        plotter.add_points(gt_pts, color='green', point_size=10, opacity=0.85, label='Ground Truth Points')
        # plotter.add_points(vert_tet_fb, color='red', point_size=10, label='FB points from SOFA')
        # plotter.show_grid()
        plotter.show()

def measure_diff(pts_gt, pts_sim, return_details: bool = False):
    """
    Measure the difference between two point clouds with unmatched indices.

    The score is the symmetric nearest-neighbor mean distance:
    ground-truth to simulation and simulation to ground-truth. Coordinates are
    assumed to use the same frame and units.

    Args:
        pts_gt (np.ndarray): Ground truth points of shape (N, 3).
        pts_sim (np.ndarray): Simulated points of shape (M, 3).
        return_details (bool): If True, also return per-direction statistics.

    Returns:
        float: The mean distance between the two sets of points.
    """
    pts_gt = _as_point_array(pts_gt, "pts_gt")
    pts_sim = _as_point_array(pts_sim, "pts_sim")

    sim_tree = cKDTree(pts_sim)
    gt_tree = cKDTree(pts_gt)
    gt_to_sim_dist, _ = sim_tree.query(pts_gt, k=1)
    sim_to_gt_dist, _ = gt_tree.query(pts_sim, k=1)

    mean_gt_to_sim = float(np.mean(gt_to_sim_dist))
    mean_sim_to_gt = float(np.mean(sim_to_gt_dist))
    diff = 0.5 * (mean_gt_to_sim + mean_sim_to_gt)

    if not return_details:
        return diff

    return {
        "mean": diff,
        "gt_to_sim_mean": mean_gt_to_sim,
        "sim_to_gt_mean": mean_sim_to_gt,
        "gt_to_sim_rmse": float(np.sqrt(np.mean(gt_to_sim_dist**2))),
        "sim_to_gt_rmse": float(np.sqrt(np.mean(sim_to_gt_dist**2))),
        "gt_to_sim_max": float(np.max(gt_to_sim_dist)),
        "sim_to_gt_max": float(np.max(sim_to_gt_dist)),
        "num_gt": int(len(pts_gt)),
        "num_sim": int(len(pts_sim)),
    }

def draw_3d(flying_carpet:Flying_carpet):
    with open("data_flying_carpet/flat_FKD.pkl", "rb") as f:
            vert_flat_list = pickle.load(f)

    for i in range(3):
        vert_flat = vert_flat_list[i]
        flying_carpet.visualize_vert(vert_flat)


if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # print("initial cable length: ", flying_carpet.get_cable_length_bary(flying_carpet.vertices))
    # get_FKD_flat(flying_carpet, cl_list)
    # get_FKD_3(flying_carpet)
    # draw_3d(flying_carpet)
    compare_SOFA_FKD(flying_carpet)

    