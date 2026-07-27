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
from scipy.spatial import cKDTree
import open3d as o3d

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
    percentile_95 = error_sorted[int(0.95 * len(error_sorted))]

    return errors, percentile_95


def downsample_random(points: np.ndarray, num_points: int, seed=None) -> np.ndarray:
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if num_points >= len(points):
        return points.copy()

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=num_points, replace=False)
    return points[indices]

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

def draw_SOFA_compare(c_srs: C_SRS_fixedEnd):
    def get_fb_verts_SOFA(vert_SOFA, feedback_vert_idx):
        fb_verts_SOFA = vert_SOFA[feedback_vert_idx]
        return fb_verts_SOFA

    file_flat = "data/fixedEnd_FKD_exp_data_forpaper_final.pkl"
    with open(file_flat, "rb") as f:
        data = pickle.load(f)
    vert_list_123 = data["vert_list"]
    pts_list_123 = data["pts_list"]
    vert_list_SOFA_123 = data["vert_list"]
    SOFA_file = "data/FKD_as_SOFA_compare.pickle"
    with open(SOFA_file, "rb") as f:
        data = pickle.load(f)
    vert_list_me = data["vert_list"]
    tetrahedra_ORIGINAL = data["tetrahedra"]
    feedback_vert_idx = data["feedback_vert_idx"]
    vert_list_SOFA = []
    for i in range(3):
        vert_SOFA, tetrahedra = read_VTK(f"data/SOFA_fixedEnd/vtk_{i+2}.vtk")
        # print("is tetrahedron original and tetrahedra same? : ", np.array_equal(tetrahedra_ORIGINAL, tetrahedra))
        vert_SOFA *= 1e-3
        vert_list_SOFA.append(vert_SOFA)
    print("length of fb verts:", len(feedback_vert_idx))
    
    with open("data/fixedEnd_FKD_exp_manual_data_1.pkl", "rb") as f:
        data_3 = pickle.load(f)
    vert_3 = data_3["vert_length"]
    pts_3 = data_3["pts"]


    vert_list = [vert_list_123[1], vert_list_123[2], vert_3]
    pts_list = [pts_list_123[1], pts_list_123[2], pts_3]
    vert_list_SOFA[0] = vert_list_me[1]
    vert_list_SOFA[1] = vert_list_me[2]
    for i in range(len(vert_list)):
        plotter = pv.Plotter()
        vert = vert_list[i]
        gt_pts = pts_list[i]
        # downsample gt_pts to 100 points for visualization
        
        vert_SOFA = vert_list_SOFA[i]
        # vert_SOFA[:, 2] -= 0.025
        # vert_SOFA[:, 0] += 0.02
        fb_vertices = c_srs.get_fb_surface(vert)
        fb_verts_SOFA = get_fb_verts_SOFA(vert_SOFA, feedback_vert_idx)
        # fb_verts_SOFA = vert_SOFA.copy()
        faces = np.hstack((np.full((c_srs.mesh_triangles.shape[0], 1), 3), c_srs.mesh_triangles))
        mesh_surface = pv.PolyData(vert, faces)
        mesh_SOFA = pv.PolyData(vert_SOFA)
        if i == 2:
            fb_triangle = construct_fb_triangle(tetrahedra, feedback_vert_idx)
            mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra])
        else:
            fb_triangle = construct_fb_triangle(tetrahedra_ORIGINAL, feedback_vert_idx)
            mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra_ORIGINAL])
        mesh_fb = pv.PolyData(fb_vertices, faces)
        diff_SOFA, diff_SOFA_95 = measure_diff_surface(gt_pts, vert_SOFA, fb_triangle)
        print("Difference between GT and SOFA FB points:", np.mean(diff_SOFA),
              " 95 percentile: ", np.mean(diff_SOFA_95))
        diff, diff_95 = measure_diff_surface(gt_pts, fb_vertices, c_srs.mesh_triangles)
        print("Difference between GT and FB points:", np.mean(diff), " 95 percentile: ", np.mean(diff_95))
        min_point = np.array([[0, -20, -100],
                              [300, 180, 100]]) * 1e-3
        plotter.add_mesh(pv.PolyData(min_point), color='white', point_size=0.1)

        # plotter.add_mesh(mesh_surface, color='lightblue', show_edges=True, opacity=0.35, label='Input Surface')
        # plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, label='FB Mid-Surface',opacity=0.85)
        plotter.add_mesh(mesh_SOFA, color='lightgrey', show_edges=True, opacity=0.75, label='SOFA Surface')
        # gt_pts = downsample_random(gt_pts, 1000, seed=42)
        plotter.add_points(gt_pts, color='green', point_size=5, label='Ground Truth Points',opacity=0.85)
        # plotter.add_points(fb_verts_SOFA, color='red', point_size=10, label='FB points from SOFA')
        # plotter.show_grid()
        plotter.show()

def read_VTK(file_path):
    """Read the vertices and tetrahedral connectivity from a VTK mesh.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to a VTK unstructured-grid file.

    Returns
    -------
    mesh_vtk : numpy.ndarray
        Vertex coordinates with shape ``(n_vertices, 3)``. Coordinates retain
        the units used by the input file.
    tetrahedron : numpy.ndarray
        Zero-based vertex indices with shape ``(n_tetrahedra, 4)``.
    """
    file_path = os.fspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"VTK file does not exist: {file_path}")

    vtk_grid = pv.read(file_path)
    mesh_vtk = np.asarray(vtk_grid.points).copy()
    if mesh_vtk.ndim != 2 or mesh_vtk.shape[1] != 3:
        raise ValueError(
            f"Expected 3-D VTK points, but received an array with shape "
            f"{mesh_vtk.shape} from {file_path}"
        )

    # VTK cell type 10 is a four-node linear tetrahedron.  Using cells_dict
    # avoids assuming that every cell has the same size or type.
    cells_dict = getattr(vtk_grid, "cells_dict", {})
    tetrahedron = cells_dict.get(pv.CellType.TETRA)
    if tetrahedron is None:
        raise ValueError(f"No tetrahedral cells were found in VTK file: {file_path}")

    tetrahedron = np.asarray(tetrahedron, dtype=np.int64).reshape(-1, 4).copy()
    return mesh_vtk, tetrahedron

def get_diff():
    pass


if __name__ == "__main__":
    cl_list_2 = (np.array([416,430 , 436, 302, 270, 286]) * 1e-3).tolist()
    cl_list_3 = (np.array([443, 442, 433, 277,264,292]) * 1e-3).tolist()
    cl_list_4 = (np.array([363,357,363, 333,305,335]) * 1e-3).tolist()
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    draw_SOFA_compare(c_srs)

