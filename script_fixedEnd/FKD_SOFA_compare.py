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
    vert_list_SOFA[0] = vert_list_SOFA_123[1]
    vert_list_SOFA[1] = vert_list_SOFA_123[2]
    for i in range(len(vert_list)):
        plotter = pv.Plotter()
        vert = vert_list[i]
        gt_pts = pts_list[i]
        vert_SOFA = vert_list_SOFA[i]
        # vert_SOFA[:, 2] -= 0.025
        # vert_SOFA[:, 0] += 0.02
        fb_vertices = c_srs.get_fb_surface(vert)
        fb_verts_SOFA = get_fb_verts_SOFA(vert_SOFA, feedback_vert_idx)
        faces = np.hstack((np.full((c_srs.mesh_triangles.shape[0], 1), 3), c_srs.mesh_triangles))
        mesh_surface = pv.PolyData(vert, faces)
        mesh_SOFA = pv.PolyData(vert_SOFA)
        mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra])
        mesh_fb = pv.PolyData(fb_vertices, faces)
        diff_SOFA = measure_diff(gt_pts, vert_SOFA, return_details=True)
        print("Difference between GT and SOFA FB points:", diff_SOFA["sim_to_gt_mean"])
        diff = measure_diff(gt_pts, fb_vertices, return_details=True)
        print("Difference between GT and FB points:", diff["sim_to_gt_mean"])

        # plotter.add_mesh(mesh_surface, color='lightblue', show_edges=True, opacity=0.35, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, label='FB Mid-Surface')
        # plotter.add_mesh(mesh_SOFA, color='lightcoral', show_edges=True, opacity=0.5, label='SOFA Surface')
        plotter.add_points(gt_pts, color='green', point_size=10, label='Ground Truth Points')
        plotter.add_points(fb_verts_SOFA, color='red', point_size=10, label='FB points from SOFA')
        plotter.show_grid()
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

