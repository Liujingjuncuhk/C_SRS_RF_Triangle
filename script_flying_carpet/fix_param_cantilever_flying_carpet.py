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
import open3d as o3d
import csv
import argparse
from scipy.spatial import cKDTree
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from flying_carpet_cantilever import Flying_carpet_fixedEnd
import pickle

DEFAULT_FILTERED_REGION = [0.02, 0.265, 0, 0.16, -0.05, 0.02]

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

def filter_points(points, region):
    """
    Filter points within a specified 3D region.

    Args:
        points (np.ndarray): Input points of shape (N, 3).
        region (list or tuple): A list or tuple of 6 values specifying the
                                bounding box [x_min, x_max, y_min, y_max, z_min, z_max].

    Returns:
        np.ndarray: Filtered points within the specified region.
    """
    points = _as_point_array(points, "points")
    x_min, x_max, y_min, y_max, z_min, z_max = region
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]

def find_best_Youngs_modulus(
    pts_gt,
    c_srs: Flying_carpet_fixedEnd,
    E_list,
    save_folder,
    filtered_region=None,
):
    """
    Find the best Young's modulus and save all fitting results.

    Args:
        pts_gt (np.ndarray): Ground truth points of shape (N, 3).
        c_srs (Flying_carpet_fixedEnd): The flying-carpet fixed-end object.
        E_list (list): Young's modulus values to evaluate.
        save_folder (str): Folder where screenshots, CSV, plot, and summary are saved.
        filtered_region (list): Optional [xmin, xmax, ymin, ymax, zmin, zmax].
    """
    E_list = np.asarray(E_list, dtype=float).reshape(-1)
    if len(E_list) == 0:
        raise ValueError("E_list must contain at least one Young's modulus value.")
    if filtered_region is None:
        filtered_region = DEFAULT_FILTERED_REGION

    os.makedirs(save_folder, exist_ok=True)
    best_E = None
    best_diff = float("inf")
    poission_ratio = c_srs.Poisson_ratio
    results = []
    for E in E_list:
        c_srs.reassemble_stiffness_matrices(E, poission_ratio)
        Q_list = c_srs.FKD_free_static()
        pts_fb = c_srs.get_fb_surface(Q_list[-1])
        pts_fb = filter_points(pts_fb, filtered_region)
        diff_info = measure_diff(pts_gt, pts_fb, return_details=True)
        diff = diff_info["mean"]
        vertices = c_srs.q_to_vertices(Q_list[-1])

        plotter = c_srs.visualize_fb_surface_w_gt(vertices, pts_gt)
        screenshot_file = os.path.join(save_folder, f"fb_surface_E_{E:.2e}.png")
        plotter.screenshot(screenshot_file)
        plotter.close()

        results.append({
            "E": float(E),
            "screenshot": screenshot_file,
            **diff_info,
        })
        print(f"E: {E:.2e}, Error: {diff:.6f}")

        if diff < best_diff:
            best_diff = diff
            best_E = E
            best_vertices = vertices
            best_screenshot_file = screenshot_file

    print(f"Best Young's modulus: {best_E}, Best difference: {best_diff}")
    best_plotter = c_srs.visualize_fb_surface_w_gt(best_vertices, pts_gt)
    best_plotter.screenshot(os.path.join(save_folder, "best_fit.png"))
    best_plotter.close()

    csv_file = os.path.join(save_folder, "difference_vs_youngs_modulus.csv")
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "E",
            "mean",
            "gt_to_sim_mean",
            "sim_to_gt_mean",
            "gt_to_sim_rmse",
            "sim_to_gt_rmse",
            "gt_to_sim_max",
            "sim_to_gt_max",
            "num_gt",
            "num_sim",
            "screenshot",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(E_list, [result["mean"] for result in results], marker='o')
    plt.axvline(best_E, color="red", linestyle="--", label=f"best E={best_E:.2e}")
    plt.xlabel("Young's Modulus (Pa)")
    plt.ylabel("Difference (m)")
    plt.title("Difference vs Young's Modulus")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_folder, "difference_vs_youngs_modulus.png"))
    plt.close()

    summary_file = os.path.join(save_folder, "best_youngs_modulus.txt")
    with open(summary_file, "w") as f:
        f.write(f"best_E,{best_E}\n")
        f.write(f"best_difference,{best_diff}\n")
        f.write(f"best_screenshot,{best_screenshot_file}\n")

    return best_E, best_diff

def view_best_fitted_param(pts_gt, c_srs: Flying_carpet_fixedEnd, best_E):
    """
    Visualize the best fitted parameters by simulating the surface with the
    best Young's modulus and comparing it to the ground truth points.

    Args:
        pts_gt (np.ndarray): Ground truth points of shape (N, 3).
        c_srs (C_SRS_fixedEnd): The C_SRS_fixedEnd object.
        best_E (float): The best Young's modulus found.
    """
    poission_ratio = c_srs.Poisson_ratio
    c_srs.reassemble_stiffness_matrices(best_E, poission_ratio)
    Q_list = c_srs.FKD_free_static()
    pts_fb = c_srs.get_fb_surface(Q_list[-1])
    filtered_region = [0.02, 0.265, 0, 0.16, -0.05, 0.02]
    pts_fb = filter_points(pts_fb, filtered_region)
    vertices = c_srs.q_to_vertices(Q_list[-1])
    c_srs.visualize_fb_surface_w_gt(vertices, pts_gt)


def save_plots_and_pics(pts_gt, c_srs: Flying_carpet_fixedEnd, E_list, save_folder="results_fitE"):
    """
    Save fitting pictures while finding the best Young's modulus.

    Args:
        pts_gt (np.ndarray): Ground truth points of shape (N, 3).
        c_srs (Flying_carpet_fixedEnd): The flying-carpet fixed-end object.
        E_list (list): List of Young's modulus values to evaluate.
        save_folder (str): Folder where results are saved.
    """
    return find_best_Youngs_modulus(pts_gt, c_srs, E_list, save_folder)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Young's modulus for the flying-carpet cantilever data and save all results."
    )
    parser.add_argument(
        "save_folder",
        nargs="?",
        default="results_fitE_flying_carpet",
        help="Folder where screenshots, CSV, plot, and best-fit summary are saved.",
    )
    parser.add_argument(
        "--description-file",
        default="./models/flying_carpet/flying_carpet_description_bary.pkl",
        help="Flying-carpet description pickle file.",
    )
    parser.add_argument(
        "--pts-gt-file",
        default="data/cantilever_gravity_fixedEnd_flying_carpet.pkl",
        help="Ground-truth point cloud pickle file.",
    )
    args = parser.parse_args()

    description_file = args.description_file
    flying_carpet = Flying_carpet_fixedEnd(description_file)
    E_range = []
    E_range.extend(np.arange(0.1e7, 3.5e7, 0.1e7))
    E_range.extend(np.arange(3.5e7+0.01e7, 3.6e7, 0.01e7))
    E_range.extend(np.arange(3.6e7+0.1e7,1.0e8, 0.1e7))

    with open(args.pts_gt_file, "rb") as f:
        pts_gt = pickle.load(f)

    best_E, best_diff = find_best_Youngs_modulus(
        pts_gt,
        flying_carpet,
        E_range,
        save_folder=args.save_folder,
    )
    print(f"Saved fitting results to: {args.save_folder}")
    print(f"Best Young's modulus: {best_E:.6e}, Best difference: {best_diff:.6f}")
    
