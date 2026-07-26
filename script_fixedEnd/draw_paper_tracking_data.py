import csv
from pathlib import Path
import numpy as np
import pyvista as pv
import os
import sys
import inspect
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle

pickleFile_name = ["data_tracking_ball/parallelogram_data_30s.pkl",
                   "data_tracking_ball/parallelogram_data_20s.pkl",
                   "data_tracking_ball/parallelogram_data_10s.pkl",
                   "data_tracking_ball/parallelogram_data_5s.pkl",
                   "data_tracking_ball/triangle_data_30s.pkl",
                   "data_tracking_ball/triangle_data_20s.pkl",
                   "data_tracking_ball/triangle_data_10s.pkl",
                   "data_tracking_ball/triangle_data_5s.pkl"]



def get_ave_diff(array1, array2):
    assert len(array1) == len(array2), "Arrays must have the same length"
    diff_list = []
    for i in range(len(array1)):
        diff_list.append(np.linalg.norm(array1[i] - array2[i]))
    print("mean diff:", np.mean(diff_list))
    return np.mean(diff_list)

def plot_xyz_target(target_position, recorded_position, total_t, save_path=None, show=True):
    """Plot target and recorded X, Y and Z positions against time."""
    target_position = np.asarray(target_position, dtype=float)
    recorded_position = np.asarray(recorded_position, dtype=float)

    if target_position.ndim != 2 or target_position.shape[1] != 3:
        raise ValueError("target_position must have shape (number_of_samples, 3)")
    if recorded_position.ndim != 2 or recorded_position.shape[1] != 3:
        raise ValueError("recorded_position must have shape (number_of_samples, 3)")
    if target_position.shape[0] == 0:
        raise ValueError("target_position and recorded_position cannot be empty")
    if target_position.shape[0] != recorded_position.shape[0]:
        raise ValueError("target_position and recorded_position must have equal length")
    if not np.isfinite(total_t) or total_t < 0:
        raise ValueError("total_t must be a non-negative finite number")

    time_values = np.linspace(0.0, total_t, target_position.shape[0])
    coordinate_names = ("X", "Y", "Z")

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for coordinate_index, (axis, coordinate_name) in enumerate(
        zip(axes, coordinate_names)
    ):
        axis.plot(
            time_values,
            target_position[:, coordinate_index],
            color="tab:blue",
            linewidth=1.5,
            label="Target",
        )
        axis.plot(
            time_values,
            recorded_position[:, coordinate_index],
            color="tab:red",
            linewidth=1.3,
            label="Recorded EE",
        )
        axis.set_ylabel(f"{coordinate_name} position (m)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Target and Recorded End-Effector Positions")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes


def plot_target_record_3d(target_list, ee_positions_recorded, save_path=None, show=True):
    """Plot the 3D trajectory of the target and recorded end-effector positions."""
    target_list = np.asarray(target_list, dtype=float)
    ee_positions_recorded = np.asarray(ee_positions_recorded, dtype=float)

    if target_list.ndim != 2 or target_list.shape[1] != 3:
        raise ValueError("target_list must have shape (number_of_samples, 3)")
    if ee_positions_recorded.ndim != 2 or ee_positions_recorded.shape[1] != 3:
        raise ValueError("ee_positions_recorded must have shape (number_of_samples, 3)")
    if target_list.shape[0] == 0 or ee_positions_recorded.shape[0] == 0:
        raise ValueError("target_list and ee_positions_recorded cannot be empty")

    # minx = 0.025, maxx = 

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection='3d')

    ax.plot(target_list[:, 0], target_list[:, 1], target_list[:, 2], label='Target', color='tab:blue')
    ax.plot(ee_positions_recorded[:, 0], ee_positions_recorded[:, 1], ee_positions_recorded[:, 2], label='Recorded EE', color='tab:red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Give X, Y and Z the same numeric span and the same visual scale.
    all_positions = np.vstack((target_list, ee_positions_recorded))
    data_min = np.min(all_positions, axis=0)
    data_max = np.max(all_positions, axis=0)
    axis_centers = (data_min + data_max) / 2.0
    equal_range = np.max(data_max - data_min)
    if equal_range == 0.0:
        equal_range = 1.0
    half_range = equal_range * 0.525

    ax.set_xlim(axis_centers[0] - half_range, axis_centers[0] + half_range)
    ax.set_ylim(axis_centers[1] - half_range, axis_centers[1] + half_range)
    ax.set_zlim(axis_centers[2] - half_range, axis_centers[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax


def draw_target_w_mesh(c_srs:C_SRS_fixedEnd, dense_target, ee_pos_recorded):
    vertices = c_srs.vertices
    mesh = pv.PolyData(vertices, np.hstack((np.full((c_srs.mesh_triangles.shape[0], 1), 3), c_srs.mesh_triangles)))

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    pp_locations = c_srs.get_pp_location_bary(vertices)
    plotter.add_points(pp_locations, color='blue', point_size=10
                        , label='Pullpoints')
    plotter.add_points(c_srs.pulley_location, color='blue', point_size=10
                        , label='Pulleys')
    # add lines between pullpoints and pulleys
    for i in range(c_srs.nCable):
        plotter.add_lines(np.array([pp_locations[i], c_srs.pulley_location[i]]), color='blue', width=2)
    # annotate ee vertices
    plotter.add_points(vertices[c_srs.ee_idx], color='red', point_size=10, label='End Effectors')
    # add all points in ws_pts as cyan points
    plotter.add_points(c_srs.ee_pos_list, color='cyan', point_size=5, label='WS Points')

    # make fixed idx black
    plotter.add_points(vertices[c_srs.fixed_idx], color='black', point_size=10, label='Fixed Vertices')
    # add target and ee_pos_recorded as lines
    for i in range(1, len(dense_target)):
        plotter.add_lines(np.array([dense_target[i-1], dense_target[i]]), color='green', width=2)
        plotter.add_lines(np.array([ee_pos_recorded[i-1], ee_pos_recorded[i]]), color='red', width=2)
    
    # add grid
    plotter.show_grid()
    plotter.show_axes()
    plotter.add_legend()
    plotter.show()

if __name__ == "__main__":
    traj_file = 'data/IKD_traj_result_triangle.pkl'
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    ave_diff_list = []
    for i in range(len(pickleFile_name)):
        picklefile = pickleFile_name[i]
        with open(picklefile, 'rb') as f:
            data = pickle.load(f)
        total_time = data["cut_time"]
        dense_target = data["dense_target"]
        ee_pos_recorded = data["ee_positions_recorded"]
        print("length of ee_positions_recorded:", len(ee_pos_recorded))
        # diff = get_ave_diff(dense_target, ee_pos_recorded)
        draw_target_w_mesh(c_srs, dense_target, ee_pos_recorded)
        # ave_diff_list.append(diff)