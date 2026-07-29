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
from matplotlib import pyplot as plt

def add_translation(ee_pos, translation_vector):
    ee_pos_translated = ee_pos.copy()
    for i in range(ee_pos.shape[0]):
        ee_pos_translated[i] += translation_vector
    return ee_pos_translated

def generate_dense_ee(ee_list, num_points=100):
    """Linearly interpolate a sequence of 8-point end-effector poses.

    Parameters
    ----------
    ee_list : sequence of array-like
        Planned poses.  Each pose must have shape ``(8, 3)``.
    num_points : int
        Total number of poses in the returned trajectory, including all
        planned poses.

    Returns
    -------
    list of numpy.ndarray
        A list of exactly ``num_points`` poses.  Each pose has shape ``(8, 3)``.
    """
    ee_waypoints = np.asarray(ee_list, dtype=float)

    if ee_waypoints.ndim != 3 or ee_waypoints.shape[1:] != (8, 3):
        raise ValueError(
            "ee_list must contain at least one waypoint with shape (8, 3)"
        )
    if not isinstance(num_points, (int, np.integer)) or isinstance(num_points, bool):
        raise TypeError("num_points must be an integer")
    if num_points < len(ee_waypoints):
        raise ValueError(
            "num_points must be at least the number of planned waypoints "
            "so that every waypoint can be included"
        )

    if len(ee_waypoints) == 1:
        return [ee_waypoints[0].copy() for _ in range(num_points)]

    # Allocate the (num_points - 1) sampling intervals as evenly as possible
    # among waypoint segments.  Sampling each segment without its endpoint
    # avoids duplicates; the next segment contributes that endpoint instead.
    num_segments = len(ee_waypoints) - 1
    intervals_per_segment = np.full(
        num_segments, (num_points - 1) // num_segments, dtype=int
    )
    intervals_per_segment[:(num_points - 1) % num_segments] += 1

    dense_segments = []
    for segment_index, num_intervals in enumerate(intervals_per_segment):
        alpha = np.arange(num_intervals, dtype=float) / num_intervals
        start = ee_waypoints[segment_index]
        end = ee_waypoints[segment_index + 1]
        dense_segments.append(
            (1.0 - alpha[:, None, None]) * start
            + alpha[:, None, None] * end
        )

    dense_segments.append(ee_waypoints[-1:])
    dense_trajectory = np.concatenate(dense_segments, axis=0)
    return [pose.copy() for pose in dense_trajectory]

def visualize_ee_target(ee_target_dense):
    """Display the interpolated 3-D trajectory of each end-effector point."""
    try:
        ee_target_dense = np.stack(ee_target_dense).astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ee_target_dense must be a non-empty list of (8, 3) arrays"
        ) from error

    if (
        ee_target_dense.ndim != 3
        or ee_target_dense.shape[1:] != (8, 3)
        or len(ee_target_dense) == 0
    ):
        raise ValueError(
            "ee_target_dense must have shape (num_points, 8, 3)"
        )

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
    ]
    plotter = pv.Plotter()

    for point_index in range(ee_target_dense.shape[1]):
        trajectory = ee_target_dense[:, point_index, :]
        color = colors[point_index]

        if len(trajectory) > 1:
            plotter.add_mesh(
                pv.lines_from_points(trajectory, close=False),
                color=color,
                line_width=4,
                label=f"Point {point_index + 1}",
            )
        plotter.add_points(
            trajectory,
            color=color,
            point_size=7,
            render_points_as_spheres=True,
        )

    plotter.show_axes()
    plotter.show_grid()
    plotter.add_legend()
    plotter.reset_camera()
    plotter.show(title="Interpolated End-Effector Trajectories")
    return plotter

def plan_pick_place(flying_carpet:Flying_carpet):
    pick_place = np.array([280, 200, 100])
    place_place = np.array([280, 565, 100])
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_bent = pickle.load(f)
    filename = "./data_flying_carpet/110mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_grasping = pickle.load(f)
    
    ee_list = []
    ee_list.append(add_translation(ee_pos_bent, [0.28, 0.38, 0.20]))
    ee_list.append(add_translation(ee_pos_bent, [0.28, 0.2, 0.20]))
    ee_list.append(add_translation(ee_pos_bent, [0.28, 0.2, 0.15]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.2, 0.12]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.2, 0.2]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.3, 0.2]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.38, 0.2]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.48, 0.2]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.565, 0.2]))
    ee_list.append(add_translation(ee_pos_grasping, [0.28, 0.565, 0.12]))
    ee_list.append(add_translation(ee_pos_bent, [0.28, 0.565, 0.2]))
    ee_list.append(add_translation(ee_pos_bent, [0.28, 0.38, 0.25]))

    # ee_target_dense = generate_dense_ee(ee_list, 30)
    # visualize_ee_target(ee_target_dense)
    cl_list = []
    vert_list = []
    for i in range(len(ee_list)):
        ee_target_pos = ee_list[i]
        # starting_vert = flying_carpet.get_fixedEE_guess_vertices(ee_target_pos)
        # starting_length = flying_carpet.get_cable_length_bary(starting_vert)
        # final_vert = starting_vert.copy()
        # Q_list, final_vert, cable_tension = flying_carpet.FKD_time(starting_length, 10, starting_vert, tol = 1e-4, h = 0.1, show_info=True)
        # final_length = flying_carpet.get_cable_length_bary(final_vert)
        # starting_vertices = flying_carpet.vertices
        final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, flying_carpet.vertices, max_iter=30, tol=8e-3, show_info = True, initial_guess=True)
        cl_list.append(final_length)
        vert_list.append(final_vert)
        # starting_vertices = final_vert.copy()
        flying_carpet.visualize_IKD_result(ee_target_pos, final_vert)

    data_2save = {"ee_target_list": ee_list, "cl_list": cl_list, "vert_list": vert_list}
    with open("data_flying_carpet/pick_place_data.pkl", "wb") as f:
        pickle.dump(data_2save, f)


def view_trajectory(flying_carpet: Flying_carpet, traj_file = "data_flying_carpet/pick_and_place_trajectory_tro_paper.pickle"):
    with open(traj_file, 'rb') as f:
        # Load the pick-and-place command list from the pickle file
        pick_place_commands = pickle.load(f)
    length_cmd = pick_place_commands['cable_length_list']

    for i in range(len(length_cmd)):
        cl_cmd = length_cmd[i]
        startineg_vertices = flying_carpet.vertices
        Q_list, starting_vertices, cable_tension = flying_carpet.FKD_time(cl_cmd, 10, startineg_vertices, tol = 1e-6, h = 0.1, show_info=True)
        flying_carpet.visualize_vert(starting_vertices)




if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    plan_pick_place(flying_carpet)
    # view_trajectory(flying_carpet)
