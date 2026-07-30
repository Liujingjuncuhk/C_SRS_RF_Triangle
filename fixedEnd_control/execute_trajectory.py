import os
import sys
import inspect
import time
import numpy as np
from fixedEnd_sys import FixedEndSystem
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

filtered_region = [0.02, 0.3, 0, 0.16, -0.1, 0.2]
# traj_file = 'data/IKD_traj_result_triangle_half_mirror_smoothed.pkl'
traj_file = 'data/IKD_traj_result_paral_new.pkl'
def get_t_list(target_list, total_time):
    cartesian_dist_list = []
    total_cartesian_dist = 0
    for i in range(1, len(target_list)):
        dist = np.linalg.norm(target_list[i] - target_list[i-1])
        cartesian_dist_list.append(dist)
        total_cartesian_dist += dist

    t_list = [0]
    for dist in cartesian_dist_list:
        ttoAppend = t_list[-1] + dist / total_cartesian_dist * total_time
        ttoAppend = float(ttoAppend)
        t_list.append(ttoAppend)

    return t_list


def smooth_cl_input(time_list, length_cmd_list, Hz = 10):
    """Interpolate six-cable commands while retaining every input waypoint.

    The returned timestamps are spaced at the requested nominal frequency
    within each waypoint interval. Each original timestamp and its associated
    cable command are copied exactly into the dense result.
    """
    waypoint_times = np.asarray(time_list, dtype=float).reshape(-1)
    waypoint_commands = np.asarray(length_cmd_list, dtype=float)

    if len(waypoint_times) < 2:
        raise ValueError("time_list must contain at least two waypoints.")
    if not np.isfinite(waypoint_times).all():
        raise ValueError("time_list must contain only finite values.")
    if np.any(np.diff(waypoint_times) <= 0):
        raise ValueError("time_list must be strictly increasing.")
    if waypoint_commands.ndim != 2 or waypoint_commands.shape[1] != 6:
        raise ValueError(
            "length_cmd_list must have shape (number_of_waypoints, 6)."
        )
    if len(waypoint_commands) != len(waypoint_times):
        raise ValueError(
            "time_list and length_cmd_list must contain the same number "
            "of waypoints."
        )
    if not np.isfinite(waypoint_commands).all():
        raise ValueError("length_cmd_list must contain only finite values.")
    if not np.isfinite(Hz) or Hz <= 0:
        raise ValueError("Hz must be a positive finite number.")

    sample_period = 1.0 / Hz
    dense_times = [waypoint_times[0]]
    waypoint_indices = [0]

    for start_time, end_time in zip(waypoint_times[:-1], waypoint_times[1:]):
        duration = end_time - start_time
        interior_offsets = (
            np.arange(1, int(np.floor(duration * Hz)) + 1) * sample_period
        )
        interior_offsets = interior_offsets[
            (interior_offsets < duration)
            & ~np.isclose(interior_offsets, duration)
        ]
        dense_times.extend(start_time + interior_offsets)
        dense_times.append(end_time)
        waypoint_indices.append(len(dense_times) - 1)

    dense_times = np.asarray(dense_times, dtype=float)
    spline = CubicSpline(waypoint_times, waypoint_commands, axis=0)
    dense_commands = spline(dense_times)

    # CubicSpline interpolates its knots, but assign the source values directly
    # so the returned trajectory contains bit-for-bit copies of all commands.
    dense_commands[waypoint_indices] = waypoint_commands

    return dense_times.tolist(), dense_commands.tolist()

def execute_traj(fixedEnd_sys:FixedEndSystem, target_list ,planned_total_time, length_cmd_list):
    t_list = get_t_list(target_list, planned_total_time)
    # print(t_list)
    fixedEnd_sys.move_to_length_timed(length_cmd_list[0], consumed_time=3)
    input("press to execute traj")
    fixedEnd_sys.execute_traj(t_list, length_cmd_list)

def view_dense_trajectory(dense_times, dense_commands):
    """Plot the dense length command for each of the six cables over time."""
    dense_commands = np.asarray(dense_commands, dtype=float)
    dense_times = np.asarray(dense_times, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cable_idx in range(6):
        ax.plot(
            dense_times,
            dense_commands[:, cable_idx],
            marker="o",
            label=f"Cable {cable_idx + 1}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cable length (m)")
    ax.set_title("Planned cable-length trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

def view_planned_trajectory(target_list, planned_total_time, length_cmd_list):
    """Plot the planned length command for each of the six cables over time."""
    t_list = get_t_list(target_list, planned_total_time)
    length_commands = np.asarray(length_cmd_list, dtype=float)

    if length_commands.ndim != 2 or length_commands.shape[1] != 6:
        raise ValueError(
            "length_cmd_list must have shape (number_of_waypoints, 6)."
        )
    if length_commands.shape[0] != len(t_list):
        raise ValueError(
            "target_list and length_cmd_list must contain the same number "
            "of waypoints."
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for cable_idx in range(6):
        ax.plot(
            t_list,
            length_commands[:, cable_idx],
            marker="o",
            label=f"Cable {cable_idx + 1}",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cable length (m)")
    ax.set_title("Planned cable-length trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax

def move_and_take_fb(fixedEnd_sys:FixedEndSystem, target_list, length_cmd_list, vert_list):
    nSample = len(target_list)
    filtered_region = [0.02, 0.3, 0, 0.16, -0.02, 0.2]
    for i in range(nSample):
        tcl = length_cmd_list[i]
        target = target_list[i]
        vert = vert_list[i]
        # print("cl cmd: ", tcl)
        fixedEnd_sys.move_to_length_timed(tcl, consumed_time=1)
        time.sleep(1)
        pcd = fixedEnd_sys.camera.get_depth_pointcloud(region = filtered_region)
        pts_points = np.asarray(pcd.points)
        plotter = fixedEnd_sys.c_srs.visualize_fb_surface_w_gt(vert, pts_points)
        input("press to next")

if __name__ == "__main__":
    
    out_region = [0.22, 0.25, 0, 0.16, 0, 0.06]
    fixedEnd_sys = FixedEndSystem()
    icl = fixedEnd_sys.get_cur_length()

    with open(traj_file, 'rb') as f:
        cl_data = pickle.load(f)
        target_list = cl_data['target_list']
        target_list = np.array(target_list)
        length_cmd_list = cl_data['length_cmd_list']
        vert_list = cl_data['vert_list']
    
    # print("length of length cmd list: ", len(length_cmd_list))
    # print("length of ", len(length_cmd_list[0]))

    planned_total_time = 10


    # t_list range from 0 to total_time, with length same as nSample
    t_list = get_t_list(target_list, planned_total_time)
    dense_times, dense_commands = smooth_cl_input(t_list, length_cmd_list, Hz = 30)
    # view_dense_trajectory(dense_times, dense_commands)
    fixedEnd_sys.move_to_length_timed(length_cmd_list[0], consumed_time=3)
    input("press to execute traj")
    fixedEnd_sys.execute_traj(dense_times, dense_commands)
    # move_and_take_fb(fixedEnd_sys, target_list, length_cmd_list, vert_list)
    # execute_traj(fixedEnd_sys,target_list, planned_total_time, length_cmd_list)
    # view_planned_trajectory(target_list, planned_total_time, length_cmd_list)

    input("press to move back")
    fixedEnd_sys.move_to_length_timed(icl, consumed_time=3)
