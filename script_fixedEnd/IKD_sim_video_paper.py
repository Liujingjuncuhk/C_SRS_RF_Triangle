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
import matplotlib.pyplot as plt

def dense_ee_target(ee_target_list, nSamples):
    """Resample a path while preserving every planned waypoint exactly."""
    waypoints = np.asarray(ee_target_list, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        raise ValueError("ee_target_list must be a non-empty 2D array.")
    if nSamples < 2:
        raise ValueError("nSamples must be at least 2 to include both endpoints.")
    if nSamples < len(waypoints):
        raise ValueError(
            "nSamples must be at least the number of planned waypoints so that "
            "every waypoint can be included."
        )
    if len(waypoints) == 1:
        return np.repeat(waypoints, nSamples, axis=0)

    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.all(segment_lengths == 0.0):
        return np.repeat(waypoints[:1], nSamples, axis=0)

    # Assign at least one interval to every segment. Distribute the remaining
    # intervals by segment length, using largest remainders to keep the total
    # number of returned samples exactly equal to nSamples.
    interval_counts = np.ones(len(segment_lengths), dtype=int)
    remaining = nSamples - len(waypoints)
    if remaining:
        shares = remaining * segment_lengths / segment_lengths.sum()
        extra_intervals = np.floor(shares).astype(int)
        intervals_left = remaining - extra_intervals.sum()
        if intervals_left:
            largest_remainders = np.argsort(-(shares - extra_intervals))
            extra_intervals[largest_remainders[:intervals_left]] += 1
        interval_counts += extra_intervals

    dense_targets = [waypoints[0].copy()]
    for start, end, count in zip(
        waypoints[:-1], waypoints[1:], interval_counts
    ):
        # Exclude the segment endpoints from interpolation. The start is
        # already present, and directly appending the end guarantees an exact
        # copy of the planned waypoint without floating-point reconstruction.
        ratios = np.arange(1, count, dtype=float) / count
        dense_targets.extend(start + ratios[:, None] * (end - start))
        dense_targets.append(end.copy())

    return np.asarray(dense_targets)

def draw_cl_cmd(length_cmd_list):
    # length_cmd_list is a list of list, draw for each element
    cl = [[] for _ in range(6)]
    for length_cmd in length_cmd_list:
        for i in range(6):
            cl[i].append(length_cmd[i])
    # use subfigures
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(cl[i], label=f"Cable {i+1}")

    plt.xlabel("Time step")
    plt.ylabel("Cable length command")
    plt.title("Cable length command over time")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    picklefile = "./data/IKD_traj_result_paral_new.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['target_list']
    length_cmd_list = traj_result['length_cmd_list']
    # draw the length cmd list using matplotlib
    draw_cl_cmd(length_cmd_list)

    vert_list = traj_result['vert_list']
    c_srs.replay_IKD_trajectory(ee_target_list, vert_list, framerate = 5, filePath = "IKD_paral_traj_new.mp4")
