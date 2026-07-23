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
    """Return ``nSamples`` points evenly spaced along the full waypoint path."""
    waypoints = np.asarray(ee_target_list, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        raise ValueError("ee_target_list must be a non-empty 2D array.")
    if nSamples < 2:
        raise ValueError("nSamples must be at least 2 to include both endpoints.")

    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]

    if total_length == 0.0:
        return np.repeat(waypoints[:1], nSamples, axis=0)

    sample_lengths = np.linspace(0.0, total_length, nSamples)
    dense_targets = np.empty((nSamples, waypoints.shape[1]))
    for coordinate in range(waypoints.shape[1]):
        dense_targets[:, coordinate] = np.interp(
            sample_lengths, cumulative_lengths, waypoints[:, coordinate]
        )
    return dense_targets

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
    picklefile = "./data/IKD_traj_result.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['target_list']
    length_cmd_list = traj_result['length_cmd_list']
    # draw the length cmd list using matplotlib
    draw_cl_cmd(length_cmd_list)

    vert_list = traj_result['vert_list']
    c_srs.replay_IKD_trajectory(ee_target_list, vert_list, framerate = 20, filePath = "IKD_parallelogram_traj.mp4")
    