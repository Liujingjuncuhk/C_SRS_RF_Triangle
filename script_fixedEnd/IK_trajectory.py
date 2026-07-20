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

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    ee_target_list = np.array([[0.25, 0.08, 0.02],
                               [0.24, 0.08, 0.06],
                               [0.23, 0.08, 0.06],
                               [0.24, 0.08, 0.02],
                               [0.25, 0.08, 0.02]])
    dense_targets = dense_ee_target(ee_target_list, nSamples=20)
    # ee_target_list = np.array([[0.25, 0.08, 0.02]])
    # c_srs.visualize_planned_traj(c_srs.vertices, ee_target_list)
    length_cmd_list = []
    vert_list = []
    starting_vert = c_srs.vertices
    for ee_target in dense_targets:
        tcl = c_srs.ikModel.predict_cable_length(ee_target)
        # print(tcl)   # → 6 cable lengths in metres
        Q_list, cable_tension = c_srs.FKD_static_length(starting_vert, tcl, tol = 1e-6)
        starting_vert = c_srs.q_to_vertices(Q_list[-1])
        cur_length, starting_vertices, Q_list = c_srs.IKD_single(ee_target, starting_vert,AA = False, tol=1e-3)
        length_cmd_list.append(cur_length)
        vert_list.append(starting_vertices)
    # save the length_cmd_list and vert_list to a pickle file
    dump_data = {'target_list': dense_targets, 'length_cmd_list': length_cmd_list, 'vert_list': vert_list}
    with open('data/IKD_traj_result.pkl', 'wb') as f:
        pickle.dump(dump_data, f)