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

def get_total_dis(ee_target_list):
    """Calculate the total distance of a path defined by waypoints."""
    waypoints = np.asarray(ee_target_list, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] == 0:
        raise ValueError("ee_target_list must be a non-empty 2D array.")
    if waypoints.shape[1] != 3:
        raise ValueError("Each waypoint must have exactly three coordinates (X, Y, Z).")
    dist = 0
    for i in range(len(waypoints) - 1):
        dist += np.linalg.norm(waypoints[i + 1] - waypoints[i])
    return dist

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    # ee_target_list = np.array([[0.26, 0.08, 0.03],
    #                            [0.25, 0.08, 0.07],
    #                            [0.22, 0.08, 0.08],
    #                            [0.23, 0.08, 0.04],
    #                            [0.26, 0.08, 0.03]]) # parallelogram

    ee_target_list = np.array([[0.26, 0.08, 0.03],
                           [0.25, 0.08, 0.07],
                           [0.23, 0.08, 0.08],
                           [0.24, 0.08, 0.04],
                           [0.26, 0.08, 0.03]]) # parallelogram (faked)

    # ee_target_list = np.array([[0.26, 0.08, 0.03],
    #                            [0.24, 0.06, 0.07],
    #                            [0.24, 0.1, 0.07],
    #                            [0.26, 0.08, 0.03]]) # triangle

    # ee_target_list = np.array([[0.26, 0.08, 0.03],
    #                                [0.24, 0.06, 0.07],
    #                                [0.24, 0.08, 0.07]]) # triangle half

    total_dis = get_total_dis(ee_target_list)
    print("total distance of the path:", total_dis)
    time_list = [30, 20, 10, 5]
    for total_time in time_list:
        print("speed:", total_dis/total_time)
    # exit(0)
    dense_targets = dense_ee_target(ee_target_list, nSamples=20)
    # ee_target_list = np.array([[0.25, 0.08, 0.02]])
    # c_srs.visualize_planned_traj(c_srs.vertices, ee_target_list)
    # exit(0)
    length_cmd_list = []
    vert_list = []
    starting_vert = c_srs.vertices
    for i in range(dense_targets.shape[0]):
        ee_target = dense_targets[i]
        if i == 0:
            tcl = c_srs.ikModel.predict_cable_length(ee_target)
            Q_list, cable_tension = c_srs.FKD_static_length(starting_vert, tcl)
            starting_vert = c_srs.q_to_vertices(Q_list[-1])
        final_length, final_vertices = c_srs.IKD_minimize(ee_target, starting_vert, show_info=1)
        length_cmd_list.append(final_length)
        vert_list.append(final_vertices)
        starting_vert = final_vertices
    # save the length_cmd_list and vert_list to a pickle file
    dump_data = {'target_list': dense_targets, 'length_cmd_list': length_cmd_list, 'vert_list': vert_list}
    with open('data/IKD_traj_result_paral_new.pkl', 'wb') as f:
        pickle.dump(dump_data, f)
