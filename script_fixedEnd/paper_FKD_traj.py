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
# from flying_carpet_cantilever import Flying_carpet_fixedEnd
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle
import matplotlib.pyplot as plt


def generate_cl_trajectory(time_list, cl_list,Hz = 50, save_path = "data/fixedend_FKD_paper.pickle"):
    """Linearly resample cable-length waypoints at ``Hz`` samples/second.

    The returned trajectory starts at time zero and includes the exact final
    waypoint.  If the final time is not on the regular sampling grid, it is
    appended as one additional sample.
    """
    times = np.asarray(time_list, dtype=float)
    cable_lengths = np.asarray(cl_list, dtype=float)

    if times.ndim != 1 or times.size == 0:
        raise ValueError("time_list must be a non-empty one-dimensional list.")
    if cable_lengths.ndim != 2 or cable_lengths.shape[0] != times.size:
        raise ValueError(
            "cl_list must be a two-dimensional list with one row per time."
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(cable_lengths)):
        raise ValueError("time_list and cl_list must contain only finite values.")
    if not np.isclose(times[0], 0.0):
        raise ValueError("time_list must start at 0.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_list must be strictly increasing.")
    if not np.isscalar(Hz) or not np.isfinite(Hz) or Hz <= 0:
        raise ValueError("Hz must be a positive finite number.")

    if times.size == 1:
        return [0.0], cable_lengths.tolist()

    sample_period = 1.0 / float(Hz)
    dense_times = np.arange(0.0, times[-1], sample_period)
    if dense_times.size == 0 or not np.isclose(dense_times[-1], times[-1]):
        dense_times = np.append(dense_times, times[-1])
    else:
        dense_times[-1] = times[-1]

    dense_cable_lengths = np.empty((dense_times.size, cable_lengths.shape[1]))
    for cable_index in range(cable_lengths.shape[1]):
        dense_cable_lengths[:, cable_index] = np.interp(
            dense_times, times, cable_lengths[:, cable_index]
        )

    return dense_times.tolist(), dense_cable_lengths.tolist()

def FKD_paper(c_srs: C_SRS_fixedEnd, dense_time_list, dense_cl_list, save_path = "data/fixedend_FKD_paper.pickle"):
    """Run FKD simulation for the given cable-length trajectory and save the results."""
    cl_return_list = []
    cable_force_list = []
    vert_list = []
    residual_list = []
    starting_vertices = c_srs.vertices.copy()
    for i, cl in enumerate(dense_cl_list):
        t = dense_time_list[i]
        Q_list, cable_tension = c_srs.FKD_static_length(starting_vertices, cl)
        residual_this = c_srs.FKD_get_residual(Q_list[-1], cable_tension)
        residual_list.append(residual_this)
        cl_return_list.append(c_srs.get_cable_length_bary(c_srs.q_to_vertices(Q_list[-1])))
        cable_force_list.append(cable_tension)
        vert_list.append(c_srs.q_to_vertices(Q_list[-1]))
        starting_vertices = c_srs.q_to_vertices(Q_list[-1])
    return cl_return_list, cable_force_list, vert_list, residual_list


def plot_FKD_results(c_srs: C_SRS_fixedEnd ):
    with open("data/FKD_trajectory_fixedEnd.pickle", 'rb') as f:
        data = pickle.load(f)
    dense_time_list = data['dense_time_list']
    dense_cl_list = data['dense_cl_list']
    dense_cl_list = np.array(dense_cl_list)
    cl_return_list = data['cl_return_list']
    cl_return_list = np.array(cl_return_list)
    cable_force_list = data['cable_force_list']
    vert_list = data['vert_list']
    residual_list = data['residual_list']
    residual_norm = [np.linalg.norm(res)/(3*c_srs.nMoving) for res in residual_list]


    # in a subplot, plot dense_cl_list and cl_return_list for each cable
    plt.figure(figsize=(6, 8))
    nCable = 6
    plot = (3,2)
    for i in range(nCable):
        plt.subplot(plot[0], plot[1], i+1)
        plt.plot(dense_time_list, dense_cl_list[:, i], label='dense_cl_list', color='blue', linestyle='--')
        plt.plot(dense_time_list, cl_return_list[:, i], label='cl_return_list', color='green')
        # plt.xlabel('Time [s]')
        # plt.ylabel(f'Cable {i + 1} Length [m]')
        plt.grid(True)
    # plt.tight_layout()
    plt.show()

    # plot cable_force_list for each cable
    plt.figure(figsize=(6, 8))
    for i in range(nCable):
        plt.subplot(plot[0], plot[1], i+1)
        plt.plot(dense_time_list, [cf[i] for cf in cable_force_list], label='cable_force_list', color='red')
        # plt.xlabel('Time [s]')
        # plt.ylabel(f'Cable {i + 1} Force [N]')
        plt.grid(True)
    # plt.tight_layout()
    plt.show()


    # plot residual norm over time
    plt.figure(figsize=(8, 6))
    max_norm = max(residual_norm)
    plt.plot(dense_time_list, residual_norm, label='Residual Norm')
    plt.xlabel('Time [s]')
    plt.ylabel('Residual Norm')
    plt.ylim(0, max_norm * 5)  # Set y-axis limit slightly above the max value
    # add grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()  

def smooth_data(data, window_size=5):
    """Smooth the data using a simple moving average."""
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if window_size > len(data):
        raise ValueError("Window size must not be larger than the data length.")
    
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Pad the beginning of the smoothed data to match the original length
    pad_size = (len(data) - len(smoothed_data)) // 2
    smoothed_data = np.pad(smoothed_data, (pad_size, len(data) - len(smoothed_data) - pad_size), mode='edge')
    
    return smoothed_data

def plot_FKD_together(c_srs, plot_id = 2):
    with open("data/FKD_trajectory_fixedEnd.pickle", 'rb') as f:
        data = pickle.load(f)
    dense_time_list = data['dense_time_list']
    dense_cl_list = data['dense_cl_list']
    dense_cl_list = np.array(dense_cl_list)
    cl_return_list = data['cl_return_list']
    cl_return_list = np.array(cl_return_list)
    cable_force_list = data['cable_force_list']
    vert_list = data['vert_list']

    cl_cmd_list = smooth_data(dense_cl_list[:, plot_id])
    cl_return_list = smooth_data(cl_return_list[:, plot_id])
    cable_force_list = smooth_data([cf[plot_id] for cf in cable_force_list])

    fig, ax_left = plt.subplots()
    line1, = ax_left.plot(dense_time_list, cl_cmd_list, label="Commanded Cable Length", color='blue', linestyle='--')
    line2, = ax_left.plot(dense_time_list, cl_return_list, label="Returned Cable Length", color='green')
    ax_right = ax_left.twinx()
    line3, = ax_right.plot(dense_time_list, cable_force_list, label="Cable Force", color='red')
    ax_left.set_xlabel('Time [s]')
    ax_left.set_ylabel('Cable Length [m]')
    ax_right.set_ylabel('Cable Force [N]')
    # no labels
    # add grid
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    plot_FKD_results(c_srs)
    # plot_FKD_together(c_srs, plot_id = 2)
    exit(0)

    
    print("initial cable length: ", c_srs.get_cable_length_bary(c_srs.vertices))
    time_list = [0.0, 3.0, 6.0, 9.0]
    icl = c_srs.get_cable_length_bary(c_srs.vertices)
    cl_list = [icl]
    cl_2 = icl.copy()
    cl_2[1] -= 0.03
    cl_list.append(cl_2)
    cl_3 = [cl_2[0] + 0.01, cl_2[1] +0.04, cl_2[2] + 0.01, cl_2[3] - 0.02, cl_2[4]-0.03, cl_2[5]-0.02]
    cl_list.append(cl_3)
    cl_4 = [cl_3[0] - 0.01, cl_3[1] -0.01, cl_3[2] - 0.02, cl_3[3] + 0.01, cl_3[4]+0.03, cl_3[5]+0.02]
    cl_list.append(cl_4)
    dense_time_list, dense_cl_list = generate_cl_trajectory(time_list, cl_list, Hz = 20)
    cl_return_list, cable_force_list, vert_list, residual_list = FKD_paper(c_srs, dense_time_list, dense_cl_list)

    data_to_save = {
        'dense_time_list': dense_time_list,
        'dense_cl_list': dense_cl_list,
        'cl_return_list': cl_return_list,
        'cable_force_list': cable_force_list,
        'vert_list': vert_list,
        'residual_list': residual_list
    }
    with open("data/FKD_trajectory_fixedEnd.pickle", 'wb') as f:
        pickle.dump(data_to_save, f)

