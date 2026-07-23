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

def generate_cl_cmd(length_cmd_list, t_total, out_file):
    """
    Generate a list of cable length commands for each time step based on the provided length_cmd_list and total time.

    Parameters:
    - length_cmd_list: List of lists, where each inner list contains the cable lengths for a specific waypoint.
    - t_total: Total time for the trajectory.

    Returns:
    - cl_cmd_list: List of cable length commands for each time step.
    - time_list: List of time steps corresponding to the cable length commands. starting from 0 to t_total with equal step between each time step.
    """
    n_waypoints = len(length_cmd_list)
    if n_waypoints < 2:
        raise ValueError("length_cmd_list must contain at least two waypoints.")
    
    # Calculate the number of time steps based on the total time and a fixed time step (e.g., 0.1s)
    step = t_total / (n_waypoints - 1)
    time_list = np.arange(0, t_total + step, step)

    saved_data = {'time_list': time_list, 'length_cmd_list': length_cmd_list, 'total_time': t_total}
    with open(out_file, 'wb') as f:
        pickle.dump(saved_data, f)

    return length_cmd_list, time_list

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    picklefile = "./data/IKD_traj_result_triangle.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['target_list']
    length_cmd_list = traj_result['length_cmd_list']
    generate_cl_cmd(length_cmd_list, t_total=10.0, out_file="./data/IKD_traj_result_triangle_control_10s.pkl")


