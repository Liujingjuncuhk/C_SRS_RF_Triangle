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
    traj_result = c_srs.FKD_trajectory(dense_time_list, dense_cl_list, tol = 1e-4, h = 0.01)
    with open(save_path, 'wb') as f:
        pickle.dump(traj_result, f)
    return traj_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best-fit Young's modulus for cantilever fixed-end structure.")
    parser.add_argument(
        "--description-file",
        default="./models/flat_tri_surface/C_SRS_description_bary.pkl",
        help="C_SRS_fixedEnd description pickle file.",
    )

    c_srs = C_SRS_fixedEnd(description_file=parser.parse_args().description_file)
    print("initial cable length: ", c_srs.get_cable_length_bary(c_srs.vertices))
    time_list = [0.0, 3.0, 6.0, 9.0]
    icl = c_srs.get_cable_length_bary(c_srs.vertices)
    cl_list = [icl]
    cl_2 = icl.copy()
    cl_2[1] -= 0.03
    cl_list.append(cl_2)
    cl_3 = [cl_2[0] + 0.01, cl_2[1] +0.04, cl_2[2] + 0.01, cl_2[3] - 0.02, cl_2[4]-0.02, cl_2[5]-0.02]
    cl_list.append(cl_3)
    cl_4 = [cl_3[0] - 0.01, cl_3[1] -0.01, cl_3[2] - 0.02, cl_3[3] + 0.01, cl_3[4]+0.3, cl_3[5]+0.02]
    cl_list.append(cl_4)
    dense_time_list, dense_cl_list = generate_cl_trajectory(time_list, cl_list, Hz = 50)

    

