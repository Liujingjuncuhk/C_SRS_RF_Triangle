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



if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    # ee_target_list = np.array([[0.25, 0.08, 0.02],
    #                            [0.24, 0.08, 0.06],
    #                            [0.23, 0.08, 0.06],
    #                            [0.24, 0.08, 0.02],
    #                            [0.25, 0.08, 0.02]])
    ee_target_list = np.array([[0.25, 0.08, 0.02]])
    c_srs.visualize_planned_traj(c_srs.vertices, ee_target_list)
    starting_vert = c_srs.vertices
    for ee_target in ee_target_list:
        tcl = c_srs.ikModel.predict_cable_length(ee_target)
        # print(tcl)   # → 6 cable lengths in metres
        Q_list, starting_vert, cable_tension = c_srs.FKD_time(tcl, 1, starting_vert, tol = 1e-5)
        cur_length, starting_vertices, Q_list = c_srs.IKD_single(ee_target, starting_vert,AA = False, tol=2e-3)
        c_srs.replay_IKD_Q_list(ee_target, Q_list, filePath="./C_SRS_IKD.mp4", framerate=1)
        c_srs.visualize_IKD_result(starting_vert, ee_target)
        # print("prediction error: ", np.linalg.norm(c_srs.get_ee_pos(starting_vert) - ee_target))
        # c_srs.visualize_IKD_result(starting_vert, ee_target)
    