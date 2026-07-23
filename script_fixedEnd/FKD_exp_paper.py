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


if __name__ == "__main__":
    cl_list_1 = (np.array([424, 425, 424, 298, 274, 298]) * 1e-3).tolist()
    cl_list_2 = (np.array([416,430 , 436, 302, 270, 286]) * 1e-3).tolist()
    cl_list_3 = (np.array([443,442,433, 277,264,292]) * 1e-3).tolist()
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)

    data_file = "data/fixedEnd_FKD_exp_data.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    pts_list = data["pts_list"]
    fcl_list = data["fcl_list"]
    cl_lists = [cl_list_1, cl_list_2, cl_list_3]
    vert_list = []
    for i in range(len(cl_lists)):
        cl_list = cl_lists[i]
        pts = pts_list[i]
        # Add your processing code here for each cl_list and pts
        Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl_list, show_info=True)
        vert_length = c_srs.q_to_vertices(Q_list[-1])
        vert_list.append(vert_length)
        c_srs.visualize_fb_surface_w_gt(vert_length, pts)

    data_2save = {"original_cl_list": fcl_list, "cl_lists": cl_lists, "vert_list": vert_list, "pts_list": pts_list}
    with open("data/fixedEnd_FKD_exp_data_forpaper_final.pkl", "wb") as f:
        pickle.dump(data_2save, f)