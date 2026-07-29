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
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    picklefile = "./data/IKD_traj_result_triangle_half.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list_array = traj_result['target_list']
    ee_target_list = ee_target_list_array.tolist()
    print("type of ee_target_list:", type(ee_target_list))
    length_cmd_list = traj_result['length_cmd_list']
    print("type of length_cmd_list:", type(length_cmd_list))
    vert_list = traj_result['vert_list']
    nEE = len(ee_target_list)
    ee_target_list_mirrorer = ee_target_list.copy()
    length_cmd_list_mirrorer = length_cmd_list.copy()
    vert_list_mirrorer = vert_list.copy()
    for i in range(1, nEE):
        ee_target_toAppend = ee_target_list[nEE - 1 - i].copy()
        ee_target_toAppend[1] = 0.16 - ee_target_toAppend[1]
        ee_target_list_mirrorer.append(ee_target_toAppend)
        length_cmd_toMirror = length_cmd_list[nEE - 1 - i]
        length_cmd_toAppend = length_cmd_toMirror.copy()
        length_cmd_toAppend[0] = length_cmd_toMirror[2]
        length_cmd_toAppend[2] = length_cmd_toMirror[0]
        length_cmd_toAppend[3] = length_cmd_toMirror[5]
        length_cmd_toAppend[5] = length_cmd_toMirror[3]

        length_cmd_list_mirrorer.append(length_cmd_toAppend)
        starting_vert = vert_list[nEE - 1 - i]
        tcl = length_cmd_toAppend
        Q_list, cable_tension = c_srs.FKD_static_length(starting_vert, tcl)
        vert_list_mirrorer.append(c_srs.q_to_vertices(Q_list[-1]))

    with open("./data/IKD_traj_result_triangle_half_mirror.pkl", "wb") as f:
        dump_data = {'target_list': ee_target_list_mirrorer, 'length_cmd_list': length_cmd_list_mirrorer, 'vert_list': vert_list_mirrorer}
        pickle.dump(dump_data, f)