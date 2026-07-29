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
    picklefile = "./data/IKD_traj_result_triangle_half_mirror_smoothed.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['target_list']
    vert_list = traj_result['vert_list']
    print(ee_target_list[19])
    
    length_cmd_list = traj_result['length_cmd_list']
    cl_cmd_middle = length_cmd_list[19]
    starting_vert = c_srs.vertices
    Q_list, cable_tension = c_srs.FKD_static_length(starting_vert, cl_cmd_middle)
    vert_list[19] = c_srs.q_to_vertices(Q_list[-1])
    dump_data = {'target_list': ee_target_list, 'length_cmd_list': length_cmd_list, 'vert_list': vert_list}
    with open('data/IKD_traj_result_triangle_half_mirror_smoothed.pkl', 'wb') as f:
        pickle.dump(dump_data, f)
    exit(0)
    ee_pos_middle = c_srs.get_ee_pos(c_srs.q_to_vertices(Q_list[-1]))
    diff = np.linalg.norm(ee_pos_middle - ee_target_list[19])
    print("Difference between middle EE position and target:", diff)
    length_middle, vert_middle = c_srs.IKD_minimize(ee_target_list[19], c_srs.q_to_vertices(Q_list[-1]), show_info=1)
    print("length_middle:", length_middle)
    

    length_cmd_list[19] = length_middle
    vert_list[19] = vert_middle
    replanned_length_idx = [14,15,16,17, 18, 20, 21,22,23,24]
    for idx in replanned_length_idx:
        length_cmd, vert_cmd = c_srs.IKD_minimize(ee_target_list[idx], vert_middle, show_info=1)
        length_cmd_list[idx] = length_cmd
        vert_list[idx] = vert_cmd

    draw_cl_cmd(length_cmd_list)
    c_srs.replay_IKD_trajectory(ee_target_list, vert_list, framerate = 5, filePath = "IKD_triangle_half_smoothed.mp4")
    dump_data = {'target_list': ee_target_list, 'length_cmd_list': length_cmd_list, 'vert_list': vert_list}
    with open('data/IKD_traj_result_triangle_half_mirror_smoothed.pkl', 'wb') as f:
        pickle.dump(dump_data, f)
    
    