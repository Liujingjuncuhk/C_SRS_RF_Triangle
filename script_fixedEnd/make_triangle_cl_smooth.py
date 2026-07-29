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
    picklefile = "./data/IKD_traj_result_triangle_half_mirror.pkl"
    with open(picklefile, "rb") as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['target_list']
    print(ee_target_list[19])
    length_cmd_list = traj_result['length_cmd_list']
    cl_speed = []
    for i in range(1, len(length_cmd_list)):
        speed = np.array(length_cmd_list[i]) - np.array(length_cmd_list[i-1])
        cl_speed.append(speed)
    print("Total number of waypoints:", len(ee_target_list))
    draw_cl_cmd(length_cmd_list)
    