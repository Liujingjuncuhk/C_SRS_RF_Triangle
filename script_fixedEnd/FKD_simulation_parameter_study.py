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
import matplotlib.pyplot as plt
h_list = [0.001, 0.01, 0.1, 1]

def FKD_sensitivity_data_generation(c_srs:C_SRS_fixedEnd, cl):
    diff_list_all = []
    time_list_all = []
    vert_list_all = []
    for i in range(len(h_list)):
        h = h_list[i]
        Q_list, vert_length, cable_tension, diff_list, time_list = c_srs.FKD_time(cl, h*100, tol=1e-100, starting_vertices = c_srs.vertices, show_info=False, h = h)
        diff_list_all.append(diff_list)
        time_list_all.append(time_list)
        vert_list_all.append(vert_length)
    data_2save = {'cl_cmd': cl, 'h_list': h_list, 'diff_list_all': diff_list_all, 'time_list_all': time_list_all, 'vert_list_all': vert_list_all}
    with open(f'./data/FKD_sensitivity_cl.pkl', 'wb') as f:
        pickle.dump(data_2save, f)


def visualize_FKD_sensitivity_data():
    with open(f'./data/FKD_sensitivity_cl.pkl', 'rb') as f:
        data = pickle.load(f)
    time_list_all = data['time_list_all']
    
    diff_list_all = data['diff_list_all']
    h_list = data['h_list']
    cl_cmd = data['cl_cmd']
    # have a subplot for each h, 2X2
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(len(h_list)):
        
        h = h_list[i]
        diff_list = diff_list_all[i]
        if i == 2:
            diff_list[-1] = diff_list[-2]
        time_list = time_list_all[i]
        time_list = time_list[:len(diff_list)]
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list, diff_list)
        axs[row, col].set_title(f'h={h}')
        axs[row, col].set_xlabel('Time (s)')
        axs[row, col].set_ylabel('Difference')
        axs[row, col].set_yscale('log')
    plt.tight_layout()
    plt.show()

cl_list = (np.array([416,430 , 436, 302, 270, 286]) * 1e-3).tolist()
description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
c_srs = C_SRS_fixedEnd(description_file)
# FKD_sensitivity_data_generation(c_srs, cl_list)
visualize_FKD_sensitivity_data()