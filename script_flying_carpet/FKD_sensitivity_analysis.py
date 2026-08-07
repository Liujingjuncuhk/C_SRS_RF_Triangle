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
from flying_carpet import Flying_carpet
import pickle
import pickle
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt

cl_1 = (np.array([495, 483, 506, 470, 450, 437, 457, 426])*1e-3).tolist()
cl_2 = (np.array([372, 550, 388, 570, 363, 518, 379, 525])*1e-3).tolist()
cl_3 = (np.array([575, 470, 580, 445, 510, 348, 498, 309])*1e-3).tolist()
cl_list = [cl_1, cl_2, cl_3]
h_list = [0.001, 0.01, 0.1, 1]
diff_list_all = []
time_list_all = []
vert_list_all = []
def FKD_sensitivity_data_generation(flying_carpet:Flying_carpet, cl):
    for i in range(len(h_list)):
        h = h_list[i]
        Q_list, cable_tension, diff_list, time_list = flying_carpet.FKD_time(cl, h*800, tol=1e-100, starting_vertices = flying_carpet.vertices, show_info=False, h = h)
        diff_list_all.append(diff_list)
        time_list_all.append(time_list)
        vert_list_all.append(flying_carpet.q_to_vertices(Q_list[-1]))
    data_2save = {'cl_cmd': cl, 'h_list': h_list, 'diff_list_all': diff_list_all, 'time_list_all': time_list_all, 'vert_list_all': vert_list_all}
    with open(f'./data_flying_carpet/FKD_sensitivity_cl.pkl', 'wb') as f:
        pickle.dump(data_2save, f)

def visualize_FKD_sensitivity_data():
    with open(f'./data_flying_carpet/FKD_sensitivity_cl.pkl', 'rb') as f:
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
        if i == 1:
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

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # Q_list, cable_tension, diff_list, time_list = flying_carpet.FKD_time(cl_1, 800, tol=1e-12, starting_vertices = flying_carpet.vertices, show_info=True, h = 1)
    # flying_carpet.visualize_vert(flying_carpet.q_to_vertices(Q_list[-1]))
    # flying_carpet.visualize_vert(flying_carpet.q_to_vertices(Q_list[-1]))
    # FKD_sensitivity_data_generation(flying_carpet, cl_1)
    visualize_FKD_sensitivity_data()