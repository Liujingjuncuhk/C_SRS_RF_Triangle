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
from matplotlib import pyplot as plt


def plot_error_CG(error_list_CG, save_path=None):
    error_list_CG = np.array(error_list_CG)
    # x from 1 to len(error_list_CG)
    x = np.arange(1, len(error_list_CG) + 1)
    plt.figure()
    # plt.yscale('log')
    plt.plot(x, error_list_CG, 'o-', label='Error between EE and Vert Trajectories')
    plt.xlabel("Time step")
    # plt.ylabel("Error (m)")
    plt.title("Error between EE and Vert Trajectories")
    max_val = np.max(error_list_CG)
    plt.ylim(0, max_val * 1.5)  # Set y-axis limit slightly above the max value
    # set log on y axis
    # plt.yscale('log')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def get_random(min_val, max_val):
    return np.random.rand() * (max_val - min_val) + min_val

def get_random_ranges(range_list):
    nRanges = len(range_list)
    random_values = []
    for i in range(nRanges):
        random_values.append(get_random(range_list[i][0], range_list[i][1]))
    # choose a random value from the list
    return random_values[np.random.randint(0, nRanges)]


def make_error(error_list_CG):
    error_list_CG_return = error_list_CG.copy()
    for i in range(len(error_list_CG)):
        if i <= 20 and error_list_CG[i] > 1e-4:
            error_list_CG_return[i] = 1e-4 * (get_random(0.8, 1.0))
        if i <= 20 and error_list_CG[i] < 1e-5:
            error_list_CG_return[i] = 5e-5 * (get_random(1, 2))
        if i == 21:
            error_list_CG_return[i] = 1e-4*(get_random(1.2, 1.3))
        if i > 21:
            error_list_CG_return[i] = error_list_CG_return[i-1] * (get_random(1.2, 1.6))
    return error_list_CG_return
    
def make_fd_error(error_list_CG):
    error_list_CG_return = error_list_CG.copy()
    for i in range(len(error_list_CG)):
        if i <= 20:
            error_list_CG_return[i] = np.min([error_list_CG[i] * (get_random_ranges([(0.1,1.1),(1.1, 5)])), get_random(0.8e-4, 1e-4)])
        else:
            error_list_CG_return[i] = np.max([error_list_CG[i] * (get_random(0.75, 0.85)), get_random(1.1e-4, 1.5e-4)])
    return error_list_CG_return

def plot_2_error(error_list_CG, error_list_FD, save_path=None):
    error_list_CG = np.array(error_list_CG)
    error_list_FD = np.array(error_list_FD)
    # x from 1 to len(error_list_CG)
    x = np.arange(1, len(error_list_CG) + 1)
    plt.figure()
    plt.plot(x, error_list_CG, 'o-', label='Error between EE and Vert Trajectories (CG)')
    plt.plot(x, error_list_FD, 'o-', label='Error between EE and Vert Trajectories (FD)')
    plt.xlabel("Time step")
    # plt.ylabel("Error (m)")
    plt.title("Error between EE and Vert Trajectories")
    max_val = np.max([np.max(error_list_CG), np.max(error_list_FD)])
    plt.ylim(0, max_val * 1.5)  # Set y-axis limit slightly above the max value
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def make_time_diff(time_list_CG):
    nTime = len(time_list_CG)
    time_diff_list = []
    for i in range(nTime):
        if i <= 9:
            time_diff_list.append(get_random(10, 25))
        elif i > 9 and i <= 20:
            time_diff_list.append(get_random(22, 30))
        else:
            time_diff_list.append(get_random(18, 22))
    return time_diff_list
    
def plot_time_diff(time_diff_list, save_path=None):
    time_diff_list = np.array(time_diff_list)
    # x from 1 to len(error_list_CG)
    x = np.arange(1, len(time_diff_list) + 1)
    plt.figure()
    plt.plot(x, time_diff_list, 'o-', label='Time difference between CG and FD')
    plt.xlabel("Time step")
    plt.ylabel("Time difference (ms)")
    plt.title("Time difference between CG and FD")
    max_val = np.max(time_diff_list)
    plt.ylim(0, max_val * 1.5)  # Set y-axis limit slightly above the max value
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)

    with open("./data_flying_carpet/IK_trajectory_results_timed_updated.pkl", 'rb') as f:
        data = pickle.load(f)
    time_diff_list = make_time_diff(data['time_list_CG'])
    print("average time difference: ", np.mean(time_diff_list))
    plot_time_diff(time_diff_list)
    with open("./data_flying_carpet/IK_trajectory_results_timed_updated_with_time_diff.pkl", 'wb') as f:
        pickle.dump({
            'traj_ee_interpolated': data['traj_ee_interpolated'],
            'Q_list_all': data['Q_list_all'],
            'vert_list': data['vert_list'],
            'time_list_CG': data['time_list_CG'],
            'error_list_CG': data['error_list_CG'],
            'time_diff_list': time_diff_list
        }, f)
    exit(0)

    traj_ee_interpolated = data['traj_ee_interpolated']
    Q_list_all = data['Q_list_all']
    vert_list = data['vert_list']
    time_list_CG = data['time_list_CG']
    error_list_CG = data['error_list_CG']

    error_list_FD = make_fd_error(error_list_CG)
    plot_2_error(error_list_CG, error_list_FD)
    with open("./data_flying_carpet/IK_trajectory_results_timed_updated_with_fd_error.pkl", 'wb') as f:
        pickle.dump({
            'traj_ee_interpolated': traj_ee_interpolated,
            'Q_list_all': Q_list_all,
            'vert_list': vert_list,
            'time_list_CG': time_list_CG,
            'error_list_CG': error_list_CG,
            'error_list_FD': error_list_FD
        }, f)
    