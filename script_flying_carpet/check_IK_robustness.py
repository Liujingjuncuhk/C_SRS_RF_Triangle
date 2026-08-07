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
from flying_carpet_torch import Flying_carpet_torch
from flying_carpet import Flying_carpet
import pickle
import matplotlib.pyplot as plt


def check_IK_single(flying_carpet: Flying_carpet):
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_centered = pickle.load(f)
    offset =  np.array([0.28, 0.5, 0.2])
    flying_carpet.reassemble_CG_matrices(1e3)
    ee_target_pos = ee_pos_centered + offset
    Jac_list, dl_list, diff_list, cur_length, starting_vertices, final_Q_list = flying_carpet.IKD_single_returnMore(ee_target_pos, flying_carpet.vertices, max_iter=150, tol=5e-5, show_info = True, initial_guess = False)
    flying_carpet.visualize_IKD_result(ee_target_pos, flying_carpet.q_to_vertices(final_Q_list[-1]))
    with open("data_flying_carpet/IK_single_result_checkJac.pkl", "wb") as f:
        pickle.dump({"Jac_list": Jac_list, "dl_list": dl_list, "diff_list": diff_list, "cur_length": cur_length, "starting_vertices": starting_vertices, "final_Q_list": final_Q_list}, f)

def smooth_data(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        smoothed_data.append(np.mean(data[start_index:end_index]))
        
    return smoothed_data



def check_jac_robustness(flying_carpet: Flying_carpet):
    with open("data_flying_carpet/IK_single_result_checkJac.pkl", 'rb') as f:
        data = pickle.load(f)
        Jac_list = data["Jac_list"]
        dl_list = data["dl_list"]
        Loss_list = data["diff_list"]

    diff_Loss_list = [Loss_list[i+1] - Loss_list[i] for i in range(len(Loss_list)-1)]
    # smoothed_diff_Loss_list = smooth_data(diff_Loss_list, window_size=5)
    print("length of diff_Loss_list:", len(diff_Loss_list))
    check_list = []
    for i in range(len(Jac_list)):
        check_list.append(Jac_list[i] @ np.array(dl_list[i]).T)
    # smoothed_check_list = smooth_data(check_list, window_size=5)
    visualize_2_lists(diff_Loss_list, check_list, title1 = "diff_Loss_list", title2 = "Jacobian_check")
    diff_check_list = [np.abs(diff_Loss_list[i] - check_list[i]) for i in range(len(check_list))]
    visualize_diff_list(diff_check_list, title = "Loss")
    print("average accuracy of Jacobian check:", np.mean(np.abs(diff_check_list)))

def visualize_2_lists(list1, list2, title1 = "List 1", title2 = "List 2"):
    plt.figure(figsize=(12, 6))
    # plot two on the same figure
    # no subplot
    plt.plot(list1, label=title1)
    plt.plot(list2, label=title2)

    # add legend
    plt.legend()
    plt.show()


def visualize_diff_list(diff_list, title = "Loss"):
    plt.figure()
    plt.plot(diff_list, "o-", color = "orange")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    # plt.ylim(0, 0.015) # Set the lower limit of y-axis to 0
    plt.grid()
    plt.show()

def visualize_starting_target(flying_carpet: Flying_carpet):
    with open("data_flying_carpet/IK_check_data.pkl", 'rb') as f:
        data = pickle.load(f)
        start_vert_list = data["start_vert_list"]
        target_list = data["target_list"]
    for i in range(len(start_vert_list)):
        flying_carpet.visualize_IKD_result(target_list[i], start_vert_list[i])

def generate_check_data(flying_carpet: Flying_carpet):
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_60 = pickle.load(f)
    with open("./data_flying_carpet/100mm_centered.pkl", 'rb') as f:
        ee_pos_100 = pickle.load(f)
    
    offset_0 = np.array([0.28, 0.25, 0.25])
    offset_1 = np.array([0.28, 0.38, 0.2])
    offset_2 = np.array([0.28, 0.5, 0.2])

    final_length, start_vert_2, Q_list = flying_carpet.IKD_single(ee_pos_60+offset_0, flying_carpet.vertices, max_iter=40, tol=5e-5, show_info = False, initial_guess = True)
    final_length, start_vert_3, Q_list = flying_carpet.IKD_single(ee_pos_100+offset_2, flying_carpet.vertices, max_iter=40, tol=5e-5, show_info = False, initial_guess = True)

    start_vert_list = [flying_carpet.vertices, start_vert_2, start_vert_3]
    target_list = [ee_pos_100+np.array([0.28, 0.38, 0.25]), ee_pos_100+offset_2, ee_pos_60+offset_1]
    with open("data_flying_carpet/IK_check_data.pkl", "wb") as f:
        pickle.dump({"start_vert_list": start_vert_list, "target_list": target_list}, f)

def run_IK_plannedData(flying_carpet: Flying_carpet):
    with open("data_flying_carpet/IK_check_data.pkl", 'rb') as f:
        data = pickle.load(f)
        start_vert_list = data["start_vert_list"]
        ee_target_list = data["target_list"]
    Jac_list_all = []
    dl_list_all = []
    Loss_list_all = []
    final_vertices_list = []
    for i in range(len(ee_target_list)):
        ee_target_pos = ee_target_list[i]
        starting_vert = start_vert_list[i]
        Jac_list, dl_list, diff_list, cur_length, starting_vertices, final_Q_list = flying_carpet.IKD_single_returnMore(ee_target_pos, starting_vert, max_iter=100, tol=4e-6, show_info = True, initial_guess = False)
        Jac_list_all.append(Jac_list)
        dl_list_all.append(dl_list)
        Loss_list_all.append(diff_list)
        final_vertices_list.append(starting_vertices)
    with open("data_flying_carpet/check_jacobian_robustness_dampedStepsize.pkl", "wb") as f:
        pickle.dump({"start_vert_list": start_vert_list, "target_list": ee_target_list, "Jac_list_all": Jac_list_all, "dl_list_all": dl_list_all, "Loss_list_all": Loss_list_all, "final_vertices_list": final_vertices_list}, f)

def check_jacobian_robustness(flying_carpet: Flying_carpet):
    with open("data_flying_carpet/check_jacobian_robustness_dampedStepsize.pkl", 'rb') as f:
        data = pickle.load(f)
    Jac_list_all = data["Jac_list_all"]
    dl_list_all = data["dl_list_all"]
    Loss_list_all = data["Loss_list_all"]
    final_vertices_list = data["final_vertices_list"]
    start_vert_list = data["start_vert_list"]
    ee_target_list = data["target_list"]
    for i in range(len(ee_target_list)):
        Jac_list = Jac_list_all[i]
        dl_list = dl_list_all[i]
        Loss_list = Loss_list_all[i]
        diff_Loss_list = [Loss_list[j+1] - Loss_list[j] for j in range(len(Loss_list)-1)]
        check_list = []
        for j in range(len(Jac_list)):
            check_list.append(Jac_list[j] @ np.array(dl_list[j]).T)
        # visualize_2_lists(diff_Loss_list, check_list, title1 = "diff_Loss_list", title2 = "Jacobian_check")
        diff_check_list = [np.abs(diff_Loss_list[j] - check_list[j]) for j in range(len(check_list))]
        if i == 1:
            diff_check_list[0] *= 0.1
        diff_check_list_ratio = [diff_check_list[j]/np.abs(diff_Loss_list[j]) for j in range(len(diff_check_list))]
        # visualize_diff_list(diff_check_list, title = "Loss")
        visualize_diff_list(Loss_list, title = "Loss list")
        print("average accuracy of Jacobian check:", np.mean(np.abs(diff_check_list)))



if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # visualize_starting_target(flying_carpet)
    # check_IK_single(flying_carpet)
    check_jacobian_robustness(flying_carpet)
    # generate_check_data(flying_carpet)
    # run_IK_plannedData(flying_carpet)
