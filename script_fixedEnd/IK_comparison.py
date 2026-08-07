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
from C_SRS_fixedEnd_torch import C_SRS_fixedEnd_torch
import pickle
import matplotlib.pyplot as plt


def get_diff(c_srs, Q_list, ee_target):
    diff_list = []
    for Q in Q_list:
        ee_pos = c_srs.get_ee_pos(Q)
        diff = 1/2 * np.linalg.norm(ee_pos - ee_target)**2
        diff_list.append(diff)
    return diff_list


def generate_IK_comparison_ref_data(c_srs:C_SRS_fixedEnd):
    ee_target = np.array([0.25, 0.08, 0.06])
    vert_1 = c_srs.vertices.copy()
    # vert_2 is the closest in the workspace to the target, but not necessarily the closest in the configuration space
    ee_pos_list = c_srs.ee_pos_list
    idx_closest = np.argmin(np.linalg.norm(ee_pos_list - ee_target, axis=1))
    vert_2 = c_srs.vertices_list[idx_closest]
    icl = c_srs.initial_cable_length
    tcl = [icl[0]-0.02, icl[1], icl[2]+0.01, icl[3], icl[4], icl[5]-0.02]
    Q_list_1, tension_1 = c_srs.FKD_static_length(vert_1, tcl)
    vert_3 = c_srs.q_to_vertices(Q_list_1[-1])
    c_srs.visualize_IKD_result(vert_3, ee_target)
    vert_list = [vert_1, vert_2, vert_3]
    n_iter_list = []
    diff_list_all = []
    for i in range(3):
        cur_length, starting_vertices, Q_list_final = c_srs.IKD_single(ee_target, vert_list[i], tol = 0.001, max_iter=100, show_info=1)
        diff_list = get_diff(c_srs, Q_list_final, ee_target)
        diff_list_all.append(diff_list)
        n_iter_list.append(len(Q_list_final))
    with open("data/IK_comparison_ref_data.pkl", "wb") as f:
        pickle.dump({"ee_target": ee_target, "vert_list": vert_list, "n_iter_list": n_iter_list, "diff_list_all": diff_list_all}, f)
    

def generate_IK_outof_ws_case(c_srs:C_SRS_fixedEnd):
    ee_target = np.array([0.3, 0.08, 0.06])
    cur_length, starting_vertices, Q_list_final = c_srs.IKD_single(ee_target, c_srs.vertices, tol = 0.001, max_iter=100, show_info=1)
    diff_list = get_diff(c_srs, Q_list_final, ee_target)
    with open("data/IK_outof_ws_case.pkl", "wb") as f:
        pickle.dump({"ee_target": ee_target, "starting_vertices": starting_vertices, "Q_list_final": Q_list_final, "diff_list": diff_list}, f)

def plot_diff_list(diff_list):
    plt.figure()
    plt.plot(diff_list)
    plt.xlabel("Iteration")
    # plt.ylabel("Diff (1/2 * ||ee_pos - ee_target||^2)")
    # plt.title("IKD Convergence")
    plt.grid()
    plt.show()

def plot_IK_comparison_ref_data(c_srs:C_SRS_fixedEnd):
    with open("data/IK_comparison_ref_data.pkl", "rb") as f:
        data = pickle.load(f)
    ee_target = data["ee_target"]
    vert_list = data["vert_list"]
    n_iter_list = data["n_iter_list"]
    diff_list_all = data["diff_list_all"]
    for i in range(3):
        # c_srs.visualize_IKD_result(vert_list[i], ee_target)
        # print("number of iterations for vert {}: {}".format(i, n_iter_list[i]))
        plot_diff_list(diff_list_all[i])

def plot_IK_outof_ws_case(c_srs:C_SRS_fixedEnd):
    with open("data/IK_outof_ws_case.pkl", "rb") as f:
        data = pickle.load(f)
    ee_target = data["ee_target"]
    starting_vertices = data["starting_vertices"]
    Q_list_final = data["Q_list_final"]
    diff_list = data["diff_list"]
    
    # plot_diff_list(diff_list)
    # find cartesian distance between starting_vertices and ee_target
    dist_IKD = np.linalg.norm(c_srs.get_ee_pos(Q_list_final[-1]) - ee_target)
    min_dist_ws_idx = np.argmin(np.linalg.norm(c_srs.ee_pos_list - c_srs.get_ee_pos(Q_list_final[-1]), axis=1))
    min_dist_ws_point = c_srs.ee_pos_list[min_dist_ws_idx]
    min_dist_ws = np.linalg.norm(min_dist_ws_point - ee_target)
    plotter = c_srs.visualize_single_target(c_srs.vertices, ee_target)
    # add min distance point in workspace as a purple point
    plotter.add_points(min_dist_ws_point.reshape((1,3)), color='purple', point_size=10, label='Min Distance in Workspace')
    plotter.show()
    print("IKD final distance to target: ", dist_IKD)
    print("min distance to target in workspace: ", min_dist_ws)

    

if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    # generate_IK_comparison_ref_data(c_srs)
    # generate_IK_outof_ws_case(c_srs)
    # plot_IK_comparison_ref_data(c_srs)
    plot_IK_outof_ws_case(c_srs)
    exit(0)
    ee_target = np.array([0.3, 0.08, 0.06])
    # c_srs.visualize_single_target(c_srs.vertices, ee_target)
    # exit(0)
    # print("bending_weight average: ", np.mean(c_srs.bending_weight_list))
    # print("mem_weight average: ", np.mean(c_srs.mem_weight_list))
    # c_srs.reassemble_CG_matrices(0.01,10)
    # icl = c_srs.initial_cable_length
    # tcl = [icl[0]-0.03, icl[1]-0.03, icl[2]-0.03, icl[3], icl[4], icl[5]]
    # Q_list, tension = c_srs.FKD_static_length(c_srs.vertices, tcl)
    # print("ee result: ", c_srs.get_ee_pos(Q_list[-1]))
    # c_srs.visualize_vert(Q_list[-1])
    # exit(0)
    cur_length, starting_vertices, Q_list_final = c_srs.IKD_single(ee_target, c_srs.vertices, tol = 0.001, max_iter=100, show_info=1)
    c_srs.visualize_IKD_result( starting_vertices, ee_target)
    exit(0)