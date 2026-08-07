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

cl_cmd_list = [[416, 430, 436,302,270, 286],
               [443, 442,433,277,264,292],
               [363,357, 363,333,305,335]]
cl_cmd_list = np.array(cl_cmd_list) * 1e-3

bending_weight_list = [10,100,200,300,400,500,600,700,800,900,1000]

def visualize_matrices(jac_fd, jac_cg):
    jac_fd = np.asarray(jac_fd, dtype=float)
    jac_cg = np.asarray(jac_cg, dtype=float)
    if jac_fd.ndim != 2 or jac_cg.ndim != 2:
        raise ValueError("Both Jacobians must be two-dimensional matrices.")
    if jac_fd.shape != jac_cg.shape:
        raise ValueError(
            f"Jacobian shapes must match, got {jac_fd.shape} and {jac_cg.shape}."
        )
    if not np.all(np.isfinite(jac_fd)) or not np.all(np.isfinite(jac_cg)):
        raise ValueError("Jacobians must contain only finite values.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    for ax, matrix, title in zip(
        axes,
        (jac_fd, jac_cg),
        ("Finite-difference Jacobian", "Shape-Up Jacobian"),
    ):
        # Scale each panel independently. Keeping symmetric limits around zero
        # preserves the meaning of color intensity: darker means a larger
        # absolute value, red is positive, and blue is negative.
        color_limit = max(np.max(np.abs(matrix)), np.finfo(float).eps)
        image = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Cable index")
        ax.set_ylabel("End-effector DOF")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                text_color = "white" if abs(value) > 0.55 * color_limit else "black"
                ax.text(
                    column,
                    row,
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6,
                )

        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.set_label(
            "Jacobian value (blue: negative, red: positive)"
        )

    plt.show()
    return fig, axes


def determine_CG_params(c_srs: C_SRS_fixedEnd, weight_list: list):
    nSamples = len(weight_list)
    Q_list = c_srs.FKD_static(c_srs.vertices, [1,1,1,1,1,1],tol = 1e-7, show_info = True)
    vert_test = c_srs.q_to_vertices(Q_list[-1])
    # c_srs.visualize_vert(vert_test)
    Jac_fd = c_srs.get_FD_Jacobian_EE(Q_list[-1])
    print("Jacobian by finite difference: ", Jac_fd)
    error_list = []
    best_weight = None
    best_error = float('inf')
    Jac_CG_list = []
    best_Jac_CG = None
    for i in range(nSamples):
        weight = weight_list[i]
        print("weight: ", weight)
        c_srs.reassemble_CG_matrices(weight)
        Jac_CG = c_srs.get_CG_Jacobian_EE(Q_list[-1])
        Jac_CG_list.append(Jac_CG)
        error = np.linalg.norm(Jac_fd - Jac_CG)
        error_list.append(error)
        if error < best_error:
            best_error = error
            best_weight = weight
            best_Jac_CG = Jac_CG
    print("best weight: ", best_weight)
    print("best error: ", best_error)
    print("best Jacobian by CG: ", best_Jac_CG)
    print("Jacobian by finite difference: ", Jac_fd)
    # plot the error_list vs weight_list
    
    plt.plot(weight_list, error_list)
    plt.xlabel('weight')
    plt.ylabel('error')
    plt.title('Error vs Weight')
    # plt.savefig('error_vs_weight_cg.png')
    plt.show()
    # save the error_list and weight_list to a csv file
    with open('error_list_cg.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['weight', 'error'])
        for i in range(nSamples):
            writer.writerow([weight_list[i], error_list[i]])


def visualize_matrices(mat1, mat2, title1 = "Matrix 1", title2 = "Matrix 2"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axs[0].imshow(mat1, cmap='viridis')
    axs[0].set_title(title1)
    fig.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(mat2, cmap='viridis')
    axs[1].set_title(title2)
    fig.colorbar(im2, ax=axs[1])
    plt.show()

def get_Jacobian_IK(c_srs: C_SRS_fixedEnd, ee_target_pos, cur_vert, Jac_ee):
    error = (
        c_srs.get_ee_pos(cur_vert) - ee_target_pos
    ).reshape(-1)

    # If objective = 0.5 ||x - target||²
    return Jac_ee.T @ error

def determine_bending_params_CG(c_srs: C_SRS_fixedEnd, bending_range):
    with open('data/vert_list_toTest_cg_params.pkl', 'rb') as f:
        data = pickle.load(f)
    vert_list_toTest = data['vert_list_toTest']
    Jac_list = data['Jac_list']
    Jac_list_noRotation = data['Jac_list_noRotation']
    diff_list_full = []
    diff_list_noRotation = []
    for bending_weight in bending_range:
        c_srs.reassemble_CG_matrices(bending_weight, 10)
        Jac_CG_list = []
        for vert in vert_list_toTest:
            Jac_CG = c_srs.get_CG_Jacobian_EE(vert)
            Jac_CG_list.append(Jac_CG)
            diff_full = np.linalg.norm(Jac_list[0] - Jac_CG)
            diff_noRotation = np.linalg.norm(Jac_list_noRotation[0] - Jac_CG)
            visualize_matrices(Jac_list[0], Jac_CG, title1 = "Jacobian by FD", title2 = f"Jacobian by CG, bending_weight={bending_weight}")
            visualize_matrices(Jac_list_noRotation[0], Jac_CG, title1 = "Jacobian by FD with fixed rotation", title2 = f"Jacobian by CG, bending_weight={bending_weight}")

        print("bending_weight: ", bending_weight, "diff_full: ", diff_full, "diff_noRotation: ", diff_noRotation)
        diff_list_full.append(diff_full)
        diff_list_noRotation.append(diff_noRotation)

def compare_FD_CG_Jacobian(c_srs: C_SRS_fixedEnd):
    with open('data/vert_list_toTest_cg_params.pkl', 'rb') as f:
        data = pickle.load(f)
    # tar_ee_pos_list = data['tar_ee_pos_list']
    
    starting_vert_list = data['starting_vert_list']
    Jac_FD_list = data['Jac_list']
    Jac_FD_noRotation_list = data['Jac_list_noRotation']
    # for i in range(len(tar_ee_pos_list)):
    ee_target_pos = tar_ee_pos_list[i]
    starting_vert = starting_vert_list[i]
    Jac_FD = Jac_FD_list[i]
    Jac_FD_noRotation = Jac_FD_noRotation_list[i]
    Jac_CG = c_srs.get_CG_Jacobian_EE(starting_vert)
    print("diff CG and FD: ", np.linalg.norm(Jac_FD - Jac_CG)/np.linalg.norm(Jac_FD))
    print("diff CG and FD with fixed rotation: ", np.linalg.norm(Jac_FD_noRotation - Jac_CG)/np.linalg.norm(Jac_FD_noRotation))

def generate_locations_for_CG_approx_comparison(c_srs: C_SRS_fixedEnd):
    vert_list_toTest = []
    Jac_list = []
    Jac_list_noRotation = []    
    for i in range(3):
        cl_cmd = cl_cmd_list[i]

        Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl_cmd, tol = 1e-6, show_info = False)
        vert_list_toTest.append(c_srs.q_to_vertices(Q_list[-1]))
        Q = Q_list[-1]
        Jac_rotation = c_srs.get_FD_Jacobian_EE(Q, delta = 1e-4)
        Jac_list.append(Jac_rotation)
        Jac_norotation = c_srs.get_Jacobian_FD_fixedRotation(Q,eps = 1e-4)
        Jac_list_noRotation.append(Jac_norotation)
    data_2save = {}
    data_2save['vert_list_toTest'] = vert_list_toTest
    data_2save['Jac_list'] = Jac_list
    data_2save['Jac_list_noRotation'] = Jac_list_noRotation
    with open('data/info_to_test_CG_approx.pkl', 'wb') as f:
        pickle.dump(data_2save, f)

def compare_FD_CG_Jacobian(c_srs: C_SRS_fixedEnd):
    with open('data/info_to_test_CG_approx.pkl', 'rb') as f:
        data = pickle.load(f)
    vert_list_toTest = data['vert_list_toTest']
    Jac_list = data['Jac_list']
    Jac_list_noRotation = data['Jac_list_noRotation']
    ee_target_pos = np.array([0.26, 0.08, 0.03])
    for i in range(3):
        starting_vert = vert_list_toTest[i]
        Jac_FD = Jac_list[i]
        Jac_FD_IK = get_Jacobian_IK(c_srs, ee_target_pos, starting_vert, Jac_FD)
        Jac_FD_noRotation = Jac_list_noRotation[i]
        Jac_FD_IK_noRotation = get_Jacobian_IK(c_srs, ee_target_pos, starting_vert, Jac_FD_noRotation)
        Jac_CG = c_srs.get_CG_Jacobian_EE(starting_vert)
        Jac_CG_IK = get_Jacobian_IK(c_srs, ee_target_pos, starting_vert, Jac_CG)
        print("diff CG and FD: ", np.linalg.norm(Jac_FD_IK - Jac_CG_IK)/np.linalg.norm(Jac_FD_IK))
        print("diff CG and FD with fixed rotation: ", np.linalg.norm(Jac_FD_IK_noRotation - Jac_CG_IK)/np.linalg.norm(Jac_FD_IK_noRotation))



    



if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    generate_locations_for_CG_approx_comparison(c_srs)
    # determine_bending_params_CG(c_srs, bending_weight_list)
    compare_FD_CG_Jacobian(c_srs)
    exit(0)

    vert_list_toTest = []
    Jac_list = []
    Jac_list_noRotation = []
    for i in range(3):
        Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl_cmd_list[i], tol = 1e-6, show_info = False)
        vert_list_toTest.append(c_srs.q_to_vertices(Q_list[-1]))
        Q = Q_list[-1]
        for delta in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]:
            J_full = c_srs.get_FD_Jacobian_EE(Q, delta=delta)
            J_fixed = c_srs.get_Jacobian_FD_fixedRotation(Q, eps=delta)
            J_CG = c_srs.get_CG_Jacobian_EE(Q)

            print(
                delta,
                np.linalg.norm(J_full),
                np.linalg.norm(J_fixed),
                # np.linalg.norm(J_full - J_fixed),
                np.linalg.norm(J_full - J_CG),
                np.linalg.norm(J_fixed - J_CG)
            )
        # Jac_rotation = c_srs.get_FD_Jacobian_EE(Q_list[-1], delta = 1e-4)
        # print("Jacobian by finite difference: ", Jac_rotation)
        # Jac_list.append(Jac_rotation)
        # Jac_norotation = c_srs.get_Jacobian_FD_fixedRotation(Q_list[-1],eps = 1e-4)
        # print("Jacobian by finite difference with fixed rotation: ", Jac_norotation)
        # Jac_list_noRotation.append(Jac_norotation)
        # print("difference between two Jacobians: ", np.linalg.norm(Jac_rotation - Jac_norotation))

    data_2save = {}
    data_2save['vert_list_toTest'] = vert_list_toTest
    data_2save['Jac_list'] = Jac_list
    data_2save['Jac_list_noRotation'] = Jac_list_noRotation

    with open('data/vert_list_toTest_cg_params.pkl', 'wb') as f:
        pickle.dump(data_2save, f)


