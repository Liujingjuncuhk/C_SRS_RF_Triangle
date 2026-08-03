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
import matplotlib.pyplot as plt

cl_shorten_list = [0.04, 0.06, 0.08]
# ratio_bending = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
ratio_bending = [1]

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

def generate_ref_data(flying_carpet: Flying_carpet, cl_shorten_list):
    ref_data = {}
    icl = flying_carpet.initial_cable_length
    vertice_list = []
    Jac_fd_list = []
    for cl_shorten in cl_shorten_list:
        cl_test = [icl[0] - cl_shorten, icl[1] - cl_shorten, icl[2] - cl_shorten, icl[3] - cl_shorten, icl[4], icl[5], icl[6], icl[7]]
        

        start_vert = flying_carpet.vertices.copy()
        Q_list, vertices, cable_tension = flying_carpet.FKD_static(cl_test, start_vert, max_iter=100, tol=1e-5, show_info=False)
        flying_carpet.visualize_vert(vertices)
        Jac_fd = flying_carpet.get_FD_Jacobian_EE(Q_list[-1], delta = 1e-3)
        vertice_list.append(vertices)
        Jac_fd_list.append(Jac_fd)
    ref_data['vertice_list'] = vertice_list
    ref_data['Jac_fd_list'] = Jac_fd_list
    with open('data_flying_carpet/Jacobian_ref_data.pkl', 'wb') as f:
        pickle.dump(ref_data, f)
    return ref_data

def visualize_Jacobian_diff(Jac_diff, ratio_bending_list):
    Jac_diff = np.asarray(Jac_diff)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(Jac_diff.shape[1]):
        ax.plot(ratio_bending_list, Jac_diff[:, i], marker='o', label=f'test_vert={[i]}')
    ax.legend()
    ax.set_xlabel('Ratio Bending')
    ax.set_ylabel('Jacobian Difference Norm')
    ax.set_title('Jacobian Difference vs Ratio Bending')
    ax.legend()
    plt.grid()
    plt.show()
    return fig, ax


def find_optimal_ratio_bending(flying_carpet: Flying_carpet, ratio_bending_list):
    with open('data_flying_carpet/Jacobian_ref_data.pkl', 'rb') as f:
        ref_data = pickle.load(f)
    Jac_fd_list = ref_data['Jac_fd_list']
    vert_list = ref_data['vertice_list']
    Jac_diff = []
    for i in range(len(ratio_bending_list)):
        diff_this = []
        ratio_bending = ratio_bending_list[i]
        flying_carpet.reassemble_CG_matrices(ratio_bending)
        for j in range(len(vert_list)):
            ref_jac = Jac_fd_list[j]
            ref_jac = ref_jac / np.linalg.norm(ref_jac)
            vertices = vert_list[j]
            # ref_jac = flying_carpet.get_FD_Jacobian_EE_fixedRotation(vertices, delta = 1e-3)
            # ref_jac = ref_jac / np.linalg.norm(ref_jac)
            # Jac_cg = flying_carpet.get_CG_Jacobian_EE(vertices)
            Jac_cg = flying_carpet.get_CG_Jacobian_EE_FD(vertices, delta = 1e-4)
            jac_cg = Jac_cg / np.linalg.norm(Jac_cg)
            visualize_matrices(ref_jac, jac_cg)
            diff_this.append(np.linalg.norm(ref_jac - jac_cg))
        ave_diff = np.mean(diff_this)
        print("ave diff for ratio_bending = ", ratio_bending, " is ", ave_diff)
        Jac_diff.append(diff_this)

    # data_2save = {'ratio_bending_list': ratio_bending_list, 'Jac_diff': Jac_diff}
    # with open('data_flying_carpet/Jacobian_diff_find.pkl', 'wb') as f:
    #     pickle.dump(data_2save, f)

    visualize_Jacobian_diff(Jac_diff, ratio_bending_list)
    return Jac_diff

def visualize_Jacobians(flying_carpet: Flying_carpet):
    with open('data_flying_carpet/Jacobian_ref_data.pkl', 'rb') as f:
        ref_data = pickle.load(f)
    Jac_fd_list = ref_data['Jac_fd_list']
    vert_list = ref_data['vertice_list']
    Jac_diff = []
    for i in range(len(vert_list)):
        vertices = vert_list[i]
        Jac_cg = flying_carpet.get_CG_Jacobian_EE(vertices)
        Jac_CG_fd = flying_carpet.get_CG_Jacobian_EE_FD(flying_carpet.vertices, delta = 1e-4)
        visualize_matrices(Jac_fd_list[i], Jac_CG_fd)

def visualize_IK_Jacobians(Jac_FD, Jac_CG):
    Jac_FD = np.asarray(Jac_FD, dtype=float)
    Jac_CG = np.asarray(Jac_CG, dtype=float)
    if Jac_FD.ndim != 1 or Jac_CG.ndim != 1:
        raise ValueError("Both IK Jacobians must be one-dimensional vectors.")
    if Jac_FD.shape != Jac_CG.shape:
        raise ValueError(
            f"IK Jacobian shapes must match, got {Jac_FD.shape} and "
            f"{Jac_CG.shape}."
        )
    # IK Jacobians are objective gradients with one value per cable. Display
    # them as one-row matrices so the horizontal axis remains the cable index.
    return visualize_matrices(Jac_FD.reshape(1, -1), Jac_CG.reshape(1, -1))

def testIK_Jacobians(flying_carpet: Flying_carpet):
    with open('data_flying_carpet/60mm_centered.pkl', 'rb') as f:
        ee_pos_centered = pickle.load(f)
    offset =  np.array([0.28, 0.2, 0.15])
    ee_target_pos = ee_pos_centered + offset
    # guess_vert = flying_carpet.get_fixedEE_guess_vertices(ee_target_pos)
    shorten_cl = 0.06
    icl = flying_carpet.initial_cable_length
    cl_test = [icl[0] - shorten_cl, icl[1] - shorten_cl, icl[2] - shorten_cl, icl[3] - shorten_cl, icl[4], icl[5], icl[6], icl[7]]
    start_vert = flying_carpet.vertices.copy()
    Q_list, vertices, cable_tension = flying_carpet.FKD_static(cl_test, start_vert, max_iter=100, tol=1e-5, show_info=False)
    Jac_fd = flying_carpet.get_jacobian_IK_FD(ee_target_pos, vertices, delta = 1e-3)
    Jac_cg = flying_carpet.get_jacobian_IK_CG(ee_target_pos, vertices)
    print("Jacobian FD = ", Jac_fd)
    print("Jacobian CG = ", Jac_cg)
    print("normalized FD Jacobian = ", Jac_fd/np.linalg.norm(Jac_fd))
    print("normalized CG Jacobian = ", Jac_cg/np.linalg.norm(Jac_cg))
    print("Jacobian FD norm = ", np.linalg.norm(Jac_fd))
    print("Jacobian CG norm = ", np.linalg.norm(Jac_cg))
    print("normalized difference = ", np.linalg.norm(Jac_fd/np.linalg.norm(Jac_fd) - Jac_cg/np.linalg.norm(Jac_cg)))
    visualize_IK_Jacobians(Jac_fd, Jac_cg)

def testIK_Jacobian_pertubateEE(flying_carpet:Flying_carpet):
    with open('data_flying_carpet/60mm_centered.pkl', 'rb') as f:
        ee_pos_centered = pickle.load(f)
    offset =  np.array([0.28, 0.3, 0.15])
    ee_target_pos = ee_pos_centered + offset
    shorten_cl = 0.06
    icl = flying_carpet.initial_cable_length
    cl_test = [icl[0] - shorten_cl, icl[1] - shorten_cl, icl[2] - shorten_cl, icl[3] - shorten_cl, icl[4], icl[5], icl[6], icl[7]]
    start_vert = flying_carpet.vertices.copy()
    Q_list, vertices, cable_tension = flying_carpet.FKD_static(cl_test, start_vert, max_iter=100, tol=1e-5, show_info=False)
    Jac_cg = flying_carpet.get_jacobian_IK_CG(ee_target_pos, Q_list[-1])
    Jac_cg_pertubate = flying_carpet.get_jacobian_IK_CG_pertubateEE(ee_target_pos, Q_list[-1], delta = 1e-4)
    print("Jacobian CG = ", Jac_cg)
    print("Jacobian CG pertubate = ", Jac_cg_pertubate)
    print("Jacobian CG norm = ", np.linalg.norm(Jac_cg))
    print("Jacobian CG pertubate norm = ", np.linalg.norm(Jac_cg_pertubate))
    visualize_IK_Jacobians(Jac_cg, Jac_cg_pertubate)


if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # generate_ref_data(flying_carpet, cl_shorten_list)
    find_optimal_ratio_bending(flying_carpet, ratio_bending)
    # visualize_Jacobians(flying_carpet)
    # testIK_Jacobians(flying_carpet)
