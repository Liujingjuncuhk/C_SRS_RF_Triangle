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
  
def determine_bending_params_CG(c_srs: C_SRS_fixedEnd, bending_range):
    pass



if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)

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


