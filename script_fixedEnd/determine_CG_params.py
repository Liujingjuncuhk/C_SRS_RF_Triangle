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

        



if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    weight_range = []
    weight_range.extend(np.arange(1, 300, 10))
    determine_CG_params(c_srs, weight_range)
    # weight_range.extend(np.arange(3.5e7+0.01e7, 3.6e7, 0.01e7))
    # weight_range.extend(np.arange(3.6e7+0.1e7,1.0e8, 0.1e7))

