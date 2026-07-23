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

def see_FKD(c_srs: C_SRS_fixedEnd, tcl):
    Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, tcl, show_info=True)
    # Q_list = c_srs.FKD_static(c_srs.vertices, [1,1,1,2,2,2])
    vert_length = c_srs.q_to_vertices(Q_list[-1])
    cl_final = c_srs.get_cable_length_bary(vert_length)
    c_srs.visualize_vert(vert_length)
    
    tcl_return = [float(x) for x in tcl]
    print("target_cable_length: ", tcl_return)
    print("final cable length is: ", cl_final)
    print("cable tension is: ", cable_tension)
    return tcl_return, cl_final, vert_length


if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    icl = c_srs.initial_cable_length
    tcl_list = []
    fcl_list = []
    verts_list = []
    tcl_1 = [icl[0]-0.02, icl[1]-0.02, icl[2]-0.02, icl[3], icl[4], icl[5]]
    # c_srs.visualize_vert(vert_length)
    tcl, fcl, vert_length = see_FKD(c_srs, tcl_1)
    tcl_list.append(tcl)
    fcl_list.append(fcl)
    verts_list.append(vert_length)

    tcl_2 = [icl[0]-0.03, icl[1]+0.1, icl[2]+0.1, icl[3], icl[4]-0.01, icl[5]-0.01]
    tcl, fcl, vert_length = see_FKD(c_srs, tcl_2)
    tcl_list.append(tcl)
    fcl_list.append(fcl)
    verts_list.append(vert_length)

    tcl_3 = [icl[0], icl[1], icl[2]-0.01, icl[3]-0.02, icl[4]-0.02, icl[5]]
    # see_FKD(c_srs, tcl_3)
    tcl, fcl, vert_length = see_FKD(c_srs, tcl_3)
    tcl_list.append(tcl)
    fcl_list.append(fcl)
    verts_list.append(vert_length)

    data_2save = {"tcl_list": tcl_list, "fcl_list": fcl_list, "verts_list": verts_list}
    with open("data/fixedEnd_FKD_paper_data.pkl", "wb") as f:
        pickle.dump(data_2save, f)

    


