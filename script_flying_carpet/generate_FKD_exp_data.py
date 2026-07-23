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


if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)

    icl = flying_carpet.initial_cable_length
    shortened_length = 0.04
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    Q_list, vert_length, cable_tension = flying_carpet.FKD_time(tcl, 1, flying_carpet.vertices, tol = 1e-6, show_info = True, h = 0.01)
    flying_carpet.visualize_vert(vert_length)
    ee_pos = flying_carpet.get_ee_poses(vert_length)
    fb_pts = flying_carpet.get_fb_surface(vert_length)
    fcl = flying_carpet.get_cable_length_bary(vert_length)
    data_2save = {
        "ee_pos": ee_pos,
        "vert_length": vert_length,
        "fb_pts": fb_pts,
        "fcl": fcl
    }

    with open("./data_flying_carpet/40mm_FKD.pkl", 'wb') as f:
        pickle.dump(data_2save, f)


