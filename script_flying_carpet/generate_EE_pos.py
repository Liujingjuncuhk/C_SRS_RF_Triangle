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
    shortened_length = 0.08
    tcl = [icl[0]-shortened_length, icl[1]-shortened_length, icl[2]-shortened_length, icl[3]-shortened_length, icl[4], icl[5], icl[6], icl[7]]
    Q_list, vert_length, cable_tension = flying_carpet.FKD_time(tcl, 1, flying_carpet.vertices, tol = 1e-6, show_info = True)
    flying_carpet.visualize_vert(vert_length)
    ee_pos = flying_carpet.get_ee_poses(vert_length)
    ee_pos_centered = ee_pos - np.mean(ee_pos, axis=0)
    print("ee center position: ", np.mean(ee_pos, axis=0))
    print("EE positions: ", ee_pos_centered)
    pickleFile = "./data_flying_carpet/80mm_centered.pkl"
    with open(pickleFile, 'wb') as f:
        pickle.dump(ee_pos_centered, f)
    