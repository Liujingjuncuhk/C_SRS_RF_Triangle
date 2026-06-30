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
    filename = "./data_f I lying_carpet/80mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_centered = pickle.load(f)
    # offset = np.array([0.28, 0.4, 0.25])
    # offset = np.array([0.27938779, 0.37983389, 0.27474488])
    offset = np.array([0.27898019, 0.37982945, 0.26301642])
    ee_target_pos = ee_pos_centered + offset
    final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, flying_carpet.vertices, max_iter=30, tol=5e-3)

    flying_carpet.replay_IKD_Q_list(ee_target_pos, Q_list)
    flying_carpet.visualize_IKD_result(ee_target_pos, final_vert)
