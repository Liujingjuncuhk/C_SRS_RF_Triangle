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
    cl_3 = (np.array([575, 470, 580, 445, 510, 348, 498, 309])*1e-3).tolist()
    start_vert = flying_carpet.vertices.copy()
    # start_vert[:,1] += 0.05
    # start_vert[:,2] += 0.05
    Q_list, vertices, cable_tension = flying_carpet.FKD_time(cl_3, 10, start_vert,h = 0.01,tol = 1e-6, show_info=1)

    flying_carpet.replay_Q_list(Q_list, filePath = "strange_FKD_FC.mp4")