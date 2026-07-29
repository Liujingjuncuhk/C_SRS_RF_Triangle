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
from matplotlib import pyplot as plt


def view_pick_place_paper():
    with open("data_flying_carpet/pick_place_data.pkl", "rb") as f:
        data_cl = pickle.load(f)
    cl_list = data_cl["cl_list"]
    vert_list = data_cl["vert_list"]
    ee_target_list = data_cl["ee_target_list"]

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    
    