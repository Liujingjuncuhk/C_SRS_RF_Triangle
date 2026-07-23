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


if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    icl = c_srs.initial_cable_length.copy()
    cl_range_1 = [[icl[0]-0.08, icl[0]-0.02], 
                [icl[1]-0.08, icl[1]-0.02],
                [icl[2]-0.08, icl[2]-0.02],
                [icl[3], icl[3]+0.05],
                [icl[4], icl[4]+0.05],
                [icl[5], icl[5]+0.05]]
    
    cl_range_2 = [[icl[0]-0.03, icl[0]+0.01], 
                [icl[1]-0.03, icl[1]+0.01],
                [icl[2]-0.03, icl[2]+0.01],
                [icl[3]-0.04, icl[3]+0.01],
                [icl[4]-0.04, icl[4]+0.01],
                [icl[5]-0.04, icl[5]+0.01]]
    
    cl_range = [[icl[0]-0.08, icl[0]+0.01], 
                [icl[1]-0.08, icl[1]+0.01],
                [icl[2]-0.08, icl[2]+0.01],
                [icl[3]-0.04, icl[3]+0.05],
                [icl[4]-0.04, icl[4]+0.05],
                [icl[5]-0.04, icl[5]+0.05]]
    c_srs.generate_ws(cl_range, total_number=2000, saveFile='training_data_all.pkl')
    # c_srs.generate_ws(cl_range_2, total_number=1000, saveFile='training_data_2.pkl')