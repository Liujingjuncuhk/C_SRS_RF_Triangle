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
from flying_carpet_torch import Flying_carpet_torch
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet_torch(description_file)
    