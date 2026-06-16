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
from C_SRS_fixedEnd import C_SRS_fixedEnd
import pickle

class IK_MLP(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,   128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,   6),
        )
    def forward(self, x):
        return self.net(x)
    
    def predict_cable_length(self,  ee_pos):
        """ee_pos: (3,) array in metres. Returns (6,) cable lengths in metres."""
        x = scaler_X.transform(ee_pos.reshape(1, 3))
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_norm = self(x_t).numpy()
        return scaler_Y.inverse_transform(y_norm).flatten()

scaler_X = joblib.load("./learning_model/scaler_X.pkl")
scaler_Y = joblib.load("./learning_model/scaler_Y.pkl")
model = IK_MLP()
model.load_state_dict(torch.load("./learning_model/ik_model_best.pth"))
model.eval()



if __name__ == "__main__":
    description_file = "./models/flat_tri_surface/C_SRS_description.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    ee_target = np.array([0.24, 0.08, 0.06])
    tcl = model.predict_cable_length(ee_target)
    print(tcl)   # → 6 cable lengths in metres
    Q_list, vert_length, cable_tension = c_srs.FKD_time(tcl, 1, c_srs.vertices, tol = 1e-5)
    c_srs.visualize_IKD_result(vert_length, ee_target)