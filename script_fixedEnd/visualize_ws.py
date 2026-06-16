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

def get_ws(filePath):
    with open(filePath, 'rb') as f:
        data = pickle.load(f)
    ee_pos_all = np.array(data['ee_pos'],       dtype=np.float32)  # (N, 3)
    return ee_pos_all

if __name__ == "__main__":


    description_file = "./models/flat_tri_surface/C_SRS_description.pkl"
    c_srs = C_SRS_fixedEnd(description_file)

    ws_all = get_ws("./data/training_data_all.pkl")
    c_srs.visualize_ws(c_srs.vertices, ws_all)
    exit(0)
    ws1_file = "./training_data_1.pkl"
    cl_list_all = []
    ee_pos_all = []
    vert_length_all = []
    cable_tension_all = []
    with open(ws1_file, 'rb') as f:
        data_1 = pickle.load(f)
    for data in data_1:
        ee_pos = data['ee_pos']
        if ee_pos[0] > 0.2:  # only visualize the points with x>0.2
            cl_list_all.append(data['cable_length'])
            ee_pos_all.append(data['ee_pos'])
            vert_length_all.append(data['vertices'])
            cable_tension_all.append(data['cable_tension'])


    ws2_file = "./training_data_2.pkl"
    with open(ws2_file, 'rb') as f:
        data_2 = pickle.load(f)
    for data in data_2:
        ee_pos = data['ee_pos']
        if ee_pos[0] > 0.2:  # only visualize the points with x>0.2
            cl_list_all.append(data['cable_length'])
            ee_pos_all.append(data['ee_pos'])
            vert_length_all.append(data['vertices'])
            cable_tension_all.append(data['cable_tension'])

    pkl_file_all = "./training_data_all.pkl"
    data_all = {
        'cable_length': cl_list_all,
        'ee_pos': ee_pos_all,
        'vertices': vert_length_all,
        'cable_tension': cable_tension_all
    }

    print("total number of data points: ", len(cl_list_all))
    
    with open(pkl_file_all, 'wb') as f:
        pickle.dump(data_all, f)

    c_srs.visualize_ws(c_srs.vertices, np.array(ee_pos_all))

    
    # data = {
    #     'cable_length': fcl,
    #     'ee_pos': ee_pos,
    #     'vertices': vert_length,
    #     'cable_tension': cable_tension
    # }
