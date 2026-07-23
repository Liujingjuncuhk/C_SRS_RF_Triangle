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

def add_translation(ee_pos, translation_vector):
    ee_pos_translated = ee_pos.copy()
    for i in range(ee_pos.shape[0]):
        ee_pos_translated[i] += translation_vector
    return ee_pos_translated


def visualize_3_shapes(flying_carpet, ee_list):
    plotter = pv.Plotter()
    plotter.add_mesh(flying_carpet.mesh, color='lightblue', opacity=0.5)
    for i, ee_pos in enumerate(ee_list):
        plotter.add_mesh(pv.PolyData(ee_pos), color='red', point_size=10, render_points_as_spheres=True)
        plotter.add_text(f"Shape {i+1}", position='upper_left', font_size=12, color='black')
    plotter.show()

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)

    with open("./data_flying_carpet/60mm_FKD.pkl", 'rb') as f:
        shape_60 = pickle.load(f)
        shape_60_vert = shape_60["vert_length"]

    with open("./data_flying_carpet/80mm_centered.pkl", 'rb') as f:
        shape_80 = pickle.load(f)
        shape_80_vert = shape_80["vert_length"]


    with open("./data_flying_carpet/100mm_centered.pkl", 'rb') as f:
        shape_100 = pickle.load(f)
        shape_100_vert = shape_100["vert_length"]


    cl_list = []
    vert_list = []

    vert_1 = add_translation(shape_60["ee_pos"], np.array([0.28, 0.25, 0.25]))
    vert_2 = add_translation(shape_80["ee_pos"], np.array([0.28, 0.38, 0.25]))
    vert_3 = add_translation(shape_100["ee_pos"], np.array([0.28, 0.51, 0.25]))

    ee_list = [vert_1, vert_2, vert_3]

    for i in range(3):
        cur_length, starting_vertices, final_Q_list = flying_carpet.IKD_single(ee_list[i], flying_carpet.vertices,tol = 3e-3, max_iter = 30, show_info = True)
        cl_list.append(cur_length)
        vert_list.append(starting_vertices)
        flying_carpet.visualize_IKD_result(ee_list[i], starting_vertices)
        
    data_to_save = {
        "cl_list": cl_list,
        "vert_list": vert_list
    }
    with open("./data_flying_carpet/3_shapes_cl_vert.pkl", 'wb') as f:
        pickle.dump(data_to_save, f)
