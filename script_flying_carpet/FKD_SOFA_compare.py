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

cl_1 = (np.array([495, 483, 506, 470, 450, 437, 457, 426])*1e-3).tolist()
cl_2 = (np.array([372, 550, 388, 570, 363, 518, 379, 525])*1e-3).tolist()
cl_3 = (np.array([575, 470, 580, 445, 510, 348, 498, 309])*1e-3).tolist()
cl_list = [cl_1, cl_2, cl_3]

def get_FKD_flat(flying_carpet:Flying_carpet, cable_lengths):
    """
    Get the flat FKD (Forward Kinematics) for a given set of cable lengths.

    Parameters
    ----------
    flying_carpet : Flying_carpet
        The flying carpet object.
    cable_lengths : list of list/array, shape (N, nCable)
        Cable length targets for each waypoint.

    Returns
    -------
    flat_FKD : list of list/array, shape (N, nVertices)
        Flattened FKD results for each waypoint.
    """
    flat_FKD = []
    start_vert_3 = flying_carpet.vertices.copy()
    start_vert_3[:,1] += 0.05
    start_vert_3[:,2] += 0.05

    for i in range(len(cable_lengths)):
        # if i == 2:
        #     start_vert = start_vert_3
        # else:
        start_vert = flying_carpet.vertices
        Q_list, vertices, cable_tension = flying_carpet.FKD_time(cable_lengths[i], 10, start_vert,h = 0.01,tol = 1e-6, show_info=1)
        flat_FKD.append(vertices)
        flying_carpet.visualize_vert(vertices)
    with open("data_flying_carpet/flat_FKD.pkl", "wb") as f:
        pickle.dump(flat_FKD, f)
    return flat_FKD

def get_FKD_3(flying_carpet:Flying_carpet):
    tcl = cl_3
    start_vert = flying_carpet.vertices
    start_vert[:, 1] += 0.1
    start_vert[:,2] += 0.05
    Q_list, vertices, cable_tension = flying_carpet.FKD_time(tcl, 10, start_vert,h = 0.01,tol = 1e-6, show_info=1)
    flying_carpet.visualize_vert(vertices)



def compare_SOFA_FKD(flying_carpet:Flying_carpet):
    def get_fb_vert(vertices, feedback_idx):
        return vertices[feedback_idx, :]

    feedback_pts_list = []
    for i in range(3):
        feedback_filename = "data_flying_carpet/feedback_points_"+str(i+1)+".pickle"
        with open(feedback_filename, "rb") as f:
            feedback_pts = pickle.load(f)
            feedback_pts_list.append(feedback_pts)
    
    with open("data_flying_carpet/vert_tets.pickle", "rb") as f:
        vert_tets = pickle.load(f)

    with open("data_flying_carpet/other_info.pickle", "rb") as f:
        other_info = pickle.load(f)
        tetrahedra = other_info["tetrahedra"]
        feedback_idx = other_info["fd_idx"]

    with open("data_flying_carpet/flat_FKD.pkl", "rb") as f:
        vert_flat_list = pickle.load(f)

    for i in range(3):
        plotter = pv.Plotter()
        vert_flat = vert_flat_list[i]
        fb_vert_flat = flying_carpet.get_fb_surface(vert_flat)
        vert_tet = vert_tets[i]
        gt_pts = feedback_pts_list[i]
        vert_tet_fb = vert_tet[feedback_idx, :]
        faces = np.hstack((np.full((flying_carpet.mesh_triangles.shape[0], 1), 3), flying_carpet.mesh_triangles))
        mesh_surface = pv.PolyData(vert_flat, faces)
        mesh_SOFA = pv.PolyData(vert_tet)
        mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra])
        mesh_fb = pv.PolyData(fb_vert_flat, faces)
        plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, label='FB Mid-Surface')
        plotter.add_mesh(mesh_SOFA, color='lightcoral', show_edges=True, opacity=0.5, label='SOFA Surface')
        plotter.add_points(gt_pts, color='green', point_size=10, label='Ground Truth Points')
        plotter.add_points(vert_tet_fb, color='red', point_size=10, label='FB points from SOFA')
        plotter.show_grid()
        plotter.show()



if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # print("initial cable length: ", flying_carpet.get_cable_length_bary(flying_carpet.vertices))
    # get_FKD_flat(flying_carpet, cl_list)
    # get_FKD_3(flying_carpet)
    compare_SOFA_FKD(flying_carpet)

    