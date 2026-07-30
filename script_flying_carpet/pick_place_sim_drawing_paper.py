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


def draw_pick_place_sim(flying_carpet, vert_list):
    """
    Draw the pick and place simulation of the flying carpet using pyvista.
    
    Parameters:
    - flying_carpet: An instance of the Flying_carpet class.
    - ee_target_list: List of end-effector target positions.
    - vert_list: List of vertex positions for each time step.
    - cl_list: List of cable length commands for each time step.
    - framerate: Frame rate for the video output.
    - filePath: Path to save the output video.
    """
    # Create a pyvista plotter
    for i in range(len(vert_list)):
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        # Get the current vertex positions
        current_vertices = vert_list[i]
        mesh = pv.PolyData(current_vertices, np.hstack((np.full((flying_carpet.mesh_triangles.shape[0], 1), 3), flying_carpet.mesh_triangles)))
        plotter.add_mesh(mesh, color='lightgray', show_edges=True)
        pp_locations = flying_carpet.get_pp_location_bary(current_vertices)
        for j in range(flying_carpet.nCable):
            plotter.add_lines(np.array([pp_locations[j], flying_carpet.pulley_location[j]]), color='blue', width=0.5)
        plate_1 = pv.Disc(center=(0.28, 0.2, 0.08), inner=0., outer=0.04, normal=(0.0, 0.0, 1.0), c_res=30)
        plotter.add_mesh(plate_1, color='orange')
        plate_2 = pv.Disc(center=(0.28, 0.565, 0.08), inner=0., outer=0.04, normal=(0.0, 0.0, 1.0), c_res=30)
        plotter.add_mesh(plate_2, color='orange')

        if i in [0,1,2,3]:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.2, 0.10))
            plotter.add_mesh(ball_2caTCH, color='orange')

        if i == 4:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.2, 0.18))
            plotter.add_mesh(ball_2caTCH, color='orange')
        if i == 5:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.3, 0.18))
            plotter.add_mesh(ball_2caTCH, color='orange')
        if i == 6:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.38, 0.18))
            plotter.add_mesh(ball_2caTCH, color='orange')
        if i == 7:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.48, 0.18))
            plotter.add_mesh(ball_2caTCH, color='orange')
        if i == 8:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.565, 0.18))
            plotter.add_mesh(ball_2caTCH, color='orange')

        if i in [9,10,11]:
            ball_2caTCH = pv.Sphere(radius=0.02, center=(0.28, 0.565, 0.10))
            plotter.add_mesh(ball_2caTCH, color='orange')

        # 
        plotter.camera.position = (1.3, 0.38, 0.3)
        plotter.camera.focal_point = (0.0, 0.38, 0.2)
        plotter.camera.up = (0, 0, 1.0)

        # plotter.camera_position = 'yz'
        
        # plotter.show_axes()
        plotter.show_grid()
        # plotter.show()
        # save the current frame as an image
        filename = f"flying_carpet_pick_place/frame_{i:03d}.png"
        plotter.screenshot(filename)
        # break


if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)

    with open("data_flying_carpet/pick_place_data.pkl", "rb") as f:
        data_cl = pickle.load(f)

    cl_list = data_cl["cl_list"]
    vert_list = data_cl["vert_list"]
    ee_target_list = data_cl["ee_target_list"]
    print("length of vert_list:", len(vert_list))
    draw_pick_place_sim(flying_carpet, vert_list)