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

def interpolate_trajectory(ee_pos_list, num_points):
    ee_pos_list = np.array(ee_pos_list)
    num_poses = ee_pos_list.shape[0]
    interpolated_trajectory = []
    
    for i in range(num_poses - 1):
        start_pose = ee_pos_list[i]
        end_pose = ee_pos_list[i + 1]
        
        for j in range(num_points):
            alpha = j / num_points
            interpolated_pose = (1 - alpha) * start_pose + alpha * end_pose
            interpolated_trajectory.append(interpolated_pose)
    
    interpolated_trajectory.append(ee_pos_list[-1])  # Add the last pose
    return np.array(interpolated_trajectory)



def visualize_IKD_paper_1(flying_carpet: Flying_carpet, tar_ee_interpolated, vert_interpolated):
    n_interpolated = len(tar_ee_interpolated)
    plotter = pv.Plotter()
    for i in range(n_interpolated):
        vertices = vert_interpolated[i]
        ee_poses = flying_carpet.get_ee_poses(vertices)
        mesh = pv.PolyData(vertices, np.hstack((np.full((flying_carpet.mesh_triangles.shape[0], 1), 3), flying_carpet.mesh_triangles)))
        if i == 0:
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        elif i == 10 or i == 20 or i == 30:
            target_ee_pos = tar_ee_interpolated[i]
            plotter.add_points(target_ee_pos, color='green', point_size=10, label='Target EE Pos')
            plotter.add_mesh(mesh, color='lightblue', show_edges=True)
            # plotter.add_points(ee_poses, color='red', point_size=10, label='End Effectors')
        else:
            plotter.add_mesh(mesh, color='lightblue', show_edges=True, opacity=0.1)
    points = np.array([[116,36,46], [116, 724, 46], [444, 36,46], [444,724, 46], [116,36,516], [116, 724, 516], [464,36,516], [464,724,516]]) * 1e-3
    # add these points:
    for point in points:
        plotter.add_mesh(pv.Sphere(radius=0.0001, center=point), color='white')
    
    # set camera
    plotter.camera_position = pv.CameraPosition(
                position=(0.4, 0.38, 0.2),
                focal_point=(0., 0.38, 0.12),
                viewup=(0.0, 0.0, 1.0),
            )
    # add axes and grid
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()
    # plotter.show(auto_close=False)
    # plotter.screenshot("./data_flying_carpet/IKD_illustration_paper_1.png")
    # plotter.close()

if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    with open("./data_flying_carpet/IK_trajectory_results_timed.pkl", "rb") as f:
        data = pickle.load(f)
    tar_ee_interpolated = data['traj_ee_interpolated']
    # tar_ee_interpolated = interpolate_trajectory(tar_ee_interpolated, num_points=10)
    # print("length of tar_ee_interpolated: ", len(tar_ee_interpolated))
    vert_interpolated = data['vert_list']
    
    # vert_interpolated = interpolate_trajectory(vert_interpolated, num_points=10)
    visualize_IKD_paper_1(flying_carpet, tar_ee_interpolated, vert_interpolated)
    # temp_script(flying_carpet, "./data_flying_carpet/IK_trajectory_results.pkl")