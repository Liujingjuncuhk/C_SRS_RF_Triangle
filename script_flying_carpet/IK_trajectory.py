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

def rotZ(thetad):
    thetar = np.deg2rad(thetad)
    return np.array([[np.cos(thetar), -np.sin(thetar), 0],
                     [np.sin(thetar), np.cos(thetar), 0],
                     [0, 0, 1]])

def add_rotation(tar_ee_pos, rotation_matrix):
    return np.dot(rotation_matrix, tar_ee_pos.T).T

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


def visualize_interpolated_ee(ee_pos_list):
    num_poses = len(ee_pos_list)
    
    # Create a PyVista plotter
    plotter = pv.Plotter()
    
    # Add the end-effector positions to the plotter
    for i in range(num_poses):
        ee_pos = ee_pos_list[i]
        for j in range(ee_pos.shape[0]):
            plotter.add_mesh(pv.Sphere(radius=0.01, center=ee_pos[j]), color='red')
    
    # Connect the end-effector positions with lines
    # for i in range(num_poses - 1):
    #     start_pose = ee_pos_list[i]
    #     end_pose = ee_pos_list[i + 1]
    #     for j in range(start_pose.shape[0]):
    #         line = pv.Line(start_pose[j], end_pose[j])
    #         plotter.add_mesh(line, color='blue', line_width=2)
    pulley_locations = np.array([[116,36,46], [116, 724, 46], [444, 36,46], [444,724, 46], [116,36,516], [116, 724, 516], [464,36,516], [464,724,516]]) * 1e-3
    for pulley in pulley_locations:
        plotter.add_mesh(pv.Sphere(radius=0.01, center=pulley), color='white')
    # Show the plot
    # add axes and grid
    plotter.show_axes()
    plotter.show_grid()
    plotter.show()

def add_translation(ee_pos, translation_vector):
    ee_pos_translated = ee_pos.copy()
    for i in range(ee_pos.shape[0]):
        ee_pos_translated[i] += translation_vector
    return ee_pos_translated

def traj_paper_IKD(flying_carpet: Flying_carpet):
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_centered = pickle.load(f)
    initial_ee = flying_carpet.get_ee_poses(flying_carpet.vertices)
    initial_ee = initial_ee - np.mean(initial_ee, axis=0)
    initial_offset =  np.array([0.28, 0.2, 0.3])
    initial_ee = add_translation(initial_ee, initial_offset)
    tar_ee_list = [initial_ee]
    ee_2 = add_translation(ee_pos_centered, np.array([0.28, 0.2, 0.15]))
    tar_ee_list.append(ee_2)
    ee_3 = add_rotation(ee_pos_centered, rotZ(30))
    ee_3 = add_translation(ee_3, np.array([0.28, 0.4, 0.25]))

    tar_ee_list.append(ee_3)

    ee_4 = add_translation(ee_pos_centered, np.array([0.28, 0.6, 0.25]))
    ee_4[3,0] += 0.05
    ee_4[3,1] += 0.05
    ee_4[3,2] -= 0.05
    tar_ee_list.append(ee_4)
    print("Target EE positions: ", tar_ee_list)

    # tar_ee_interpolated = interpolate_trajectory(tar_ee_list, num_points=10)
    # visualize_interpolated_ee(tar_ee_interpolated)  
    return tar_ee_list

def tarj_paper_IKD_2(flying_carpet: Flying_carpet):
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_60 = pickle.load(f)
    filename = "./data_flying_carpet/40mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_40 = pickle.load(f)
    tar_ee_list = []
    tar_ee_list.append(add_translation(ee_40, np.array([0.28, 0.2, 0.3])))
    tar_ee_list.append(add_translation(ee_60, np.array([0.28, 0.3, 0.3])))
    tar_ee_list.append(add_translation(ee_40, np.array([0.28, 0.4, 0.3])))
    tar_ee_list.append(add_translation(ee_60, np.array([0.28, 0.5, 0.3])))
    tar_ee_list.append(add_translation(ee_40, np.array([0.28, 0.6, 0.3])))
    return tar_ee_list

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
    
            
            

def IKD_illustration_paper(flying_carpet: Flying_carpet, IK_traj_filename):
    with open(IK_traj_filename, 'rb') as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['traj_ee_interpolated']
    # Q_list_all = traj_result['Q_list_all']
    vert_list = traj_result['vert_list']
    ntarget = len(ee_target_list)
    tar_ee_interpolated = interpolate_trajectory(ee_target_list, num_points=10)
    vert_interpolated = interpolate_trajectory(vert_list, num_points=10)
    n_interpolated = len(tar_ee_interpolated)
    error_list = []
    # for i in range(n_interpolated):
    #     flying_carpet.visualize_IKD_result(tar_ee_interpolated[i], vert_interpolated[i])
    for i in range(n_interpolated):
        ee_pos = flying_carpet.get_ee_poses(vert_interpolated[i])
        diff = 1/2*np.linalg.norm(ee_pos - tar_ee_interpolated[i])**2
        error_list.append(diff)
    # visualize_IKD_paper_1(flying_carpet, tar_ee_interpolated, vert_interpolated)
    # draw the error list using matplotlib
    plt.figure()
    # set log scale for y axis
    # plt.yscale('log')
    plt.plot(error_list[1:], marker='o', linestyle='-')
    
    # show grid
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error vs Iteration')
    plt.show()
        
    
    
def temp_script(flying_carpet: Flying_carpet, IK_traj_filename):
    with open(IK_traj_filename, 'rb') as f:
        traj_result = pickle.load(f)
    ee_target_list = traj_result['traj_ee_interpolated']
    Q_list_all = traj_result['Q_list_all']
    
    vert_list = [flying_carpet.q_to_vertices(Q_list_all[0])]
    # vert_list = traj_result['vert_list']
    print("length of Q_list_all: ", len(Q_list_all))
    print("type of each Q_list: ", [type(Q_list) for Q_list in Q_list_all])
    print("length of each Q_list: ", [len(Q_list) for Q_list in Q_list_all])
    print("length of ee_target_list: ", len(ee_target_list))
    # vert_list = [flying_carpet.q_to_vertices(Q_list[-1]) for Q_list in Q_list_all]
    for i in range(4):
        if i > 0:
            vert = flying_carpet.q_to_vertices(Q_list_all[i][-1])
            vert_list.append(vert)
            # flying_carpet.visualize_IKD_result(ee_target_list[i], vert)
    save_data = {
        "traj_ee_interpolated": ee_target_list,
        "Q_list_all": Q_list_all,
        "vert_list": vert_list
    }
    with open("./data_flying_carpet/IK_trajectory_results.pkl", 'wb') as f:
        pickle.dump(save_data, f)


def IK_trajectory(flying_carpet: Flying_carpet, tar_ee_interpolated):
    starting_vert = flying_carpet.vertices - np.mean(flying_carpet.vertices, axis=0)
    starting_vert = starting_vert + np.array([0.28, 0.2, 0.3])
    # flying_carpet.visualize_vert(starting_vert)
    # exit(0)
    Q_list_all = [flying_carpet.vertices_to_q(starting_vert)]
    vert_list = [starting_vert.copy()]
    starting_vert = flying_carpet.vertices.copy()
    diff_list = [0]
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_centered = pickle.load(f)
    ee_4 = add_translation(ee_pos_centered, np.array([0.28, 0.6, 0.25]))
    for i in range(len(tar_ee_interpolated)):
        if i == 0:
            continue
        ee_target_pos = tar_ee_interpolated[i]
        if i != len(tar_ee_interpolated) - 1:
            starting_vert = flying_carpet.get_fixedEE_guess_vertices(ee_target_pos)
            final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, starting_vert, max_iter=50, tol=5e-3, show_info = True)
        else: 
            # starting_vert = flying_carpet.vertices.copy()
            starting_vert = flying_carpet.get_fixedEE_guess_vertices(ee_4)
            final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, starting_vert, max_iter=30, tol=5e-3, show_info = True, initial_guess=False)
        flying_carpet.visualize_IKD_result(ee_target_pos, final_vert)
        ee_pos = flying_carpet.get_ee_poses(final_vert)
        diff = 1/2*np.linalg.norm(ee_pos - ee_target_pos)**2
        vert_list.append(final_vert)
        Q_list_all.append(Q_list)
        diff_list.append(diff)
        starting_vert = final_vert
        print(f"Step {i}: Error = {diff:.6f}, Final cable length = {final_length}")
    return Q_list_all, diff_list, vert_list

def get_error(flying_carpet: Flying_carpet, ee_target_pos, vert):
    ee_pos = flying_carpet.get_ee_poses(vert)
    diff = 0
    for i in range(ee_pos.shape[0]):
        diff += 1/2*np.linalg.norm(ee_pos[i] - ee_target_pos[i])**2
    return diff

def IK_trajectory_time_comparison(flying_carpet: Flying_carpet, tar_ee_interpolated):
    starting_vert = flying_carpet.vertices - np.mean(flying_carpet.vertices, axis=0)
    starting_vert = starting_vert + np.array([0.28, 0.2, 0.3])
    Q_list_all = [flying_carpet.vertices_to_q(starting_vert)]
    vert_list = [starting_vert.copy()]
    starting_vert = flying_carpet.vertices.copy()
    diff_list = [0]
    filename = "./data_flying_carpet/60mm_centered.pkl"
    with open(filename, 'rb') as f:
        ee_pos_centered = pickle.load(f)
    ee_4 = add_translation(ee_pos_centered, np.array([0.28, 0.6, 0.25]))
    time_list_CG = []
    time_list_FD = []
    error_list_CG = []
    error_list_FD = []
    vert_list_FD = [starting_vert.copy()]
    Q_list_all_FD = [flying_carpet.vertices_to_q(starting_vert)]
    for i in range(len(tar_ee_interpolated)):
        if i == 0:
            continue
        ee_target_pos = tar_ee_interpolated[i]
        start_time = time.time()
        if i <= 20:
            starting_vert = flying_carpet.get_fixedEE_guess_vertices(ee_target_pos)
            # starting_vert = vert_list[-1]
            start_time = time.time()
            final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, starting_vert, max_iter=50, tol=1e-4, show_info = True, initial_guess = False)
            time_list_CG.append(time.time() - start_time)
            error_list_CG.append(get_error(flying_carpet, ee_target_pos, final_vert))
            vert_list.append(final_vert)
            Q_list_all.append(Q_list)
            # starting_vert = vert_list_FD[-1]
            # start_FD_time = time.time()
            # final_length, final_vert, Q_list = flying_carpet.IKD_single_minimize(ee_target_pos, starting_vert, max_iter=50, tol=1e-4, show_info = True, initial_guess = False)
            # vert_list_FD.append(final_vert)
            # Q_list_all_FD.append(Q_list)
            # time_list_FD.append(time.time() - start_FD_time)
            # error_list_FD.append(get_error(flying_carpet, ee_target_pos, final_vert))
            print(f"Step {i}: CG Time taken: {time_list_CG[-1]:.4f} seconds, CG Error = {error_list_CG[-1]:.6f}")
        else: 
            # starting_vert = flying_carpet.get_fixedEE_guess_vertices(ee_4)
            # starting_vert = vert_list[-1]
            starting_vert = flying_carpet.vertices.copy()
            start_time = time.time()
            final_length, final_vert, Q_list = flying_carpet.IKD_single(ee_target_pos, starting_vert, max_iter=30, tol=1e-4, show_info = True, initial_guess=False)
            time_list_CG.append(time.time() - start_time)
            error_list_CG.append(get_error(flying_carpet, ee_target_pos, final_vert))
            vert_list.append(final_vert)
            Q_list_all.append(Q_list)   

            # starting_vert = vert_list_FD[-1]
            # start_FD_time = time.time()
            # final_length, final_vert, Q_list = flying_carpet.IKD_single_minimize(ee_target_pos, starting_vert, max_iter=30, tol=1e-4, show_info = True, initial_guess=False)
            # time_list_FD.append(time.time() - start_FD_time)
            # vert_list_FD.append(final_vert)
            # Q_list_all_FD.append(Q_list)
            # error_list_FD.append(get_error(flying_carpet, ee_target_pos, final_vert))
            print(f"Step {i}: CG Time taken: {time_list_CG[-1]:.4f} seconds, CG Error = {error_list_CG[-1]:.6f}")

        # print(f"Step {i}: Error = {diff:.6f}, Final cable length = {final_length}, Time taken: {time_list_CG[-1]:.4f} seconds, CG Error = {error_list_CG[-1]:.6f}")
    return Q_list_all, vert_list, time_list_CG, error_list_CG


if __name__ == "__main__":
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    # IKD_illustration_paper(flying_carpet, "./data_flying_carpet/IK_trajectory_results.pkl")
    # # temp_script(flying_carpet, "./data_flying_carpet/IK_trajectory_results.pkl")
    # exit(0)

    # traj_ee_interpolated = traj_paper_IKD(flying_carpet)
    traj_ee_interpolated = interpolate_trajectory(traj_paper_IKD(flying_carpet), num_points=10)
    # print("length of traj_ee_interpolated: ", len(traj_ee_interpolated))
    # exit(0)

    Q_list_all, vert_list, time_list_CG, error_list_CG = IK_trajectory_time_comparison(flying_carpet, traj_ee_interpolated)
    # print("Error list: ", diff_list)
    dict2save = {
        "traj_ee_interpolated": traj_ee_interpolated,
        "Q_list_all": Q_list_all,
        "vert_list": vert_list,
        "time_list_CG": time_list_CG,
        "error_list_CG": error_list_CG
    }

    with open("./data_flying_carpet/IK_trajectory_results_timed.pkl", 'wb') as f:
        pickle.dump(dict2save, f)
    




