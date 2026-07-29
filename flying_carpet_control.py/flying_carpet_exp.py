import os
import sys
import inspect
import time
import numpy as np
from flying_carpet_sys import Flying_carpet_sys
import pickle

def execute_traj(flying_carpet_sys: Flying_carpet_sys, cl_list, time_list):
    assert len(cl_list) == len(time_list)
    flying_carpet_sys.move_to_length_timed(cl_list[0], 2)
    flying_carpet_sys.checkpoint("press to start trajectory")
    flying_carpet_sys.execute_traj(time_list, cl_list)


def execute_traj_feedback(flying_carpet_sys: Flying_carpet_sys, cl_list, vert_list, target_ee_list):
    flying_carpet_sys.move_to_length_timed(cl_list[0], 2)
    flying_carpet_sys.checkpoint("press to start trajectory")
    pts_fb_list = []
    for i in range(len(cl_list)):
        flying_carpet_sys.move_to_length_timed(cl_list[i], 2)
        vert_sim = vert_list[i]
        tar_ee = target_ee_list[i]
        xrange = (np.min(vert_sim[:,0]-0.01), np.max(vert_sim[:,0]+0.01))
        yrange = (np.min(vert_sim[:,1]-0.01), np.max(vert_sim[:,1]+0.01))
        zrange = (np.min(vert_sim[:,2]-0.01), np.max(vert_sim[:,2]+0.01))
        flying_carpet_sys.checkpoint("press to take feedback")
        pts_fb = flying_carpet_sys.get_feedback_pts(x_range=xrange, y_range=yrange, z_range=zrange)
        pts_fb_list.append(pts_fb)
        flying_carpet_sys.robot.visualize_vert_w_fb(vert_sim, pts_fb)
    # with open("data_flying_carpet/pick_place_feedback.pkl", "wb") as f:
    #     pickle.dump({"pts_fb_list": pts_fb_list, "cl_list": cl_list, "vert_list": vert_list, "ee_target_list": target_ee_list}, f)


def generate_time_list(ee_target_list, total_time):
    time_list_ori = [0]
    ave_dist_between = []
    for i in range(1, len(ee_target_list)):
        dist_ee_all = 0
        for j in range(len(ee_target_list[i])):
            dist_ee_all += np.linalg.norm(np.array(ee_target_list[i][j]) - np.array(ee_target_list[i-1][j]))
        ave_dist_between.append(dist_ee_all / len(ee_target_list[i]))

    total_ave_dist = np.sum(ave_dist_between)
    for i in range(len(ave_dist_between)):
        time_list_ori.append(time_list_ori[-1] + ave_dist_between[i] / total_ave_dist * total_time)
    
    return time_list_ori
    

if __name__ == "__main__":
    flying_carpet_sys = Flying_carpet_sys()
    with open("data_flying_carpet/pick_place_data.pkl", "rb") as f:
        data_cl = pickle.load(f)

    cl_list = data_cl["cl_list"]
    vert_list = data_cl["vert_list"]
    ee_target_list = data_cl["ee_target_list"]
    # execute_traj_feedback(flying_carpet_sys, cl_list, vert_list, ee_target_list)
    # flying_carpet_sys.checkpoint("press to move back")
    # flying_carpet_sys.exit_all()
    # exit(0)
    total_time = 20

    time_list_ori = generate_time_list(ee_target_list, total_time)
    

    # print(cl_list)
    # time_list_ori = [0,2,4,6,8,10,12,14,16,18,20,22]

    time_ratio = 1
    time_list = [t * time_ratio for t in time_list_ori]

    execute_traj(flying_carpet_sys, cl_list, time_list)

    flying_carpet_sys.checkpoint("press to move back")
    flying_carpet_sys.exit_all()



