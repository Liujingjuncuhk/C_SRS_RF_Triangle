import os
import sys
import inspect
import time
import numpy as np
from fixedEnd_sys import FixedEndSystem
import pickle

cl_list_1 = [424, 425, 424, 298, 274, 298]
cl_list_2 = [416,430 , 436, 302, 270, 286]
cl_list_3 = [443,442,433, 277,264,292]

if __name__ == "__main__":
    filtered_region = [0.02, 0.3, 0, 0.16, -0.1, 0.1]
    fixedEnd_sys = FixedEndSystem()
    icl = fixedEnd_sys.get_cur_length()

    FKD_file = "data/fixedEnd_FKD_paper_data.pkl"

    with open(FKD_file, "rb") as f:
        data = pickle.load(f)
    tcl_list = data["tcl_list"]
    fcl_list = data["fcl_list"]
    verts_list = data["verts_list"]
    print("fcl_list: ", fcl_list)
    pts_list = []
    for i in range(len(fcl_list)):
        cl_cmd = fcl_list[i]
        cl_cmd_list = [float(x) for x in cl_cmd]
        verts = verts_list[i]
        fixedEnd_sys.move_to_length_timed(cl_cmd_list, 5)
        time.sleep(3)
        cl_feedback = fixedEnd_sys.get_cur_length()
        motor_pos = fixedEnd_sys.motor_controller.read_positions()
        pcd = fixedEnd_sys.camera.get_depth_pointcloud(region = filtered_region)
        pts = np.asarray(pcd.points)
        pts_list.append(pts)
        print("command cable length: ", cl_cmd_list)
        print("feedback cable length: ", cl_feedback)
        print("motor pos: ", motor_pos)
        fixedEnd_sys.c_srs.visualize_fb_surface_w_gt(verts, pts)
        fixedEnd_sys.move_to_length_timed(icl, 2)
        
        input("press Enter to continue...")

    with open("data/fixedEnd_FKD_exp_data.pkl", "wb") as f:
        pickle.dump({"tcl_list": tcl_list, "fcl_list": fcl_list, "verts_list": verts_list, "pts_list": pts_list}, f)