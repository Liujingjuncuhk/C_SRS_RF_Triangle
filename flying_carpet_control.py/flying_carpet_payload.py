import os
import sys
import inspect
import time
import numpy as np
from flying_carpet_sys import Flying_carpet_sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from flying_carpet import Flying_carpet
import pickle
# from force_sensor_single import TouchForceSensor
import matplotlib.pyplot as plt

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
    

def perform_experiment():
    flying_carpet_sys = Flying_carpet_sys()
    cur_cl = flying_carpet_sys.get_cur_length()
    icl = flying_carpet_sys.robot.initial_cable_length
    shorten = 0.13
    tcl = [icl[0]-shorten, icl[1]-shorten, icl[2]-shorten, icl[3]-shorten, 
           icl[4], icl[5], icl[6], icl[7]]
    flying_carpet_sys.move_to_length_timed(tcl, 3)

    flying_carpet_sys.checkpoint("press OK to continue")
    flying_carpet_sys.exit_all()

def plot_force():
    with open("data_flying_carpet/payload_limit.pkl", "rb") as f:
        forces = pickle.load(f)
    plt.plot(forces)
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.title("Force vs Time")
    plt.show()


def execute_traj(flying_carpet_sys: Flying_carpet_sys, cl_list, time_list):
    assert len(cl_list) == len(time_list)
    flying_carpet_sys.move_to_length_timed(cl_list[0], 2)
    flying_carpet_sys.checkpoint("press to start trajectory")
    flying_carpet_sys.execute_traj(time_list, cl_list)

if __name__ == "__main__":
    # generate_grasping_cl()
    
    flying_carpet_sys = Flying_carpet_sys()
    with open("data_flying_carpet/grasping_payload_test.pkl", "rb") as f:
        data_cl = pickle.load(f)

    cl_list = data_cl["cl_list"]
    vert_list = data_cl["vert_list"]
    ee_target_list = data_cl["ee_target_list"]
    # exit(0)
    # execute_traj_feedback(flying_carpet_sys, cl_list, vert_list, ee_target_list)
    # flying_carpet_sys.checkpoint("press to move back")
    # flying_carpet_sys.exit_all()
    # exit(0)
    total_time = 8

    time_list_ori = generate_time_list(ee_target_list, total_time)
    

    # print(cl_list)
    # time_list_ori = [0,2,4,6,8,10,12,14,16,18,20,22]

    time_ratio = 1
    time_list = [t * time_ratio for t in time_list_ori]

    execute_traj(flying_carpet_sys, cl_list, time_list)

    flying_carpet_sys.checkpoint("press to move back")
    flying_carpet_sys.exit_all()
    # plot_force()

    # force_sensor = TouchForceSensor()
    # print("Press 'Q' to stop collecting force data.")
    # forces = force_sensor.read_until_pressedKey()
    # # plot_force(forces)
    # print("Force data collection stopped.")
    # print("length of force data:", len(forces))
    # print("Force data:", forces)    