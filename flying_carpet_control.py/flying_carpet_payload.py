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
from force_sensor_single import TouchForceSensor
import matplotlib.pyplot as plt

def generate_grasping_cl():
    description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"
    flying_carpet = Flying_carpet(description_file)
    icl = flying_carpet.initial_cable_length
    shorten = 0.1
    tcl = [icl[0]-shorten, icl[1]-shorten, icl[2]-shorten, icl[3]-shorten, 
           icl[4], icl[5], icl[6], icl[7]]

def perform_experiment():
    flying_carpet_sys = Flying_carpet_sys()
    force_sensor = TouchForceSensor()
    cur_cl = flying_carpet_sys.get_cur_length()
    icl = flying_carpet_sys.robot.initial_cable_length
    shorten = 0.12
    tcl = [icl[0]-shorten, icl[1]-shorten, icl[2]-shorten, icl[3]-shorten, 
           icl[4], icl[5], icl[6], icl[7]]
    flying_carpet_sys.move_to_length_timed(tcl, 3)
    input("press to start collect force data")
    forces = force_sensor.read_until_pressedKey()
    with open("data_flying_carpet/payload_limit.pkl", "wb") as f:
        pickle.dump(forces, f)

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

if __name__ == "__main__":
    # generate_grasping_cl()
    
    # perform_experiment()
    plot_force()

    # force_sensor = TouchForceSensor()
    # print("Press 'Q' to stop collecting force data.")
    # forces = force_sensor.read_until_pressedKey()
    # # plot_force(forces)
    # print("Force data collection stopped.")
    # print("length of force data:", len(forces))
    # print("Force data:", forces)    