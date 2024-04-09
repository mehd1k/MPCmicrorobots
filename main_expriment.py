

import pandas as pd

import numpy as np
import sys
import Learning_module_2d as GP # type: ignore
from utils import readfile,test_gp,find_alpha_corrected
from utils import plot_xy,plot_traj,plot_vel,plot_bounded_curves
from scipy.ndimage import uniform_filter1d
from MR_simulator import Simulator
import math 
# from MPC import mpc_control
import matplotlib.pyplot as plt
def load_exp_data(dir):
    ###loads the data from expriment env

    return px_idle,py_idle,alpha_idle,time_idle,freq_idle




#first we do nothing
time_steps = 100 #do nothing for 100/30 seconds
actions_idle = np.zeros((time_steps, 2))

gp_sim = GP.LearningModule()
noise_vars= [0.0]



def control_action_creater(freq_range = 40, num_frq = 40):

# Creating a DataFrame with the specified tags (columns)
    columns = ["Frame", "Bx", "By", "Bz", "Alpha", "Gamma", "Rolling Frequency", "Psi", "Gradient", "Acoustic Frequency", "Sensor Bx", "Sensor By", "Sensor Bz"]
    df = pd.DataFrame(columns=columns)
    freq_ls = np.linspace(0.1, freq_range, num_frq)
    
    for  freq in freq_ls:
        time_steps = 300 #train for 10s at 30 hz
        cycles = 3 #train my moving in 3 circles

        steps = (int)(time_steps / cycles)

        #generate actions to move in a circle at a constant frequency
        actions_circle = np.zeros( (steps, 2))
        actions_circle[:,0] = freq
        actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)
        
        #stack the circle actions to get our learning set
        actions_learn = np.vstack([actions_circle]*cycles)
        for ai in range(len(actions_learn)):
            new_row = {"Frame": len(df)+1, "Bx": 0, "By": 0, "Bz": 0, "Alpha": actions_learn[ai,1], "Gamma": 0, "Rolling Frequency": actions_learn[ai,0], "Psi": 0, "Gradient": 0, "Acoustic Frequency": 0, "Sensor Bx": 0, "Sensor By": 0, "Sensor Bz": 0}
            df = df.append(new_row, ignore_index=True)
    df.to_excel("control_actions.xlsx", index=False)
        

def control_action_idle():

# Creating a DataFrame with the specified tags (columns)
    columns = ["Frame", "Bx", "By", "Bz", "Alpha", "Gamma", "Rolling Frequency", "Psi", "Gradient", "Acoustic Frequency", "Sensor Bx", "Sensor By", "Sensor Bz"]
    df = pd.DataFrame(columns=columns)
    for  i in range(100):
            new_row = {"Frame": len(df)+1, "Bx": 0, "By": 0, "Bz": 0, "Alpha": 0, "Gamma": 0, "Rolling Frequency": 0, "Psi": 0, "Gradient": 0, "Acoustic Frequency": 0, "Sensor Bx": 0, "Sensor By": 0, "Sensor Bz": 0}
            df = df.append(new_row, ignore_index=True)
    df.to_excel("idle_actions.xlsx", index=False)
        
# control_action_idle()
# control_action_creater()




#first we do nothing

gp_sim = GP.LearningModule()

#first we will do absolutely nothing to try and calculate the drift term
px_idle,py_idle,alpha_idle,time_idle,freq_idle = load_exp_data('idle_dir')
gp_sim.estimateDisturbance(px_idle, py_idle, time_idle)


px_sim,py_sim,alpha_sim,time_sim,freq_sim = load_exp_data('control_ol')


# learn noise and a0 -- note px_desired and py_desired need to be at the same time
a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim, freq_sim, time_sim)
print("Estimated a0 value is " + str(a0_sim))

gp_sim.visualize()




