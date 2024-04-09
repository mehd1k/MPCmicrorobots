


import numpy as np
import sys
import Learning_module_2d as GP # type: ignore
from utils import readfile,test_gp,find_alpha_corrected
from utils import plot_xy,plot_traj,plot_vel,plot_bounded_curves
from scipy.ndimage import uniform_filter1d
from MR_simulator import Simulator

import math 
from MPC import mpc_control, MPC
import matplotlib.pyplot as plt

def generate_in_between_points(node_ls):
    """
    Generates in-between points for a given list of segment endpoints.

    Parameters:
    - node_ls: Array of shape (number_of_segments, 2, 2), where each entry represents
               a segment with [start_point, end_point] and each point is [x, y].
    - num_points_per_segment: Number of in-between points to generate per segment.

    Returns:
    - full_trajectory: Array of points representing the full trajectory, including
                       the original endpoints and the newly generated in-between points.
    """
    full_trajectory = []

    for i in range(len(node_ls)-1):
        start_point, end_point = node_ls[i], node_ls[i+1]
        length = np.linalg.norm(end_point-start_point)
        num_points_per_segment= int(length/2)
        # Generate a sequence of numbers between 0 and 1, which will serve as interpolation factors.
        interpolation_factors = np.linspace(0, 1, num_points_per_segment + 2)
        
        # Interpolate x and y separately
        x_points = (1 - interpolation_factors) * start_point[0] + interpolation_factors * end_point[0]
        y_points = (1 - interpolation_factors) * start_point[1] + interpolation_factors * end_point[1]
        
        # Combine the x and y coordinates
        segment_points = np.vstack((x_points, y_points)).T
        full_trajectory.extend(segment_points[:-1].tolist())

    # Ensure the last point of the last segment is included
    full_trajectory.append(node_ls[-1].tolist())

    return np.array(full_trajectory)






#note: timestep is 1/30 seconds, the rate we get data at in the experiment


gp_sim = GP.LearningModule()

gp_sim.load_GP()
a0_sim = np.load('a_sim.npy')

# freq = 4
# a0_def = 1.5
dt = 0.030 #assume a timestep of 30 ms
noise_var = 0.0
sim = Simulator()

#### ref Trjactory


#########Ref Trajectory
# r0 = 2
# node_ls = np.load('node_path.npy')
# node_ls[3] = np.array([1500, 1600])
# node_ls = np.delete(node_ls, 1, 0)
# gpath_planner_traj = generate_in_between_points(node_ls)
# ref = gpath_planner_traj
# np.save('ref.npy', ref)
ref = np.load('ref.npy')

time_steps = len(ref)


x0 = [ref[0,0],ref[0,1]]


sim.reset_start_pos(x0)
sim.noise_var = noise_var
sim.a0 = a0_sim
sim.is_mismatched = True
########MPC parameters
B  = a0_sim*dt*np.array([[1,0],[0,1]])
A = np.eye(2)
# Weight matrices for state and input
Q = np.array([[1,0],[0,1]])
R = 0.01*np.array([[1,0],[0,1]])
N = 2
mpc = MPC(A= A, B=B, ref=ref, N=N, Q=Q, R=R)

# Simulates the dynamic system over T time steps using MPC control.
# 
#     Parameters:
    # - All parameters are as previously described.
    # - T: Total number of time steps to simulate.

    # Returns:
    # - x_traj: Trajectory of states.
    # - u_traj: Trajectory of control inputs.
    # """
x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

u_traj = np.zeros((time_steps, 2))
x_traj[0, :] = x0

alpha_t = 0
freq_t =0
for t in range(time_steps):
    # Update reference for the current time step
    current_ref = ref[t:min(t+N, time_steps), :]
    if current_ref.shape[0] < N:
        # Pad the reference if it's shorter than the prediction horizon
        current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))

    ### Disturbance Compensator 
    muX,sigX = gp_sim.gprX.predict(np.array([[alpha_t, freq_t]]), return_std=True)
    muY,sigY = gp_sim.gprY.predict(np.array([[alpha_t, freq_t]]), return_std=True)
    v_e = np.array([muX[0], muY[0]])
    
    Qz = 0*R
    

    u_mpc,pred_traj = mpc.control(x_traj[t, :], current_ref, (v_e)*dt)
    z0 = x_traj[t,:]-current_ref[0,:]



    u_current = u_mpc
    u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
    f_t = np.linalg.norm(u_current)
    alpha_t = math.atan2(u_current[1], u_current[0])


    # action = np.array([ [f_t], [alpha_t], [t*dt]])


    sim.step(f_t=f_t, alpha_t= alpha_t)
    x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
    # x_traj_lin[t+1, :] =x_traj_lin[t, :] + B @ u_opt

















import planner.plot
from planner.env import Map
start = (200, 200)
goal = (2200, 1800)
env = Map(2448, 2048)

plot =  planner.plot.Plot(start, goal, env)
plot.plot_traj( ref, x_traj) 
# plt.show()







# Plotting
# time_span = np.arange(time_steps+1)
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.plot(time_span, x_traj[:, 0], label='Actual Trajectory (x1)')
# plt.plot(time_span[1:], ref[:, 0], 'r--', label='Desired Trajectory (x1)')
# plt.xlabel('Time step')
# plt.ylabel('State x1')
# plt.title('Trajectory of State x1')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.plot(time_span, x_traj[:, 1], label='Actual Trajectory (x2)')
# plt.plot(time_span[1:], ref[:, 1], 'r--', label='Desired Trajectory (x2)')
# plt.xlabel('Time step')
# plt.ylabel('State x2')
# plt.title('Trajectory of State x2')
# plt.legend()
# plt.grid(True)


# plt.subplot(1, 3, 3)
# plt.plot(x_traj[:,0], x_traj[:, 1], label='Actual Trajectory (x1)')
# plt.plot(ref[:,0], ref[:, 1], 'r--', label='Desired Trajectory (x1)')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Trajectory')
# plt.legend()
# plt.axis('equal')
# plt.grid(True)
# plt.show()
