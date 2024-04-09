import numpy as np
import Learning_module as GP # type: ignore
from utils import run_sim,find_alpha_corrected
from utils import plot_xy,plot_traj,plot_bounded_curves,plot_curve
from scipy.ndimage import uniform_filter1d
from MR_simulator import Simulator
from MPC import mpc_control
import matplotlib.pyplot as plt
import math


#frequency of the magnetic field in Hz and the nominal value of a0
time_steps = 100
freq_ls = np.zeros(time_steps)
alpha_ls = np.zeros(time_steps)
a0_def = 1.5
dt = 0.030 #assume a timestep of 30 ms
noise_var = 0.5
sim = Simulator()
time_steps = 50
x0 = [0.0,0.0]
ref = np.ones((time_steps,2)) 
sim.reset_start_pos(x0)
sim.noise_var = noise_var
sim.a0 = a0_def
sim.is_mismatched = False
        
########MPC parameters
B  = a0_def*dt*np.array([[1,0],[0,1]])

# Weight matrices for state and input
Q = np.array([[1,0],[0,1]])
R = 0.1*np.array([[1,0],[0,1]])

N = 2


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
x_traj_lin = np.zeros((time_steps+1, 2))
u_traj = np.zeros((time_steps, 2))
x_traj[0, :] = x0
x_traj_lin[0,:] = x0

for t in range(time_steps):
    # # Update reference for the current time step
    # current_ref = ref[t:min(t+N, time_steps), :]
    # if current_ref.shape[0] < N:
    #     # Pad the reference if it's shorter than the prediction horizon
    #     current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))
    
    # u_opt = mpc_control(B, x_traj[t, :], current_ref, N, Q, R)
    # u_traj[t, :] = u_opt  # Assuming u_opt is the control input for the next step
    # f_t = np.linalg.norm(u_opt)
    # alpha_t = math.atan2(u_opt[1], u_opt[0])
    sim.step(f_t=freq_ls[t], alpha_t= alpha_ls[t])
    x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
    # x_traj_lin[t+1, :] =x_traj_lin[t, :] + B @ u_opt


# Plotting
time_span = np.arange(time_steps+1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_span, x_traj[:, 0], label='Actual Trajectory (x1)')
plt.plot(time_span, np.vstack((ref[:1, 0], np.ones((time_steps, 1)))), 'r--', label='Desired Trajectory (x1)')
plt.xlabel('Time step')
plt.ylabel('State x1')
plt.title('Trajectory of State x1')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_span, x_traj[:, 1], label='Actual Trajectory (x2)')
plt.plot(time_span, np.vstack((ref[:1, 1], np.ones((time_steps, 1)))), 'r--', label='Desired Trajectory (x2)')
plt.xlabel('Time step')
plt.ylabel('State x2')
plt.title('Trajectory of State x2')
plt.legend()
plt.grid(True)


# plt.subplot(1, 4, 3)
# plt.plot(time_span, x_traj_lin[:, 0], label='Actual Trajectory (x1)')
# plt.plot(time_span, np.vstack((ref[:1, 0], np.ones((time_steps, 1)))), 'r--', label='Desired Trajectory (x1)')
# plt.xlabel('Time step')
# plt.ylabel('State x1')
# plt.title('Trajectory of State x1')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 4, 4)
# plt.plot(time_span, x_traj_lin[:, 1], label='Actual Trajectory (x2)')
# plt.plot(time_span, np.vstack((ref[:1, 1], np.ones((time_steps, 1)))), 'r--', label='Desired Trajectory (x2)')
# plt.xlabel('Time step')
# plt.ylabel('State x2')
# plt.title('Trajectory of State x2')


plt.legend()
plt.grid(True)


# plt.subplot(1, 3, 3)
# plt.plot(time_span[0:time_steps], u_traj[:, 1]**2+u_traj[:, 0]**2, label='control effort')

# plt.xlabel('Time step')
# plt.ylabel('control effort')
# plt.title('control effort')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
plt.show()



























#first we do nothing
# time_steps = 100 #do nothing for 100/30 seconds
# actions_idle = np.zeros((time_steps, 3))
# actions_idle[:,2] = np.arange(0, dt*time_steps, dt)


# #note: timestep is 1/30 seconds, the rate we get data at in the experiment
# time_steps = 1800 #train for 60s at 30 hz
# cycles = 3 #train my moving in 3 circles

# steps = (int)(time_steps / cycles)

#generate actions to move in a circle at a constant frequency
# actions_circle = np.zeros( (steps, 3))
# actions_circle[:,0] = freq
# actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)

#stack the circle actions to get our learning set
# actions_learn = np.vstack([actions_circle]*cycles)
# actions_learn[:,2] = np.arange(0, dt*time_steps, dt)

# t = np.linspace(0, time_steps, time_steps)


#generate actions for testing (1/30 hz for 30 seconds)
# time_steps = 1000

# actions = np.zeros( (time_steps, 3) )

# actions[0:200,1]   = np.linspace(0, np.pi/2, 200)
# actions[200:400,1] = np.linspace(np.pi/2, -np.pi/2, 200)
# actions[400:600,1] = np.linspace(-np.pi/2, 0, 200)
# actions[600:800,1] = np.linspace(0, np.pi/8, 200)
# actions[800::,1]  = np.linspace(np.pi/8, -np.pi, 200)

# actions[:,0] = freq # np.linspace(3, 4, time_steps)
# actions[:,2] = np.arange(0, dt*time_steps, dt)



# gp_sim = GP.LearningModule()

#first we will do absolutely nothing to try and calculate the drift term
# px_idle,py_idle,alpha_idle,time_idle,freq_idle = run_sim(actions_idle,
#                                                             init_pos = np.array([0,0]),
#                                                             noise_var = noise_var,
#                                                             a0=a0_def,is_mismatched=True)


# gp_sim.estimateDisturbance(px_idle, py_idle, time_idle)


# # THIS IS WHAT THE SIMULATION ACTUALLY GIVES US -- model mismatch && noise
# px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions_learn,
# #                                                      init_pos = np.array([0,0]),
#                                                      noise_var = noise_var,
#                                                      a0=a0_def,is_mismatched=True)


# learn noise and a0 -- note px_desired and py_desired need to be at the same time
# a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim, time_sim, actions)
# print("Estimated a0 value is " + str(a0_sim))
# gp_sim.visualize()


# THIS CALCULATES THE DESIRED TRAJECTORY FROM OUR a0 ESTIMATE
# px_desired,py_desired,alpha_desired,time_desired,freq_desired = run_sim(actions_learn,
#                                                                         init_pos = np.array([0,0]),
#                                                                         noise_var = 0.0,a0=a0_sim)

# # plot the desired vs achieved velocities
# /xys  = [(px_desired,py_desired),
#         (px_sim,py_sim),
#        ]
# legends =["Desired Trajectory","Simulated Trajectory (no learning)"
#           ]
# fig_title   = ["Learning Dataset"]
# plot_xy(xys,legends =legends,fig_title =fig_title) 


