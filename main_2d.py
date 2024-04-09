



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


def run_sim(actions,init_pos,noise_var = 1,a0 =1.5, is_mismatched = False):
    # state_prime = np.empty((0,2))
    # states      = np.empty((0,2))
    sim = Simulator()
    sim.reset_start_pos(init_pos)
    sim.noise_var = noise_var
    sim.a0 = a0
    sim.is_mismatched = is_mismatched
    time_steps = len(actions)
    X = np.zeros(time_steps)
    Y = np.zeros(time_steps)
    # X[0] = init_pos[0]
    # Y[0] = init_pos[1]
    # state       = env.reset(init = init_pos,noise_var = noise_var,a0=a0, is_mismatched = is_mismatched)
    # init
    # states      = np.append(states, env.last_pos, axis=0)
    # state_prime = np.append(state_prime, np.array([0,0]), axis=0)
    counter = 0
    for action in actions:
        sim.step(f_t=action[0], alpha_t=action[1])
        X[counter] = sim.last_state[0]
        Y[counter] = sim.last_state[1]
        counter += 1
    alpha   = actions[:,1]
    freq    = actions[:,0]
    time    = np.linspace(0, (len(X) - 1)/30.0, len(X)) # (np.arange(len(X))) / 30.0 #timestep is 1/30
    return X,Y,alpha,time,freq

freq = 4
a0_def = 1.5

#first we do nothing
time_steps = 100 #do nothing for 100/30 seconds
actions_idle = np.zeros((time_steps, 2))

gp_sim = GP.LearningModule()
noise_vars= [0.0]
#first we will do absolutely nothing to try and calculate the drift term
px_idle,py_idle,alpha_idle,time_idle,freq_idle = run_sim(actions_idle,
                                                            init_pos = np.array([0,0]),
                                                            noise_var = noise_vars[0],
                                                            a0=a0_def,is_mismatched=True)



gp_sim.estimateDisturbance(px_idle, py_idle, time_idle)

#note: timestep is 1/30 seconds, the rate we get data at in the experiment
time_steps = 300 #train for 10s at 30 hz
cycles = 3 #train my moving in 3 circles

steps = (int)(time_steps / cycles)

#generate actions to move in a circle at a constant frequency
actions_circle = np.zeros( (steps, 2))
actions_circle[:,0] = freq
actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)

#stack the circle actions to get our learning set
actions_learn = np.vstack([actions_circle]*cycles)

t = np.linspace(0, time_steps, time_steps)

actions_learn[:,0] = (np.cos(t / 5) + 1)/2 * 4.9 + 0.1

#generate actions for testing (1/30 hz for 30 seconds)
time_steps = 1000



freq_ls = np.linspace(0.1, 40, 40)
#actions = np.array([[1, 0.3*np.pi*((t/time_steps)-1)*(-1)**(t//300)] 
#                        for t in range(1,time_steps)]) # [T,action_dim]
actions_learn_ls = []


# for fi, freq in enumerate(freq_ls):
#     time_steps = 300 #train for 10s at 30 hz
#     cycles = 3 #train my moving in 3 circles

#     steps = (int)(time_steps / cycles)

#     #generate actions to move in a circle at a constant frequency
#     actions_circle = np.zeros( (steps, 2))
#     actions_circle[:,0] = freq
#     actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)

#     #stack the circle actions to get our learning set
#     actions_learn = np.vstack([actions_circle]*cycles)

#     t = np.linspace(0, time_steps, time_steps)

#     # actions_learn[:,0] = (np.cos(t / 5) + 1)/2 * 4.9 + 0.1

#     #generate actions for testing (1/30 hz for 30 seconds)
#     time_steps = 1000

  

#     # THIS IS WHAT THE SIMULATION ACTUALLY GIVES US -- model mismatch && noise
#     px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions_learn,
#                                                          init_pos = np.array([0,0]),
#                                                          noise_var = noise_vars[0],
#                                                          a0=a0_def,is_mismatched=True)


#     # learn noise and a0 -- note px_desired and py_desired need to be at the same time
#     a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim, freq_sim, time_sim)
#     print("Estimated a0 value is " + str(a0_sim))







actions_learn_ls = np.array([])
lf = len(freq_ls)
for fi, freq in enumerate(freq_ls):
    time_steps = 300 #train for 10s at 30 hz
    cycles = 3 #train my moving in 3 circles

    steps = (int)(time_steps / cycles)

    #generate actions to move in a circle at a constant frequency
    actions_circle = np.zeros( (steps, 2))
    actions_circle[:,0] = freq
    actions_circle[:,1] = np.linspace(-np.pi, np.pi, steps)

    #stack the circle actions to get our learning set
    actions_learn = np.vstack([actions_circle]*cycles)
    if len(actions_learn_ls ) ==0:
        actions_learn_ls = actions_learn
    else:
        actions_learn_ls = np.vstack([actions_learn_ls, actions_learn])



    # actions_learn[:,0] = (np.cos(t / 5) + 1)/2 * 4.9 + 0.1

    #generate actions for testing (1/30 hz for 30 seconds)
    time_steps = 1000

  

    # THIS IS WHAT THE SIMULATION ACTUALLY GIVES US -- model mismatch && noise
px_sim,py_sim,alpha_sim,time_sim,freq_sim = run_sim(actions_learn_ls,
                                                        init_pos = np.array([0,0]),
                                                        noise_var = noise_vars[0],
                                                        a0=a0_def,is_mismatched=True)


    # learn noise and a0 -- note px_desired and py_desired need to be at the same time
a0_sim = gp_sim.learn(px_sim, py_sim, alpha_sim, freq_sim, time_sim)
print("Estimated a0 value is " + str(a0_sim))

    

    
gp_sim.visualize()







# gp_sim.load_GP()

# a0_sim = np.load('a_sim.npy')

# freq = 4
# # a0_def = 1.5
# dt = 0.030 #assume a timestep of 30 ms
# noise_var = 0.0
# sim = Simulator()
# time_steps = 500
# x0 = [0.0,0.0]

# sim.reset_start_pos(x0)
# sim.noise_var = noise_var
# sim.a0 = a0_def
# sim.is_mismatched = True
# #########Ref Trajectory
# r0 = 2
# theta_ls = np.linspace(0, 2*np.pi,time_steps)
# x_ls = r0*(1-np.cos(theta_ls))
# y_ls = r0*np.sin(theta_ls)
# ref = np.ones((time_steps,2))
# ref[:,0]= x_ls
# ref[:,1]=y_ls
        
# ########MPC parameters
# B  = a0_sim*dt*np.array([[1,0],[0,1]])

# # Weight matrices for state and input
# Q = np.array([[1,0],[0,1]])
# R = 0.001*np.array([[1,0],[0,1]])

# N = 2


# # Simulates the dynamic system over T time steps using MPC control.
# # 
# #     Parameters:
#     # - All parameters are as previously described.
#     # - T: Total number of time steps to simulate.

#     # Returns:
#     # - x_traj: Trajectory of states.
#     # - u_traj: Trajectory of control inputs.
#     # """
# x_traj = np.zeros((time_steps+1, 2))  # +1 to include initial state

# u_traj = np.zeros((time_steps, 2))
# x_traj[0, :] = x0

# alpha_t = 0
# freq_t =0
# for t in range(time_steps):
#     # Update reference for the current time step
#     current_ref = ref[t:min(t+N, time_steps), :]
#     if current_ref.shape[0] < N:
#         # Pad the reference if it's shorter than the prediction horizon
#         current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))

#     ### Disturbance Compensator 
#     muX,sigX = gp_sim.gprX.predict(np.array([[alpha_t, freq_t]]), return_std=True)
#     muY,sigY = gp_sim.gprY.predict(np.array([[alpha_t, freq_t]]), return_std=True)
#     # D = np.array([gp_sim.Dx, gp_sim.Dy])
#     v_e = np.array([muX[0], muY[0]])
#     # u_c = -(D)/a0_def
#     A = np.zeros((2,2))
#     Qz = 0*R
    
#     u_mpc,pred_traj = mpc_control(B, x_traj[t, :], current_ref, N, Q, R,(v_e)*dt)
#     z0 = x_traj[t,:]-current_ref[0,:]
#     # u_mpc = mpc_with_integral_action(A, B, Q, R, Qz, x0, z0, ref, N,np.hstack((v_e*dt, [0,0])) )


#     u_current = u_mpc
#     u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
#     f_t = np.linalg.norm(u_current)
#     alpha_t = math.atan2(u_current[1], u_current[0])


#     action = np.array([ [f_t], [alpha_t], [t*dt]])


#     sim.step(f_t=f_t, alpha_t= alpha_t)
#     x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
#     # x_traj_lin[t+1, :] =x_traj_lin[t, :] + B @ u_opt


# # Plotting
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
# plt.savefig('cycle.png', dpi = 600)
# plt.show()

