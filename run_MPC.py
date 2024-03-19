import numpy as np
import Learning_module as GP # type: ignore
from utils import find_alpha_corrected
from utils import plot_xy,plot_traj,plot_bounded_curves,plot_curve
from scipy.ndimage import uniform_filter1d
from MR_simulator import Simulator
import math 
from MPC import mpc_control, mpc_with_integral_action
import matplotlib.pyplot as plt




def run_sim(actions,init_pos=None,noise_var = 1,a0 =1.5, is_mismatched = False):
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
#frequency of the magnetic field in Hz and the nominal value of a0
freq = 4
a0_def = 1.5
dt = 0.030 #assume a timestep of 30 ms


#first we do nothing
time_steps = 100 #do nothing for 100/30 seconds
actions_idle = np.zeros((time_steps, 3))
actions_idle[:,2] = np.arange(0, dt*time_steps, dt)


#note: timestep is 1/30 seconds, the rate we get data at in the experiment


gp_sim = GP.LearningModule()

#######first we will do absolutely nothing to try and calculate the drift term


# #######THIS CALCULATES THE DESIRED TRAJECTORY FROM OUR a0 ESTIMATE
# px_desired,py_desired,alpha_desired,time_desired,freq_desired = run_sim(actions_learn,
#                                                                         init_pos = np.array([0,0]),
#                                                                         noise_var = 0.0,a0=a0_sim)

gp_sim.load_GP()
a0_sim = np.load('a_sim.npy')

freq = 4
# a0_def = 1.5
dt = 0.030 #assume a timestep of 30 ms
noise_var = 0.0
sim = Simulator()
time_steps = 500

#########Ref Trajectory
# r0 = 2
# theta_ls = np.linspace(0, 2*np.pi,time_steps)
# x_ls = r0*(1-np.cos(theta_ls))
# y_ls = r0*np.sin(theta_ls)
# ref = np.ones((time_steps,2))

def generate_in_between_points(node_ls, num_points_per_segment):
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

    for segment in node_ls:
        start_point, end_point = segment[0], segment[1]
        # Generate a sequence of numbers between 0 and 1, which will serve as interpolation factors.
        interpolation_factors = np.linspace(0, 1, num_points_per_segment + 2)
        
        # Interpolate x and y separately
        x_points = (1 - interpolation_factors) * start_point[0] + interpolation_factors * end_point[0]
        y_points = (1 - interpolation_factors) * start_point[1] + interpolation_factors * end_point[1]
        
        # Combine the x and y coordinates
        segment_points = np.vstack((x_points, y_points)).T
        full_trajectory.extend(segment_points[:-1].tolist())

    # Ensure the last point of the last segment is included
    full_trajectory.append(node_ls[-1, 1].tolist())

    return np.array(full_trajectory)
node_ls = np.load('node.npy')
seg_ls = []
for i in range(len(node_ls)-1):
    seg_ls.append([node_ls[i],node_ls[i+1]])
ref = np.array([])

ref = generate_in_between_points(np.array(seg_ls), 30)
time_steps = len(ref)
# ref[:,0]= x_ls
# ref[:,1]=y_ls
x0 = ref[0]
########MPC parameters
B  = a0_sim*dt*np.array([[1,0],[0,1]])

# Weight matrices for state and input
Q = np.array([[1,0],[0,1]])
R = 0.001*np.array([[1,0],[0,1]])

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

u_traj = np.zeros((time_steps, 2))
x_traj[0, :] = x0


sim.reset_start_pos(x0)
sim.noise_var = noise_var
sim.a0 = a0_def
sim.is_mismatched = True

alpha_t = 0
for t in range(time_steps):
    # Update reference for the current time step
    current_ref = ref[t:min(t+N, time_steps), :]
    if current_ref.shape[0] < N:
        # Pad the reference if it's shorter than the prediction horizon
        current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))

    ### Disturbance Compensator 
    muX,sigX = gp_sim.gprX.predict(np.reshape(alpha_t, (-1, 1)), return_std=True)
    muY,sigY = gp_sim.gprY.predict(np.reshape(alpha_t, (-1, 1)), return_std=True)
    # D = np.array([gp_sim.Dx, gp_sim.Dy])
    v_e = np.array([muX[0], muY[0]])
    # u_c = -(D)/a0_def
    A = np.zeros((2,2))
    Qz = 0*R
    
    u_mpc,pred_traj = mpc_control(B, x_traj[t, :], current_ref, N, Q, R,(v_e)*dt)
    z0 = x_traj[t,:]-current_ref[0,:]
    # u_mpc = mpc_with_integral_action(A, B, Q, R, Qz, x0, z0, ref, N,np.hstack((v_e*dt, [0,0])) )


    u_current = u_mpc
    u_traj[t, :] = u_current # Assuming u_opt is the control input for the next step
    f_t = np.linalg.norm(u_current)
    alpha_t = math.atan2(u_current[1], u_current[0])


    action = np.array([ [f_t], [alpha_t], [t*dt]])


    sim.step(f_t=f_t, alpha_t= alpha_t)
    x_traj[t+1, :] =  sim.last_state# Update state based on non_linear dynamics
    # x_traj_lin[t+1, :] =x_traj_lin[t, :] + B @ u_opt


# Plotting
time_span = np.arange(time_steps+1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(time_span, x_traj[:, 0], label='Actual Trajectory (x1)')
plt.plot(time_span[1:], ref[:, 0], 'r--', label='Desired Trajectory (x1)')
plt.xlabel('Time step')
plt.ylabel('State x1')
plt.title('Trajectory of State x1')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(time_span, x_traj[:, 1], label='Actual Trajectory (x2)')
plt.plot(time_span[1:], ref[:, 1], 'r--', label='Desired Trajectory (x2)')
plt.xlabel('Time step')
plt.ylabel('State x2')
plt.title('Trajectory of State x2')
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 3)
plt.plot(x_traj[:,0], x_traj[:, 1], label='Actual Trajectory (x1)')
plt.plot(ref[:,0], ref[:, 1], 'r--', label='Desired Trajectory (x1)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
