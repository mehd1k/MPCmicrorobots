import numpy as np
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
def mpc_control(B,x0, ref, N, Q, R,Dist):
    """
    MPC controller using Gurobi optimization.

    Parameters:
    - A, B, C: System matrices.
    - x0: Initial state.
    - ref: Reference trajectory (Nx1 vector).
    - N: Prediction horizon.
    - Q, R: Weight matrices for the state and input.
    - umin, umax: Minimum and maximum control inputs.
    - xmin, xmax: Minimum and maximum state values.

    Returns:
    - u_opt: Optimal control input for the current time step.
    """
    nx = 2  # Number of states
    nu = 2 # Number of inputs

    # Create a new model
    m = gp.Model("mpc")

    # Decision variables for states and inputs
    x = m.addMVar((N+1, nx), lb=-GRB.INFINITY, name="x")
    u = m.addMVar((N, nu), lb=-35, ub=35, name="u")

    # Initial state constraint
    m.addConstr(x[0, :] == x0, name="init")

    # Dynamics constraints
    for t in range(N):
        m.addConstr(x[t+1, :] ==  x[t, :] + B @ u[t, :]+Dist, name=f"dyn_{t}")
        # m.addConstr(u[t, 0]**2+u[t,1]**2 ==  1, name=f"sincos_{t}")

    # State constraints
    # for t in range(N+1):
    #     m.addConstr(x[t, :] >= xmin, name=f"xmin_{t}")
    #     m.addConstr(x[t, :] <= xmax, name=f"xmax_{t}")

    # Objective: Minimize cost function
    cost = 0
    gamma = 1
    for t in range(N):
        cost += gamma**t*(x[t, :] - ref[t, :]) @ Q @ (x[t, :] - ref[t, :]) + u[t, :] @ R @ u[t, :]
        
    # cost+=1000* (x[t, :] - ref[t, :]) @ Q @ (x[t, :] - ref[t, :])
    # for t in range(N-1):
    #     cost += (u[t, :]-u[t+1,:]) @ R @ (u[t, :]-u[t+1,:])
    m.setObjective(cost, GRB.MINIMIZE)

    # Optimize model
    # m.params.NonConvex = 2
    m.optimize()

    u_opt = u.X[0, :]  # Get optimal control input for the current time step
    predict_traj = x.X
    return u_opt, predict_traj





import numpy as np
import gurobipy as gp
from gurobipy import GRB

def mpc_with_integral_action(A, B, Q, R, Qz, x0, z0, ref, N,Dist):
    # Augment system matrices for integral action
    nx = A.shape[1]
    A_aug = np.block([
        [A, np.zeros((nx, nx))],
        [-np.eye(nx), np.eye(nx)]
    ])
    B_aug = np.vstack([B, np.zeros_like(B)])
    nx_aug = A_aug.shape[1]  # Number of augmented states

    x_aug = A_aug.shape[1]  # Number of augmented states
    nu = B_aug.shape[1]  # Number of inputs
   
    
    m = gp.Model("mpc_integral")
    x_aug = m.addMVar((N+1, nx_aug), name="x_aug")
    u = m.addMVar((N, nu), lb=-50, ub=50, name="u")
  

    # Create a new model
    m = gp.Model("mpc")

    # Decision variables for augmented states and inputs
    x_aug = m.addMVar((N+1, nx_aug), lb=-GRB.INFINITY, name="x_aug")
    u = m.addMVar((N, nu), lb=-50, ub=50, name="u")
    x0_aug = np.hstack((x0,z0))
    # Initial state constraint
    m.addConstr(x_aug[0, :] == x0_aug, name="init_aug")

    # Dynamics constraints (augmented)
    for t in range(N):
        m.addConstr(x_aug[t+1, :] == A_aug @ x_aug[t, :] + B_aug @ u[t, :]+Dist, name=f"dyn_aug_{t}")

    # State constraints (only on original states, not on integral states)
    # for t in range(N+1):
    #     m.addConstr(x_aug[t, :self.A.shape[1]] >= self.xmin, name=f"xmin_{t}")
    #     m.addConstr(x_aug[t, :A.shape[1]] <= self.xmax, name=f"xmax_{t}")

    # Objective: Minimize cost function including integral of error
    cost = 0
    for t in range(N):
        # Only original states are considered for the reference tracking error
        cost += (x_aug[t, :A.shape[1]] - ref[t, :]) @ Q @ (x_aug[t, :A.shape[1]] - ref[t, :])
        # Integral of error state cost
        cost += x_aug[t, A.shape[1]:] @ Qz @ x_aug[t, A.shape[1]:]
        # Control input cost
        cost += u[t, :] @ R @ u[t, :]
    m.setObjective(cost, GRB.MINIMIZE)

    # Optimize model
    m.optimize()

    u_opt = np.array(u.X)[0, :]  # Get optimal control input for the current time step

    return u_opt


 

# # Example usage parameters
# A = np.array([[1, 0], [0, 1]])
# B = np.array([[1, 0], [0, 1]])
# C = np.array([[1, 0], [0, 1]])  # Not directly used in this function
# Q = np.eye(2)  # State weighting
# R = 0.01 * np.eye(2)  # Input weighting
# Qz = np.eye(2)  # Integral error weighting
# x0 = np.array([0, 0])  # Initial state
# z0 = np.array([0, 0])  # Initial integral error
# ref = np.ones((5, 2))  # Reference trajectory
# N = 5  # Prediction horizon
# umin = np.array([-10, -10])  # Min control input
# umax = np.array([10, 10])  # Max control input
# xmin = np.array([-100, -100])  # Min state values
# xmax = np.array([100, 100])  # Max state values
# T = 20  # Simulation time steps

# The actual call to the function would be like this:
# x_traj, u_traj, z_traj = mpc_with_integral_action(A, B, C, Q, R, Qz, x0, z0, ref, N, umin, umax,












# def simulate_system( B, x0, ref, N, Q, R, T):
#     """
#     Simulates the dynamic system over T time steps using MPC control.

#     Parameters:
#     - All parameters are as previously described.
#     - T: Total number of time steps to simulate.

#     Returns:
#     - x_traj: Trajectory of states.
#     - u_traj: Trajectory of control inputs.
#     """
#     x_traj = np.zeros((T+1, 2))  # +1 to include initial state
#     u_traj = np.zeros((T, 2))
#     x_traj[0, :] = x0

#     for t in range(T):
#         # Update reference for the current time step
#         current_ref = ref[t:min(t+N, T), :]
#         if current_ref.shape[0] < N:
#             # Pad the reference if it's shorter than the prediction horizon
#             current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))
        
#         u_opt = mpc_control(B, x_traj[t, :], current_ref, N, Q, R)
#         u_traj[t, :] = u_opt  # Assuming u_opt is the control input for the next step
#         x_traj[t+1, :] =x_traj[t, :] + B @ u_opt  # Update state based on dynamics

#     return x_traj, u_traj

# # Simulation parameters
# T = 50  # Total number of time steps to simulate
# dt = 0.05
# B  = 6*dt*np.array([[1,0],[0,1]])
# ref = np.ones((5, 2))
# # Prediction horizon
# N = 2
# x0 = 0

# # Weight matrices for state and input
# Q = np.array([[1,0],[0,1]])
# R = 0.1*np.array([[1,0],[0,1]])

# # Simulate the system
# x_traj, u_traj = simulate_system(B, x0, ref, N, Q, R, T)

# # Plotting
# time_steps = np.arange(T+1)
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.plot(time_steps, x_traj[:, 0], label='Actual Trajectory (x1)')
# plt.plot(time_steps, np.vstack((ref[:1, 0], np.ones((T, 1)))), 'r--', label='Desired Trajectory (x1)')
# plt.xlabel('Time step')
# plt.ylabel('State x1')
# plt.title('Trajectory of State x1')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.plot(time_steps, x_traj[:, 1], label='Actual Trajectory (x2)')
# plt.plot(time_steps, np.vstack((ref[:1, 1], np.ones((T, 1)))), 'r--', label='Desired Trajectory (x2)')
# plt.xlabel('Time step')
# plt.ylabel('State x2')
# plt.title('Trajectory of State x2')
# plt.legend()
# plt.grid(True)


# plt.subplot(1, 3, 3)
# plt.plot(time_steps[0:T], u_traj[:, 1]**2+u_traj[:, 0]**2, label='control effort')

# plt.xlabel('Time step')
# plt.ylabel('control effort')
# plt.title('control effort')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()