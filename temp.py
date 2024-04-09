import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB


# def nonlinear_dynamic_model(alpha_0,t , a0=1):
#     f =1
#     D = np.array([0,1])
#     dx = a0*f*np.cos(alpha_0)+D[0]
#     dy = a0*f*np.sin(alpha_0)+D[1]
#     return dx, dy

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Model parameters
a = 1.0  # Example value
f = 1.0  # Example value
alpha = np.pi / 4  # 45 degrees, as an example

# The dynamic model
def model(x, t):
    u_opt = mpc_control(B, x, current_ref, N, Q, R)
    
    dxdt = a * f * np.array([np.cos(alpha), np.sin(alpha)])
    return dxdt

# Initial condition
x0 = np.array([0, 0])  # Starting at the origin

# Time points where the solution is computed
t = np.linspace(0, 10, 100)  # From t=0 to t=10

# Solve ODE
x = odeint(model, x0, t)

# # Plotting
# plt.plot(t, x[:, 0], label='x1(t)')
# plt.plot(t, x[:, 1], label='x2(t)')
# plt.xlabel('Time')
# plt.ylabel('x(t)')
# plt.legend()
# plt.show()


def mpc_control(B,x0, ref, N, Q, R):
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
    u = m.addMVar((N, nu), lb=-1, ub=1, name="u")

    # Initial state constraint
    m.addConstr(x[0, :] == x0, name="init")

    # Dynamics constraints
    for t in range(N):
        m.addConstr(x[t+1, :] ==  x[t, :] + B @ u[t, :], name=f"dyn_{t}")
        m.addConstr(u[t, 0]**2+u[t,1]**2 ==  1, name=f"sincos_{t}")

    # State constraints
    # for t in range(N+1):
    #     m.addConstr(x[t, :] >= xmin, name=f"xmin_{t}")
    #     m.addConstr(x[t, :] <= xmax, name=f"xmax_{t}")

    # Objective: Minimize cost function
    cost = 0
    for t in range(N):
        cost += (x[t, :] - ref[t, :]) @ Q @ (x[t, :] - ref[t, :]) + u[t, :] @ R @ u[t, :]
    # cost+=1000* (x[t, :] - ref[t, :]) @ Q @ (x[t, :] - ref[t, :])
    # for t in range(N-1):
    #     cost += (u[t, :]-u[t+1,:]) @ R @ (u[t, :]-u[t+1,:])
    m.setObjective(cost, GRB.MINIMIZE)

    # Optimize model
    m.params.NonConvex = 2
    m.optimize()

    u_opt = u.X[0, :]  # Get optimal control input for the current time step

    return u_opt











def simulate_system( B, x0, ref, N, Q, R, T):
    """
    Simulates the dynamic system over T time steps using MPC control.

    Parameters:
    - All parameters are as previously described.
    - T: Total number of time steps to simulate.

    Returns:
    - x_traj: Trajectory of states.
    - u_traj: Trajectory of control inputs.
    """
    x_traj = np.zeros((T+1, 2))  # +1 to include initial state
    u_traj = np.zeros((T, 2))
    x_traj[0, :] = x0

    for t in range(T):
        # Update reference for the current time step
        current_ref = ref[t:min(t+N, T), :]
        if current_ref.shape[0] < N:
            # Pad the reference if it's shorter than the prediction horizon
            current_ref = np.vstack((current_ref, np.ones((N-current_ref.shape[0], 1)) * ref[-1, :]))
        
        u_opt = mpc_control(B, x_traj[t, :], current_ref, N, Q, R)
        u_traj[t, :] = u_opt  # Assuming u_opt is the control input for the next step
        x_traj[t+1, :] =x_traj[t, :] + B @ u_opt  # Update state based on dynamics

    return x_traj, u_traj
