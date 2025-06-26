"""
DR_robot_routing_simulate.py

This script implements a simulation for a robot routing/control experiment using the Robotarium.
It loads a Gaussian Process (GP) model to predict robot state transitions, computes costs that include 
goal tracking and obstacle avoidance, and uses a distributionally robust (DR) algorithm for control.

Key functionalities:
    - System models: actual and nominal models for state updates.
    - Cost computation for goals and obstacles, including KL divergence.
    - Control step that computes control inputs based on predicted next states.
    - Visualization of goal/obstacle markers and robot trajectories.
"""

# %% Import Libraries and Modules
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from time import time

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import time
import pickle
import requests
from io import BytesIO
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.gaussian_process import GaussianProcessRegressor 
from multiprocessing import Pool

from Maxdiff2 import SimpleMaxDiff

from scipy.interpolate import RegularGridInterpolator
import pandas as pd

# %%
# Small grid for quick demo
x1_vals = np.linspace(-1.5, 1.5, 50)
x2_vals = np.linspace(-1.0, 1.0, 30)
u1_vals = np.linspace(-.5, .5, 5)
u2_vals = np.linspace(-.5, 0.5, 5)

state_grid = np.array([(x1, x2) for x1 in x1_vals for x2 in x2_vals])
input_grid = np.array([(u1, u2) for u1 in u1_vals for u2 in u2_vals])

# %% Define Global Parameters and Action Spaces

control_space_size = 5  # Number of discrete control inputs per dimension

# Define action space for each control input using a linspace between -0.5 and 0.5
U_space_1 = np.linspace(-.50, .50, control_space_size)
U_space_2 = np.linspace(-.50, .50, control_space_size)

time_step = 0.033  # Simulation time step (seconds)

# %% Load Gaussian Process Model
# The GP model is loaded from a local .dump file.
GP_nominal = pickle.load(open(r'D:\Network Security\KL Control\robotarium_python_simulator\rps\examples\DR_FREE\Experiments\GP_nominal_1.dump', 'rb'))

# %% Define Environment Constants for Wind Effects (if used)
WIND_DIRECTION = np.array([1, 1])  # Wind direction vector (blowing in both positive x and y directions)
WIND_SPEED = 0.8                 # Wind speed
DRAG_COEFFICIENT = 1.0           # Drag coefficient
AIR_DENSITY = 1.25               # Air density
ROBOT_AREA = 1.0                 # Approximate frontal area of the robot in m^2

def compute_wind_force():
    """
    Compute the wind force based on environmental parameters.
    Returns:
        wind_force (np.array): Force vector due to wind.
    """
    wind_force_magnitude = 1.0 * AIR_DENSITY * DRAG_COEFFICIENT * ROBOT_AREA * WIND_SPEED**2
    # Normalize wind direction and compute force vector
    wind_force = wind_force_magnitude * WIND_DIRECTION / np.linalg.norm(WIND_DIRECTION)
    return wind_force

# %% System Dynamics Models

def model_step(x, velocities, time_step=0.1):
    """
    Actual system dynamics model P(.)
    
    Args:
        x (np.array): Current state (position)
        velocities (np.array): Control velocities
        time_step (float): Time step for integration
    Returns:
        np.array: Updated state (position)
    """
    poses = np.zeros((2, 1))
    poses[0] = x[0] + time_step * velocities[0]
    poses[1] = x[1] + time_step * velocities[1]
    return poses

def nominal_model_step(x, velocities, time_step=0.1):
    """
    Nominal system dynamics model P̄(.)
    Introduces a bias (e.g., modeling error) in the state update.
    
    Args:
        x (np.array): Current state
        velocities (np.array): Control velocities
        time_step (float): Time step for integration
    Returns:
        np.array: Nominal updated state
    """
    poses = np.zeros((2, 1))
    poses[0] = 1.0 * x[0] + time_step * velocities[0] + 0.1 * x[0]
    poses[1] = 1.0 * x[1] + time_step * velocities[1] + 0.1 * x[1]
    return poses

# %% Cost Functions

def logpdf(x, u, covar):
    """
    Compute the Gaussian probability density (kernel) at x with mean u and covariance covar.
    
    Args:
        x (np.array): Current state (vector)
        u (np.array): Obstacle point (vector)
        covar (np.array): Covariance matrix
    Returns:
        float: Gaussian PDF value
    """
    k = len(x)  # Dimensionality
    diff = x - u
    inv_covar = np.linalg.inv(covar)
    exponent = -0.5 * (diff.T @ inv_covar @ diff)
    denom = np.sqrt((2 * np.pi) ** k * np.linalg.det(covar))
    pdf = np.exp(exponent) / denom
    return pdf

def goal_cost(state, goal_points):
    """
    Compute cost based on the squared distance between the current state and the goal.
    
    Args:
        state (np.array): Current state
        goal_points (np.array): Goal point(s)
    Returns:
        float: Cost value
    """
    cost = 30 * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2)
    return cost

def state_cost(state, goal_points, obs_points):
    """
    Compute a composite cost based on distance to the goal and proximity to obstacles.
    
    Args:
        state (np.array): Current state
        goal_points (np.array): Goal point(s)
        obs_points (np.array): Obstacle points
    Returns:
        float: Combined state cost
    """
    def logpdf(x, u, covar):
        """
        Compute the Gaussian probability density (kernel) at x with mean u and covariance covar.
        
        Args:
            x (np.array): Current state (vector)
            u (np.array): Obstacle point (vector)
            covar (np.array): Covariance matrix
        Returns:
            float: Gaussian PDF value
        """
        k = len(x)  # Dimensionality
        diff = x - u
        inv_covar = np.linalg.inv(covar)
        exponent = -0.5 * (diff.T @ inv_covar @ diff)
        denom = np.sqrt((2 * np.pi) ** k * np.linalg.det(covar))
        pdf = np.exp(exponent) / denom
        return pdf
    
    v = np.array([0.025, 0.025], dtype=np.float32)
    covar = np.diag(v)
    
    gauss_sum = 0
    for i in range(obs_points.shape[1]):
        gauss_sum += 20 * logpdf(state[:2], obs_points[:2, i], covar)
    
    cost = (50 * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2) +
            gauss_sum +
            5 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi))))
    return cost

def calculate_kl_divergence(mu1, cov1, mu2, cov2):
    """
    Calculate the Kullback-Leibler divergence between two multivariate Gaussian distributions.
    
    Args:
        mu1 (np.array): Mean vector of the first distribution.
        cov1 (np.array): Covariance matrix of the first distribution.
        mu2 (np.array): Mean vector of the second distribution.
        cov2 (np.array): Covariance matrix of the second distribution.
    Returns:
        float: KL divergence.
    """
    d = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    trace_term = np.trace(cov2_inv @ cov1)
    mean_diff = mu2 - mu1
    mean_diff_term = mean_diff.T @ cov2_inv @ mean_diff
    log_det_term = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    kl_divergence = 0.5 * (trace_term + mean_diff_term - d + log_det_term)
    return kl_divergence

def is_inside_rectangle(state, bottom_left, top_right):
    """
    Check if the state lies within a specified rectangular region.
    
    Args:
        state (list or np.array): 2D state [x, y].
        bottom_left (list): [x_min, y_min] of the rectangle.
        top_right (list): [x_max, y_max] of the rectangle.
    Returns:
        bool: True if state is inside the rectangle, False otherwise.
    """
    return bottom_left[0] <= state[0] <= top_right[0] and bottom_left[1] <= state[1] <= top_right[1]

# %% Define Goal and Obstacle Points
# Define goal points (remove orientation if provided)
goal_points = np.array(np.mat('-1.4; -0.8; 0'))  # Modify as needed

# Define obstacle points
# First row: x coordinates, second row: y coordinates, third row: pose (set to zero)
obs_points = np.array(np.mat('0 0 0 0 0 -0.8;0 0.2 0.4 0.6 0.8 -0.8;0 0 0 0 0 0'))

# %% compute cost-to-go

# c_hat, interpolator_c_hat = compute_receding_horizon_cost_to_go(
#     x0=np.array([0.0, 0.0]),
#     H=2,
#     x1_vals=x1_vals,
#     x2_vals=x2_vals,
#     input_grid=input_grid,
#     GP_nominal=GP_nominal,
#     x_goal=goal_points,
#     c_fn=state_cost
# )
# np.save(r'C_Hat_N3.npy', c_hat)

c_hat = np.load(r'D:\Network Security\KL Control\robotarium_python_simulator\rps\examples\DR_FREE\Experiments\C_Hat_N3.npy',allow_pickle=True).item()
interpolator_c_hat = RegularGridInterpolator((x1_vals, x2_vals), c_hat[0], bounds_error=False, fill_value=0.0)


# %% Setup Simulation Parameters

N = 1  # Number of robots
M = 12  # Number of simulation runs (or experiments)

# Initial conditions for each simulation run (modify as needed)
# initial_conditions = [
#     np.array(np.mat('1.32;0.9; 0.')),
#     np.array(np.mat('0.2;0.9; 0')),
#     np.array(np.mat('1.3;-0.5; 0')),
#     np.array(np.mat('-1.0;0.8; 0'))
# ]

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Grid search parameters

# horizon_list = [2, 5, 10, 20, 30, 50]
horizon_list = [0.01, 0.1, 1.0, 10, 100]
samples_list = [10, 20, 30, 50, 100, 200]
param_grid = list(product(horizon_list, samples_list))

# Store results as {(h, s): count_success}
results = {}

# Initial conditions
initial_conditions = [
    np.array(np.mat('1.32;0.9; 0')), np.array(np.mat('0.5;-0.2; 0')), np.array(np.mat('1.2;-0.5; 0')), np.array(np.mat('-0.5;0.25; 0')),
    np.array(np.mat('1.0;0.8; 0')), np.array(np.mat('0.5;-0.2; 0')), np.array(np.mat('-0.5;-0.5; 0')), np.array(np.mat('-0.9;0.25; 0')),
    np.array(np.mat('-1.1;0.9; 0')), np.array(np.mat('0.9;-0.2; 0')), np.array(np.mat('-0.7;0.5; 0')), np.array(np.mat('-0.5;0.25; 0'))
]

def run_single_simulation(init_cond,agent):
    try:
        r = robotarium.Robotarium(number_of_robots=1, show_figure=True,
                                  initial_conditions=init_cond, sim_in_real_time=True)

        _, uni_to_si_states = create_si_to_uni_mapping()
        si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

        x = r.get_poses()
        x_si = uni_to_si_states(x)

        k = 0
        r.step()

        while np.size(at_pose(np.vstack((x_si, x[2, :])), goal_points, position_error=0.2, rotation_error=100)) != 1:
            x = r.get_poses()
            x_si = uni_to_si_states(x)

            # Crash/Fail condition
            if any([
                x_si[0] <= -1.5, x_si[0] >= 1.5,
                x_si[1] <= -1.0, x_si[1] >= 1.0,
                is_inside_rectangle(x_si, [-0.25, -0.15], [0.15, 0.85]),
                is_inside_rectangle(x_si, [-1.1, -1.0], [-0.51, -0.65])
            ]):
                r.call_at_scripts_end()
                del r
                return False

            dxi = np.clip(agent.rollout(x_si.reshape(2,)).reshape(2, 1),-0.5,0.5)
            dxu = si_to_uni_dyn(dxi, x)
            r.set_velocities(np.arange(1), dxu)
            r.step()

            k += 1
            if k >= 3000:
                r.call_at_scripts_end()
                del r
                return False

        r.call_at_scripts_end()
        del r
        return True

    except Exception as e:
        print(f"Simulation failed: {e}")
        return False


# Run grid search

h, s = 0.10, 50
success = 0
print(f"Running Horizon={h}, Samples={s}")
agent = SimpleMaxDiff(model_fn=model_step, state_dim=2, action_dim=2,
                            horizon=20, gamma=0.98, samples=s,
                            alpha=h, lam=0.1)

for cond in initial_conditions:
    if run_single_simulation(cond, agent=agent):
        print(f"Simulation succeeded for initial condition {cond}")
        success += 1
    else:
        print(f"Simulation failed for initial condition {cond}")    
results[(h, s)] = success
print(f"Result for A={h}, S={s}: {100*success/12}/12 successes\n")

# # Heatmap Visualization
# horizons = sorted(set(h for h, s in results))
# samples = sorted(set(s for h, s in results))
# heatmap_matrix = np.zeros((len(horizons), len(samples)))

np.save('maxdiff_grid_search_results001_10.npy', results)
# np.save('maxdiff_grid_search_results50100.npz', results)
 
# for i, h in enumerate(horizons):
#     for j, s in enumerate(samples):
#         heatmap_matrix[i, j] = results.get((h, s), 0)

# plt.figure(figsize=(8, 6))
# sns.heatmap(heatmap_matrix, annot=True, fmt=".0f", xticklabels=samples, yticklabels=horizons, cmap="YlGnBu")
# plt.xlabel("Samples")
# plt.ylabel("Horizon (H)")
# plt.title("Grid Search: Success Rate of DR-MaxDiff (out of 12)")
# plt.tight_layout()
# plt.show()

