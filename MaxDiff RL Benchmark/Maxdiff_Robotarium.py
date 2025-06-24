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

from Maxdiff import SimpleMaxDiff

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

def state_cost_with_weights(state, goal_points, obs_points, weights):
    """
    Compute state cost using weighted contributions from the goal and obstacles.
    
    Args:
        state (np.array): Current state
        goal_points (np.array): Goal point(s)
        obs_points (np.array): Obstacle points
        weights (np.array): Weight matrix where first element is for goal cost and subsequent elements for obstacles
    Returns:
        float: Weighted state cost
    """
    v = np.array([0.025, 0.025], dtype=np.float32)
    covar = np.diag(v)
    
    gauss_sum = 0
    for i in range(obs_points.shape[1]):
        gauss_sum += -weights[0, i+1] * logpdf(state[:2], obs_points[:2, i], covar)
    
    cost = (-weights[0, 0] * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2) +
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




# %% Setup Simulation Parameters

N = 1  # Number of robots
M = 4  # Number of simulation runs (or experiments)

# Initial conditions for each simulation run (modify as needed)
# initial_conditions = [
#     np.array(np.mat('1.32;0.9; 0.')),
#     np.array(np.mat('0.2;0.9; 0')),
#     np.array(np.mat('1.3;-0.5; 0')),
#     np.array(np.mat('-1.0;0.8; 0'))
# ]



# initial_conditions = [np.array(np.mat('1.0;0.8; 0')),np.array(np.mat('0.2;0.9; 0')),np.array(np.mat('1.3;-0.5; 0')),np.array(np.mat('-1.0;0.8; 0'))]
initial_conditions = [ np.array(np.mat('1.32;0.9; 0')),np.array(np.mat('0.5;-0.2; 0')),np.array(np.mat('1.2;-0.5; 0')),np.array(np.mat('-0.5;0.25; 0')),
                      np.array(np.mat('1.0;0.8; 0')),np.array(np.mat('0.5;-0.2; 0')),np.array(np.mat('-0.5;-0.5; 0')),np.array(np.mat('-0.9;0.25; 0')),
                       np.array(np.mat('-1.1;0.9; 0')),np.array(np.mat('0.9;-0.2; 0')),np.array(np.mat('-0.7;0.5; 0')),np.array(np.mat('-0.5;0.25; 0'))] # can change robot initial condition in this line
# initial_conditions2 = [np.array(np.mat('1.0;0.8; 0')),np.array(np.mat('0.5;-0.2; 0')),np.array(np.mat('-0.5;-0.5; 0')),np.array(np.mat('-0.9;0.25; 0'))] # can change robot initial condition in this line
# initial_conditions3 = [np.array(np.mat('-1.1;0.9; 0')),np.array(np.mat('0.9;-0.2; 0')),np.array(np.mat('-0.7;0.5; 0')),np.array(np.mat('-0.5;0.25; 0'))]

# initial_conditions = np.column_stack((initial_conditions1, initial_conditions2, initial_conditions3))

agent = SimpleMaxDiff(model_fn=model_step, state_dim=2, action_dim=2, horizon=15,gamma=0.9,samples=50,alpha=0.01,lam=1.0)

# Lists to store simulation trajectories and control inputs
XX = [0] * M
UU = [0] * M

XN = [0] * M
COVN = [0] * M

# %% Run Simulation

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Grid search parameters
# horizon_list = [3, 5, 7]
# samples_list = [5, 10, 15]
# param_grid = list(product(horizon_list, samples_list))

# Store results as {(h, s): count_success}
results = {}



for I in range(M):
    D_xi = []       # Store control inputs
    X_si = []       # Store actual state trajectories
    X_si_nom = []   # Store nominal state predictions from GP model
    Cov_si_nom = [] # Store covariances of nominal state predictions

    # Initialize the Robotarium object for a single simulation run
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, 
                              initial_conditions=initial_conditions[I+8], sim_in_real_time=False)

    # Create controllers and mappings
    single_integrator_position_controller = create_si_position_controller()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

    # Get initial state
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    X_si_nom.append(x_si)

    # Plotting parameters and markers for goals and obstacles
    CM = np.random.rand(N+10, 3)  # Random colors for markers
    goal_marker_size_m = 0.1
    obs_marker_size_m = 0.1
    marker_size_goal = determine_marker_size(r, goal_marker_size_m)
    marker_size_obs = determine_marker_size(r, obs_marker_size_m)
    font_size = determine_font_size(r, 0.1)
    line_width = 5

    # Create goal markers with captions
    goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
    goal_points_text = [r.axes.text(goal_points[0, ii], goal_points[1, ii], goal_caption[ii],
                                    fontsize=font_size, color='k', fontweight='bold',
                                    horizontalalignment='center', verticalalignment='center', zorder=-2)
                        for ii in range(goal_points.shape[1])]
    goal_markers = [r.axes.scatter(goal_points[0, ii], goal_points[1, ii], s=marker_size_goal, marker='s',
                                   facecolors='none', edgecolors=CM[ii, :], linewidth=line_width, zorder=-2)
                    for ii in range(goal_points.shape[1])]

    # Create obstacle markers with captions
    obs_caption = ['OBS{0}'.format(ii) for ii in range(obs_points.shape[1])]
    obs_points_text = [r.axes.text(obs_points[0, ii], obs_points[1, ii], obs_caption[ii],
                                   fontsize=font_size, color='k', fontweight='bold',
                                   horizontalalignment='center', verticalalignment='center', zorder=-2)
                       for ii in range(obs_points.shape[1])]
    obs_markers = [r.axes.scatter(obs_points[0, ii], obs_points[1, ii], s=marker_size_obs, marker='s',
                                  facecolors='none', edgecolors=CM[ii+1, :], linewidth=line_width, zorder=-2)
                   for ii in range(obs_points.shape[1])]

    # Uncomment below to load and display a background image if available
    # gt_img = plt.imread(r'robotarium_environemnt.jpg')
    # r.axes.imshow(gt_img, extent=(-1.7, 1.67, -1.091, 1.07))

    # position_history = np.empty((2, 0))
    k = 0
    r.step()

    # Main simulation loop: run until robot reaches goal or encounters a termination condition
    while np.size(at_pose(np.vstack((x_si, x[2, :])), goal_points, position_error=0.25, rotation_error=100)) != N:
        # Get updated robot poses
        x = r.get_poses()
        x_si = uni_to_si_states(x)
        X_si.append(x_si)

        # Check for collision with obstacles
        for i in range(obs_points.shape[1]):
            column = obs_points[:, i]
            if np.array_equal(column, x_si):
                print(f"Crashed into obstacle at {i}: {column}")
                break

        # Check for boundary collisions
        if x_si[0] <= -1.5 or x_si[0] >= 1.5 or x_si[1] <= -1.0 or x_si[1] >= 1.0:
            print('Touched the boundary wall')
            break

        # Check for entry into specific rectangular regions
        if is_inside_rectangle(x_si, [-0.25, -0.1], [0.15, 0.85]):
            print(f"State entered the rectangle at: {x_si}")
            break
        if is_inside_rectangle(x_si, [-1.1, -1.0], [-0.51, -0.65]):
            print(f"State entered the rectangle at: {x_si}")
            break

        # Update and plot position history
        # position_history = np.append(position_history, x[:2], axis=1)
        # r.axes.scatter(position_history[0, :], position_history[1, :], s=1, linewidth=4, color='b', linestyle='dashed')

        # Update marker sizes (in case the figure scale changes)
        for j in range(goal_points.shape[1]):
            goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])
        for j in range(obs_points.shape[1]):
            obs_markers[j].set_sizes([determine_marker_size(r, obs_marker_size_m)])

        # Compute control inputs using the Control_step function
        # dxi, u_pf = Control_step(x_si, U_space_1, U_space_2, goal_points, obs_points, interpolator_c_hat)
        # D_xi.append(dxi)
        
        dxi = agent.rollout(x_si.reshape(2,)).reshape(2,1)

        # Predict next state using the GP model (for logging purposes)
        test_input = np.hstack((x_si.reshape(-1,), dxi.reshape(-1,))).reshape(1, -1)
        x_nom, sigma_nom = GP_nominal.predict(test_input, return_std=True)
        X_si_nom.append(x_nom)
        Cov_si_nom.append(sigma_nom)

        # Map single-integrator velocities to unicycle dynamics
        dxu = si_to_uni_dyn(dxi, x)

        k += 1
        if k == 3000:
            break

        # Set robot velocities and iterate the simulation step
        r.set_velocities(np.arange(N), dxu)
        r.step()

    # Save trajectories for this simulation run
    UU[I] = D_xi
    XX[I] = X_si
    XN[I] = X_si_nom
    COVN[I] = Cov_si_nom

    # End of simulation run: ensure proper shutdown of the Robotarium instance
    r.call_at_scripts_end()

# %% Save Simulation Data
XX = np.array(XX, dtype=object)
UU = np.array(UU, dtype=object)
XN = np.array(XN, dtype=object)
COVN = np.array(COVN, dtype=object)

np.save(r'State_Data_Simulation_Maxdiff_2050_nocap_3.npy', XX)
np.save(r'Input_Data_Simulation_Maxdiff_2050_nocap_3.npy', UU)
# # np.save(r'State_Data_nom.npy', XN)
# np.save(r'COV_Data_nom.npy', COVN)
