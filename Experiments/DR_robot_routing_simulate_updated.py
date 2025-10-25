import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from time import time
from io import BytesIO
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.gaussian_process import GaussianProcessRegressor
from multiprocessing import Pool
import pickle
import sys

# Constants
control_space_size = 3  # define action space size
U_space_1 = np.array(np.linspace(-1, 1, control_space_size))  # action space for 1st input
U_space_2 = np.array(np.linspace(-1, 1, control_space_size))  # action space for 2nd input
time_step = 0.033

# Load GP model with error handling
try:
    GP_nominal = pickle.load(open(r'D:\Network Security\KL Control\robotarium_python_simulator\rps\examples\DR_FREE\Experiments\GP_nominal_1.dump', 'rb'))
    print("GP model loaded successfully")
except Exception as e:
    print(f"Error loading GP model: {e}")
    sys.exit(1)

def model_step(x, velocities, time_step):
    """
    Actual System Model P(.)
    Args:
        x (array): State
        velocities (array): Control velocities
        time_step (float): Time step
    Returns:
        array: Next state
    """
    x = np.asarray(x, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    
    poses = np.zeros((2, 1), dtype=np.float64)
    poses[0] = x[0] + time_step * velocities[0]
    poses[1] = x[1] + time_step * velocities[1]
    
    return poses

def reference_model_step(x, velocities, time_step):
    """
    Reference System Model Q(.)
    Args:
        x (array): State
        velocities (array): Control velocities
        time_step (float): Time step
    Returns:
        array: Next state
    """
    x = np.asarray(x, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    
    poses = np.zeros((2, 1), dtype=np.float64)
    poses[0] = 0.80 * x[0] + time_step * velocities[0]
    poses[1] = 0.80 * x[1] + time_step * velocities[1]
    
    return poses

def nominal_model_step(x, velocities, time_step):
    """
    Nominal System Model P^{bar}(.)
    Args:
        x (array): State
        velocities (array): Control velocities
        time_step (float): Time step
    Returns:
        array: Nominal next state
    """
    x = np.asarray(x, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    
    poses = np.zeros((2, 1), dtype=np.float64)
    poses[0] = 1.0 * x[0] + time_step * velocities[0] + 0.1 * x[0]
    poses[1] = 1.0 * x[1] + time_step * velocities[1] + 0.1 * x[1]
    
    return poses

def worst_model_step(x, velocities, time_step):
    """
    Worst System Model
    Args:
        x (array): State
        velocities (array): Control velocities
        time_step (float): Time step
    Returns:
        array: Worst case next state
    """
    x = np.asarray(x, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    
    poses = np.zeros((2, 1), dtype=np.float64)
    poses[0] = 1.5 * x[0] + time_step * velocities[0]
    poses[1] = 1.5 * x[1] + time_step * velocities[1]
    
    return poses

def logpdf(x, u, covar):
    """
    Gaussian kernel
    Args:
        x (array): Current state
        u (array): Obstacle points
        covar (array): Covariance
    Returns:
        float: Probability of hitting the obstacle
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        covar = np.asarray(covar, dtype=np.float64)
        
        k = len(x)  # dimension
        a = np.transpose(x - u)
        b = np.linalg.inv(covar)
        c = x - u
        d = np.matmul(a, b)
        e = np.matmul(d, c)
        numer = np.exp(-0.5 * e)
        f = (2 * np.pi) ** k
        g = np.linalg.det(covar)
        denom = np.sqrt(f * g)
        pdf = numer / denom
        return pdf
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return 0.0

def goal_cost(state, goal_points):
    """
    Calculate cost based on the distance between goal point and current robot state
    Args:
        state (array): Current state
        goal_points (array): Goal points
    Returns:
        float: Cost
    """
    state = np.asarray(state, dtype=np.float64)
    goal_points = np.asarray(goal_points, dtype=np.float64)
    
    cost = 30 * ((state[0] - goal_points[0]) ** 2 + (state[1] - goal_points[1]) ** 2)
    return cost

def state_cost(state, goal_points, obs_points):
    """
    Calculate state cost considering goal and obstacles
    Args:
        state (array): Current state
        goal_points (array): Goal points
        obs_points (array): Obstacle points
    Returns:
        float: State cost
    """
    state = np.asarray(state, dtype=np.float64)
    goal_points = np.asarray(goal_points, dtype=np.float64)
    obs_points = np.asarray(obs_points, dtype=np.float64)
    
    v = np.array([0.025, 0.025], dtype=np.float64)
    covar = np.diag(v)
    
    gauss_sum = 0
    
    for i in range(np.size(obs_points, axis=1)):
        gauss_sum += 20 * logpdf(state[:2], obs_points[:2, i], covar)
    
    cost = (50 * ((state[0] - goal_points[0]) ** 2 + (state[1] - goal_points[1]) ** 2) + 
            gauss_sum + 
            5 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi))))
    
    return cost

def state_cost_with_weights(state, goal_points, obs_points, weights):
    """
    Calculate state cost considering goal and obstacles using weights
    Args:
        state (array): Current state
        goal_points (array): Goal points
        obs_points (array): Obstacle points
        weights (array): Weights for cost computation
    Returns:
        float: State cost
    """
    state = np.asarray(state, dtype=np.float64)
    goal_points = np.asarray(goal_points, dtype=np.float64)
    obs_points = np.asarray(obs_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    v = np.array([0.025, 0.025], dtype=np.float64)
    covar = np.diag(v)
    
    gauss_sum = 0
    
    for i in range(np.size(obs_points, axis=1)):
        gauss_sum += -weights[0, i + 1] * logpdf(state[:2], obs_points[:2, i], covar)
    
    cost = (-weights[0, 0] * ((state[0] - goal_points[0]) ** 2 + (state[1] - goal_points[1]) ** 2) + 
            gauss_sum + 
            1 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03) ** 2) / (0.03 * np.sqrt(2 * np.pi))))
    
    return cost

def calculate_kl_divergence(mu1, cov1, mu2, cov2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.
    
    Args:
        mu1: numpy array of shape (d,), the mean vector of the first Gaussian distribution
        cov1: numpy array of shape (d, d), the covariance matrix of the first Gaussian distribution
        mu2: numpy array of shape (d,), the mean vector of the second Gaussian distribution
        cov2: numpy array of shape (d, d), the covariance matrix of the second Gaussian distribution
    
    Returns:
        kl_divergence: float, the KL divergence between the two Gaussian distributions
    """
    try:
        mu1 = np.asarray(mu1, dtype=np.float64)
        mu2 = np.asarray(mu2, dtype=np.float64)
        cov1 = np.asarray(cov1, dtype=np.float64)
        cov2 = np.asarray(cov2, dtype=np.float64)
        
        d = len(mu1)
        
        # Invert the covariance matrix of the second Gaussian distribution
        cov2_inv = np.linalg.inv(cov2)
        
        # Calculate the trace term
        trace_term = np.trace(np.matmul(cov2_inv, cov1))
        
        # Calculate the squared difference in means
        mean_diff = mu2 - mu1
        mean_diff_term = np.dot(np.dot(mean_diff, cov2_inv), mean_diff)
        
        # Calculate the log-determinant term
        log_det_term = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
        
        # Calculate the KL divergence
        kl_divergence = 0.5 * (trace_term + mean_diff_term - d + log_det_term)
        
        return kl_divergence
    except (np.linalg.LinAlgError, ZeroDivisionError):
        print("Warning: KL divergence calculation failed")
        return 0.0

def C_tilde(Costs, eta, nominal_prob, reference_prob):
    """
    Calculate C_tilde for cost transformation
    Args:
        Costs (array): Costs array
        eta (float): radius of the KL ball
        nominal_prob (array): Nominal probability distribution
        reference_prob (array): Reference probability distribution
    Returns:
        float: Transformed cost
    """
    try:
        Costs = np.asarray(Costs, dtype=np.float64)
        nominal_prob = np.asarray(nominal_prob, dtype=np.float64)
        reference_prob = np.asarray(reference_prob, dtype=np.float64)
        
        def objective(alpha):
            alpha = alpha[0]
            if alpha <= 0.0:
                return np.inf
            term1 = alpha * eta
            term2 = alpha * np.log(np.sum(nominal_prob * ((nominal_prob / reference_prob) * np.exp(np.array(Costs))) ** (1 / alpha)))
            return term1 + term2
        
        # Initial guess for alpha
        initial_guess = [1.0]
        
        # Constraint definition
        constraints = [
            {'type': 'ineq', 'fun': lambda alpha: alpha[0] - 0.0}  # alpha > 0
        ]
        
        if eta != 0.0:  # Solve the problem
            result = minimize(objective, initial_guess, constraints=constraints)
            c_tilde = np.min([np.max(np.log((nominal_prob / reference_prob) * np.exp(np.array(Costs)))), result.fun])
        else:
            c_tilde = np.max(np.array(Costs))
        
        return c_tilde
    except Exception as e:
        print(f"Warning: C_tilde calculation failed: {e}")
        return np.max(Costs) if len(Costs) > 0 else 0.0

def reference_input(state, U_space_1, U_space_2, goal_points):
    """
    Calculate reference input distribution q_u(.) for reaching the goal point
    Args:
        state (array): Current state
        U_space_1 (array): Action space for 1st input
        U_space_2 (array): Action space for 2nd input
        goal_points (array): Goal points
    Returns:
        array: Reference input distribution
    """
    state = np.asarray(state, dtype=np.float64)
    goal_points = np.asarray(goal_points, dtype=np.float64)
    
    time_step = 0.033
    
    qpf = np.zeros((control_space_size, control_space_size))
    for i in range(control_space_size):
        for j in range(control_space_size):
            next_state_reference = reference_model_step(state, [U_space_1[i], U_space_2[j]], time_step)
            cov_nominal = np.array([[0.001, 0.0002], [0.0002, 0.001]], dtype=np.float64)
            f = st.multivariate_normal(next_state_reference.reshape((2,)), cov_nominal)
            N_samples = 20
            next_sample = f.rvs(N_samples)
            
            cost = [goal_cost(next_sample[k, :], goal_points) for k in range(N_samples)]
            policy = np.exp(-np.sum(cost) / N_samples)
            
            qpf[i, j] = policy
    
    S2 = np.sum(qpf)
    qpf = np.array([x / S2 for x in qpf])
    
    return qpf

def Control_step(state, U_space_1, U_space_2, goal_points, obs_points, method='DR', model_known=False):
    """
    Perform a control step
    Args:
        state (array): Current state
        U_space_1 (array): Action space for 1st input
        U_space_2 (array): Action space for 2nd input
        goal_points (array): Goal points
        obs_points (array): Obstacle points
        method (str): 'DR' or 'FPD'
        model_known (bool): Whether to use known model
    Returns:
        tuple: (action, policy)
    """
    # Input validation and type conversion
    state = np.asarray(state, dtype=np.float64)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    
    goal_points = np.asarray(goal_points, dtype=np.float64)
    obs_points = np.asarray(obs_points, dtype=np.float64)
    
    try:
        exponent = np.zeros((control_space_size, control_space_size))
        
        for i in range(control_space_size):
            for j in range(control_space_size):
                # Get next state prediction
                if model_known:
                    next_state_nominal = model_step(state, np.array([U_space_1[i], U_space_2[j]]), time_step)
                    cov_nom = np.array([[0.0001, 0.00002], [0.00002, 0.0001]], dtype=np.float64)
                else:
                    test_input = np.hstack((state.reshape(-1,), np.array([U_space_1[i], U_space_2[j]]))).reshape(1, -1)
                    next_state_nominal, sigma_nom = GP_nominal.predict(test_input, return_cov=True)
                    cov_nom = np.diag(sigma_nom.reshape((2,)))
                
                # Create nominal distribution
                p_bar = st.multivariate_normal(next_state_nominal.reshape((2,)), cov_nom)
                N_samples = 50
                next_sample = p_bar.rvs(N_samples)
                nominal_pdf = p_bar.pdf(next_sample)
                nominal_prob = nominal_pdf / np.sum(nominal_pdf)
                
                # Create reference distribution
                next_state_reference = goal_points[:-1]
                cov_reference = np.array([[0.001, 0.0002], [0.0002, 0.001]], dtype=np.float64)
                q = st.multivariate_normal(next_state_reference.reshape((2,)), cov_reference)
                reference_pdf = q.pdf(next_sample)
                reference_prob = reference_pdf / np.sum(reference_pdf)
                
                # Calculate costs
                cost = [state_cost(next_sample[k, :], goal_points, obs_points) for k in range(N_samples)]
                
                # Calculate policy based on method
                if method == 'DR':
                    eta = np.clip(calculate_kl_divergence(
                        goal_points[:-1].reshape((2,)), cov_reference,
                        next_state_nominal.T.reshape((2,)), cov_nom
                    ), 0.0, 100.0)
                    c_t = C_tilde(cost, eta, nominal_prob, reference_prob)
                    exponent[i, j] = -eta - c_t
                elif method == 'FPD':
                    DKL = calculate_kl_divergence(
                        next_state_nominal.T.reshape((2,)), cov_nom,
                        goal_points[:-1].reshape((2,)), cov_reference
                    )
                    exponent[i, j] = -DKL - np.sum(cost) / N_samples
        
        # Calculate policy
        exp_max = np.max(exponent)
        pf = np.exp(exponent - exp_max)
        pf = pf / np.sum(pf)
        
        # Sample action
        flat = pf.flatten()
        sample_index = np.random.choice(a=flat.size, p=flat)
        adjusted_index = np.unravel_index(sample_index, pf.shape)
        
        action = np.array([[U_space_1[adjusted_index[0]]],
                          [U_space_2[adjusted_index[1]]]], dtype=np.float64)
        
        return action, pf
        
    except Exception as e:
        print(f"Error in Control_step: {e}")
        import traceback
        traceback.print_exc()
        return np.array([[-1.], [-1.]], dtype=np.float64), np.zeros((control_space_size, control_space_size))

# Main simulation code
if __name__ == "__main__":
    # Instantiate Robotarium object
    N = 1
    M = 4
    
    # Define goal and obstacle points
    goal_points = np.array([[-1.4], [-0.8], [0]], dtype=np.float64)
    obs_points = np.array([[0, 0, 0, 0, 0, -0.8],
                          [0, 0.2, 0.4, 0.6, 0.8, -0.8],
                          [0, 0, 0, 0, 0, 0]], dtype=np.float64)
    
    initial_conditions = [
        np.array([[1.3], [0.9], [0]], dtype=np.float64),
        np.array([[0.5], [-0.2], [0]], dtype=np.float64),
        np.array([[1.2], [-0.5], [0]], dtype=np.float64),
        np.array([[-0.5], [0.25], [0]], dtype=np.float64)
    ]
    
    XX = [0] * M
    UU = [0] * M
    XN = [0] * M
    COVN = [0] * M
    Time = [0] * M
    
    for I in range(M):
        print(f"\n=== Starting Trial {I + 1}/{M} ===")
        
        try:
            r = robotarium.Robotarium(
                number_of_robots=N,
                show_figure=True,
                initial_conditions=initial_conditions[I],
                sim_in_real_time=False
            )
            
            # Create single integrator barrier certificate
            si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
            
            # Create SI to UNI dynamics transformation
            _, uni_to_si_states, si_to_uni_dyn = create_si_to_uni_mapping()
            
            # Lists to store data
            X = []
            U = []
            Xn = []
            Covn = []
            t = []
            
            # Main control loop
            while (np.size(at_pose(np.vstack((uni_to_si_states(r.get_poses()), r.get_poses()[2, :])), 
                                   goal_points, position_error=0.25, rotation_error=100)) != N):
                
                # Get poses of agents
                x = r.get_poses()
                x = np.asarray(x, dtype=np.float64)
                x_si = uni_to_si_states(x)
                x_si = np.asarray(x_si, dtype=np.float64).reshape(2, 1)
                
                # Store state
                X.append(x_si)
                
                # Get control input
                dxi, u_pf = Control_step(x_si, U_space_1, U_space_2, goal_points, obs_points, 
                                       method='FPD', model_known=True)
                
                # Store control
                U.append(dxi)
                
                # Apply barrier certificate
                dxi = si_barrier_cert(dxi, x_si)
                
                # Transform to unicycle dynamics
                dxu = si_to_uni_dyn(dxi, x)
                dxu = np.asarray(dxu, dtype=np.float64)
                
                # Set velocities
                r.set_velocities(np.arange(N), dxu)
                
                # Step simulation
                r.step()
            
            # Store trial data
            XX[I] = X
            UU[I] = U
            XN[I] = Xn
            COVN[I] = Covn
            Time[I] = t
            
            print(f"Trial {I + 1} completed successfully")
            
        except Exception as e:
            print(f"Error in trial {I + 1}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            try:
                r.call_at_scripts_end()
            except:
                pass
    
    # Save results
    print("\n=== Saving Results ===")
    try:
        np.save('State_Data_Simulation_DR_noamb_eta_0.npy', XX)
        np.save('Input_Data_Simulation_DR_nomab_eta_0.npy', UU)
        np.save('State_Data_nom_nomab_eta_0.npy', XN)
        np.save('COV_Data_nom_noamb_eta_0.npy', COVN)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error saving data: {e}")
    
    print("\n=== Simulation Complete ===")