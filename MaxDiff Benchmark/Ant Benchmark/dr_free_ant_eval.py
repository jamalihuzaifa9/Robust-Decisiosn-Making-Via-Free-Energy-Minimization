import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from termcolor import cprint
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from collections import deque
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
import pickle
warnings.filterwarnings('ignore')

# Try to import newer gym version first
try:
    import gymnasium as gym
    print("Using gymnasium")
except ImportError:
    import gym
    print("Using gym")


class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func
    def forward(self, x): return self.func(x)

class Model(nn.Module):
    _activations = {
        "sin": Lambda(lambda x: torch.sin(x)),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "softplus": nn.Softplus(),
        "swish": Lambda(lambda x: x * F.sigmoid(x)),
        "none" : None
    }
    def __init__(self, num_states, num_actions,init_w=3e-3,
                 model_layers=[200, 200],reward_layers=[200,200],std=1e-6,
                 model_AF='relu', reward_AF='relu',stoch=False,
                 log_std_min=-10, log_std_max=2, print_nets=False, reward_fn=None):

        super(Model, self).__init__()
        self.num_states  = nn.Parameter(torch.tensor(num_states),requires_grad=False)
        self.num_actions = nn.Parameter(torch.tensor(num_actions),requires_grad=False)
        self.stoch = stoch
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.reward_fun = reward_fn

        '''
        Set activation function based on input
        '''
        _AF = self._activations[model_AF.lower()]
        _AF_rew = self._activations[reward_AF.lower()]
        out_multiplier = 2 if stoch else 1


        '''
        Model representation of dynamics as specified
        '''
        modules = []
        modules.append(nn.Linear(num_states+num_actions, model_layers[0]))
        for i in range(len(model_layers)-1):
            modules.append(_AF)
            modules.append(nn.Linear(model_layers[i], model_layers[i+1]))
        modules.append(_AF)
        modules.append(nn.Linear(model_layers[-1], num_states*out_multiplier))

        self.mu = nn.Sequential(*modules)

        # initialize weights
        self.mu[-1].weight.data.uniform_(-init_w, init_w)
        self.mu[-1].bias.data.uniform_(-init_w, init_w)
        if print_nets:
            cprint(['model',self.mu],'cyan')
            
        
        if _AF_rew is not None:
            '''
            The reward function is modeled as specified
            '''
            rew_modules = []
            rew_modules.append(nn.Linear(num_states+num_actions, reward_layers[0]))
            for i in range(len(reward_layers)-1):
                rew_modules.append(_AF_rew)
                rew_modules.append(nn.Linear(reward_layers[i], reward_layers[i+1]))
            rew_modules.append(_AF_rew)
            rew_modules.append(nn.Linear(reward_layers[-1], 1))

            self.reward_fun = nn.Sequential(*rew_modules)

            # initialize weights
            self.reward_fun[-1].weight.data.uniform_(-init_w, init_w)
            self.reward_fun[-1].bias.data.uniform_(-init_w, init_w)
        else:
            self.reward_fun = nn.Sequential(nn.Linear(num_states+num_actions, 1))
        if print_nets:
            cprint(['reward_fun',self.reward_fun],'magenta')

        if not self.stoch:
            self.log_std = nn.Parameter(torch.randn(1, num_states) * std)
        else:
            self.log_std = torch.zeros(1)

    def forward(self, s, a):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """
        _in   = torch.cat([s, a], dim=1)
        if self.stoch:
            x,log_std  = torch.split(self.mu(_in),[self.num_states,self.num_states],dim=-1)
            std = torch.clamp(log_std, self.log_std_min, self.log_std_max).exp()
        else:
            x = self.mu(_in)
            std = torch.clamp(self.log_std,self.log_std_min, self.log_std_max).exp().expand_as(x)
        rew = self.reward_fun(_in)
        return x+s, std, rew, torch.zeros(0)

    def step(self, x, u):
        mean, std, rew, done = self.forward(x, u)
        if self.stoch:
            return self.sample(mean,std), rew, done
        else:
            return mean, rew, done

    @torch.jit.ignore
    def sample(self,mean,std):
        dist = Normal(mean, std)
        return dist.sample()


SEED = 113
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STATE_DIM, ACTION_DIM = 29, 8
np.random.seed(SEED)
torch.manual_seed(SEED)

class DRFreeAgent:
    """DR-FREE agent for HalfCheetah with pretrained model and fixed covariance"""
    
    def __init__(self, env_name='Ant-v5', n_action_samples=100, n_state_samples=20, 
                 fixed_cov_value=0.01, model_path=None):
        self.env = gym.make(env_name, render_mode='human', exclude_current_positions_from_observation=False)
        self.state_dim = 29 #self.env.observation_space.shape[0]
        self.action_dim = 8 #self.env.action_space.shape[0]
        self.action_bounds = (self.env.action_space.low, self.env.action_space.high)
        
        # Sampling parameters
        self.n_action_samples = n_action_samples
        self.n_state_samples = n_state_samples
        
        # Load pretrained dynamics model
        self.dynamics_model = Model(STATE_DIM, ACTION_DIM, model_layers=[512, 512, 512], reward_layers=[512, 512, 512]).to(DEVICE)
        if model_path:
            self.dynamics_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Loaded pretrained model from {model_path}")
        else:
            # Default path from your code
            model_path = r"D:\MaxDiffRL\data\maxdiff\AntEnv_v3_orig_H20_alpha5\seed_13\model_final.pt"
            self.dynamics_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"✅ Loaded pretrained model from default path")
        
        self.dynamics_model.eval()
        
        # Fixed covariance matrix
        self.fixed_cov_value = fixed_cov_value
        self.fixed_cov = np.eye(self.state_dim) * fixed_cov_value
        
        # Reference distribution parameters
        self.reference_cov = np.eye(self.state_dim) * 0.01
        
        # DR-FREE parameters
        self.eta_max = 10.0
        
    def predict_next_state(self, state, action):
        """Predict next state using pretrained model with learned covariance"""
        state_tensor = torch.FloatTensor(state[:29].reshape(1, -1)).to(DEVICE)
        action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(DEVICE)
        
        with torch.no_grad():
            # Get prediction from pretrained model
            next_state_pred, std, reward, _ = self.dynamics_model.forward(state_tensor, action_tensor)
            next_state_mean = next_state_pred.cpu().numpy().flatten()
            # Convert std to covariance matrix
            std_np = std.cpu().numpy().flatten()
            next_state_cov = np.diag(std_np**2)  # Convert std to variance for diagonal elements
        
        return next_state_mean, next_state_cov, reward
    
    def compute_cost(self, state, action, goal_state=None):
        """Cost function for HalfCheetah (negative reward)"""
        # HalfCheetah reward: forward velocity - control cost
        velocity = state[8] if len(state) > 8 else 0
        control_cost = 0.1 * np.sum(np.square(action))  # Control cost
        cost = -1*velocity + control_cost  # Negative reward is cost
        return cost
    
    def calculate_kl_divergence(self, mu1, cov1, mu2, cov2):
        """KL divergence between two multivariate Gaussians"""
        d = len(mu1)
        
        try:
            cov2_inv = np.linalg.inv(cov2 + 1e-6 * np.eye(d))
            cov1_det = np.linalg.det(cov1 + 1e-6 * np.eye(d))
            cov2_det = np.linalg.det(cov2 + 1e-6 * np.eye(d))
            
            trace_term = np.trace(np.dot(cov2_inv, cov1))
            mean_diff = mu2 - mu1
            mean_term = np.dot(np.dot(mean_diff, cov2_inv), mean_diff)
            log_det_term = np.log(cov2_det / cov1_det)
            
            kl_div = 0.5 * (trace_term + mean_term - d + log_det_term)
            return np.clip(kl_div, 0.0, self.eta_max)
        except:
            return self.eta_max
        
    def calculate_kl_divergence(self, mu1, cov1, mu2, cov2):
        """KL divergence between desired forward velocity and current forward velocity"""
        # Extract x-velocity components (index 13 in Ant state)
        x_vel_mean1 = mu1[13]  # Current x-velocity mean
        x_vel_var1 = cov1[13, 13]  # Current x-velocity variance
        
        # Desired positive x-velocity (you can adjust this value)
        desired_x_vel = 1.0  # Desired forward velocity
        desired_x_vel_var = 0.1  # Small variance for desired velocity
        
        try:
            # KL divergence between two 1D Gaussians
            var_ratio = desired_x_vel_var / (x_vel_var1 + 1e-6)
            mean_diff = desired_x_vel - x_vel_mean1
            
            kl_div = 0.5 * (var_ratio + (mean_diff**2)/(desired_x_vel_var + 1e-6) - 1 - np.log(var_ratio))
            return np.clip(kl_div, 0.0, self.eta_max)
        except:
            return self.eta_max
    
    def c_tilde_optimization(self, costs, eta, nominal_probs, reference_probs):
        """Solve the C_tilde optimization problem"""
        costs = np.array(costs)
        
        def objective(alpha):
            alpha_val = alpha[0]
            if alpha_val <= 0:
                return np.inf
            
            try:
                # Compute robust cost
                ratio = nominal_probs / (reference_probs + 1e-10)
                exp_term = nominal_probs*((ratio * np.exp(costs))**alpha_val)
                
                if np.any(~np.isfinite(exp_term)):
                    return np.inf
                
                return alpha_val * eta + alpha_val * np.log(np.sum(exp_term))
            except:
                return np.inf
        
        # Optimize
        result = minimize(objective, [1.0], bounds=[(1e-6, 1e10)], method='L-BFGS-B')
        
        return np.min([np.max(np.log((nominal_probs/reference_probs)*np.exp(np.array(costs)))),result.fun])
    
    def sample_actions(self):
        """Sample actions from continuous action space"""
        actions = np.random.uniform(
            self.action_bounds[0], 
            self.action_bounds[1], 
            size=(self.n_action_samples, self.action_dim)
        )
        return actions
    
    # def dr_free_control_step(self, state, goal_state=None):
    #     """DR-FREE control step with sampling using pretrained model"""
    #     # Sample candidate actions
    #     action_samples = 0.5*self.sample_actions()
        
    #     # Reference distribution mean (goal-oriented or state-keeping)
    #     if goal_state is not None:
    #         ref_mean = goal_state
    #     else:
    #         ref_mean = state[:29]
        
    #     # Compute policy for each action sample
    #     log_probs = []
        
    #     for action in action_samples:
    #         # Predict next state distribution using pretrained model
    #         next_state_mean, next_state_cov, reward = self.predict_next_state(state, action)
            
    #         # Sample next states from nominal distribution
    #         nominal_dist = multivariate_normal(next_state_mean, next_state_cov)
    #         next_state_samples = nominal_dist.rvs(self.n_state_samples)
    #         if next_state_samples.ndim == 1:
    #             next_state_samples = next_state_samples.reshape(1, -1)
            
    #         # Compute probabilities
    #         nominal_probs = nominal_dist.pdf(next_state_samples)
    #         nominal_probs = nominal_probs / np.sum(nominal_probs)
            
    #          # Create structured reference covariance matrix for ant locomotion
    #         reference_variances = np.zeros(self.state_dim)
            
    #         # State indices for Ant-v5 (first 29 dimensions)
    #         # Torso height and orientation (very small variance for stability)
    #         reference_variances[2] = 0.001  # z-pos (height)
    #         reference_variances[3:7] = 0.001  # quaternion orientation
            
    #         # Forward velocity and yaw (moderate variance)
    #         reference_variances[14:17] = 0.05  # linear velocities
    #         reference_variances[20] = 0.05  # yaw velocity
            
    #         # Angular velocities for roll/pitch (small variance)
    #         reference_variances[18:20] = 0.01  # roll and pitch velocities
            
    #         # Joint angles and velocities (larger variance for gait exploration)
    #         reference_variances[7:14] = 0.1  # joint angles
    #         reference_variances[21:29] = 0.1  # joint velocities
            
    #         # Create diagonal covariance matrix
    #         self.reference_cov = np.diag(reference_variances)
            
    #         # Add small off-diagonal terms for joint angle-velocity coupling
    #         for i in range(7):  # 7 joints
    #             joint_pos_idx = 7 + i  # joint position index
    #             joint_vel_idx = 21 + i  # joint velocity index
    #             coupling = 0.01  # coupling strength
    #             self.reference_cov[joint_pos_idx, joint_vel_idx] = coupling
    #             self.reference_cov[joint_vel_idx, joint_pos_idx] = coupling
            
    #         print("✅ Created structured reference covariance matrix for ant locomotion")
                
    #         # Reference distribution
    #         reference_dist = multivariate_normal(next_state_mean, self.reference_cov)
    #         reference_probs = reference_dist.pdf(next_state_samples)
    #         reference_probs = reference_probs / (np.sum(reference_probs) + 1e-10)
            
    #         # Compute KL divergence
    #         eta = self.calculate_kl_divergence(
    #             next_state_mean, next_state_cov, ref_mean, self.reference_cov
    #         )
            
    #         # Compute costs
    #         costs = [self.compute_cost(sample, action, goal_state) for sample in next_state_samples]
            
    #         # Solve DR optimization
    #         # c_tilde = self.c_tilde_optimization(costs, eta, nominal_probs, reference_probs)
            
    #         # Store log probability
    #         log_prob = -eta + float(reward) - 0.*np.mean(costs)
    #         log_probs.append(log_prob)
        
    #     # Convert to probabilities
    #     log_probs = np.array(log_probs)
    #     log_probs = log_probs - np.max(log_probs)  # Numerical stability
    #     probs = np.exp(log_probs)
    #     probs = probs / np.sum(probs)
        
    #     # Sample action according to policy
    #     action_idx = np.random.choice(len(action_samples), p=probs)
    #     selected_action = action_samples[action_idx]
        
    #     return selected_action, probs
    
    def dr_free_control_step(self, state, goal_state=None):
        """DR-FREE control step with sampling using pretrained model"""
        # Sample candidate actions
        action_samples = 0.5*self.sample_actions()
        
        # Reference distribution mean (goal-oriented or state-keeping)
        if goal_state is not None:
            ref_mean = goal_state
        else:
            ref_mean = state[:29]
        
        # Compute policy for each action sample
        log_probs = []
        
        for action in action_samples:
            # Predict next state distribution using pretrained model
            next_state_mean, next_state_cov, reward = self.predict_next_state(state, action)
            
            # Sample next states from nominal distribution
            nominal_dist = multivariate_normal(next_state_mean, next_state_cov + 1e-6 * np.eye(29))
            next_state_samples = nominal_dist.rvs(self.n_state_samples)
            if next_state_samples.ndim == 1:
                next_state_samples = next_state_samples.reshape(1, -1)
            
            # Compute probabilities
            nominal_probs = nominal_dist.pdf(next_state_samples)
            nominal_probs = nominal_probs / np.sum(nominal_probs)
            
            # Create structured reference covariance matrix
            reference_variances = np.zeros(self.state_dim)
    
            # Torso height (very small variance for stability)
            reference_variances[0] = 0.001  # z-pos (height)
            reference_variances[1:5] = 0.001  # quaternion orientation
            
            # Forward velocity and yaw (moderate variance)
            reference_variances[12:15] = 0.05  # linear velocities
            reference_variances[18] = 0.05  # yaw velocity
            
            # Angular velocities for roll/pitch (small variance)
            reference_variances[16:18] = 0.01  # roll and pitch velocities
            
            # Joint angles and velocities (larger variance for gait exploration)
            reference_variances[5:12] = 0.1  # joint angles (8 joints)
            reference_variances[19:27] = 0.1  # joint velocities
            
            # Set very large variance for unused x-y positions
            reference_variances[27:29] = 1000.0  # Excluded x-y positions
            
            # Create base diagonal covariance matrix
            ref_cov = np.diag(reference_variances)
    
            # Add small off-diagonal terms for joint angle-velocity coupling
            coupling = 0.01
            for i in range(7):  # 7 joints
                joint_pos_idx = 5 + i  # joint angle index (starting at 5)
                joint_vel_idx = 19 + i  # joint velocity index (starting at 19)
                ref_cov[joint_pos_idx, joint_vel_idx] = coupling
                ref_cov[joint_vel_idx, joint_pos_idx] = coupling
            
            # Ensure matrix is positive definite
            ref_cov = ref_cov + 1e-6 * np.eye(self.state_dim)
            
            # Create reference distribution
            reference_dist = multivariate_normal(next_state_mean, ref_cov)
            reference_probs = reference_dist.pdf(next_state_samples)
            reference_probs = reference_probs / (np.sum(reference_probs) + 1e-10)
            
            # Compute KL divergence using the same covariance
            eta = self.calculate_kl_divergence(
                next_state_mean, next_state_cov, ref_mean, ref_cov
            )
            
            # Compute costs and log probability
            costs = [self.compute_cost(sample, action, goal_state) for sample in next_state_samples]
            log_prob = -eta + float(reward) - 0.*np.mean(costs)
            log_probs.append(log_prob)
        
        # Rest of the method remains the same
        log_probs = np.array(log_probs)
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)
        
        action_idx = np.random.choice(len(action_samples), p=probs)
        selected_action = action_samples[action_idx]
        
        return selected_action, probs
    
    # def evaluate_episode(self, max_steps=1000, render=False):
    #     """Evaluate a single episode and store timestep rewards"""
    #     state = self.env.reset()
    #     # Handle different gym versions
    #     if isinstance(state, tuple):
    #         state = state[0]
                
    #     total_reward = 0
    #     timestep_rewards = []  # List to store rewards at each timestep
        
    #     for step in range(max_steps):
    #         # Get action from DR-FREE policy
    #         action, _ = self.dr_free_control_step(state)
            
    #         # Step environment
    #         step_result = self.env.step(action)
    #         if len(step_result) == 5:
    #             # New gym API (>= 0.26)
    #             next_state, reward, terminated, truncated, _ = step_result
    #             done = terminated or truncated
    #         else:
    #             # Old gym API
    #             next_state, reward, done, _ = step_result
            
    #         # Store reward for this timestep
    #         timestep_rewards.append(reward)
    #         total_reward += reward
    #         state = next_state
            
    #         if done:
    #             break
        
    #         if render:
    #             self.env.render()
    
        return total_reward, step + 1, timestep_rewards
    
    def calculate_prediction_kl_divergence(self, predicted_mean, predicted_cov, actual_state):
        """
        Compute KL divergence between predicted and actual states
        Args:
            predicted_mean: Model's predicted next state mean
            predicted_cov: Model's predicted next state covariance
            actual_state: Actual next state from environment
        """
        # Use only the first 27 states (excluding x-y position)
        predicted_mean = predicted_mean[:27]
        predicted_cov = predicted_cov[:27, :27]
        actual_state = actual_state[:27]

        # Create small diagonal covariance for actual state (treating it as a sharp distribution)
        actual_cov = np.eye(27) * 1e-6

        try:
            # KL(actual || predicted)
            d = len(predicted_mean)
            predicted_cov_inv = np.linalg.inv(predicted_cov + 1e-6 * np.eye(d))
            actual_cov_det = np.linalg.det(actual_cov + 1e-6 * np.eye(d))
            predicted_cov_det = np.linalg.det(predicted_cov + 1e-6 * np.eye(d))
            
            trace_term = np.trace(np.dot(predicted_cov_inv, actual_cov))
            mean_diff = predicted_mean - actual_state
            mean_term = np.dot(np.dot(mean_diff, predicted_cov_inv), mean_diff)
            log_det_term = np.log(predicted_cov_det / actual_cov_det)
            
            kl_div = 0.5 * (trace_term + mean_term - d + log_det_term)
            return np.clip(kl_div, 0.0, self.eta_max)
        except:
            return self.eta_max

    def evaluate_episode(self, max_steps=1000, render=False):
        """Evaluate a single episode and store predictions and actual states"""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        timestep_rewards = []
        predicted_states = []  # Store predicted states
        predicted_covs = []   # Store predicted covariances
        actual_states = []    # Store actual states

        for step in range(max_steps):
            # Get action from DR-FREE policy
            action, _ = self.dr_free_control_step(state)
            
            # Get model prediction before stepping environment
            predicted_next_state, predicted_cov, _ = self.predict_next_state(state, action)
            
            # Step environment
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            
            # Store data
            timestep_rewards.append(reward)
            predicted_states.append(predicted_next_state)
            predicted_covs.append(predicted_cov)
            actual_states.append(next_state)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
            
            if render:
                self.env.render()

        return total_reward, step + 1, timestep_rewards, predicted_states, predicted_covs, actual_states


def main():
    # Initialize agent with pretrained model
    agent = DRFreeAgent(
        env_name='Ant-v5',
        n_action_samples=100,   # Number of action samples
        n_state_samples=50,    # Number of state samples per action
        fixed_cov_value=0.05,  # Fixed covariance value for state uncertainty
        model_path= None     # Uses default path or specify custom path
    )
    
    print("✅ DR-FREE agent initialized with pretrained model and fixed covariance")
    print(f"Fixed covariance value: {agent.fixed_cov_value}")
    print("No training required - using pretrained dynamics model")
    
    # Evaluate DR-FREE policy directly
    print("\nEvaluating DR-FREE policy...")
    episode_rewards = []
    prediction_data = []  # Store prediction data for each episode
    
    for i in range(1):
        reward, steps, timestep_rewards, pred_states, pred_covs, actual_states = agent.evaluate_episode()
        episode_rewards.append(reward)
        
        # Store all data for this episode
        episode_data = {
            'timestep_rewards': timestep_rewards,
            'predicted_states': pred_states,
            'predicted_covs': pred_covs,
            'actual_states': actual_states
        }
        prediction_data.append(episode_data)
        
        print(f"Episode {i+1}: Reward = {reward:.2f}, Steps = {steps}")

    # Save all data
    np.save('ant_prediction_data.npy', {
        'episode_rewards': episode_rewards,
        'prediction_data': prediction_data
    })
    
    print(f"\nAverage reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Data saved to 'ant_prediction_data.npy'")
    
    
    # Close environment
    agent.env.close()


if __name__ == "__main__":
    main()
    
# import numpy as np
# import pickle

# def convert_numpy_to_pickle():
#     # Load the numpy file
#     data = np.load(r'D:\DR_FREE_ICRA\code\ant_drfree_rewards_log.npy', allow_pickle=True).item()
    
#     # Extract only the timestep data
#     timestep_rewards = data['timestep_data']
    
#     # Save to pickle file
#     with open('ant_timestep_rewards.pkl', 'wb') as f:
#         pickle.dump(timestep_rewards, f)
    
#     print(f"Successfully converted timestep rewards to pickle format")
#     print(f"Number of episodes: {len(timestep_rewards)}")
#     print(f"Average episode length: {np.mean([len(ep) for ep in timestep_rewards]):.1f} steps")

# if __name__ == "__main__":
#     convert_numpy_to_pickle()    