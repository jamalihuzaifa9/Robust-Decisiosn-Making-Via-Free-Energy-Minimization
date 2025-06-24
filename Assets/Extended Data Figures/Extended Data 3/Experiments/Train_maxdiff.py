import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import random
from termcolor import cprint
import pickle

from nn_model import Model
from nn_model import KnownRewardModel
from MaxDiff3 import SimpleMaxDiff
from optimizer import ModelOptimizer
from replaybuffer import ReplayBuffer
from Env_Robotarium import RobotariumEnv

goal = np.array([[-1.4], [-0.8], [0.0]])
obs = np.array([[0, 0, 0, 0, 0, -0.8],
                [0, 0.2, 0.4, 0.6, 0.8, -0.8],
                [0, 0, 0, 0, 0, 0]])

env = RobotariumEnv(goal_points=goal, obs_points=obs)

# 1. --- Reward function ---
def state_cost(state, goal_points=goal, obs_points=obs):
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
    
    v = np.array([0.035, 0.035], dtype=np.float32)
    covar = np.diag(v)
    
    gauss_sum = 0
    for i in range(obs_points.shape[1]):
        gauss_sum += 100 * logpdf(state[:2], obs_points[:2, i], covar)
    
    cost = (30 * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2) +
            gauss_sum +
            10 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi))))
    return -cost

# 2. --- Hyperparameters ---
SEED = 42
MAX_FRAMES = 100000
MAX_STEPS = 1000
BATCH_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. --- Set seeds ---
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 4. --- Environment setup ---
args = {
    'env': 'Env_Robotarium',   # Or 'RobotariumEnv'
    'render': False,
    'pointmass': True,
    'start_mode': 'one_corner',
    'beta': 0.01
}
config = {
    'model_layers': [128, 128],
    'model_activation_fun': 'relu',
    'reward_layers': [128,128],
    'reward_activation_fun': 'relu',
    'planner': {
        'samples': 50,
        'horizon': 2,
        'alpha': 0.5,
        'bound': 0.5
    },
    'model_lr': 5e-4,
    'model_iter': 5,
    'batch_size': BATCH_SIZE
}

# 5. --- Build env + replay buffer ---
# env, env_name, action_dim, state_dim = 
action_dim, state_dim = 2, 2  # For RobotariumEnv
replay_buffer = ReplayBuffer(int(1e5), state_dim=2, action_dim=2)
replay_buffer.seed(SEED)

# 6. --- Model and planner ---
# model = Model(state_dim, action_dim,
#               model_layers=config['model_layers'],
#               reward_layers=config['reward_layers']).to(DEVICE)

model = KnownRewardModel(state_dim, action_dim, model_layers=config['model_layers'],
                                reward_fn=state_cost).to(DEVICE)

with torch.no_grad():
    inputs = (torch.rand(config['planner']['samples'], state_dim, device=DEVICE),
              torch.rand(config['planner']['samples'], action_dim, device=DEVICE))
    jit_model_plan = torch.jit.trace(model, inputs)
    prime = jit_model_plan(*inputs)

inputs = (torch.rand(config['batch_size'], state_dim, device=DEVICE),
              torch.rand(config['batch_size'], action_dim, device=DEVICE))
jit_model_opt = torch.jit.trace(model, inputs)
prime = jit_model_opt(*inputs)
model_optim = ModelOptimizer(jit_model_opt, replay_buffer, lr=config['model_lr'], device=DEVICE)

# MaxDiff expects model to give (next_state, reward)
class DynamicsWithKnownReward(torch.nn.Module):
    def __init__(self, model, reward_fn):
        super().__init__()
        self.model = model
        self.reward_fn = reward_fn

    def forward(self, s, a):
        s_next = self.model(s, a)
        r = torch.tensor([self.reward_fn(s[i].cpu().numpy(), a[i].cpu().numpy())
                          for i in range(s.shape[0])], device=s.device)
        return s_next, r.unsqueeze(1)

planner = SimpleMaxDiff(model_fn=jit_model_plan, state_dim=2, action_dim=2,
                            horizon=config['planner']['horizon'], gamma=0.95, samples=config['planner']['samples'],
                            alpha=config['planner']['alpha'], lam=1.0, device=DEVICE)

# 7. --- Training Loop ---
frame_idx = 0
while frame_idx < MAX_FRAMES:
    x = np.random.uniform(low=-1.4, high=1.4)
    y = np.random.uniform(low=-0.9, high=0.9)
    theta = 0.0

    init_state = np.array([[x], [y], [theta]])
    state = env.reset(init_state)
    
    # state = env.reset(np.array([[0.7], [0.5], [0.0]]))  # Reset the environment with a specific initial state
    planner.reset()
    action = planner(state.copy())
    episode_reward = 0

    for step in range(MAX_STEPS):

        next_state, reward, done, _ = env.step(action.copy())

        # reward = known_reward_fn(state, action)
        next_action = planner(next_state.copy())

        replay_buffer.push(state, action, reward, next_state, next_action, done)

        if len(replay_buffer) > BATCH_SIZE:
            model_optim.update_model(BATCH_SIZE,
                                     mini_iter=config['model_iter'],
                                     debug=(frame_idx % 256 == 0),
                                     calc_eig=False)

        state = next_state
        action = next_action
        print(f"[Frame {frame_idx}] Step {step} | Action: {action} | Reward: {reward:.2f} | Done: {done} | state: {state}")
        episode_reward += reward
        
        frame_idx += 1

        if done:
            break

    print(f"[Frame {frame_idx}] Episode Reward: {episode_reward:.2f}")
env.close()
# Save trained dynamics model
torch.save(model.state_dict(), "dynamics_trained_nn_nocap.pt")
pickle.dump(model_optim.log, open('optim_data'+ '.pkl', 'wb'))
print("✅ Dynamics and reward model saved as trained_model.pt")
  