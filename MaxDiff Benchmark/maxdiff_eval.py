"""
DR_robot_routing_simulate_maxdiff_eval.py

This script loads a trained MaxDiff model for robot control using a known reward
and a learned dynamics model, then evaluates its behavior on a batch of
initial conditions within the Robotarium simulator.

It wraps the dynamics model with a known reward function and deploys the
SimpleMaxDiff planner for evaluating closed-loop control performance.
"""

# %% Imports and Setup
import os
import torch
import numpy as np
from termcolor import cprint
from Env_Robotarium import RobotariumEnv
from nn_model import Model
from nn_model import KnownRewardModel
from MaxDiff3 import SimpleMaxDiff

# %% Load Environment and Define Cost

goal = np.array([[-1.4], [-0.8], [0.0]])
obs = np.array([[0, 0, 0, 0, 0, -0.8],
                [0, 0.2, 0.4, 0.6, 0.8, -0.8],
                [0, 0, 0, 0, 0, 0]])

def state_cost(state, goal_points=goal, obs_points=obs):
    def logpdf(x, u, covar):
        k = len(x)
        diff = x - u
        inv_covar = np.linalg.inv(covar)
        exponent = -0.5 * (diff.T @ inv_covar @ diff)
        denom = np.sqrt((2 * np.pi) ** k * np.linalg.det(covar))
        return np.exp(exponent) / denom

    v = np.array([0.035, 0.035], dtype=np.float32)
    covar = np.diag(v)

    gauss_sum = 0
    for i in range(obs_points.shape[1]):
        gauss_sum += 100 * logpdf(state[:2], obs_points[:2, i], covar)

    cost = (50 * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2) +
            gauss_sum + 10 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi))))
    return -cost

# %% Config and Init

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STATE_DIM, ACTION_DIM = 2, 2
np.random.seed(SEED)
torch.manual_seed(SEED)

env = RobotariumEnv(goal_points=goal, obs_points=obs)

# %% Load trained model
# model = Model(STATE_DIM, ACTION_DIM, model_layers=[128, 128], reward_layers=[128,128]).to(DEVICE)
model = KnownRewardModel(STATE_DIM, ACTION_DIM, model_layers=[128, 128], reward_fn=state_cost).to(DEVICE)
model.load_state_dict(torch.load(r"D:\Network Security\KL Control\robotarium_python_simulator\rps\examples\DR_FREE\Experiments\dynamics_trained_nn_nocap.pt", map_location=DEVICE))
model.eval()
print("\n✅ Loaded trained dynamics model with known reward.")

# %% JIT tracing for use in planner
with torch.no_grad():
    inputs = (torch.rand(100, STATE_DIM, device=DEVICE),
              torch.rand(100, ACTION_DIM, device=DEVICE))
    jit_model = torch.jit.trace(model, inputs)

# %% Wrap in planner
planner = SimpleMaxDiff(model_fn=jit_model, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                        horizon=20, gamma=0.95, samples=100, alpha=0.1, lam=1.0, device=DEVICE)

# %% Evaluate Policy on Multiple Initial Conditions
initial_conditions = [ np.array(np.mat('1.32;0.9; 0')),np.array(np.mat('0.5;-0.2; 0')),np.array(np.mat('1.2;-0.5; 0')),np.array(np.mat('-0.5;0.25; 0')),
                      np.array(np.mat('1.0;0.8; 0')),np.array(np.mat('0.5;-0.2; 0')),np.array(np.mat('-0.5;-0.5; 0')),np.array(np.mat('-0.9;0.25; 0')),
                       np.array(np.mat('-1.1;0.9; 0')),np.array(np.mat('0.9;-0.2; 0')),np.array(np.mat('-0.7;0.5; 0')),np.array(np.mat('-0.5;0.25; 0'))] 

MAX_STEPS = 3000
M=12
XX = []*M

total_rewards = []
for i, init_state in enumerate(initial_conditions):
    state = env.reset(init_state)
    planner.reset()
    episode_reward = 0
    X = []

    for step in range(MAX_STEPS):
        action = planner(state.copy())
        next_state, reward, done, _ = env.step(action.copy())
        episode_reward += reward
        state = next_state
        X.append(state.copy())
        if done:
            break
    XX.append(X)
    total_rewards.append(episode_reward)
    print(f"\n▶ Episode {i+1} Reward: {episode_reward:.2f}")



print("\n✅ Evaluation completed.")
print(f"Average reward: {np.mean(total_rewards):.2f} over {len(initial_conditions)} runs")

# Save logs if needed
np.save("eval_rewards_maxdiff.npy", np.array(total_rewards))
# Save trajectories
np.save("eval_trajectories_maxdiff_dynamics_nocap.npy", np.array(XX))
