import torch
from torch.distributions import Normal
import numpy as np
from termcolor import cprint
from entropy import get_entropy_params
from utils import _batch_mv

goal_points = np.array(np.mat('-1.4; -0.8; 0'))  # Modify as needed

obs_points = np.array(np.mat('0 0 0 0 0 -0.8;0 0.2 0.4 0.6 0.8 -0.8;0 0 0 0 0 0'))


def state_cost(state, goal_points=goal_points, obs_points=obs_points):
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
    
    cost = (100 * ((state[0] - goal_points[0])**2 + (state[1] - goal_points[1])**2) +
            gauss_sum +
            10 * (np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[0] - 1.5) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) 
                 +
                 np.exp(-0.5 * ((state[1] - 1.0) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi)) +
                 np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03)**2) / (0.03 * np.sqrt(2 * np.pi))))
    return cost


class MaxDiff:
    def __init__(self, model_fn, state_dim, action_dim, samples=10, horizon=10,
                 lam=1.0, alpha=0.01, eps=0.3, bound=1e10, gamma=1.0, device='cpu', use_real_env=False):     

        self.model = model_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.samples = samples
        self.horizon = horizon
        self.lam = lam
        self.alpha = alpha
        self.bound = bound
        self.gamma = gamma
        self.device = device
        self.use_real_env = use_real_env

        self.a = torch.zeros(horizon, action_dim, device=device)
        self.gammas = (gamma ** torch.arange(horizon, device=device)).unsqueeze(-1).repeat(1, samples)

        self.entropy_fn = get_entropy_params(
        horizon=self.horizon,
        num_states=self.state_dim,
        device=self.device,
        explr_dim=None,                # or a list of indices like [0,1]
        angle_idx=None,
        weights=None,                  # or custom [1., 1.]
        window=False,                  # or True if you want time-windowed entropy
        logdet_method='abs',           # safest option
        weight_method='quad'           # quadratic form for covariance weighting
        )

        self.noise_dist = Normal(
            torch.zeros(samples, action_dim, device=device),
            torch.ones(samples, action_dim, device=device) * eps
        )

    def reset(self):
        self.a.zero_()

    def __call__(self, state):
        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            sk = torch.zeros(self.horizon, self.samples, device=self.device)
            log_prob = torch.zeros(self.horizon, self.samples, device=self.device)
            da = torch.zeros(self.horizon, self.samples, self.action_dim, device=self.device)
            states = torch.zeros(self.horizon, self.samples, self.state_dim, device=self.device)

            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.samples, 1)
            eta = torch.zeros(self.samples, self.action_dim, device=self.device)
            states[0] = s
            
            if self.use_real_env:    
                for t in range(self.horizon-1):
                    eps = self.noise_dist.sample()
                    eta = 0.5 * eta + 0.5 * eps
                    log_prob[t] = self.noise_dist.log_prob(eta).sum(1)
                    da[t] = eta

                    actions = self.a[t].expand_as(eta) + eta
                    
                    for j in range(self.samples):
                        # If using a real environment, set the state and get the reward
                        next_state = self.model(states[t,j].cpu().numpy(), actions[j].cpu().numpy()).reshape(self.state_dim,)
                        states[t+1, j] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                        rewards = -state_cost(next_state)  # cost-to-reward
                        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                        sk[t,j] = rewards.squeeze()
            
            else:
                for t in range(self.horizon):
                    states[t] = s
                    eps = self.noise_dist.sample()
                    eta = 0.5 * eta + 0.5 * eps
                    log_prob[t] = self.noise_dist.log_prob(eta).sum(1)
                    da[t] = eta

                    actions = self.a[t].expand_as(eta) + eta
                    s_next, _, rewards, done = self.model(s, actions)
                    s = torch.clamp(s_next, -self.bound, self.bound)
                    sk[t] = rewards.squeeze()

            sk_discounted = sk * self.gammas
            sk_total = sk_discounted.sum(0)
            entropy = self.entropy_fn(states)

            sk_total = sk_total + self.alpha * entropy
            sk_total = sk_total - torch.max(sk_total)
            w = torch.exp(sk_total / self.lam) + 1e-5
            w = w / w.sum()

            self.a = self.a + torch.transpose(da, -1, -2) @ w

            return self.a[0].cpu().numpy()
