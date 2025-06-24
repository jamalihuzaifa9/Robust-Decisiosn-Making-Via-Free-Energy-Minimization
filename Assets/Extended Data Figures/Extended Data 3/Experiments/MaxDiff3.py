import torch
from torch.distributions import Normal
import numpy as np
from termcolor import cprint
from entropy import get_entropy_params
from utils import _batch_mv

class SimpleMaxDiff:
    def __init__(self, model_fn, state_dim, action_dim, samples=10, horizon=10,
                 lam=0.5, alpha=0.1, eps=0.3, bound=1e10, gamma=1.0, device='cpu', use_real_env=False):     

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
            
            # if self.use_real_env:
            #     s = state[None,:].repeat(self.samples,0)
            #     self.model.set_state(s)
            #     sk = []
            #     states = []
            # else:
            #     if not self.tensor:
            #         s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            #         s = s0.repeat(self.samples, 1)
            #     else:
            #         s = state.repeat(self.samples, 1)

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
