import numpy as np


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
    
    v = np.array([0.025, 0.025], dtype=np.float32)
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
    return cost

class SimpleMaxDiff:
    def __init__(self, model_fn, state_dim, action_dim, horizon=10, samples=20, lam=1.0,
                 alpha=0.1, eps=0.3, gamma=1.0, cost_to_go=True, window=10, use_log_prob=True):
        self.H = horizon
        self.N = samples
        self.d_x = state_dim
        self.d_u = action_dim
        self.lam = lam
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.model_fn = model_fn
        self.cost_to_go = cost_to_go
        self.window = window
        self.use_log_prob = use_log_prob

        self.a = np.zeros((self.H, self.d_u))  # Nominal action plan

    def rollout(self, s0, state_cost_fn=state_cost):
        eta = np.random.normal(0, self.eps, size=(self.H, self.N, self.d_u))
        u = self.a[:, None, :] + eta  # shape: [H, N, d_u]

        s = np.zeros((self.H + 1, self.N, self.d_x))
        s[0] = s0[None, :]
        rewards = np.zeros((self.H, self.N))
        log_prob = np.zeros((self.H, self.N))

        for t in range(self.H):
            for i in range(self.N):
                s[t + 1, i] = self.model_fn(s[t, i], u[t, i]).reshape(self.d_x,)
                rewards[t, i] = -state_cost_fn(s[t + 1, i])  # cost-to-reward

                if self.use_log_prob:
                    # Log probability under isotropic Gaussian with std = eps
                    log_prob[t, i] = -0.5 * np.sum((eta[t, i] / self.eps) ** 2) \
                                     - 0.5 * self.d_u * np.log(2 * np.pi * self.eps**2)

        gammas = self.gamma ** np.arange(self.H)[:, None]

        if self.cost_to_go:
            # discounted cost-to-go at each time t
            discounted_rewards = rewards * gammas
            R = np.flip(np.cumsum(np.flip(discounted_rewards, axis=0), axis=0), axis=0)
            if self.use_log_prob:
                R += self.lam * log_prob  # shape: [H, N]
        else:
            R = np.sum(rewards * gammas, axis=0)  # shape: [N]
            if self.use_log_prob:
                R += self.lam * np.sum(log_prob, axis=0)

        # Compute entropy over a time window
        entropies = np.zeros((self.H, self.N))
        for t in range(self.H - self.window + 1):
            seg = s[t + 1:t + 1 + self.window]  # shape: [window, N, d_x]
            for i in range(self.N):
                cov = np.cov(seg[:, i, :].T) + 1e-6 * np.eye(self.d_x)
                entropies[t, i] = 0.5 * np.log(np.linalg.det(cov))

        if self.cost_to_go:
            entropy_bonus = np.mean(entropies, axis=0)
            R[0] += self.alpha * entropy_bonus
            R = R[0]  # shape: [N]
        else:
            R += self.alpha * np.mean(entropies, axis=0)

        # Normalize for numerical stability
        R -= np.max(R)
        w = np.exp(R / self.lam)
        w /= np.sum(w)

        # Update nominal action plan
        for t in range(self.H):
            self.a[t] += np.sum(w[:, None] * eta[t], axis=0)

        return np.clip(self.a[0],-0.5,0.5)  # Return first action in the plan

    def shift_plan(self):
        self.a[:-1] = self.a[1:]
        self.a[-1] = 0.0
