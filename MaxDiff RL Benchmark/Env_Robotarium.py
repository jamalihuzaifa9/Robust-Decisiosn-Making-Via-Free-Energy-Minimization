import numpy as np
from rps.robotarium import Robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *

class RobotariumEnv:
    def __init__(self, goal_points, obs_points, time_step=0.033, max_steps=3000,show_figure=False):
        self.goal = goal_points[:2]
        self.obs = obs_points[:2]
        self.dt = time_step
        self.state_dim = 2
        self.action_dim = 2
        self.max_steps = max_steps

        self.r = None
        self.step_count = 0
        self.si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()
        _, self.uni_to_si_states = create_si_to_uni_mapping()
        self.x_si = None
        self.x_raw = None
        self.show_figure = show_figure

    def reset(self, init_state=None):
        if self.r is not None:
            self.r.call_at_scripts_end()
            del self.r

        if init_state is None:
            init_state = np.array([[0.0], [0.0], [0.0]])

        self.r = Robotarium(number_of_robots=1,
                            show_figure=self.show_figure,
                            initial_conditions=init_state,
                            sim_in_real_time=False)
        self.step_count = 0
        
        self.x_raw = self.r.get_poses()
        self.x_si = self.uni_to_si_states(self.x_raw)
        
        self.r.step()

        
        return self.x_si.flatten()

    def reward(self, state):
        v = np.array([0.025, 0.025])
        covar = np.diag(v)
        obs_penalty = 0
        for i in range(self.obs.shape[1]):
            diff = state - self.obs[:, i]
            exponent = -0.5 * diff.T @ np.linalg.inv(covar) @ diff
            denom = np.sqrt((2 * np.pi)**2 * np.linalg.det(covar))
            obs_penalty += 20 * np.exp(exponent) / denom

        wall_penalty = 5 * (
            np.exp(-0.5 * ((state[0] - (-1.5)) / 0.03)**2) +
            np.exp(-0.5 * ((state[0] - 1.5) / 0.03)**2) +
            np.exp(-0.5 * ((state[1] - 1.0) / 0.03)**2) +
            np.exp(-0.5 * ((state[1] - (-1.0)) / 0.03)**2)
        ) / (0.03 * np.sqrt(2 * np.pi))

        goal_term = 50 * np.sum((state - self.goal) ** 2)
        return -goal_term - obs_penalty - wall_penalty

    def _is_inside(self, state, bottom_left, top_right):
        return bottom_left[0] <= state[0] <= top_right[0] and bottom_left[1] <= state[1] <= top_right[1]

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -0.5, 0.5).reshape(2, 1)
        
        self.x_raw = self.r.get_poses()
        self.x_si = self.uni_to_si_states(self.x_raw)
        dxu = self.si_to_uni_dyn(action, self.x_raw)
        
        self.r.set_velocities(np.arange(1), dxu)
        self.r.step()

        # self.x_raw = self.r.get_poses()
        # self.x_si = self.uni_to_si_states(self.x_raw)
        state = self.x_si.flatten()

        reward = self.reward(state)

        done = False
        if self.step_count >= self.max_steps:
            done = True
        if np.linalg.norm(state - self.goal.flatten()) <= 0.2:
            done = True
            # reward += 1000
        if any([
            state[0] <= -1.5, state[0] >= 1.5,
            state[1] <= -1.0, state[1] >= 1.0,
            self._is_inside(state, [-0.25, -0.15], [0.15, 0.85]),
            self._is_inside(state, [-1.1, -1.0], [-0.51, -0.65])
        ]):
            done = True
            # reward = -2000

        return state, reward, done, {}

    def close(self):
        if self.r is not None:
            self.r.call_at_scripts_end()
            del self.r


