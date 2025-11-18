import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    USE_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    USE_GYMNASIUM = False

# Use new mujoco instead of mujoco_py
try:
    import mujoco
    USE_NEW_MUJOCO = True
except ImportError:
    try:
        import mujoco_py
        USE_NEW_MUJOCO = False
    except ImportError:
        print("WARNING: Neither mujoco nor mujoco_py found. Some features may not work.")
        USE_NEW_MUJOCO = None


class AntContactsWrapper(gym.Wrapper):
    """Wrapper for Ant environment that adds contact information and task-specific rewards."""
    
    def __init__(self, env, task='motion', contact_threshold=0.0, **kwargs):
        super().__init__(env)
        self.task = task
        self.contact_threshold = contact_threshold
        
        # Get original observation space
        if USE_GYMNASIUM:
            orig_obs_space = env.observation_space
        else:
            orig_obs_space = env.observation_space
            
        # Add contact forces to observation space
        # Ant has 4 legs with 2 contact points each = 8 contact forces
        num_contacts = 8
        low = np.concatenate([orig_obs_space.low, np.zeros(num_contacts)])
        high = np.concatenate([orig_obs_space.high, np.full(num_contacts, np.inf)])
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.num_contacts = num_contacts
        
    def reset(self, **kwargs):
        """Reset the environment."""
        if USE_GYMNASIUM:
            obs, info = self.env.reset(**kwargs)
            contact_forces = self._get_contact_forces()
            augmented_obs = np.concatenate([obs, contact_forces])
            return augmented_obs, info
        else:
            obs = self.env.reset(**kwargs)
            contact_forces = self._get_contact_forces()
            augmented_obs = np.concatenate([obs, contact_forces])
            return augmented_obs
    
    def step(self, action):
        """Step the environment."""
        if USE_GYMNASIUM:
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            obs, reward, done, info = self.env.step(action)
            
        # Get contact forces
        contact_forces = self._get_contact_forces()
        augmented_obs = np.concatenate([obs, contact_forces])
        
        # Modify reward based on task
        if self.task == 'motion':
            # Encourage forward motion
            reward = reward
        elif self.task == 'contact':
            # Penalize excessive contact forces
            contact_penalty = np.sum(np.abs(contact_forces)) * 0.01
            reward = reward - contact_penalty
        elif self.task == 'upright':
            # Reward staying upright
            if USE_GYMNASIUM:
                # Get z-coordinate from unwrapped environment
                torso_height = self.env.unwrapped.data.qpos[2]
            else:
                torso_height = self.env.unwrapped.sim.data.qpos[2]
            upright_reward = torso_height * 0.1
            reward = reward + upright_reward
            
        if USE_GYMNASIUM:
            return augmented_obs, reward, terminated, truncated, info
        else:
            return augmented_obs, reward, done, info
    
    def _get_contact_forces(self):
        """Extract contact forces from MuJoCo simulation."""
        if USE_NEW_MUJOCO is None:
            # No MuJoCo available, return zeros
            return np.zeros(self.num_contacts)
            
        try:
            if USE_NEW_MUJOCO:
                # New mujoco API
                unwrapped = self.env.unwrapped
                
                # Access MuJoCo data
                if hasattr(unwrapped, 'data'):
                    data = unwrapped.data
                    model = unwrapped.model
                else:
                    # For gymnasium environments
                    data = unwrapped._data if hasattr(unwrapped, '_data') else unwrapped.data
                    model = unwrapped._model if hasattr(unwrapped, '_model') else unwrapped.model
                
                # Get contact forces
                contact_forces = []
                
                # Iterate through active contacts
                for i in range(data.ncon):
                    contact = data.contact[i]
                    # Get contact force magnitude
                    force = np.linalg.norm(contact.force[:3])  # Use first 3 components (xyz forces)
                    contact_forces.append(force)
                
                # Pad or truncate to expected number of contacts
                if len(contact_forces) < self.num_contacts:
                    contact_forces.extend([0.0] * (self.num_contacts - len(contact_forces)))
                else:
                    contact_forces = contact_forces[:self.num_contacts]
                    
                return np.array(contact_forces, dtype=np.float32)
                
            else:
                # Old mujoco_py API
                sim = self.env.unwrapped.sim
                contact_forces = []
                
                for i in range(sim.data.ncon):
                    contact = sim.data.contact[i]
                    force = np.linalg.norm(sim.data.cfrc_ext[contact.geom1][:3])
                    contact_forces.append(force)
                
                # Pad or truncate
                if len(contact_forces) < self.num_contacts:
                    contact_forces.extend([0.0] * (self.num_contacts - len(contact_forces)))
                else:
                    contact_forces = contact_forces[:self.num_contacts]
                    
                return np.array(contact_forces, dtype=np.float32)
                
        except Exception as e:
            print(f"Warning: Could not extract contact forces: {e}")
            return np.zeros(self.num_contacts)


class MultiAntWrapper(gym.Wrapper):
    """Wrapper for training multiple Ant agents simultaneously."""
    
    def __init__(self, env, num_agents=1):
        super().__init__(env)
        self.num_agents = num_agents
        
        # Multiply observation and action spaces
        orig_obs_space = env.observation_space
        orig_action_space = env.action_space
        
        obs_dim = orig_obs_space.shape[0]
        action_dim = orig_action_space.shape[0]
        
        self.observation_space = spaces.Box(
            low=np.tile(orig_obs_space.low, num_agents),
            high=np.tile(orig_obs_space.high, num_agents),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.tile(orig_action_space.low, num_agents),
            high=np.tile(orig_action_space.high, num_agents),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        """Reset all agents."""
        if USE_GYMNASIUM:
            obs, info = self.env.reset(**kwargs)
            multi_obs = np.tile(obs, self.num_agents)
            return multi_obs, info
        else:
            obs = self.env.reset(**kwargs)
            multi_obs = np.tile(obs, self.num_agents)
            return multi_obs
    
    def step(self, action):
        """Step all agents (using mean action for now)."""
        # Simple implementation: use mean of all agent actions
        single_action = np.mean(action.reshape(self.num_agents, -1), axis=0)
        
        if USE_GYMNASIUM:
            obs, reward, terminated, truncated, info = self.env.step(single_action)
            multi_obs = np.tile(obs, self.num_agents)
            multi_reward = reward * self.num_agents
            return multi_obs, multi_reward, terminated, truncated, info
        else:
            obs, reward, done, info = self.env.step(single_action)
            multi_obs = np.tile(obs, self.num_agents)
            multi_reward = reward * self.num_agents
            return multi_obs, multi_reward, done, info