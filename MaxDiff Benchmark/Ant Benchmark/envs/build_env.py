#!/usr/bin/env python3

__all__=['env_list','getlist','build_env']

''' load environments '''
# Try new gymnasium first, fallback to legacy gym
try:
    import gymnasium as gym
    USE_GYMNASIUM = True
    print("Using gymnasium (MuJoCo 3.x)")
    
    env_list = {
        'HalfCheetahEnv': 'HalfCheetah-v4',
        'AntEnv': 'Ant-v4',
        'SwimmerEnv_v3': 'Swimmer-v4',
        'AntEnv_v3': 'Ant-v4',
        'AntEnv_v4': 'Ant-v4',
        'HalfCheetahEnv_v3': 'HalfCheetah-v4',
        'HalfCheetahEnv_v4': 'HalfCheetah-v4',
        'SwimmerEnv_v4': 'Swimmer-v4',
    }
    
except ImportError:
    import gym
    from gym.envs.mujoco.swimmer_v3 import SwimmerEnv
    from gym.envs.mujoco.ant_v3 import AntEnv
    from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
    USE_GYMNASIUM = False
    print("Using legacy gym (MuJoCo 2.x)")
    
    env_list = {
        'HalfCheetahEnv': gym.envs.mujoco.HalfCheetahEnv,
        'AntEnv': gym.envs.mujoco.AntEnv,
        'SwimmerEnv_v3': SwimmerEnv,
        'AntEnv_v3': AntEnv,
        'HalfCheetahEnv_v3': HalfCheetahEnv,
    }

def getlist():
    out_str = ''
    for env_name in env_list.keys():
        out_str += env_name + '\n'
    return out_str


''' build environment '''
from .normalized_actions import NormalizedActions
import os

def build_env(args, config, device):
    # initialize environment
    env_name = args.env
    traj = None
    viewer = None
    print('PointMass' in args.env,'PointMass', args.env)
    
    if 'PointMass' in args.env:
        from envs.pointmass_lib import env_list_pm, TrajBuffer, PM_Viewer
        env = env_list_pm[env_name](beta=args.beta, start_mode=args.start_mode, **config['env'])
        action_dim = env.num_actions
        state_dim = env.num_states
        # set up plotting
        for name in ['alpha','horizon','env']:
            args.__dict__[name] = 'n/a'
            if name in config.keys():
                args.__dict__[name] = config[name]
            elif 'planner' in config.keys():
                if name in config['planner'].keys():
                    args.__dict__[name] = config['planner'][name]
            elif ('sac' in args.method) and name == 'alpha':
                args.__dict__[name] = config['reward_scale']
        traj = TrajBuffer(args, capacity=11)
        if args.render:
            viewer = PM_Viewer(args)
    else:
        env_args = {}
        
        # Handle xml_file argument
        if 'xml_file' in args.__dict__.keys():
            env_args['xml_file'] = args.xml
            
        if not('done_util' in args.__dict__.keys()):
            args.done_util = True
        
        # Handle v3/v4 specific settings
        if args.v3 or 'v4' in env_name:
            if USE_GYMNASIUM:
                # Gymnasium v4 parameters
                env_args['exclude_current_positions_from_observation'] = False
                env_args['terminate_when_unhealthy'] = args.done_util
            else:
                # Legacy gym v3 parameters
                env_args['exclude_current_positions_from_observation'] = False
                env_args['terminate_when_unhealthy'] = args.done_util
            
            # Handle specific environments
            if env_name == 'MultiSwimmer':
                args.env = 'SwimmerEnv_v4' if USE_GYMNASIUM else 'SwimmerEnv_v3'
                env_name = args.env
                multi_swimmer_wrapper = True
                
            elif 'SwimmerEnv' in env_name:
                if args.mod_weight.lower() == 'light':
                    xml = os.path.abspath('./envs/light_swimmer.xml')
                    env_args['xml_file'] = xml
                    config['name_mod'] = '_Light' + config['name_mod']
                    print('loading light model')
                else:
                    print('loading default model')
                    
            elif env_name == 'MultiAnt':
                env_args['ctrl_cost_weight'] = 0.
                env_args['contact_cost_weight'] = 0.
                if USE_GYMNASIUM:
                    env_args['healthy_reward'] = 0.
                else:
                    env_args['healthy_reward'] = 0.
                args.env = 'AntEnv_v4' if USE_GYMNASIUM else 'AntEnv_v3'
                env_name = args.env
                multi_ant_wrapper = True
                
            elif 'AntEnv' in env_name:
                env_args['ctrl_cost_weight'] = 0.
                env_args['contact_cost_weight'] = 0.
                if USE_GYMNASIUM:
                    env_args['healthy_reward'] = 0.
                else:
                    env_args['healthy_reward'] = 0.
                    
            elif 'HalfCheetahEnv' in env_name:
                pass
            else:
                raise ValueError(f'invalid env name: {env_name}')
        
        # Create environment based on gymnasium or gym
        if USE_GYMNASIUM:
            # For gymnasium, use make with string ID
            if args.render:
                env_args['render_mode'] = 'human'
            
            # Get the environment ID from env_list
            env_id = env_list[env_name] if env_name in env_list else env_name
            
            try:
                base_env = gym.make(env_id, **env_args)
                env = NormalizedActions(base_env)
            except Exception as e:
                print(f"Error creating environment: {e}")
                # Try without render_mode if it fails
                if 'render_mode' in env_args:
                    del env_args['render_mode']
                base_env = gym.make(env_id, **env_args)
                env = NormalizedActions(base_env)
        else:
            # For legacy gym, use class instantiation
            env_args['render'] = args.render
            try:
                env = NormalizedActions(env_list[env_name](**env_args))
            except TypeError as err:
                print(f"Warning: {err}")
                del env_args['render']
                try:
                    env = NormalizedActions(env_list[env_name](**env_args))
                except TypeError as err:
                    print(f"Warning: {err}")
                    del env_args['terminate_when_unhealthy']
                    env = NormalizedActions(env_list[env_name](**env_args))

        # Apply task-specific wrappers
        if 'AntEnv' in env_name:
            from envs.wrappers import AntContactsWrapper
            env = AntContactsWrapper(env, **config['task_info'])
            env_name = env_name + '_' + config['task_info']['task']

        # Get action and state dimensions
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]

    # Set random seed - different methods for gymnasium vs gym
    if USE_GYMNASIUM:
        env.reset(seed=args.seed)
    else:
        env.seed(args.seed)
        
    return env, env_name, action_dim, state_dim, traj, viewer