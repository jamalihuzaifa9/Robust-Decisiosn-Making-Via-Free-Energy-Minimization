#!/usr/bin/env python3

import yaml
from termcolor import cprint
import time
from utils import get_duration

import torch
import numpy as np
import random
import pickle
import os
from pathlib import Path

# local imports
from envs import build_env

import argparse

parser = argparse.ArgumentParser()
# these params are the learned model/policy to load
parser.add_argument('--env',   type=str,   default='AntEnv_v4', help="PointMass2D_DoubleIntEnv,SwimmerEnv_v3, AntEnv_v3, etc.")
parser.add_argument('--method', type=str, default='maxdiff', help='maxdiff, mppi, or sac_orig')
parser.add_argument('--seed', type=int, default=1, help='any positive integer')
parser.add_argument('--done_util', dest='done_util', action='store_true', help='ends epoch with done signal from environment')
parser.add_argument('--no_done_util', dest='done_util', action='store_false', help='ignores done signal from environment and runs for max_steps')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true',help='render each epoch in figure window as running')
parser.add_argument('--no_render', dest='render', action='store_false',help='run offline / without showing plots')
parser.set_defaults(render=False)
parser.add_argument('--cpu', dest='cpu', action='store_true',help='only use CPU')
parser.add_argument('--no_cpu', dest='cpu', action='store_false',help='try to use GPU if available')
parser.set_defaults(cpu=False)
parser.add_argument('--mod', type=str, default='_H2',help="end of file name for specfic config after beta (e.g. '_H40' or '_H40_alpha5')")
parser.add_argument('--iters',   type=int,   default='10',help="how many test iterations to run")
parser.add_argument('--final_only', dest='final_only', action='store_true',help='only test final saved model')
parser.add_argument('--all_frames', dest='final_only', action='store_false',help='save checkpoints and final saved model')
parser.set_defaults(final_only=True)
# this specifies the xml to load
parser.add_argument('--physics_model_eval', type=str, default='orig',help="specify alternate xml file for testing (gym envs only, can be same or different from training)")
parser.add_argument('--base_dir',   type=str,   default='./data/',help="directory where enviroment folder with data")
parser.add_argument('--start_mode', type=str, default='one_corner',help="(PointMass envs only) one_corner, four_corners, circle10, random")
parser.add_argument('--beta', type=float, default=0.01, help='(PointMass envs only) weights pointmass control matrix (e.g. 1.0, 0.1, 0.01, 0.001) ')
parser.add_argument('--use_raw_env', dest='use_raw_env', action='store_true', help='use raw Ant-v4 without custom wrappers')
parser.set_defaults(use_raw_env=True)

args = parser.parse_args()
cprint(args,'cyan') 
args.v3 = 'v3' in args.env
args.pointmass = 'PointMass' in args.env

# added to stop rendering when exiting
from signal import signal, SIGINT
from sys import exit

if args.pointmass:
    def end_test():
        env.close()
        try:
            print('saving data set')
            pickle.dump(rewards, open(state_dict_path + args.start_mode + '_final_eval_reward_data' + '.pkl', 'wb'))
        except NameError:
            print('no rewards to save, closing simulation')
        fig_path = state_dict_path + args.start_mode + "_eval_" + "final_fig" 
        if args.render:
            viewer.save(fig_path)
        else:
            try:
                traj.save_fig(fig_path + '.svg')
            except:
                traj.save_buff(fig_path + '.pkl')

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    args.render = False
    print('Exiting gracefully')
    exit(0)

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    if args.pointmass:
        pm_mod = '_beta'+ '{:0.0e}'.format(args.beta).replace('+','').replace('-','_')
        args.mod = pm_mod + args.mod

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Convert base_dir to absolute path relative to script location
    if not os.path.isabs(args.base_dir):
        args.base_dir = str(script_dir / args.base_dir)
    
    # load config
    state_dict_path = os.path.join(args.base_dir, args.method, 
                                   args.env + args.mod, 
                                   f'seed_{args.seed}/')
    print(f"Looking for data in: {state_dict_path}")

    base_method = args.method[:3]
    config_path = os.path.join(state_dict_path, 'config.yaml')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at: {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"\nPlease ensure the data directory structure is correct.")
        print(f"Expected path: {config_path}")
        exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'H_sequence' in config.keys():
        if 'horizon' in config['H_sequence'].keys():
            config['planner']['horizon'] = config['H_sequence']['horizon'][-1]
    if 'alpha_sequence' in config.keys():
        if 'alpha' in config['alpha_sequence'].keys():
            config['planner']['alpha'] = config['alpha_sequence']['alpha'][-1]

    # set seeds / torch config
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set torch config
    device ='cpu'
    if not args.cpu:
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            device  = 'cuda:0'
            print('Using GPU Accel')
        else:
            args.cpu = True

    # Check if using gymnasium
    try:
        import gymnasium as gym
        USE_GYMNASIUM = True
        print("✅ Using gymnasium (MuJoCo 3.x)")
    except ImportError:
        import gym
        USE_GYMNASIUM = False
        print("✅ Using legacy gym (MuJoCo 2.x)")

    if args.use_raw_env and ('AntEnv' in args.env):
        
        from envs.normalized_actions import NormalizedActions
        
        # Force Ant-v4 specifically (not v3 or v5)
        ant_env_id = 'Ant-v4'
        
        print(f"📦 Creating {ant_env_id} environment...")
        
        # Use same settings as original build_env - CRITICAL: Set rewards to 0!
        env_kwargs = {
            'exclude_current_positions_from_observation': False,
            'terminate_when_unhealthy': args.done_util,
            # Set all reward shaping to 0 (matching original build_env)
            'ctrl_cost_weight': 0.,
            'contact_cost_weight': 0.,
            'healthy_reward': 0.,
        }
        
        # Add render mode for gymnasium
        if USE_GYMNASIUM and args.render:
            env_kwargs['render_mode'] = 'human'
        
        try:
            # Create base environment
            base_env = gym.make(ant_env_id, **env_kwargs)
            
            # Check observation shape before wrapping
            if USE_GYMNASIUM:
                test_obs, _ = base_env.reset()
            else:
                test_obs = base_env.reset()
            
            print(f"🔍 Base Ant-v4 observation shape: {test_obs.shape}")
            print(f"   Observation length: {len(test_obs)}")
            
            # Wrap only with NormalizedActions (no AntContactsWrapper)
            env = NormalizedActions(base_env)
            
            env_name = f'{args.env}_orig'
            action_dim = env.action_space.shape[0]
            state_dim = env.observation_space.shape[0]
            traj = None
            viewer = None
            
            # Set random seed
            if USE_GYMNASIUM:
                env.reset(seed=args.seed)
            else:
                env.seed(args.seed)
            
            # Verify actual observation after wrapping
            if USE_GYMNASIUM:
                actual_obs, _ = env.reset()
            else:
                actual_obs = env.reset()
            
            actual_state_dim = len(actual_obs)
            
            print(f"\n✅ Environment created successfully!")
            print(f"   - Environment: {ant_env_id}")
            print(f"   - State dim (space): {state_dim}")
            print(f"   - State dim (actual): {actual_state_dim}")
            print(f"   - Action dim: {action_dim}")
            print(f"   - Total input: {actual_state_dim} + {action_dim} = {actual_state_dim + action_dim}")
            
            # Verify dimensions match pretrained model expectations
            expected_state_dim = 29  # Standard Ant-v4 with exclude_current_positions=False
            
            if actual_state_dim != expected_state_dim:
                print(f"\n❌ DIMENSION MISMATCH!")
                print(f"   Expected: {expected_state_dim}")
                print(f"   Got: {actual_state_dim}")
                print(f"\n   Model expects: {expected_state_dim} state + 8 action = 37 total")
                print(f"   Current setup: {actual_state_dim} state + {action_dim} action = {actual_state_dim + action_dim} total")
                print(f"\n   This WILL cause model loading to fail!")
                print(f"\n💡 Try using --no_use_raw_env to use build_env() instead")
                exit(1)
            else:
                print(f"✅ Dimension check PASSED! State dim = {actual_state_dim}")
            
            print("="*70)
            
        except Exception as e:
            print(f"❌ Error creating environment: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to build_env...")
            args.use_raw_env = False
    
    # Fallback to original build_env if not using raw env or not Ant
    if not args.use_raw_env or ('AntEnv' not in args.env):
        print("📦 Using build_env() for environment creation...")
        args.mod_weight = args.physics_model_eval
        env, env_name, action_dim, state_dim, traj, viewer = build_env(args, config, device)
        
        print(f"✅ Environment built:")
        print(f"   - State dim: {state_dim}")
        print(f"   - Action dim: {action_dim}")
    
    cprint(env,'green')
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # load models / policies / controllers
    if base_method == 'sac':
        from sac_lib import PolicyNetwork
        policy_net = PolicyNetwork(state_dim, action_dim, config['hidden_dim'], device=device).to(device)
    else:
        from mpc_lib import Model
        model_kwargs = {'model_layers':config['model_layers'],'model_AF':config['model_activation_fun'],
                        'reward_layers':config['reward_layers'],'reward_AF':config['reward_activation_fun']}
        model = Model(state_dim, action_dim, **model_kwargs).to(device)
        if base_method == 'mpp':
            from mpc_lib import PathIntegral
            planner = PathIntegral(model, device=device, **config['planner'])
        elif base_method == 'max':
            from mpc_lib import MaxDiff
            planner = MaxDiff(model, device=device, **config['planner'])

    start_time = time.time()
    # main simulation loop
    max_steps = config['max_steps']
    rewards = []

    if args.final_only:
        test_frames = ['final']
    else:
        test_frames = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]

    for test_frame in test_frames:
        # load model/policy for particular frame
        model_path = os.path.join(state_dict_path, f'model_{test_frame}.pt')
        policy_path = os.path.join(state_dict_path, f'policy_{test_frame}.pt')
        
        if base_method == 'mpp':
            if not os.path.exists(model_path):
                print(f"WARNING: Model file not found: {model_path}")
                continue
            print(f"📥 Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif base_method == 'sac':
            if not os.path.exists(policy_path):
                print(f"WARNING: Policy file not found: {policy_path}")
                continue
            print(f"📥 Loading policy from: {policy_path}")
            policy_net.load_state_dict(torch.load(policy_path, map_location=device))
        elif base_method == 'max':
            if not os.path.exists(model_path):
                print(f"WARNING: Model file not found: {model_path}")
                continue
            print(f"📥 Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError('method not found')

        print(f"\n{'='*70}")
        print(f"🧪 Testing frame: {test_frame}")
        print(f"{'='*70}")

        # test for fixed number of iters
        for ep_num in range(args.iters):
            # Handle gymnasium vs gym reset differences
            if USE_GYMNASIUM:
                state, info = env.reset()
            else:
                state = env.reset()
                
            if base_method == 'sac':
                action = policy_net.get_action(state.copy())
            else:
                planner.reset()
                action = planner(state.copy())

            episode_reward = 0
            states = []
            for step in range(max_steps):
                if base_method == 'sac':
                    action = policy_net.get_action(state.copy())
                else:
                    action = planner(state.copy())
                
                # Handle gymnasium vs gym step differences
                step_result = env.step(action.copy())
                if USE_GYMNASIUM:
                    state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, info = step_result
                
                if args.pointmass:
                    states.append(state)
                else:
                    if args.render:
                        try:
                            env.render()  # gymnasium render() takes no arguments
                        except TypeError:
                            try:
                                env.render(mode="human")  # legacy gym
                            except:
                                pass
                            
                episode_reward += reward

                if args.done_util:
                    if done:
                        break
                        
            if args.pointmass:
                if args.render:
                    viewer.render(states, ep_num)
                traj.push(states, ep_num)
            step += 1
            if ep_num % max(1, args.iters//10) == 0:
                get_duration(start_time)
            print(f"{test_frame} | Episode {ep_num+1}/{args.iters} | Reward: {episode_reward:.2f} | Steps: {step} | Final x: {state[0]:.2f}")
            rewards.append([test_frame, episode_reward, ep_num, step])
    
    env.close()
    print('\n' + '='*70)
    print('💾 Saving reward log...')
    
    if args.pointmass:
        end_test()
    else:
        reward_log_path = os.path.join(state_dict_path, f'{args.physics_model_eval}_eval_reward_log.pkl')
        pickle.dump(rewards, open(reward_log_path, 'wb'))
        print(f"✅ Saved reward log to: {reward_log_path}")
    
    # Print summary statistics
    if len(rewards) > 0:
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array[:, 1])
        std_reward = np.std(rewards_array[:, 1])
        mean_steps = np.mean(rewards_array[:, 3])
        
        print('\n' + '='*70)
        print('📊 EVALUATION SUMMARY')
        print('='*70)
        print(f"Total episodes: {len(rewards)}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean steps: {mean_steps:.2f}")
        print(f"Best reward: {np.max(rewards_array[:, 1]):.2f}")
        print(f"Worst reward: {np.min(rewards_array[:, 1]):.2f}")
        print('='*70)