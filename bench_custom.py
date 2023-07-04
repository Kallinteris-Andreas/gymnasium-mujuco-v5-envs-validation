import gymnasium as gym
from envs import ant_v4_fixed
from envs import ant_v5a
from envs import ant_v5
import numpy as np
import copy
import os
import argparse

import stable_baselines3
from stable_baselines3 import TD3, PPO, A2C, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
#from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.experimental.wrappers import RescaleActionV0
from my_eval import EvalCallback

assert stable_baselines3.__version__ == '2.0.0a5'


RUNS = 1  # Number of Statistical Runs
TOTAL_TIME_STEPS = 2_000_000
parser = argparse.ArgumentParser()
parser.add_argument("--algo")
args = parser.parse_args()
EVAL_SEED = 1234
EVAL_FREQ = 5000
EVAL_ENVS = 20

match args.algo:
    case "TD3":
        ALGO = TD3
    case "PPO":
        ALGO = PPO
    case "SAC":
        ALGO = SAC
    case "A2C":
        ALGO = A2C
ALGO_NAME = str(ALGO).split('.')[-1][:-2]

n_actions = 8
#action_noise  = None
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.10 * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.00 * np.ones(n_actions))
print(f"Using aglorithm: {ALGO_NAME}")

for run in range(0, RUNS):
    #env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/casie-scene.xml', healthy_z_range=(1.0, 2.1), ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    #env = gym.make('Humanoid-v5', xml_file='~/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    #env = gym.make('Ant-v5', xml_file='~/mujoco_menagerie/anybotics_anymal_b/scene.xml', include_cfrc_ext_in_observation=False, healthy_z_range=(0.48, 0.68), ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    env = gym.make('Ant-v5', xml_file='~/mujoco_menagerie/unitree_go1/scene.xml', include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, healthy_z_range=(0.295, 1), frame_skip=25, render_mode=None)
    env = RescaleActionV0(env, min_action=-1, max_action=1)
    eval_env = copy.deepcopy(env)

    #eval_path = 'results/cassie/run_' + str(run)
    eval_path = f'results/op3/{ALGO_NAME}/run_' + str(run)
    #eval_path = f"results/anymal_b/{ALGO_NAME}/run_" + str(run)
    #eval_path = f"results/unitree_go1/{ALGO_NAME}/run_" + str(run)

    assert not os.path.exists(eval_path)

    #eval_callback = None
    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

