import gymnasium as gym
from envs import ant_v4_fixed
from envs import ant_v5a
from envs import ant_v5
import numpy as np
import copy
import os

import stable_baselines3
from stable_baselines3 import TD3, PPO, A2C, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback

assert stable_baselines3.__version__ == '2.0.0a5'

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 2_000_000
ALGO = TD3
#ALGO = PPO
#ALGO = SAC
#ALGO = A2C  # Note: sucks for Ant
ALGO_NAME = str(ALGO).split('.')[-1][:-2]
EVAL_SEED = 1234
EVAL_FREQ = 5000

n_actions = 8
#action_noise  = None
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.10 * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.00 * np.ones(n_actions))
print(f"Using aglorithm: {ALGO_NAME}")

for run in range(0, RUNS):
    #env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=False), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5a.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5a.AntEnv(healthy_reward=0.0), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5.AntEnv(healthy_reward=0.1), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5.AntEnv(healthy_reward=0.0), max_episode_steps=1000)
    env = gym.wrappers.TimeLimit(ant_v5.AntEnv(include_cfrc_ext_in_observation=False), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5.AntEnv(), max_episode_steps=1000)
    eval_env = copy.deepcopy(env)

    #eval_path = f'results/ant_v4_fixed_with_ctn_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v4_fixed_without_ctn_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5a_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_policy_std_0.1_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_policy_std_0.25_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_policy_std_1_{ALGO_NAME}/run_' + str(run)
    eval_path = f'results/ant_v5_without_ctn_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_hr_0.1_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5_hr_0_{ALGO_NAME}/run_' + str(run)
    #eval_path = f'results/ant_v5a_hr_0_{ALGO_NAME}/run_' + str(run)
    #eval_path = 'results/temp' + str(run)

    assert not os.path.exists(eval_path)

    #eval_callback = None
    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=10, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

