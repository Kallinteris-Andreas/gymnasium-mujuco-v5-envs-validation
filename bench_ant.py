import gymnasium as gym
from envs import ant_v4_fixed
from envs import ant_v5a
from envs import ant_v5
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 2_000_000
ALGO = TD3
EVAL_SEED = 1234
EVAL_FREQ = 5000

for run in range(0, RUNS):
    #env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=False), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(ant_v4_fixed.AntEnv(use_contact_forces=False), max_episode_steps=1000)
    #env = gym.wrappers.TimeLimit(ant_v5a.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(ant_v5a.AntEnv(use_contact_forces=True), max_episode_steps=1000)
    env = gym.wrappers.TimeLimit(ant_v5.AntEnv(), max_episode_steps=1000)
    eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(), max_episode_steps=1000)

    #eval_path = 'results/ant_v4_fixed_with_ctn_TD3/run_' + str(run)
    #eval_path = 'results/ant_v4_fixed_without_ctn_TD3/run_' + str(run)
    #eval_path = 'results/ant_v5a_TD3/run_' + str(run)
    eval_path = 'results/ant_v5_TD3/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=10, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

