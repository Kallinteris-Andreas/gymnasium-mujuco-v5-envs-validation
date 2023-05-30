import gymnasium as gym
import envs.reacher_v5 as reacher_v5 
import numpy as np

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 400_000
#ALGO = TD3
ALGO = PPO
EVAL_SEED = 1234
EVAL_FREQ = 1000

for run in range(0, RUNS):
    #env = gym.make('Reacher-v4')
    #eval_env = gym.make('Reacher-v4')
    env = gym.wrappers.TimeLimit(reacher_v5.ReacherEnv(), max_episode_steps=50)
    eval_env = gym.wrappers.TimeLimit(reacher_v5.ReacherEnv(), max_episode_steps=50)

    #eval_path = 'results/reacher_v4_TD3/run_' + str(run)
    eval_path = 'results/reacher_v5_TD3/run_' + str(run)
    #eval_path = 'results/reacher_v4_PPO/run_' + str(run)
    #eval_path = 'results/reacher_v5_PPO/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=10, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

