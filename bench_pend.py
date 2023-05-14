import gymnasium as gym
from envs import inverted_pendulum_v5
from envs import inverted_double_pendulum_v5
from envs import inverted_double_pendulum_v4_fixed
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 400_000
ALGO = TD3
EVAL_SEED = 1234
EVAL_FREQ = 1000

for run in range(0, RUNS):
    #env = gym.make('InvertedPendulum-v4')
    #eval_env = gym.make('InvertedPendulum-v4')
    #env = gym.wrappers.TimeLimit(inverted_pendulum_v5.InvertedPendulumEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(inverted_pendulum_v5.InvertedPendulumEnv(), max_episode_steps=1000)
    #env = gym.make('InvertedDoublePendulum-v4')
    #eval_env = gym.make('InvertedDoublePendulum-v4')
    #env = gym.wrappers.TimeLimit(inverted_double_pendulum_v5.InvertedDoublePendulumEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(inverted_double_pendulum_v5.InvertedDoublePendulumEnv(), max_episode_steps=1000)
    env = gym.wrappers.TimeLimit(inverted_double_pendulum_v4_fixed.InvertedDoublePendulumEnv(), max_episode_steps=1000)
    eval_env = gym.wrappers.TimeLimit(inverted_double_pendulum_v4_fixed.InvertedDoublePendulumEnv(), max_episode_steps=1000)

    #eval_path = 'results/InvertedPendulum_v4_TD3/run_' + str(run)
    #eval_path = 'results/InvertedPendulum_v5_TD3/run_' + str(run)
    #eval_path = 'results/InvertedDoublePendulum_v4_TD3/run_' + str(run)
    #eval_path = 'results/InvertedDoublePendulum_v5_TD3/run_' + str(run)
    eval_path = 'results/InvertedDoublePendulum_v4_fixed_TD3/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=10, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cpu')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

