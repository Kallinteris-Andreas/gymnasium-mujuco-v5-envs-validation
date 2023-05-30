import gymnasium as gym
from envs import humanoid_v5
from envs import humanoid_v4_fixed
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from my_eval import EvalCallback

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 2_000_000
#ALGO = TD3
ALGO = PPO
EVAL_SEED = 1234
EVAL_FREQ = 5000
EVAL_ENVS = 20

for run in range(0, RUNS):
    #env = gym.make('Humanoid-v3')
    #eval_env = gym.make('Humanoid-v3')
    #env = gym.make('Humanoid-v4')
    #eval_env = gym.make('Humanoid-v4')
    #env = gym.wrappers.TimeLimit(humanoid_v4_fixed.HumanoidEnv(), max_episode_steps=1000)
    #eval_env = gym.wrappers.TimeLimit(humanoid_v4_fixed.HumanoidEnv(), max_episode_steps=1000)
    env = gym.wrappers.TimeLimit(humanoid_v5a.HumanoidEnv(), max_episode_steps=1000)
    eval_env = gym.wrappers.TimeLimit(humanoid_v5a.HumanoidEnv(), max_episode_steps=1000)

    #eval_path = 'results/Humanoid_v4_PPO/run_' + str(run)
    #eval_path = 'results/Humanoid_v4_fixed_reward_PPO/run_' + str(run)
    #eval_path = 'results/Humanoid_v4_fixed_reward_on_eval_PPO/run_' + str(run)
    eval_path = 'results/Humanoid_v5s_PPO/run_' + str(run)
    #eval_path = 'results/Humanoid_v3_TD3/run_' + str(run)
    #eval_path = 'results/Humanoid_v4_TD3/run_' + str(run)
    #eval_path = 'results/Humanoid_v4_fixed_reward_TD3/run_' + str(run)
    #eval_path = 'results/Humanoid_v5a_TD3/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

