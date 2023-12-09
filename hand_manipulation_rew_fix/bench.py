import gymnasium
import numpy as np
import copy
import os
import argparse

import stable_baselines3
from stable_baselines3 import TD3, PPO, A2C, SAC, DQN
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback
from gymnasium.wrappers import TransformReward, PassiveEnvChecker, OrderEnforcing, TimeLimit
from stable_baselines3.common.monitor import Monitor
from typing import Tuple

import reacher_v5
import reacher_v5_fix_rew
import pusher_v5
import pusher_v5_fix_rew


def make_env(env_id: str) -> Tuple[gymnasium.Env, gymnasium.Env]:
    if env_id == "Reacher-v5":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(reacher_v5.ReacherEnv())), max_episode_steps=50)
    elif env_id == "Reacher-v5_fix_rew":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(reacher_v5_fix_rew.ReacherEnv())), max_episode_steps=50)
    elif env_id == "Pusher-v5":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(pusher_v5.PusherEnv())), max_episode_steps=100)
    elif env_id == "Pusher-v5_fix_rew":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(pusher_v5_fix_rew.PusherEnv())), max_episode_steps=100)
    else:
        assert False
        return gymnasium.make(env_id, render_mode=None)

def make_eval_env(env_id: str) -> Tuple[gymnasium.Env, gymnasium.Env]:
    if env_id == "Reacher-v5" or env_id == "Reacher-v5_fix_rew":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(reacher_v5_fix_rew.ReacherEnv())), max_episode_steps=50)
    elif env_id == "Pusher-v5" or env_id == "Pusher-v5_fix_rew":
        return TimeLimit(OrderEnforcing(PassiveEnvChecker(pusher_v5_fix_rew.PusherEnv())), max_episode_steps=100)
    else:
        assert False
        return gymnasium.make(env_id, render_mode=None)


def make_model(algorithm: str):
    match args.algo:
        case "TD3":  # note does not work with Discrete
            return TD3("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)
        case "PPO":
            return PPO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "SAC":  # note does not work with Discrete
            return SAC("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)
        case "A2C":
            return A2C("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "DQN":
            return DQN("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100)


parser = argparse.ArgumentParser()
parser.add_argument("--algo", default="DQN")
parser.add_argument("--env_id", default="CartPole-v1")
parser.add_argument("--starting_run", default=0, type=int)
args = parser.parse_args()

RUNS = 10  # Number of Statistical Runs
TOTAL_TIME_STEPS = 200_000
EVAL_SEED = 1234
EVAL_FREQ = 500
EVAL_ENVS = 20


for run in range(args.starting_run, RUNS):
    env = Monitor(make_env(args.env_id))
    eval_env = Monitor(make_eval_env(args.env_id))
    eval_path = f"results/{args.env_id}/{args.algo}/run_" + str(run)

    assert not os.path.exists(eval_path)

    #eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True, seed=EVAL_SEED)


    model = make_model(args.algo)
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

