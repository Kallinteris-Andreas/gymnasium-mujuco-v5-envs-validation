import gymnasium as gym
import numpy as np
import copy
from gymnasium.experimental.wrappers import RescaleActionV0

from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from my_eval import EvalCallback

RUNS = 1  # Number of Statistical Runs
TOTAL_TIME_STEPS = 4_000_000
#ALGO = TD3
ALGO = PPO
EVAL_SEED = 1234
EVAL_FREQ = 5000
EVAL_ENVS = 20

for run in range(0, RUNS):
    #env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/casie-scene.xml', healthy_z_range=(1.0, 2.1), ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    #env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/anybotics_anymal_b/scene.xml', healthy_z_range=(0.48, 0.68), ctrl_cost_weight=0, contact_cost_weight=0, render_mode=None)
    env = RescaleActionV0(env, min_action=-1, max_action=1)
    eval_env = copy.deepcopy(env)

    #eval_path = 'results/cassie/run_' + str(run)
    eval_path = 'results/op3/run_' + str(run)
    #3eval_path = 'results/anymal_b/run_' + str(run)

    eval_callback = EvalCallback(eval_env, seed=EVAL_SEED, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)

    model = ALGO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

