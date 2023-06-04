import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import time
from envs import inverted_double_pendulum_v4_fixed
from envs import ant_v5


import numpy as np

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

#eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(use_contact_forces=False, render_mode='human'), max_episode_steps=1000)
#model = TD3.load(path='/home/master-andreas/rl/project/results/ant_v4_fixed_without_ctn_TD3/run_0/best_model.zip', env=eval_env, device='cpu')

#eval_env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/agility_cassie/scene.xml', healthy_z_range=(1.0, 2.1), render_mode='human')
#eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = PPO.load(path='/home/master-andreas/rl/project/results/cassie/run_0/best_model.zip', env=eval_env, device='cpu')

eval_env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, contact_cost_weight=0, render_mode='human')
eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
model = PPO.load(path='/home/master-andreas/rl/project/results/op3/run_0/best_model.zip', env=eval_env, device='cpu')

#eval_env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/anybotics_anymal_b/scene.xml', healthy_z_range=(0.48, 0.68), render_mode='human')
#eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = PPO.load(path='/home/master-andreas/rl/project/results/anymal_b/run_0/best_model.zip', env=eval_env, device='cpu')

eval_env = gym.make('Ant-v5', render_mode=None)


#eval_env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
#eval_env = gym.wrappers.TimeLimit(inverted_double_pendulum_v4_fixed.InvertedDoublePendulumEnv(render_mode='human'), max_episode_steps=1000)
#eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(use_contact_forces=True, render_mode='human'), max_episode_steps=1000)


#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_fixed_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
#model = TD3.load(path='/home/master-andreas/rl/project/results/ant_v5_TD3/run_8/best_model.zip', env=eval_env, device='cpu')

#avg_return, std_return = evaluate_policy(model, eval_env, n_eval_episodes=1000)
#print(f"the average return is {avg_return}")

##breakpoint()

vec_env = model.get_env()
obs = vec_env.reset()
for step in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(action)
    print(info)
    time.sleep(0.050)


