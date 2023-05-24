import gymnasium as gym
import time
from envs import inverted_double_pendulum_v4_fixed
from envs import ant_v5

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure


#eval_env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
#eval_env = gym.wrappers.TimeLimit(inverted_double_pendulum_v4_fixed.InvertedDoublePendulumEnv(render_mode='human'), max_episode_steps=1000)
eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(use_contact_forces=False, render_mode=None), max_episode_steps=1000)
#eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(use_contact_forces=True, render_mode='human'), max_episode_steps=1000)

#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_fixed_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
model = TD3.load(path='/home/master-andreas/rl/project/results/ant_v4_fixed_without_ctn_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
#model = TD3.load(path='/home/master-andreas/rl/project/results/ant_v5_TD3/run_8/best_model.zip', env=eval_env, device='cpu')

vec_env = model.get_env()
obs = vec_env.reset()
re = 0
for ep in range(1000):
    for step in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        re += reward
        if done:
            break;

    #print(str(i) + '--' + str(reward))
    #time.sleep(0.050)

print(re/1000)

