import gymnasium as gym
from gymnasium.experimental.wrappers import RescaleActionV0
import time
from envs import inverted_double_pendulum_v4_fixed
from envs import ant_v5
from stable_baselines3.common.vec_env import VecVideoRecorder

import numpy as np

from stable_baselines3 import TD3, PPO, A2C, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy


RENDER_MODE="human"


#eval_env = gym.make('Ant-v5', include_cfrc_ext_in_observation=False, width=1920, height=1080, render_mode=RENDER_MODE)
#model = SAC.load(path='/home/master-andreas/rl/project/results/ant_v5_without_ctn_SAC/run_9/best_model.zip', env=eval_env, device='cpu')

#eval_env = gym.make('Humanoid-v5', xml_file='/home/master-andreas/mujoco_menagerie/agility_cassie/scene.xml', healthy_z_range=(1.0, 2.1), render_mode='human')
#eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = PPO.load(path='/home/master-andreas/rl/project/results/cassie/run_0/best_model.zip', env=eval_env, device='cpu')

# OP3
#eval_env = gym.make('Humanoid-v5', xml_file='~/mujoco_menagerie/robotis_op3/scene.xml', healthy_z_range=(0.275, 0.5), include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False, ctrl_cost_weight=0, render_mode=RENDER_MODE, width=1920, height=1080)
#eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = SAC.load(path='/home/master-andreas/rl/project/results/op3/SAC/run_0/best_model.zip', env=eval_env, device='cpu')
# anymal b
#eval_env = gym.make('Ant-v5', xml_file='~/mujoco_menagerie/anybotics_anymal_b/scene.xml', ctrl_cost_weight=0.001, include_cfrc_ext_in_observation=False, healthy_z_range=(0.48, 0.68), width=1920, height=1080, render_mode=RENDER_MODE)
#eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = SAC.load(path='/home/master-andreas/rl/project/results/anymal_b/SAC/run_0/best_model.zip', env=eval_env, device='cpu')
# GO1
#eval_env = gym.make('Ant-v5', xml_file='~/mujoco_menagerie/unitree_go1/scene.xml', include_cfrc_ext_in_observation=False, healthy_z_range=(0.345, 1), ctrl_cost_weight=0.01, render_mode=RENDER_MODE, width=1920, height=1080, camera_id=0)
eval_env = gym.make('Ant-v5', xml_file='~/mujoco_menagerie/unitree_go1/scene.xml', include_cfrc_ext_in_observation=False, ctrl_cost_weight=0.05, healthy_z_range=(0.295, 1), frame_skip=25, render_mode=RENDER_MODE, width=1920, height=1080, camera_id=0)
eval_env = RescaleActionV0(eval_env, min_action=-1, max_action=1)
#model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_005_SAC/run_9/best_model.zip', env=eval_env, device='cpu')
#model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_001_SAC/run_5/best_model.zip', env=eval_env, device='cpu')
#model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_0001_SAC/run_0/best_model.zip', env=eval_env, device='cpu')
#model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_0_SAC/run_0/best_model.zip', env=eval_env, device='cpu')
#model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_005_z_032_SAC/run_3/best_model.zip', env=eval_env, device='cpu')
model = SAC.load(path='/home/master-andreas/rl/project/results/go1/without_ctn_ctrl_005_z_0345SAC/run_0/best_model.zip', env=eval_env, device='cpu')

#eval_env = gym.make('Ant-v5', render_mode='human')
#model = TD3.load(path='/home/master-andreas/rl/project/results/ant_v5_TD3/run_8/best_model.zip', env=eval_env, device='cpu')


#eval_env = gym.make('Ant-v5', render_mode='human')

#eval_env = gym.make('InvertedDoublePendulum-v4', render_mode='human')
#eval_env = gym.wrappers.TimeLimit(inverted_double_pendulum_v4_fixed.InvertedDoublePendulumEnv(render_mode='human'), max_episode_steps=1000)
#eval_env = gym.wrappers.TimeLimit(ant_v5.AntEnv(use_contact_forces=True, render_mode='human'), max_episode_steps=1000)


#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_TD3/run_0/best_model.zip', env=eval_env, device='cpu')
#model = TD3.load(path='/home/master-andreas/rl/project/results/InvertedDoublePendulum_v4_fixed_TD3/run_0/best_model.zip', env=eval_env, device='cpu')


#
# RECORD VIDEO
#
"""
video_folder = "videos/"
video_length = 1000
vec_env = VecVideoRecorder(model.get_env(), video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"video0")
obs = vec_env.reset()
for _ in range(video_length + 1):
  action, _state = model.predict(obs, deterministic=True)
  obs, _, _, _ = vec_env.step(action)
# Save the video
vec_env.close()
exit()
breakpoint()
"""


#
# Evaluate Policy
#

"""
avg_return, std_return = evaluate_policy(model, eval_env, n_eval_episodes=1000)
print(f"the average return is {avg_return}")
breakpoint()
"""



#
# Render Human
#
vec_env = model.get_env()
obs = vec_env.reset()
for step in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(action)
    print(info)
    time.sleep(0.150)


