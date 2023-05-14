import os
import hopper_v5
import walker2d_v5
import gymnasium
import time

env = hopper_v5.HopperEnv(xml_file=os.getcwd() + '/assets/hopper_v5.xml', render_mode='rgb_array')
env_old = gymnasium.make("Hopper-v4", render_mode='rgb_array')
env.reset(seed=5)
env_old.reset(seed=5)

for runs in range(10):
    for i in range(10_000):
        action = env.action_space.sample()
        env.step(action)
        env_old.step(action)
        assert (env.render() == env_old.render()).all()

env = walker2d_v5.Walker2dEnv(xml_file=os.getcwd() + '/assets/walker2d_v5_old.xml', render_mode='rgb_array')
env_old = gymnasium.make("Walker2d-v4", render_mode='rgb_array')
env.reset(seed=5)
env_old.reset(seed=5)

for runs in range(10):
    for i in range(10_000):
        action = env.action_space.sample()
        env.step(action)
        env_old.step(action)
        assert (env.render() == env_old.render()).all()
