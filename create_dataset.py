import gymnasium as gym
import minari
import numpy as np
from minari import DataCollectorV0, StepDataCallback
from stable_baselines3 import A2C, PPO, SAC, TD3


class AddExcludedObservationElements(StepDataCallback):
    """Add Excluded observation elements like cfrc_ext to the observation space."""
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        # if getattr(env, "_include_cinert_in_observation", None) is False:
        # if getattr(env, "_include_cvel_in_observation ", None) is False:
        # if getattr(env, "_include_qfrc_actuator_in_observation ", None) is False:
        if env.unwrapped._include_cfrc_ext_in_observation is False:
            step_data["observations"] = np.concatenate([step_data["observations"], env.unwrapped.contact_forces[1:].flat.copy()])


        return step_data


SEED = 12345
NUM_STEPS = int(1e3)
POLICY_NOISE = 0.1

dataset_name = "Ant-v5-6_7k_return-v0"
dataset = None

# Check if dataset already exist
assert dataset_name not in minari.list_local_datasets()

# Create Environment
env = gym.make("Ant-v5", include_cfrc_ext_in_observation=False, max_episode_steps=1e9)
# TODO add callback to add cfrc_ext to the obs space
collector_env = DataCollectorV0(env, step_data_callback=AddExcludedObservationElements, record_infos=True)
obs, _ = collector_env.reset(seed=SEED)

# load policy model
model = SAC.load(
    path="./results/ant_v5_without_ctn_SAC/run_9/best_model.zip",
    env=env,
    device="cuda",
)


for n_step in range(NUM_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    # Add some noise to each step action
    action += np.random.randn(*action.shape) * POLICY_NOISE

    obs, rew, terminated, truncated, info = collector_env.step(action)
    # Checkpoint
    if (n_step + 1) % 200e3 == 0:
        print(f"STEPS RECORDED: {n_step}")
        if dataset is None:
            dataset = minari.create_dataset_from_collector_env(
                collector_env=collector_env,
                dataset_id=dataset_name,
                algorithm_name="SB3/SAC",
                code_permalink="https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation/blob/main/create_dataset.py",
                author="Kallinteris Andreas",
                author_email="kallinteris@protonmail.com",
            )
        dataset.update_dataset_from_collector_env(collector_env)

    if terminated or truncated:
        env.reset()

#dataset.update_dataset_from_collector_env(collector_env)
#collector_env.save_to_disk("test.hdf5")
