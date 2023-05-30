import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs

steps = np.load('results/Humanoid_v4_PPO/run_0/evaluations.npz')['timesteps']
returns_PPO_v4 = np.average(np.array([np.load('results/Humanoid_v4_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v4_fixed = np.average(np.array([np.load('results/Humanoid_v4_fixed_reward_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v4_fixed_eval = np.average(np.array([np.load('results/Humanoid_v4_fixed_reward_on_eval_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v5a = np.average(np.array([np.load('results/Humanoid_v5a_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v4 = np.average(np.array([np.load('results/Humanoid_v4_fixed_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5a = np.average(np.array([np.load('results/Humanoid_v5a_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)

#breakpoint()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_PPO_v4, axis=0), label='v4')
ax.plot(steps, np.average(returns_PPO_v4_fixed, axis=0), label='v4 (fixed reward)')
ax.plot(steps, np.average(returns_PPO_v4_fixed_eval, axis=0), label='v4 (fixed reward on eval only)')
ax.plot(steps, np.average(returns_PPO_v5a, axis=0), label='v5 (alpha)')
#ax.plot(steps, np.average(returns_TD3_v4, axis=0), label='v4 (fixed reward)')
#ax.plot(steps, np.average(returns_TD3_v5, axis=0), label='v5 with ctn')
ax.fill_between(steps, np.min(returns_PPO_v4, axis=0), np.max(returns_PPO_v4, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_PPO_v4_fixed, axis=0), np.max(returns_PPO_v4_fixed, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_PPO_v4_fixed_eval, axis=0), np.max(returns_PPO_v4_fixed_eval, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_PPO_v5a, axis=0), np.max(returns_PPO_v5a, axis=0), alpha=0.2)
#ax.fill_between(steps, np.min(returns_TD3_v4, axis=0), np.max(returns_TD3_v4, axis=0), alpha=0.2)
#ax.fill_between(steps, np.min(returns_TD3_v5, axis=0), np.max(returns_TD3_v5, axis=0), alpha=0.2)

ax.set_title('SB3/PPO on MuJoCo/Humanoid, for ' + str(RUNS) + ' Runs')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig("figures/PPO_Humanoid" + ".eps", bbox_inches="tight")
plt.savefig("figures/PPO_Humanoid" + ".png", bbox_inches="tight")

fig.show()
breakpoint()

