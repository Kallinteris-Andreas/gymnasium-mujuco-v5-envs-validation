import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs

steps = np.load('results/HumanoidStandup_v4_PPO/run_0/evaluations.npz')['timesteps']
returns_PPO_v4 = np.average(np.array([np.load('results/HumanoidStandup_v4_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v5 = np.average(np.array([np.load('results/HumanoidStandup_v5_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)

#breakpoint()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_PPO_v4, axis=0), label='v4')
ax.plot(steps, np.average(returns_PPO_v5, axis=0), label='v5')
ax.fill_between(steps, np.min(returns_PPO_v4, axis=0), np.max(returns_PPO_v4, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_PPO_v5, axis=0), np.max(returns_PPO_v5, axis=0), alpha=0.2)

ax.set_title('SB3/PPO on MuJoCo/Humanoid, for ' + str(RUNS) + ' Runs')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig("figures/PPO_HumanoidStandup" + ".eps", bbox_inches="tight")
plt.savefig("figures/PPO_HumanoidStandup" + ".png", bbox_inches="tight")

fig.show()
breakpoint()

