import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs

steps = np.load('results/ant_v4_fixed_with_ctn_TD3/run_0/evaluations.npz')['timesteps']
returns_TD3_v4_w_ctn = np.average(np.array([np.load('results/ant_v4_fixed_with_ctn_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v4_wo_ctn = np.average(np.array([np.load('results/ant_v4_fixed_without_ctn_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v5_w_ctn = np.average(np.array([np.load('results/ant_v5_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)

# breakpoint()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_TD3_v4_w_ctn, axis=0), label='v4 (fixed reward) with ctn')
ax.plot(steps, np.average(returns_TD3_v4_wo_ctn, axis=0), label='v4 (fixed reward)/v5 without ctn')
ax.plot(steps, np.average(returns_TD3_v5_w_ctn, axis=0), label='v5 with ctn')
ax.fill_between(steps, np.min(returns_TD3_v4_w_ctn, axis=0), np.max(returns_TD3_v4_w_ctn, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_TD3_v4_wo_ctn, axis=0), np.max(returns_TD3_v4_wo_ctn, axis=0), alpha=0.2)
ax.fill_between(steps, np.min(returns_TD3_v5_w_ctn, axis=0), np.max(returns_TD3_v5_w_ctn, axis=0), alpha=0.2)

ax.set_title('TD3 on MuJoCo/Ant, for ' + str(RUNS) + ' Runs')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig("figures/TD3_Ant" + ".eps", bbox_inches="tight")
plt.savefig("figures/TD3_Ant" + ".png", bbox_inches="tight")

fig.show()
breakpoint()

