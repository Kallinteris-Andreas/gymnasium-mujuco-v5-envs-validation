import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs
FILE_NAME = "figures/go1"

steps = np.load('results/ant_v4_fixed_with_ctn_TD3/run_0/evaluations.npz')['timesteps']

returns_SAC_w_ctn = np.average(np.array([np.load('results/go1/with_ctn_ctrl_005_z0295_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_SAC_w_ctn_len = np.average(np.array([np.load('results/go1/with_ctn_ctrl_005_z0295_SAC/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)

returns_SAC_wo_ctn = np.average(np.array([np.load('results/go1/without_ctn_ctrl_005_z0295_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_SAC_wo_ctn_len = np.average(np.array([np.load('results/go1/without_ctn_ctrl_005_z0295_SAC/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)


#breakpoint()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_SAC_w_ctn, axis=0), label='with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_w_ctn, axis=0), np.max(returns_SAC_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_wo_ctn, axis=0), label='without ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_wo_ctn, axis=0), np.max(returns_SAC_wo_ctn, axis=0), alpha=0.2)

ax.set_title('SB3 on MuJoCo/go1, for ' + str(RUNS) + ' Runs, episodic returns')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig(FILE_NAME + ".eps", bbox_inches="tight")
plt.savefig(FILE_NAME + ".png", bbox_inches="tight")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(steps, np.average(returns_SAC_w_ctn_len, axis=0), label='with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_w_ctn_len, axis=0), np.max(returns_SAC_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_wo_ctn_len, axis=0), label='without ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_wo_ctn_len, axis=0), np.max(returns_SAC_wo_ctn_len, axis=0), alpha=0.2)

ax.set_title('SB3 on MuJoCo/go1, for ' + str(RUNS) + ' Runs, episodic lengths')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig(FILE_NAME + "_len" + ".eps", bbox_inches="tight")
plt.savefig(FILE_NAME + "_len" + ".png", bbox_inches="tight")

#fig.show()
#breakpoint()

