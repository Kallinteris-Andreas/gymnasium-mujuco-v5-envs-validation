import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs

steps = np.load('results/ant_v4_fixed_with_ctn_TD3/run_0/evaluations.npz')['timesteps']
returns_TD3_v4_w_ctn = np.average(np.array([np.load('results/ant_v4_fixed_with_ctn_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v4_wo_ctn = np.average(np.array([np.load('results/ant_v4_fixed_without_ctn_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_SAC_v4_w_ctn = np.average(np.array([np.load('results/ant_v4_fixed_with_ctn_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_SAC_v4_wo_ctn = np.average(np.array([np.load('results/ant_v4_fixed_without_ctn_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v5a_w_ctn = np.average(np.array([np.load('results/ant_v5a_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v5_w_ctn = np.average(np.array([np.load('results/ant_v5_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_wo_ctn = np.average(np.array([np.load('results/ant_v5_without_ctn_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_01 = np.average(np.array([np.load('results/ant_v5_policy_std_0.1_TD3/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_025 = np.average(np.array([np.load('results/ant_v5_policy_std_0.25_TD3//run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_100 = np.average(np.array([np.load('results/ant_v5_policy_std_1_TD3//run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v5a_w_ctn = np.average(np.array([np.load('results/ant_v5a_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_PPO_v5_w_ctn = np.average(np.array([np.load('results/ant_v5_PPO/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_SAC_v5a_w_ctn = np.average(np.array([np.load('results/ant_v5a_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_SAC_v5_w_ctn = np.average(np.array([np.load('results/ant_v5_SAC/run_' + str(run) + '/evaluations.npz')['results'] for run in range(RUNS)]), axis=2)
returns_TD3_v4_w_ctn_len = np.average(np.array([np.load('results/ant_v4_fixed_with_ctn_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_TD3_v4_wo_ctn_len = np.average(np.array([np.load('results/ant_v4_fixed_without_ctn_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_TD3_v5a_w_ctn_len = np.average(np.array([np.load('results/ant_v5a_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_TD3_v5_w_ctn_len = np.average(np.array([np.load('results/ant_v5_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_wo_ctn_len = np.average(np.array([np.load('results/ant_v5_without_ctn_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_01_len = np.average(np.array([np.load('results/ant_v5_policy_std_0.1_TD3/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_025_len = np.average(np.array([np.load('results/ant_v5_policy_std_0.25_TD3//run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
#returns_TD3_v5_w_ctn_std_100_len = np.average(np.array([np.load('results/ant_v5_policy_std_1_TD3//run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_PPO_v5a_w_ctn_len = np.average(np.array([np.load('results/ant_v5a_PPO/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_PPO_v5_w_ctn_len = np.average(np.array([np.load('results/ant_v5_PPO/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_SAC_v5a_w_ctn_len = np.average(np.array([np.load('results/ant_v5a_SAC/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)
returns_SAC_v5_w_ctn_len = np.average(np.array([np.load('results/ant_v5_SAC/run_' + str(run) + '/evaluations.npz')['ep_lengths'] for run in range(RUNS)]), axis=2)

#breakpoint()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.plot(steps, np.average(returns_TD3_v4_w_ctn, axis=0), label='v4 (fixed reward) with ctn (TD3)')
#ax.fill_between(steps, np.min(returns_TD3_v4_w_ctn, axis=0), np.max(returns_TD3_v4_w_ctn, axis=0), alpha=0.2)
#ax.plot(steps, np.average(returns_TD3_v4_wo_ctn, axis=0), label='v4 (fixed reward)/v5a without ctn (TD3)')
#ax.fill_between(steps, np.min(returns_TD3_v4_wo_ctn, axis=0), np.max(returns_TD3_v4_wo_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_TD3_v5a_w_ctn, axis=0), label='v5 (alpha) with ctn (TD3)')
ax.fill_between(steps, np.min(returns_TD3_v5a_w_ctn, axis=0), np.max(returns_TD3_v5a_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_TD3_v5_w_ctn, axis=0), label='v5 with ctn (TD3)')
ax.fill_between(steps, np.min(returns_TD3_v5_w_ctn, axis=0), np.max(returns_TD3_v5_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_PPO_v5a_w_ctn, axis=0), label='v5 (alpha) with ctn (PPO)')
ax.fill_between(steps, np.min(returns_PPO_v5a_w_ctn, axis=0), np.max(returns_PPO_v5a_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_PPO_v5_w_ctn, axis=0), label='v5 with ctn (PPO)')
ax.fill_between(steps, np.min(returns_PPO_v5_w_ctn, axis=0), np.max(returns_PPO_v5_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_v5a_w_ctn, axis=0), label='v5 (alpha) with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_v5a_w_ctn, axis=0), np.max(returns_SAC_v5a_w_ctn, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_v5_w_ctn, axis=0), label='v5 with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_v5_w_ctn, axis=0), np.max(returns_SAC_v5_w_ctn, axis=0), alpha=0.2)

ax.set_title('SB3 on MuJoCo/Ant, for ' + str(RUNS) + ' Runs, episodic returns')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

file_name = "figures/Ant"
plt.savefig(file_name + ".eps", bbox_inches="tight")
plt.savefig(file_name + ".png", bbox_inches="tight")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.plot(steps, np.average(returns_TD3_v4_w_ctn_len, axis=0), label='v4 (fixed reward) with ctn (TD3)')
#ax.fill_between(steps, np.min(returns_TD3_v4_w_ctn_len, axis=0), np.max(returns_TD3_v4_w_ctn_len, axis=0), alpha=0.2)
#ax.plot(steps, np.average(returns_TD3_v4_wo_ctn_len, axis=0), label='v4 (fixed reward)/v5a without ctn (TD3)')
#ax.fill_between(steps, np.min(returns_TD3_v4_wo_ctn_len, axis=0), np.max(returns_TD3_v4_wo_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_TD3_v5a_w_ctn_len, axis=0), label='v5 (alpha) with ctn (TD3)')
ax.fill_between(steps, np.min(returns_TD3_v5a_w_ctn_len, axis=0), np.max(returns_TD3_v5a_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_TD3_v5_w_ctn_len, axis=0), label='v5 with ctn (TD3)')
ax.fill_between(steps, np.min(returns_TD3_v5_w_ctn_len, axis=0), np.max(returns_TD3_v5_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_PPO_v5a_w_ctn_len, axis=0), label='v5 (alpha) with ctn (PPO)')
ax.fill_between(steps, np.min(returns_PPO_v5a_w_ctn_len, axis=0), np.max(returns_PPO_v5a_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_PPO_v5_w_ctn_len, axis=0), label='v5 with ctn (PPO)')
ax.fill_between(steps, np.min(returns_PPO_v5_w_ctn_len, axis=0), np.max(returns_PPO_v5_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_v5a_w_ctn_len, axis=0), label='v5 (alpha) with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_v5a_w_ctn_len, axis=0), np.max(returns_SAC_v5a_w_ctn_len, axis=0), alpha=0.2)
ax.plot(steps, np.average(returns_SAC_v5_w_ctn_len, axis=0), label='v5 with ctn (SAC)')
ax.fill_between(steps, np.min(returns_SAC_v5_w_ctn_len, axis=0), np.max(returns_SAC_v5_w_ctn_len, axis=0), alpha=0.2)

ax.set_title('SB3 on MuJoCo/Ant, for ' + str(RUNS) + ' Runs, episodic lengths')
ax.legend()

fig.set_figwidth(16)
fig.set_figheight(9)

file_name = "figures/Ant_len"
plt.savefig(file_name + ".eps", bbox_inches="tight")
plt.savefig(file_name + ".png", bbox_inches="tight")

#fig.show()
#breakpoint()

