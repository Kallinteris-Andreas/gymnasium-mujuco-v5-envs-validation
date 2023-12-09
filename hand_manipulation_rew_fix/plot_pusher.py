import numpy as np
import matplotlib.pyplot as plt

RUNS = 10  # Number of statistical runs
NAME = "Pusher"
FILE_NAME = f"figures/{NAME}"

steps = np.load('results/Reacher-v5/TD3/run_0/evaluations.npz')['timesteps']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for algorithm in ["TD3"]:
    returns = np.average(np.array([np.load(f'results/Pusher-v5/{algorithm}/run_{run}/evaluations.npz')['results'][:steps.size] for run in range(RUNS)]), axis=2)
    returns_len = np.average(np.array([np.load(f'results/Pusher-v5/{algorithm}/run_{run}/evaluations.npz')['ep_lengths'][:steps.size] for run in range(RUNS)]), axis=2)
    returns_fix = np.average(np.array([np.load(f'results/Pusher-v5_fix_rew/{algorithm}/run_{run}/evaluations.npz')['results'][:steps.size] for run in range(RUNS)]), axis=2)
    returns_fix_len = np.average(np.array([np.load(f'results/Pusher-v5_fix_rew/{algorithm}/run_{run}/evaluations.npz')['ep_lengths'][:steps.size] for run in range(RUNS)]), axis=2)

    ax.plot(steps, np.average(returns, axis=0), label=f'{NAME}-v5 {algorithm}')
    ax.fill_between(steps, np.min(returns, axis=0), np.max(returns, axis=0), alpha=0.2)
    ax.plot(steps, np.average(returns_fix, axis=0), "--", label=f'{NAME}-v5-fixed {algorithm}')
    ax.fill_between(steps, np.min(returns_fix, axis=0), np.max(returns_fix, axis=0), alpha=0.2)


ax.set_title(f'SB3 on Gymnasium/MuJoCo/{NAME}, for {RUNS} Runs, episodic returns')
ax.legend()
ax.set_ylim([-100,-30])

fig.set_figwidth(16)
fig.set_figheight(9)

plt.savefig(FILE_NAME + ".eps", bbox_inches="tight")
plt.savefig(FILE_NAME + ".png", bbox_inches="tight")
