import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from jax import numpy as jnp


from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

# numpyro.set_host_device_count(4)

from lqg import LQG, Actor, Dynamics, System
from lqg import xcorr


from load import load_data, preprocess_data
from constants import sampling_rate

if __name__ == "__main__":
    df = load_data(pos=(12, 22))

    xy_array = preprocess_data(df)

    axis_labels = ['X', 'Y']
    kinarm_rate = 1000  # Hz
    dt = 1 / kinarm_rate
    trajectory_ids = sorted(df['Trajectory_ID'].unique())
    min_count = df.groupby('Trajectory_ID').size().min()
    time = np.linspace(0, dt * (min_count - 1), min_count)

    # Initialize plots
    n_blocks = len(trajectory_ids)
    fig, axs = plt.subplots(n_blocks, 3, figsize=(10, 2 * n_blocks), squeeze=False)

    for i, block_num in enumerate(trajectory_ids):
        temp_df = df[df['Trajectory_ID'] == block_num].iloc[:min_count]

        # Plot X and Y hand trajectories
        for j, ax in enumerate(axis_labels):
            right_hand = temp_df[f'Right_Hand{ax}'].to_numpy()
            left_hand = temp_df[f'Left_Hand{ax}'].to_numpy()
            axs[i, j].plot(time, right_hand, label='Right Hand')
            axs[i, j].plot(time, left_hand, label='Left Hand')
            axs[i, j].set_title(f'Block {block_num} - {ax} Trajectory')
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Position')
            axs[i, j].legend()
            # axs[i, j].set_ylim(-10, 10)

        lags_x, corr_x = xcorr(jnp.diff(xy_array[i, :, 1]), jnp.diff(xy_array[i, :, 0]), maxlags=10)
        lags_y, corr_y = xcorr(jnp.diff(xy_array[i, :, 1]), jnp.diff(xy_array[i, :, 0]), maxlags=10)
        axs[i, 2].plot(lags_x/sampling_rate, corr_x, label='X Vel Corr')
        axs[i, 2].plot(lags_y/sampling_rate, corr_y, label='Y Vel Corr')

        axs[i, 2].set_title(f'Block {block_num} - Cross-Correlation (Velocities)')
        axs[i, 2].set_xlabel('Lag (s)')
        axs[i, 2].set_ylabel('Correlation')
        axs[i, 2].legend()
        axs[i, 2].set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig("trajectories.png")