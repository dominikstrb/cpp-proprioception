from typing import Tuple
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from jax import numpy as jnp, random

import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

numpyro.set_host_device_count(4)


from cppp.load import load_data, preprocess_data
from cppp.constants import sampling_rate
from cppp.utils import compute_crosscorr_stats
import cppp.models as models
from cppp.models import (
    CorrelatedRelativeObservationBoundedActor,
    CorrelatedObservationBoundedActor,
)


def lqg_model(x, ModelType):
    # get the number of time steps from the data
    T = x.shape[1]
    # get the dimensionality from the data
    dim = x.shape[2] // 2

    # priors
    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5), sample_shape=(dim,))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5), sample_shape=(dim,))

    # prior on correlation
    if dim == 1:  # if the data is one-dimensional, we don't need correlations
        L = jnp.array([[1.0]])

    else:  # for higher-dimensional data
        # This is a LKJCholesky prior for a Cholesky decomposed correlation matrix
        L = numpyro.sample("L", dist.LKJCholesky(dim, concentration=1.0))

    if dim == 2:
        # extract the correlation from the correlation matrix and add it to the chain
        # (just for logging / visualization later)
        numpyro.deterministic("rho", L[1, 0])

    if ModelType == CorrelatedObservationBoundedActor:
        sigma_target = numpyro.sample("sigma_target", dist.HalfNormal(1.0), sample_shape=(dim,))
        sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfNormal(1.0), sample_shape=(dim,))

        # pass the parameters to the model
        model = ModelType(
            action_variability=action_variability,
            action_cost=action_cost,
            sigma_target=sigma_target,
            sigma_cursor=sigma_cursor,
            corr_chol=L,
            T=T,
            dt=1 / sampling_rate,
            dim=dim,
        )

    elif ModelType == CorrelatedRelativeObservationBoundedActor:
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0), sample_shape=(dim,))

        # pass the parameters to the model
        model = ModelType(
            action_variability=action_variability,
            action_cost=action_cost,
            sigma=sigma,
            corr_chol=L,
            T=T,
            dt=1 / sampling_rate,
            dim=dim,
        )
        
    elif ModelType == models.CorrelatedObservationJerkBoundedActor:
        sigma_target = numpyro.sample("sigma_target", dist.HalfNormal(1.0), sample_shape=(dim,))
        sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfNormal(1.0), sample_shape=(dim,))
        jerk_cost = numpyro.sample("jerk_cost", dist.HalfNormal(0.5))

        # pass the parameters to the model
        model = ModelType(
            action_variability=action_variability,
            action_cost=action_cost,
            sigma_target=sigma_target,
            sigma_cursor=sigma_cursor,
            corr_chol=L,
            jerk_cost=jerk_cost,
            T=T,
            dt=1 / sampling_rate,
            dim=dim,
        )
    elif ModelType == models.CorrelatedObservationSubjectiveActor:
        sigma_target = numpyro.sample("sigma_target", dist.HalfNormal(1.0), sample_shape=(dim,))
        sigma_cursor = numpyro.sample("sigma_cursor", dist.HalfNormal(1.0), sample_shape=(dim,))
        subj_noise = numpyro.sample("subj_noise", dist.HalfNormal(1.0))
        subj_vel_noise = numpyro.sample("subj_vel_noise", dist.HalfNormal(5.0))

        model = ModelType(
            action_variability=action_variability,
            action_cost=action_cost,
            sigma_target=sigma_target,
            sigma_cursor=sigma_cursor,
            corr_chol=L,
            subj_noise=subj_noise,
            subj_vel_noise=subj_vel_noise,
            T=T,
            dt=1 / sampling_rate,
            dim=dim,
        )

    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])


def parse_args():
    parser = argparse.ArgumentParser(description="Model fitting")
    parser.add_argument("--pos", type=int, nargs=2, default=[12, 22], help="Position in the workspace")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--nwarmup", type=int, default=1_000, help="Number of warump steps for NUTS.")
    parser.add_argument("--nsamp", type=int, default=1_000, help="Number of samples for NUTS.")
    parser.add_argument("--nchain", type=int, default=4, help="Number of chains.")
    parser.add_argument(
        "--model",
        type=str,
        default="CorrelatedObservationBoundedActor",
        help="Model type (lqg.tracking)",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Load and preprocess data ---
    df = load_data(pos=args.pos)
    xy_array = preprocess_data(df, lag=2)

    ModelType = getattr(models, args.model)

    # --- Run MCMC for model fitting ---
    nuts_kernel = NUTS(lqg_model)
    mcmc_xy = MCMC(nuts_kernel, num_warmup=args.nwarmup, num_samples=args.nsamp, num_chains=4)
    mcmc_xy.run(random.PRNGKey(args.seed), xy_array, ModelType)

    inference_data_xy = az.from_numpyro(mcmc_xy)

    if args.save:
        inference_data_xy.to_netcdf(f"results/mcmc-{args.pos}-{args.model}-{args.seed}.nc")

    summary_xy = az.summary(inference_data_xy)
    print(summary_xy)

    # --- Extract posterior mean parameters ---
    posterior_mean = summary_xy["mean"].to_dict()
    if ModelType == CorrelatedRelativeObservationBoundedActor:
        params = {
            "sigma": jnp.stack([posterior_mean["sigma[0]"], posterior_mean["sigma[1]"]]),
            "corr_chol": jnp.array(
                [
                    [posterior_mean["L[0, 0]"], posterior_mean["L[0, 1]"]],
                    [posterior_mean["L[1, 0]"], posterior_mean["L[1, 1]"]],
                ]
            ),
            "action_cost": jnp.stack([posterior_mean["action_cost[0]"], posterior_mean["action_cost[1]"]]),
            "action_variability": jnp.stack(
                [
                    posterior_mean["action_variability[0]"],
                    posterior_mean["action_variability[1]"],
                ]
            ),
        }
    elif ModelType == CorrelatedObservationBoundedActor:
        params = {
            "sigma_target": jnp.stack([posterior_mean["sigma_target[0]"], posterior_mean["sigma_target[1]"]]),
            "sigma_cursor": jnp.stack([posterior_mean["sigma_cursor[0]"], posterior_mean["sigma_cursor[1]"]]),
            "corr_chol": jnp.array(
                [
                    [posterior_mean["L[0, 0]"], posterior_mean["L[0, 1]"]],
                    [posterior_mean["L[1, 0]"], posterior_mean["L[1, 1]"]],
                ]
            ),
            "action_cost": jnp.stack([posterior_mean["action_cost[0]"], posterior_mean["action_cost[1]"]]),
            "action_variability": jnp.stack(
                [
                    posterior_mean["action_variability[0]"],
                    posterior_mean["action_variability[1]"],
                ]
            ),
        }
    elif ModelType == models.CorrelatedObservationJerkBoundedActor:
        params = {
            "sigma_target": jnp.stack([posterior_mean["sigma_target[0]"], posterior_mean["sigma_target[1]"]]),
            "sigma_cursor": jnp.stack([posterior_mean["sigma_cursor[0]"], posterior_mean["sigma_cursor[1]"]]),
            "corr_chol": jnp.array(
                [
                    [posterior_mean["L[0, 0]"], posterior_mean["L[0, 1]"]],
                    [posterior_mean["L[1, 0]"], posterior_mean["L[1, 1]"]],
                ]
            ),
            "action_cost": jnp.stack([posterior_mean["action_cost[0]"], posterior_mean["action_cost[1]"]]),
            "action_variability": jnp.stack(
                [
                    posterior_mean["action_variability[0]"],
                    posterior_mean["action_variability[1]"],
                ]
            ),
            "jerk_cost": posterior_mean["jerk_cost"],
        }
    elif ModelType == models.CorrelatedObservationSubjectiveActor:
        params = {
            "sigma_target": jnp.stack([posterior_mean["sigma_target[0]"], posterior_mean["sigma_target[1]"]]),
            "sigma_cursor": jnp.stack([posterior_mean["sigma_cursor[0]"], posterior_mean["sigma_cursor[1]"]]),
            "corr_chol": jnp.array(
                [
                    [posterior_mean["L[0, 0]"], posterior_mean["L[0, 1]"]],
                    [posterior_mean["L[1, 0]"], posterior_mean["L[1, 1]"]],
                ]
            ),
            "action_cost": jnp.stack([posterior_mean["action_cost[0]"], posterior_mean["action_cost[1]"]]),
            "action_variability": jnp.stack(
                [
                    posterior_mean["action_variability[0]"],
                    posterior_mean["action_variability[1]"],
                ]
            ),
            "subj_noise": posterior_mean["subj_noise"],
            "subj_vel_noise": posterior_mean["subj_vel_noise"],
        }


    # --- Simulate data from the model given posterior mean parameters ---
    model = ModelType(**params, T=xy_array.shape[1], dt=1 / sampling_rate, dim=2)
    sim_data = model.simulate(
        rng_key=random.PRNGKey(0), n=250
    )  # I am simulating more trials to get smoother simulated CCGs without influence of random fluctuations

    # --- Compute CCGs for X and Y arrays ---
    lag_times_x, avg_data_x, std_data_x, lag_data_x, corr_data_x = compute_crosscorr_stats(
        xy_array[..., [0, 1]], sampling_rate=sampling_rate
    )
    lag_times_y, avg_data_y, std_data_y, lag_data_y, corr_data_y = compute_crosscorr_stats(
        xy_array[..., [2, 3]], sampling_rate=sampling_rate
    )

    lag_times_x, avg_model_x, std_model_x, lag_model_x, corr_model_x = compute_crosscorr_stats(
        sim_data[..., [0, 1]], sampling_rate=sampling_rate
    )
    lag_times_y, avg_model_y, std_model_y, lag_model_y, corr_model_y = compute_crosscorr_stats(
        sim_data[..., [2, 3]], sampling_rate=sampling_rate
    )

    if args.plot:
        # --- Plotting: Two Panels (Data | Model) ---
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

        # --- Panel 1: Data (X and Y) ---
        axs[0].plot(lag_times_x, avg_data_x, label="X data", color="C0")
        axs[0].fill_between(
            lag_times_x,
            avg_data_x - std_data_x,
            avg_data_x + std_data_x,
            color="C0",
            alpha=0.2,
        )
        axs[0].plot(lag_data_x, corr_data_x, "o", color="C0")
        axs[0].annotate(
            f"{corr_data_x:.2f} @ {lag_data_x:.2f}s",
            (lag_data_x, corr_data_x),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="C0",
        )

        axs[0].plot(lag_times_y, avg_data_y, label="Y data", color="C1")
        axs[0].fill_between(
            lag_times_y,
            avg_data_y - std_data_y,
            avg_data_y + std_data_y,
            color="C1",
            alpha=0.2,
        )
        axs[0].plot(lag_data_y, corr_data_y, "o", color="C1")
        axs[0].annotate(
            f"{corr_data_y:.2f} @ {lag_data_y:.2f}s",
            (lag_data_y, corr_data_y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="C1",
        )

        axs[0].axvline(0, color="black", linestyle="--", linewidth=1)
        axs[0].set_title("Cross-Correlation: Data")
        axs[0].set_xlabel("Time lag [s]")
        axs[0].set_ylabel("Correlation")
        axs[0].legend(frameon=False)
        axs[0].set_ylim(-0.25, 0.75)

        # --- Panel 2: Model (X and Y) ---
        axs[1].plot(lag_times_x, avg_model_x, label="X model", color="C0", linestyle="--")
        axs[1].fill_between(
            lag_times_x,
            avg_model_x - std_model_x,
            avg_model_x + std_model_x,
            color="C0",
            alpha=0.2,
        )
        axs[1].plot(lag_model_x, corr_model_x, "o", color="C0")
        axs[1].annotate(
            f"{corr_model_x:.2f} @ {lag_model_x:.2f}s",
            (lag_model_x, corr_model_x),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="C0",
        )

        axs[1].plot(lag_times_y, avg_model_y, label="Y model", color="C1", linestyle="--")
        axs[1].fill_between(
            lag_times_y,
            avg_model_y - std_model_y,
            avg_model_y + std_model_y,
            color="C1",
            alpha=0.2,
        )
        axs[1].plot(lag_model_y, corr_model_y, "o", color="C1")
        axs[1].annotate(
            f"{corr_model_y:.2f} @ {lag_model_y:.2f}s",
            (lag_model_y, corr_model_y),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="C1",
        )

        axs[1].axvline(0, color="black", linestyle="--", linewidth=1)
        axs[1].set_title("Cross-Correlation: Model")
        axs[1].set_xlabel("Time lag [s]")
        axs[1].legend(frameon=False)

        plt.suptitle("Cross-Correlation: Data vs Model (Separated Panels)")
        plt.tight_layout()
        plt.savefig(f"results/ccgs-{args.pos}-{args.model}-{args.seed}.png")
