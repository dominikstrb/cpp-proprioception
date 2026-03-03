import argparse
from jax import random, numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import arviz as az

numpyro.set_host_device_count(4)

from cppp.load import load_multisensory_data, preprocess_multisensory_data
from cppp.models.multisensory import UnisensoryDelayModel, MultisensoryDelayModel


def parse_args():
    parser = argparse.ArgumentParser(description="Model fitting")
    parser.add_argument(
        "--participant", type=int, default=135033060, help="Participant ID"
    )
    parser.add_argument(
        "--prop_delay",
        type=int,
        default=1,
        help="Delay in proprioceptive signal (in the model)",
    )
    parser.add_argument(
        "--vis_delay", type=int, default=2, help="Delay in visual signal (in the model)"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--nwarmup", type=int, default=2_500, help="Number of warump steps for NUTS."
    )
    parser.add_argument(
        "--nsamp", type=int, default=2_500, help="Number of samples for NUTS."
    )
    parser.add_argument("--nchain", type=int, default=4, help="Number of chains.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--model", type=str, default="optimal"
    )  # TODO: model with separate parameters for multisensory condition, to test whether the same noise parameters can explain both unisensory and multisensory conditions
    return parser.parse_args()


def filename_from_args(args):
    indicator = "_".join(map(str, vars(args).values()))
    return indicator


def optimal_integration_model(data, delays, dt=0.075):

    # priors
    sigma_vis = numpyro.sample("sigma_vis", dist.HalfNormal(10.0).expand([2]))
    sigma_prop = numpyro.sample("sigma_prop", dist.HalfNormal(10.0))
    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            model = MultisensoryDelayModel(
                process_noise=1.2,
                sigmas=[sigma_prop, sigma_vis[vis_noise - 1]],
                action_variability=action_variability,
                action_cost=action_cost,
                delays=[delays["prop"], delays["vis"]],
                dt=dt,
                T=T - 1,
            )
        else:
            delay = delays[condition]
            model = UnisensoryDelayModel(
                process_noise=1.2,
                sigma=(sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]),
                action_variability=action_variability,
                action_cost=action_cost,
                delay=delay,
                dt=dt,
                T=T - 1,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.conditional_distribution(x),
            obs=x[:, 1:],
        )


def no_integration_model(data, delays, dt=0.075):
    # priors
    sigma_vis = numpyro.sample("sigma_vis", dist.HalfNormal(10.0).expand([2]))
    sigma_prop = numpyro.sample("sigma_prop", dist.HalfNormal(10.0))

    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            model = UnisensoryDelayModel(
                process_noise=1.2,
                sigma=sigma_prop,
                action_variability=action_variability,
                action_cost=action_cost,
                delay=delays["prop"],
                dt=dt,
                T=T - 1,
            )
        else:
            delay = delays[condition]
            model = UnisensoryDelayModel(
                process_noise=1.2,
                sigma=(sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]),
                action_variability=action_variability,
                action_cost=action_cost,
                delay=delay,
                dt=dt,
                T=T - 1,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.conditional_distribution(x),
            obs=x[:, 1:],
        )


def equal_integration_model(data, delays, dt=0.075):
    # priors
    sigma_vis = numpyro.sample("sigma_vis", dist.HalfNormal(10.0).expand([2]))
    sigma_prop = numpyro.sample("sigma_prop", dist.HalfNormal(10.0))

    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            sigma = jnp.sqrt((sigma_vis[vis_noise - 1] ** 2 + sigma_prop**2) / 2)
            model = MultisensoryDelayModel(
                process_noise=1.2,
                sigmas=[sigma, sigma],
                action_variability=action_variability,
                action_cost=action_cost,
                delays=[delays["prop"], delays["vis"]],
                dt=dt,
                T=T - 1,
            )
        else:
            delay = delays[condition]
            model = UnisensoryDelayModel(
                process_noise=1.2,
                sigma=(sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]),
                action_variability=action_variability,
                action_cost=action_cost,
                delay=delay,
                dt=dt,
                T=T - 1,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.conditional_distribution(x),
            obs=x[:, 1:],
        )


models = {"optimal": optimal_integration_model, "no_integration": no_integration_model, "equal_integration": equal_integration_model}

if __name__ == "__main__":
    args = parse_args()

    # load data
    df = load_multisensory_data()

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    # load data for all conditions and visual noise levels
    data = {}
    for condition in df["type"].unique():
        for vis_noise in df["vis_noise"].unique():
            data[(condition, vis_noise)] = preprocess_multisensory_data(
                df,
                participant=args.participant,
                condition=condition,
                vis_noise=vis_noise,
            )

    # fit joint model
    nuts_kernel = NUTS(models[args.model])
    mcmc_xy = MCMC(
        nuts_kernel, num_warmup=args.nwarmup, num_samples=args.nsamp, num_chains=4
    )
    mcmc_xy.run(
        random.PRNGKey(args.seed),
        data,
        delays=delays,
        dt=0.075,
    )

    # save model fit
    inference_data = az.from_numpyro(mcmc_xy)
    inference_data.to_netcdf(f"results/multisensory-mcmc-{filename_from_args(args)}.nc")
