import argparse
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro import distributions as dist
import arviz as az

numpyro.set_host_device_count(4)

from cppp.load import load_multisensory_data, preprocess_multisensory_data
from cppp.models.multisensory import UnisensoryDelayModel


def parse_args():
    parser = argparse.ArgumentParser(description="Model fitting")
    parser.add_argument(
        "--participant", type=int, default=135033060, help="Participant ID"
    )
    parser.add_argument("--vis_noise", type=int, default=1, help="Visual noise level")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--nwarmup", type=int, default=1_000, help="Number of warump steps for NUTS."
    )
    parser.add_argument(
        "--nsamp", type=int, default=1_000, help="Number of samples for NUTS."
    )
    parser.add_argument("--nchain", type=int, default=4, help="Number of chains.")
    parser.add_argument(
        "--model",
        type=str,
        default="UnisensoryDelayModel",
        help="Model type (lqg.tracking)",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def unisensory_model(x, ModelType, delay=1, dt=0.075):
    T = x.shape[1]

    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    # priors
    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    model = ModelType(
        process_noise=1.05,
        sigma=sigma,
        action_variability=action_variability,
        action_cost=action_cost,
        delay=delay,
        dt=dt,
        T=T - 1,
    )
    # Placeholder for the actual model implementation
    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])


if __name__ == "__main__":
    args = parse_args()

    df = load_multisensory_data()

    delays = {"vis": 2, "prop": 1}

    for condition in ["vis", "prop"]:
        data = preprocess_multisensory_data(
            df,
            participant=args.participant,
            condition=condition,
            vis_noise=args.vis_noise,
        )

        # --- Run MCMC for model fitting ---
        nuts_kernel = NUTS(unisensory_model)
        mcmc_xy = MCMC(
            nuts_kernel, num_warmup=args.nwarmup, num_samples=args.nsamp, num_chains=4
        )
        mcmc_xy.run(
            random.PRNGKey(args.seed),
            data,
            UnisensoryDelayModel,
            delay=delays[condition],
            dt=0.075,
        )

        inference_data = az.from_numpyro(mcmc_xy)

        if args.save:
            inference_data.to_netcdf(
                f"results/multisensory-mcmc-{condition}-{args.vis_noise}-{args.model}-{args.seed}.nc"
            )

  