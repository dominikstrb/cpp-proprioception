import argparse
from jax import random
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
        "--vis_delay", type=int, default=2, help="Delay in visual signal (in the model)"
    )
    parser.add_argument(
        "--prop_delay",
        type=int,
        default=1,
        help="Delay in proprioceptive signal (in the model)",
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
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def unisensory_model(x, delay=1, dt=0.075):
    T = x.shape[1]

    # priors
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    model = UnisensoryDelayModel(
        process_noise=1.2,
        sigma=sigma,
        action_variability=action_variability,
        action_cost=action_cost,
        delay=delay,
        dt=dt,
        T=T - 1,
    )
    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])


def multisensory_model(x, delays, dt=0.075):
    T = x.shape[1]

    # priors
    sigmas = numpyro.sample("sigmas", dist.HalfNormal(10.0).expand([2]))
    action_variability = numpyro.sample("action_variability", dist.HalfNormal(0.5))
    action_cost = numpyro.sample("action_cost", dist.HalfNormal(0.5))

    model = MultisensoryDelayModel(
        process_noise=1.2,
        sigmas=sigmas,
        action_variability=action_variability,
        action_cost=action_cost,
        delays=[delays["prop"], delays["vis"]],
        dt=dt,
        T=T - 1,
    )
    # likelihood
    numpyro.sample("x", model.conditional_distribution(x), obs=x[:, 1:])


if __name__ == "__main__":
    args = parse_args()

    # load data
    df = load_multisensory_data()

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    # for both levels of visual noise in the experiment
    for vis_noise in [1, 2]:

        # load data from multisensory condition
        data = preprocess_multisensory_data(
            df,
            participant=args.participant,
            condition="multi",
            vis_noise=vis_noise,
        )

        # fit multisensory model
        nuts_kernel = NUTS(multisensory_model)
        mcmc_xy = MCMC(
            nuts_kernel,
            num_warmup=args.nwarmup,
            num_samples=args.nsamp,
            num_chains=4,
        )
        mcmc_xy.run(
            random.PRNGKey(args.seed),
            data,
            delays=delays,
            dt=0.075,
        )

        # save model fit
        inference_data = az.from_numpyro(mcmc_xy)
        if args.save:
            inference_data.to_netcdf(
                f"results/multisensory-mcmc-{args.participant}-multi-{vis_noise}-{args.seed}.nc"
            )

        # fit unisensory conditions
        for condition in ["vis", "prop"]:
            # load unisensory data
            data = preprocess_multisensory_data(
                df,
                participant=args.participant,
                condition=condition,
                vis_noise=vis_noise,
            )

            # fit model
            nuts_kernel = NUTS(unisensory_model)
            mcmc_xy = MCMC(
                nuts_kernel,
                num_warmup=args.nwarmup,
                num_samples=args.nsamp,
                num_chains=4,
            )
            mcmc_xy.run(
                random.PRNGKey(args.seed),
                data,
                delay=delays[condition],
                dt=0.075,
            )

            # save model fit
            inference_data = az.from_numpyro(mcmc_xy)

            if args.save:
                inference_data.to_netcdf(
                    f"results/multisensory-mcmc-{args.participant}-{condition}-{vis_noise}-{args.seed}.nc"
                )
