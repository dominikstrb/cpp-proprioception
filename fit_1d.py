import argparse
from jax import random

import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

numpyro.set_host_device_count(4)

from cppp.load import load_data, preprocess_data
from cppp.constants import sampling_rate
from cppp.models import multisensory


def filename_from_args(args):
    indicator = "_".join(map(str, vars(args).values()))
    return indicator


default_priors = {
    "sigma_vis": dist.HalfNormal(20.0).expand([2]),
    "sigma_prop": dist.HalfNormal(20.0),
    "action_variability": dist.HalfNormal(1.0),
    "action_cost": dist.HalfNormal(2.0),
}


def lqg_model(x, process_noise, model_class, priors=default_priors, obs=True):
    # get the number of time steps from the data
    T = x.shape[1]

    # priors
    sigma_prop = numpyro.sample("sigma_prop", priors["sigma_prop"])

    motor_params = {}
    if model_class in [
        multisensory.OptimalActorPointMassDynamics,
        multisensory.OptimalActor,
        multisensory.BoundedActorPointMassDynamics,
        multisensory.BoundedActor,
    ]:
        motor_params["action_variability"] = numpyro.sample(
            "action_variability", priors["action_variability"]
        )

    if model_class in [
        multisensory.BoundedActorPointMassDynamics,
        multisensory.BoundedActor,
    ]:
        motor_params["action_cost"] = numpyro.sample(
            "action_cost", priors["action_cost"]
        )

    model = model_class(
        process_noise=1.2,
        sigmas=[sigma_prop],
        delays=[delays["prop"]],
        dt=1 / sampling_rate,
        T=T - 1,
        **motor_params,
    )

    numpyro.sample(
        "x",
        model.to_numpyro(xdim=x.shape[-1]),
        obs=x if obs else None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Model fitting")
    parser.add_argument(
        "--pos", type=int, nargs=2, default=[12, 22], help="Position in the workspace"
    )
    parser.add_argument("--dim", type=str, default="x", help="Dimension")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--nwarmup", type=int, default=2500, help="Number of warump steps for NUTS."
    )
    parser.add_argument(
        "--nsamp", type=int, default=2500, help="Number of samples for NUTS."
    )
    parser.add_argument("--nchain", type=int, default=4, help="Number of chains.")
    parser.add_argument(
        "--model",
        type=str,
        default="BoundedActorPointMassDynamics",
        help="Model type",
    )
    parser.add_argument(
        "--prop_delay",
        type=int,
        default=1,
        help="Delay in proprioceptive signal (in the model)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Load and preprocess data ---
    df = load_data(pos=args.pos)
    data = preprocess_data(df, lag=1, dim=args.dim)

    print(data.shape)

    process_noise = 1.2

    ModelType = getattr(multisensory, args.model)

    delays = {"prop": args.prop_delay}

    # --- Run MCMC for model fitting ---
    nuts_kernel = NUTS(lqg_model, init_strategy=numpyro.infer.init_to_median)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=args.nwarmup,
        num_samples=args.nsamp,
        num_chains=args.nchain,
    )
    mcmc.run(random.PRNGKey(args.seed), data, process_noise, ModelType)

    idata = az.from_numpyro(mcmc)

    idata.to_netcdf(f"results/1d/{filename_from_args(args)}.nc")

    summary = az.summary(idata)
    print(summary)
