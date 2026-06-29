import argparse
from jax import random, numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import distributions as dist
import arviz as az

numpyro.set_host_device_count(4)

from cppp.load import load_multisensory_data, preprocess_multisensory_data
from cppp.models.multisensory import BoundedActor
from cppp.models import multisensory


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
    parser.add_argument("--seed", type=int, default=7453, help="Random seed")
    parser.add_argument(
        "--nwarmup", type=int, default=1000, help="Number of warump steps for NUTS."
    )
    parser.add_argument(
        "--nsamp", type=int, default=2_500, help="Number of samples for NUTS."
    )
    parser.add_argument("--nchain", type=int, default=4, help="Number of chains.")
    parser.add_argument(
        "--model", type=str, default="optimal", help="Kind of integration model to fit"
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="MultisensoryDelayModel",
        help="Model class to fit",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["prop", "vis", "multi"],
        help="Conditions to include in the fit",
    )
    return parser.parse_args()


def filename_from_args(args):
    indicator = "_".join(map(str, vars(args).values()))
    return indicator

default_priors = {
    "sigma_vis": dist.HalfNormal(40.0).expand([2]),
    "sigma_prop": dist.HalfNormal(40.0),
    "action_variability": dist.HalfNormal(1.),
    "action_cost": dist.HalfNormal(1.),
}

def optimal_integration_model(
    data, delays, dt=0.075, obs=True, model_class=BoundedActor, priors=default_priors,
):

    # priors
    sigma_vis = numpyro.sample("sigma_vis", priors["sigma_vis"])
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

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            model = model_class(
                process_noise=1.2,
                sigmas=[sigma_prop, sigma_vis[vis_noise - 1]],
                delays=[delays["prop"], delays["vis"]],
                dt=dt,
                T=T - 1,
                **motor_params,
            )
        else:
            delay = delays[condition]
            model = model_class(
                process_noise=1.2,
                sigmas=[
                    sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]
                ],
                delays=[delay],
                dt=dt,
                T=T - 1,
                **motor_params,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.to_numpyro(xdim=x.shape[-1]),
            obs=x if obs else None,
        )


def no_integration_model(data, delays, dt=0.075, obs=True, model_class=BoundedActor, priors=default_priors):
    # priors
    sigma_vis = numpyro.sample("sigma_vis", priors["sigma_vis"])
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

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            model = model_class(
                process_noise=1.2,
                sigmas=[sigma_prop],
                delays=[delays["prop"]],
                dt=dt,
                T=T - 1,
                **motor_params,
            )
        else:
            delay = delays[condition]
            model = model_class(
                process_noise=1.2,
                sigmas=[
                    sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]
                ],
                delays=[delay],
                dt=dt,
                T=T - 1,
                **motor_params,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.to_numpyro(xdim=x.shape[-1]),
            obs=x if obs else None,
        )


def equal_integration_model(data, delays, dt=0.075, obs=True, model_class=BoundedActor, priors=default_priors):
    # priors
    sigma_vis = numpyro.sample("sigma_vis", priors["sigma_vis"])
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

    for (condition, vis_noise), x in data.items():
        T = x.shape[1]

        if condition == "multi":
            sigma = jnp.sqrt((sigma_vis[vis_noise - 1] ** 2 + sigma_prop**2) / 2)
            model = model_class(
                process_noise=1.2,
                sigmas=[sigma, sigma],
                delays=[delays["prop"], delays["vis"]],
                dt=dt,
                T=T - 1,
                **motor_params,
            )
        else:
            delay = delays[condition]
            model = model_class(
                process_noise=1.2,
                sigmas=[
                    sigma_prop if condition == "prop" else sigma_vis[vis_noise - 1]
                ],
                delays=[delay],
                dt=dt,
                T=T - 1,
                **motor_params,
            )

        # likelihood
        numpyro.sample(
            f"x_{condition}_{vis_noise}",
            model.to_numpyro(xdim=x.shape[-1]),
            obs=x if obs else None,
        )


models = {
    "optimal": optimal_integration_model,
    "no_integration": no_integration_model,
    "equal_integration": equal_integration_model,
}



if __name__ == "__main__":
    args = parse_args()

    # load data
    df = load_multisensory_data()

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    # load data for all conditions and visual noise levels
    data = {}
    for condition in args.conditions:
        for vis_noise in df["vis_noise"].unique():
            cond_data = preprocess_multisensory_data(
                df,
                participant=args.participant,
                condition=condition,
                vis_noise=vis_noise,
            )
            print(f"Data shape for condition {condition} and vis_noise {vis_noise}: {cond_data.shape}")

            data[(condition, vis_noise)] = cond_data

    # fit joint model
    nuts_kernel = NUTS(models[args.model])
    mcmc = MCMC(
        nuts_kernel, num_warmup=args.nwarmup, num_samples=args.nsamp, num_chains=4
    )
    mcmc.run(
        random.PRNGKey(args.seed),
        data,
        delays=delays,
        dt=0.075,
        model_class=getattr(multisensory, args.model_class),
    )

    predictive = Predictive(models[args.model], mcmc.get_samples())
    samples_predictive = predictive(
        random.PRNGKey(args.seed),
        data,
        delays=delays,
        dt=0.075,
        obs=False,
        model_class=getattr(multisensory, args.model_class),
    )

    # save model fit
    inference_data = az.from_numpyro(mcmc, posterior_predictive=samples_predictive)
    inference_data.to_netcdf(f"results/multisensory-mcmc-{filename_from_args(args)}.nc")

    print(f"Finished fitting model {args.model} for participant {args.participant}!")
