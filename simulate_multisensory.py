from functools import partial

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from lqg import xcorr
from jax import random, numpy as jnp, jit, vmap
from jax.nn import logsumexp
import pandas as pd

from cppp.load import load_multisensory_data
from cppp.models import multisensory
from fit_multisensory import filename_from_args, parse_args

dt = 0.075


def get_model(
    vis_noise,
    model_name,
    params,
    delays,
    dt,
    model_class,
    T,
    knows_about_bias=False,
):
    if model_name == "optimal":
        delay_list = [delays["prop"], delays["vis"]]
        model = model_class(
            process_noise=1.2,
            sigmas=[params["sigma_prop"], params[f"sigma_vis[{vis_noise - 1}]"]],
            action_variability=params["action_variability"],
            action_cost=params["action_cost"],
            delays=delay_list,
            dt=dt,
            T=T,
            knows_about_bias=knows_about_bias,
        )
    elif model_name == "no_integration":
        delay_list = [delays["prop"]]
        model = model_class(
            process_noise=1.2,
            sigmas=[params["sigma_prop"]],
            action_variability=params["action_variability"],
            action_cost=params["action_cost"],
            delays=delay_list,
            dt=dt,
            T=T,
            obs_indices=(0,),
            knows_about_bias=knows_about_bias,
        )
    elif model_name == "vision_only":
        delay_list = [delays["vis"]]
        model = model_class(
            process_noise=1.2,
            sigmas=[params[f"sigma_vis[{vis_noise - 1}]"]],
            action_variability=params["action_variability"],
            action_cost=params["action_cost"],
            delays=delay_list,
            dt=dt,
            T=T,
            obs_indices=(1,),
            knows_about_bias=knows_about_bias,
        )
    elif model_name == "equal_integration":
        delay_list = [delays["prop"], delays["vis"]]
        sigma = jnp.sqrt(
            (params["sigma_prop"] ** 2 + params[f"sigma_vis[{vis_noise - 1}]"] ** 2) / 2
        )
        model = model_class(
            process_noise=1.2,
            sigmas=[sigma, sigma],
            action_variability=params["action_variability"],
            action_cost=params["action_cost"],
            delays=delay_list,
            dt=dt,
            T=T,
            knows_about_bias=knows_about_bias,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model, delay_list


if __name__ == "__main__":
    args = parse_args()
    # we are going to load all models, so we don't need to specify one here
    del args.model

    print(args)

    model_class = getattr(multisensory, "Bias" + args.model_class)
    print(model_class)

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    # jit_simulate = jit(
    #     simulate_calibration_phase,
    #     static_argnames=["model", "model_class"],
    # )

    # load the inference data for all models
    models = {
        model_name: az.from_netcdf(
            f"results/multisensory_fits/multisensory-mcmc-{args.participant}_{args.prop_delay}_{args.vis_delay}_{args.seed}_{args.nwarmup}_{args.nsamp}_{args.nchain}_{model_name}_{args.model_class}_{args.conditions}.nc"
        )
        for model_name in [
            "optimal",
            "no_integration",
            "equal_integration",
            "vision_only",
        ]
    }

    # create arviz summaries for all models
    summaries = {
        model_name: az.summary(inference_data)
        for model_name, inference_data in models.items()
    }

    # assert that all r-hats are below 1.1
    for model_name, summary in summaries.items():
        assert (summary["r_hat"] < 1.1).all(), (
            f"Model {model_name} has r-hat values above 1.1!"
        )

    df = load_multisensory_data()
    df = df[df["participant"] == args.participant]

    sim_ppc_dfs = []
    for k, (model_name, model) in enumerate(models.items()):
        print(f"Simulating model: {model_name}")

        for i, condition in enumerate(["multi", "vis", "prop"]):
            for j, vis_noise in enumerate([1, 2]):
                sim_data = model.posterior_predictive[f"x_{condition}_{vis_noise}"]

                sim_vels = np.diff(sim_data, axis=-2)
                lags, sim_correls = xcorr(
                    sim_vels[..., 1], sim_vels[..., 0], maxlags=50
                )

                sim_ppc_dfs.append(
                    pd.DataFrame(
                        {
                            "participant": args.participant,
                            "model": model_name,
                            "model_class": args.model_class,
                            "condition": condition,
                            "vis_noise": vis_noise,
                            "lag": lags * dt,
                            "correlation": sim_correls.mean(axis=(0, 1)),
                        }
                    )
                )

    sim_ppc_df = pd.concat(sim_ppc_dfs, ignore_index=True)
    sim_ppc_df.to_csv(
        f"results/ppc/multisensory-simulations-ppc-{filename_from_args(args)}.csv",
        index=False,
    )

    print("Simulating calibration phase for all models...")
    # simulate calibration phase
    dfs = []
    likelihoods = []
    for k, (model_name, model_fit) in enumerate(models.items()):
        print(f"Simulating and evaluating model: {model_name}")

        # extract posterior samples into dict
        samples = {
            k: model_fit.posterior[k].values.flatten()
            for k in ["action_variability", "action_cost", "sigma_prop"]
        }
        samples.update(
            {
                f"sigma_vis[{i}]": model_fit.posterior["sigma_vis"]
                .values[..., i]
                .flatten()
                for i in range(2)
            }
        )

        # get random samples from the posterior
        samples = {
            k: v[np.random.choice(v.shape[0], size=100)] for k, v in samples.items()
        }

        # @partial(jit, static_argnums=(1,))
        # def simulate_calibration_phase(
        #     bias,
        #     vis_noise,
        #     params,
        #     key,
        # ):
        #     model, delay_list = get_model(
        #         vis_noise=vis_noise,
        #         model_name=model_name,
        #         params=params,
        #         delays=delays,
        #         dt=dt,
        #         model_class=model_class,
        #         T=168,
        #     )

        #     if model_class == multisensory.BiasBoundedActorPointMassDynamics:
        #         x0 = jnp.array([0.0, 0.0, 0.0, 0.0, bias] * (max(delay_list) + 1))
        #     elif model_class == multisensory.BiasBoundedActor:
        #         x0 = jnp.array([0.0, 0.0, bias] * (max(delay_list) + 1))
        #     x = model.simulate(
        #         rng_key=key,
        #         n=1,
        #         x0=x0,
        #         xhat0=jnp.zeros(model.bdim),
        #     )
        #     return x[0]

        @partial(jit, static_argnums=(1,))
        def log_likelihood_fn(x, vis_noise, params, offset):

            model, delay_list = get_model(
                vis_noise=vis_noise,
                model_name=model_name,
                params=params,
                delays=delays,
                dt=dt,
                model_class=model_class,
                T=x.shape[0] - 1,
            )
            if model_class == multisensory.BiasBoundedActorPointMassDynamics:
                x0 = jnp.array(
                    [x[0, 0], x[0, 1], 0.0, 0.0, offset] * (max(delay_list) + 1)
                )
                xhat0 = jnp.array([x[0, 0], x[0, 1], 0.0, 0.0] * (max(delay_list) + 1))

            elif model_class == multisensory.BiasBoundedActor:
                x0 = jnp.array([x[0, 0], x[0, 1], offset] * (max(delay_list) + 1))
                xhat0 = jnp.array([x[0, 0], x[0, 1]] * (max(delay_list) + 1))

            return model.log_likelihood(x[None], x0=x0[None], xhat0=xhat0[None])

        for (trial_number, vis_noise), trial_df in df[
            df["phase"] == "calibration"
        ].groupby(["trial_number", "vis_noise"]):
            # get current offset
            offset = trial_df["cursor_offset"].iloc[0]

            # get current trial data
            x = jnp.stack(
                [trial_df["cursory_pos"].to_numpy(), trial_df["lefty_pos"].to_numpy()]
            ).T

            # summary = az.summary(model_fit)
            # # get posterior mean
            # posterior_mean = summary["mean"].to_dict()

            # x_sim = vmap(
            #     lambda params, key: simulate_calibration_phase(
            #         offset, vis_noise, params, key
            #     )
            # )(samples, random.split(random.PRNGKey(0), 100))

            # for rep, x_i in enumerate(x_sim):
            #     dfs.append(
            #         pd.DataFrame(
            #             {
            #                 "participant": args.participant,
            #                 "phase": "calibration",
            #                 "cursor_offset": offset,
            #                 "vis_noise": vis_noise,
            #                 "trial_number": trial_number,
            #                 "model": model_name,
            #                 "model_class": args.model_class,
            #                 "righty_pos": x_i[:, 0],
            #                 "lefty_pos": x_i[:, 1],
            #                 "repetition": rep,
            #                 "time": np.arange(x_i.shape[0]) * dt,
            #             }
            #         )
            #     )

            log_likelihood = vmap(
                lambda params: log_likelihood_fn(x, vis_noise, params, offset)
            )(samples)
            likelihoods.append(
                {
                    "participant": args.participant,
                    "model": model_name,
                    "model_class": args.model_class,
                    "trial_number": trial_number,
                    "vis_noise": vis_noise,
                    "log_likelihood": (logsumexp(log_likelihood) - jnp.log(log_likelihood.shape[0])).item(),
                }
            )

    # df = pd.concat(dfs, ignore_index=True)
    # df.to_csv(
    #     f"results/calibration/multisensory-simulations-calibration-{filename_from_args(args)}.csv",
    #     index=False,
    # )

    likelihood_df = pd.DataFrame(likelihoods)
    likelihood_df.to_csv(
        f"results/calibration/multisensory-simulations-calibration-likelihoods-{filename_from_args(args)}.csv",
        index=False,
    )

    print("Success! Simulated calibration phase data saved to CSV.")
