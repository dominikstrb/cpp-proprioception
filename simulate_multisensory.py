import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from lqg import xcorr
from jax import random, numpy as jnp, jit
import pandas as pd

from cppp.load import preprocess_multisensory_data, load_multisensory_data
from cppp.models import multisensory
from fit_multisensory import filename_from_args, parse_args

dt = 0.075

if __name__ == "__main__":
    args = parse_args()
    # we are going to load all models, so we don't need to specify one here
    del args.model

    print(args)

    model_class = getattr(multisensory, "Bias" + args.model_class)
    print(model_class)

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    def simulate_calibration_phase(
        bias,
        vis_noise,
        model_name,
        params,
        model_class=multisensory.BiasBoundedActorPointMassDynamics,
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
                T=168,
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
                T=168,
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
                T=168,
            )
        elif model_name == "equal_integration":
            delay_list = [delays["prop"], delays["vis"]]
            sigma = jnp.sqrt(
                (params["sigma_prop"] ** 2 + params[f"sigma_vis[{vis_noise - 1}]"] ** 2)
                / 2
            )
            model = model_class(
                process_noise=1.2,
                sigmas=[sigma, sigma],
                action_variability=params["action_variability"],
                action_cost=params["action_cost"],
                delays=delay_list,
                dt=dt,
                T=168,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if model_class == multisensory.BiasBoundedActorPointMassDynamics:
            x0 = jnp.array([0.0, 0.0, 0.0, 0.0, bias] * (max(delay_list) + 1))
        elif model_class == multisensory.BiasBoundedActor:
            x0 = jnp.array([0.0, 0.0, bias] * (max(delay_list) + 1))
        x = model.simulate(
            rng_key=random.PRNGKey(0),
            n=20,
            # TODO: this does not work anymore for the standard model without point mass dynamics, because it has a different state space dimensionality
            x0=x0,
            xhat0=jnp.zeros(model.bdim),
        )
        return x

    jit_simulate = jit(
        simulate_calibration_phase,
        static_argnames=["model_name", "vis_noise", "model_class"],
    )

    # load the inference data for all models
    models = {
        model_name: az.from_netcdf(
            f"results/multisensory-mcmc-{args.participant}_{args.prop_delay}_{args.vis_delay}_{args.seed}_{args.nwarmup}_{args.nsamp}_{args.nchain}_{model_name}_{args.model_class}_{args.conditions}.nc"
        )
        for model_name in ["optimal", "no_integration", "equal_integration"]
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

    data = {}
    for condition in df["type"].unique():
        for vis_noise in df["vis_noise"].unique():
            data[(condition, vis_noise)] = preprocess_multisensory_data(
                df,
                participant=args.participant,
                condition=condition,
                vis_noise=vis_noise,
            )

    ppc_dfs = []
    for k, (model_name, model) in enumerate(models.items()):
        # print(model.log_likelihood["x_multi_1"].shape)
        print(f"Simulating model: {model_name}")
        # print(model.posterior)

        summary = az.summary(model)
        # get posterior mean
        posterior_mean = summary["mean"].to_dict()

        for i, condition in enumerate(["multi", "vis", "prop"]):
            for j, vis_noise in enumerate([1, 2]):
                sim_data = model.posterior_predictive[f"x_{condition}_{vis_noise}"]

            sim_vels = np.diff(sim_data, axis=-2)
            lags, sim_correls = xcorr(sim_vels[..., 1], sim_vels[..., 0], maxlags=50)

            vels = np.diff(data[(condition, vis_noise)], axis=-2)
            lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=50)

            ppc_dfs.append(
                pd.DataFrame(
                    {
                        "participant": args.participant,
                        "model": model_name,
                        "condition": condition,
                        "vis_noise": vis_noise,
                        "lag": lags * dt,
                        "correlation": sim_correls.mean(axis=(0, 1)),
                    }
                )
            )

            if k == 0:  # only save the real data once (when plotting the first model)
                ppc_dfs.append(
                    pd.DataFrame(
                        {
                            "participant": args.participant,
                            "model": "data",
                            "condition": condition,
                            "vis_noise": vis_noise,
                            "lag": lags * dt,
                            "correlation": correls.mean(axis=0),
                        }
                    )
                )

    ppc_df = pd.concat(ppc_dfs, ignore_index=True)
    ppc_df.to_csv(
        f"results/ppc/multisensory-simulations-ppc-{filename_from_args(args)}.csv",
        index=False,
    )

    # get current participant's data
    df = df[df["participant"] == args.participant]

    # compute tracking error
    df["tracking_error"] = df["righty_pos"] - df["lefty_pos"]
    df["tracking_mse"] = df["tracking_error"] ** 2

    print("Simulating calibration phase for all models...")
    # simulate calibration phase
    dfs = []
    for (trial_number, vis_noise), trial_df in df[df["phase"] == "calibration"].groupby(
        ["trial_number", "vis_noise"]
    ):
        offset = trial_df["cursor_offset"].iloc[0]

        for k, (model_name, model) in enumerate(models.items()):
            summary = az.summary(model)
            # get posterior mean
            posterior_mean = summary["mean"].to_dict()

            for vis_noise in [1, 2]:
                x = jit_simulate(
                    offset,
                    vis_noise,
                    model_name,
                    posterior_mean,
                    model_class=model_class,
                )
                error = jnp.mean((x[..., 0] - x[..., 1]))

                for rep, x_i in enumerate(x):
                    dfs.append(
                        pd.DataFrame(
                            {
                                "participant": args.participant,
                                "phase": "calibration",
                                "cursor_offset": offset,
                                "vis_noise": vis_noise,
                                "trial_number": trial_number,
                                "model": model_name,
                                "righty_pos": x_i[:, 0],
                                "lefty_pos": x_i[:, 1],
                                "repetition": rep,
                                "time": np.arange(x_i.shape[0]) * dt,
                            }
                        )
                    )

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        f"results/calibration/multisensory-simulations-calibration-{filename_from_args(args)}.csv",
        index=False,
    )

    print("Succes")