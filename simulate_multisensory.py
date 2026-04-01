import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from lqg import xcorr
from jax import random, numpy as jnp, jit
import pandas as pd

from cppp.load import preprocess_multisensory_data, load_multisensory_data
from cppp.models.multisensory import BiasMultisensoryDelayModel
from fit_multisensory import filename_from_args, parse_args


if __name__ == "__main__":
    args = parse_args()
    # we are going to load all models, so we don't need to specify one here
    del args.model

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    def simulate_calibration_phase(bias, vis_noise, model_name, params):
        if model_name == "optimal":
            model = BiasMultisensoryDelayModel(
                process_noise=1.2,
                sigmas=[params["sigma_prop"], params[f"sigma_vis[{vis_noise - 1}]"]],
                action_variability=params["action_variability"],
                action_cost=params["action_cost"],
                delays=[delays["prop"], delays["vis"]],
                dt=0.075,
                T=168,
            )
        elif model_name == "no_integration":
            model = BiasMultisensoryDelayModel(
                process_noise=1.2,
                sigmas=[params["sigma_prop"]],
                action_variability=params["action_variability"],
                action_cost=params["action_cost"],
                delays=[delays["prop"]],
                dt=0.075,
                T=168,
            )
        elif model_name == "equal_integration":
            sigma = jnp.sqrt(
                (params["sigma_prop"] ** 2 + params[f"sigma_vis[{vis_noise - 1}]"] ** 2)
                / 2
            )
            model = BiasMultisensoryDelayModel(
                process_noise=1.2,
                sigmas=[sigma, sigma],
                action_variability=params["action_variability"],
                action_cost=params["action_cost"],
                delays=[delays["prop"], delays["vis"]],
                dt=0.075,
                T=168,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        x = model.simulate(
            rng_key=random.PRNGKey(0),
            n=20,
            x0=jnp.array([0.0, 0.0, bias] + [0.0] * (model.xdim - 3)),
        )
        return x

    jit_simulate = jit(
        simulate_calibration_phase, static_argnames=["model_name", "vis_noise"]
    )

    # load the inference data for all models
    models = {
        model_name: az.from_netcdf(
            f"results/multisensory-mcmc-{args.participant}_{args.prop_delay}_{args.vis_delay}_{args.seed}_{args.nwarmup}_{args.nsamp}_{args.nchain}_{model_name}.nc"
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
        assert (
            summary["r_hat"] < 1.1
        ).all(), f"Model {model_name} has r-hat values above 1.1!"

    # perform model comparison for all data variables separately
    var_names = ["x_multi_1", "x_multi_2", "x_vis_1", "x_vis_2", "x_prop_1", "x_prop_2"]
    for var_name in var_names:
        comp_df = az.compare(models, var_name=var_name)
        print(comp_df)
        axes = az.plot_compare(comp_df=comp_df)
        axes.set_title(f"Model comparison for {var_name}")

    # TODO: also perform model comparison for all data variables together (at the moment seems to take up too much memory)
    # for name, inference_data in models.items():
    #     print(inference_data.log_likelihood)
    #     inference_data.log_likelihood["x_all"] = xr.concat(
    #                     [inference_data.log_likelihood[var_name].rename({f"{var_name}_dim_0": "x_dim_0"}) for
    #                      var_name
    #                      in
    #                      var_names],
    #                     dim="x_dim_0"
    #                 )

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

    f, ax = plt.subplots(4, 2, figsize=(10, 16), sharey=True, sharex=True)

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
                lags, sim_correls = xcorr(
                    sim_vels[..., 1], sim_vels[..., 0], maxlags=50
                )

                vels = np.diff(data[(condition, vis_noise)], axis=-2)
                lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=50)

                # plot the cross-correlations for the real data and the simulated data
                if (
                    k == 0
                ):  # only plot the real data one (when plotting the first model)
                    ax[0, j].plot(
                        lags[51:] * 0.075,
                        correls.mean(axis=0)[51:],
                        label=f"{condition}",
                        color=f"C{i}",
                    )
                ax[k + 1, j].plot(
                    lags[51:] * 0.075,
                    sim_correls.mean(axis=(0, 1))[51:],
                    label=f"{condition}",
                    color=f"C{i}",
                )

                ax[0, j].set_xlim(0, 2)
                ax[k + 1, j].set_xlim(0, 2)

                ax[0, j].set_title(f"Data - noise level {vis_noise}")
                ax[k + 1, j].set_title(
                    f"{model_name.capitalize()} model - noise level {vis_noise}"
                )

    ax[k + 1, 0].set_xlabel("Lag (s)")
    ax[k + 1, 1].set_xlabel("Lag (s)")
    ax[0, 0].set_ylabel("Cross-correlation")
    ax[k + 1, 0].set_ylabel("Cross-correlation")

    ax[0, 0].legend()
    ax[k + 1, 0].legend()
    f.suptitle(f"Participant: {args.participant}")
    f.tight_layout()
    f.savefig(f"results/multisensory-simulations-{filename_from_args(args)}.png")

    # get current participant's data
    df = df[df["participant"] == args.participant]

    # compute tracking error
    df["tracking_error"] = df["righty_pos"] - df["lefty_pos"]
    df["tracking_mse"] = df["tracking_error"] ** 2

    # simulate calibration phase
    f, ax = plt.subplots(4, 1, figsize=(5, 16), sharey=True, sharex=True)
    dfs = []
    for (trial_number, vis_noise), trial_df in df[df["phase"] == "calibration"].groupby(
        ["trial_number", "vis_noise"]
    ):
        offset = trial_df["cursor_offset"].iloc[0]

        ax[0].scatter(
            trial_number,
            trial_df["tracking_error"].mean(),
            color="C0" if vis_noise == 1 else "C1",
        )
        ax[0].set_ylabel("Mean tracking error")

        for k, (model_name, model) in enumerate(models.items()):

            summary = az.summary(model)
            # get posterior mean
            posterior_mean = summary["mean"].to_dict()

            for vis_noise in [1, 2]:

                x = jit_simulate(offset, vis_noise, model_name, posterior_mean)
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
                            }
                        )
                    )

                ax[k + 1].scatter(
                    trial_number, error, color="C0" if vis_noise == 1 else "C1"
                )
                ax[k + 1].set_ylabel("Mean tracking error")
                ax[k + 1].set_title(f"{model_name.capitalize()} model")

                # plt.axhline(0, color="black", linestyle="--")
    ax[-1].set_xlabel("Trial number")
    ax[0].set_title("Data")
    f.suptitle(f"Participant: {args.participant}")
    f.tight_layout()
    f.savefig(
        f"results/multisensory-simulations-calibration-{filename_from_args(args)}.png"
    )

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(f"results/multisensory-simulations-calibration-{filename_from_args(args)}.csv", index=False)

