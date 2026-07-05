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

    # TODO: get the seeds that we're actually using for each model and participant
    

    print(args)

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}


    # load the inference data for all models
    model =  az.from_netcdf(
            f"results/multisensory-mcmc-{filename_from_args(args)}.nc"
        )

    # create arviz summaries for all models
    summary = az.summary(model)

    # assert that all r-hats are below 1.1
    assert (summary["r_hat"] < 1.1).all(), (
        f"Model has r-hat values above 1.1!"
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


    # get posterior mean
    posterior_mean = summary["mean"].to_dict()

    for i, condition in enumerate(args.conditions):
        for j, vis_noise in enumerate([1, 2]):
            sim_data = model.posterior_predictive[f"x_{condition}_{vis_noise}"]

            sim_vels = np.diff(sim_data, axis=-2)
            lags, sim_correls = xcorr(
                sim_vels[..., 1], sim_vels[..., 0], maxlags=50
            )

            vels = np.diff(data[(condition, vis_noise)], axis=-2)
            lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=50)

            ppc_dfs.append(
                pd.DataFrame(
                    {
                        "participant": args.participant,
                        "model": args.model_class,
                        "condition": condition,
                        "vis_noise": vis_noise,
                        "lag": lags * dt,
                        "correlation": sim_correls.mean(axis=(0, 1)),
                    }
                )
            )

            # only save the real data once (when plotting the first model)
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

 