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


def load_model(participant, model_name, model_class, seed=7453, num_warmup=2000):

    model = az.from_netcdf(
        f"results/multisensory-mcmc-{participant}_1_2_{seed}_{num_warmup}_2500_4_{model_name}_{model_class}_['prop', 'vis', 'multi'].nc"
    )
    return model


if __name__ == "__main__":
    args = parse_args()
    # we are going to load all models, so we don't need to specify one here
    del args.seed

    seeds = [7452, 1]
    print(args)

    delays = {"vis": args.vis_delay, "prop": args.prop_delay}

    seed_idx = 0
    while seed_idx < len(seeds):
        seed = seeds[seed_idx]
        model = load_model(
            args.participant, args.model, model_class=args.model_class, seed=seed
        )

        if az.summary(model)["r_hat"].max() > 1.1:
            print(
                f"Model {args.model} for participant {args.participant} with seed {seed} has r_hat > 1.1."
            )
            seed_idx += 1
        else:
            break


    ppc_dfs = []
    # print(model.log_likelihood["x_multi_1"].shape)
    print(f"Simulating model: {args.model}, class: {args.model_class}, participant: {args.participant}")
    # print(model.posterior)

    summary = az.summary(model)
    # get posterior mean
    posterior_mean = summary["mean"].to_dict()

    for i, condition in enumerate(["multi", "vis", "prop"]):
        for j, vis_noise in enumerate([1, 2]):
            sim_data = model.posterior_predictive[f"x_{condition}_{vis_noise}"]

            sim_vels = np.diff(sim_data, axis=-2)
            lags, sim_correls = xcorr(sim_vels[..., 1], sim_vels[..., 0], maxlags=50)

            ppc_dfs.append(
                pd.DataFrame(
                    {
                        "participant": args.participant,
                        "model": args.model,
                        "model_class": args.model_class,
                        "condition": condition,
                        "vis_noise": vis_noise,
                        "lag": lags * dt,
                        "correlation": sim_correls.mean(axis=(0, 1)),
                    }
                )
            )



    ppc_df = pd.concat(ppc_dfs, ignore_index=True)
    ppc_df.to_csv(
        f"results/ppc/multisensory-simulations-ppc-{filename_from_args(args)}.csv",
        index=False,
    )
