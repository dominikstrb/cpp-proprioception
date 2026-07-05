import arviz as az
import pandas as pd


from cppp.load import load_multisensory_data
from cppp.utils import concat_log_likelihoods


def load_model(participant, model_name, model_class, seeds=None, num_warmups=None):

    if seeds is None:
        # seeds = [1, 2, 7452]
        seeds = [3]

    if num_warmups is None:
        num_warmups = [2500, 2500]
    seed_idx = 0
    while seed_idx < len(seeds):
        seed = seeds[seed_idx]

        for num_warmup in num_warmups:
            try:
                model = az.from_netcdf(
                    f"results/multisensory-mcmc-{participant}_1_2_{seed}_{num_warmup}_2500_4_{model_name}_{model_class}_['prop', 'vis', 'multi'].nc"
                )
                break
            except FileNotFoundError:
                print(
                    f"Model {model_name} for participant {participant} with seed {seed} and num_warmup {num_warmup} not found."
                )

        if az.summary(model)["r_hat"].max() > 1.1:
            print(
                f"Model {model_name} for participant {participant} with seed {seed} has r_hat > 1.1."
            )
            seed_idx += 1
            # model = None
        else:
            break

    if model is None:
        raise ValueError(
            f"Model {model_name} for participant {participant} not found or has r_hat > 1.1 for all seeds."
        )

    return model


if __name__ == "__main__":
    participants = load_multisensory_data(base_path="data")["participant"].unique()

    participants_to_exclude = []

    participants = participants[:3]
    participants = [p for p in participants if p not in participants_to_exclude]

    model_names = ["equal_integration", "no_integration", "vision_only"]

    model_classes = ["BoundedActorPointMassDynamics", "BoundedActor"]

    summaries = []
    loos = []
    for participant in participants:
        # dict for model comparisons
        models = {}

        # load the models for this participant
        for model_name in model_names:
            for model_class in model_classes:
                seed_idx = 0

                model = load_model(participant, model_name, model_class)

                model.posterior["action_cost"] = model.posterior["action_cost"] * 100.0
                model.posterior["vis_prop_diff"] = (
                    model.posterior["sigma_vis"] - model.posterior["sigma_prop"]
                )

                summary = az.summary(model)

                # stack up log likelihoods for all 6 conditions into a single array for each model so that we can perform model comparison using all conditions together
                model = concat_log_likelihoods(model)
                # put model into dict
                models[model_name] = model

                # add the model name and participant to the summary dataframe
                summary["integration"] = model_name
                summary["model"] = model_class
                summary["participant"] = participant
                summaries.append(summary)

                # compute the Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO) for the model
                loo = az.loo(model)
                pareto_k = loo.pareto_k
                p_problematic = (loo.pareto_k.to_numpy() > 0.7).sum() / len(
                    loo.pareto_k.to_numpy()
                )
                loo["integration"] = model_name
                loo["model"] = model_class
                loo["participant"] = participant
                loos.append(
                    {
                        "participant": participant,
                        "integration": model_name,
                        "model": model_class,
                        "elpd": loo.elpd_loo,
                        "se": loo.se,
                        "p_problematic_k": p_problematic,
                    }
                )

    loo_df = pd.DataFrame(loos)

    loo_df["elpd_low"] = loo_df["elpd"] - 1.96 * loo_df["se"]
    loo_df["elpd_high"] = loo_df["elpd"] + 1.96 * loo_df["se"]

    loo_df.to_csv("results/aggregated/loo_all_participants.csv")

    df = pd.concat(summaries).reset_index().rename(columns={"index": "parameter"})
    df.to_csv("results/aggregated/summaries_all_participants.csv")