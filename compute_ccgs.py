import numpy as np
from lqg import xcorr
import pandas as pd

from cppp.load import load_multisensory_data, preprocess_multisensory_data

dt = 0.075

if __name__ == "__main__":
    df = load_multisensory_data()

    data = {}
    for participant in df["participant"].unique():
        print(participant)
        ppc_dfs = []
        for condition in df["type"].unique():
            for vis_noise in df["vis_noise"].unique():
                data[(condition, vis_noise)] = preprocess_multisensory_data(
                    df,
                    participant=participant,
                    condition=condition,
                    vis_noise=vis_noise,
                    cutoff=12,
                )

                vels = np.diff(data[(condition, vis_noise)], axis=-2)
                lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=50)

                # only save the real data once (when plotting the first model)
                ppc_dfs.append(
                    pd.DataFrame(
                        {
                            "participant": participant,
                            "model": "data",
                            "model_class": "data",
                            "condition": condition,
                            "vis_noise": vis_noise,
                            "lag": lags * dt,
                            "correlation": correls.mean(axis=0),
                        }
                    )
                )

        ppc_df = pd.concat(ppc_dfs, ignore_index=True)
        ppc_df.to_csv(
            f"results/ppc/multisensory-data-ppc-{participant}.csv",
            index=False,
        )
