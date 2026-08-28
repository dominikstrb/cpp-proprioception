import numpy as np
import xarray as xr


def concat_log_likelihoods(model, condition=None):
    var_names = model.log_likelihood.keys()

    if condition is not None:
        var_names = [name for name in var_names if condition in name]

    x = np.concatenate(
        [model.log_likelihood[f"{var_name}"].to_numpy() for var_name in var_names],
        axis=-1,
    )

    model.log_likelihood = xr.Dataset(
        data_vars={
            "x": (
                [
                    "chain",
                    "draw",
                    "trial",
                ],
                x,
            )
        },
        coords={
            "chain": range(x.shape[0]),
            "draw": range(x.shape[1]),
            "trial": range(x.shape[-1]),
        },
    )

    return model
