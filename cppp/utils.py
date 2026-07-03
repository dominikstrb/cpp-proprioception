import numpy as np
from lqg import xcorr
import xarray as xr

from cppp import models


# --- Cross-correlation computation ---
def compute_crosscorr_stats(array_data, sampling_rate, max_lag=10):

    cursor_vel = np.diff(array_data[..., 1])
    target_vel = np.diff(array_data[..., 0])

    lags, corr = xcorr(cursor_vel, target_vel, maxlags=max_lag)

    lag_times = lags / sampling_rate

    avg_data = corr.mean(axis=0)
    std_data = corr.std(axis=0)

    peak_idx_data = np.argmax(avg_data)

    lag = float(lag_times[peak_idx_data])
    corr = float(avg_data[peak_idx_data])

    return lag_times, avg_data, std_data, lag, corr


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
