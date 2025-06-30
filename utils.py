import numpy as np
from lqg import xcorr


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
