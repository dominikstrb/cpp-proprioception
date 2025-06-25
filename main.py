import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from jax import numpy as jnp


from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

# numpyro.set_host_device_count(4)

from lqg import LQG, Actor, Dynamics, System
from lqg import xcorr


from load import load_data, preprocess_data
from constants import sampling_rate

if __name__ == "__main__":
    df = load_data(pos=(12, 22))

    xy_array = preprocess_data(df)

