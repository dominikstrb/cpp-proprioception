import jax.numpy as jnp
from jax import vmap
from jax.scipy import linalg
from lqg import Actor, System
from cppp.models.dynamics import tracking_point_mass_dynamics_matrices


class CorrelatedObservationModel(System):
    def __init__(
        self,
        sigma,
        corr_chol,
        action_variability,
        action_cost,
        process_noise=1.0,
        dt=1 / 60.0,
        T=1000,
        dim=2,  # TODO: generalize this for higher dimensions (currently hardcoded for 2D)
        damping=0.1,
        m=1.0,
        tau=0.066,
    ):

        # dynamics model
        A, B, V = vmap(
            lambda action_var: tracking_point_mass_dynamics_matrices(
                process_noise, damping, m, tau, action_var, dt
            )
        )(action_variability)

        A = linalg.block_diag(*A)
        B = linalg.block_diag(*B)
        V = linalg.block_diag(*V)

        # Define permutation indices, so that we have target and cursor first
        perm_indices = jnp.array(
            [0, 1, 4, 5, 2, 3, 6, 7]
        )  # TODO: generalize this for higher dimensions

        # Construct permutation matrix
        identity_matrix = jnp.eye(dim * 4)
        permutation_matrix = identity_matrix[perm_indices]

        A = permutation_matrix @ A @ permutation_matrix.T
        B = permutation_matrix @ B
        V = permutation_matrix @ V @ permutation_matrix.T

        # observation model
        F = linalg.block_diag(*[jnp.array([[1.0, -1.0, 0.0, 0.0]])] * dim)
        F = (
            F @ permutation_matrix.T
        )  # permute the observation model to match the state ordering

        W = jnp.diag(sigma) @ corr_chol

        # cost function
        Q = linalg.block_diag(
            *[
                jnp.array(
                    [
                        [1.0, -1.0, 0.0, 0.0],
                        [-1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ]
            * dim
        )
        Q = (
            permutation_matrix @ Q @ permutation_matrix.T
        )  # permute the cost matrix to match the state ordering
        R = jnp.diag(jnp.ones(dim) * action_cost)

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


if __name__ == "__main__":
    from lqg import xcorr
    from jax import random

    import matplotlib.pyplot as plt

    dt_rw = 1.0 / 18.18

    params = {
        "process_noise": 1.0,
        "sigma": jnp.array([10.0, 1.0]),
        "action_variability": jnp.array([0.1, 0.1]),
        "action_cost": jnp.array([0.01, 0.01]),
    }
    rho = 0.5
    corr_chol = jnp.array([[1.0, rho], [0.0, jnp.sqrt(1 - rho**2)]])
    params["corr_chol"] = corr_chol
    model = CorrelatedObservationModel(
        **params,
        dt=dt_rw,
        T=168,
    )

    x = model.simulate(n=50, rng_key=random.PRNGKey(0))

    vels = jnp.diff(x, axis=-2)

    lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=50)
    plt.plot(lags[50:] * dt_rw, correls.mean(axis=0)[50:])

    lags, correls = xcorr(vels[..., 3], vels[..., 2], maxlags=50)
    plt.plot(lags[50:] * dt_rw, correls.mean(axis=0)[50:])

    plt.savefig("two_dim_correls.png")
