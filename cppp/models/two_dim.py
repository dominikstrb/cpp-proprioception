import jax.numpy as jnp
from jax.scipy import linalg
from lqg import Actor, System
from cppp.models.dynamics import point_mass_dynamics_matrices


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

        self.process_noise = process_noise

        # dynamics model
        A, B, V = point_mass_dynamics_matrices(damping, m, tau, action_variability, dt)

        A = linalg.block_diag(jnp.eye(1), A)
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        A = linalg.block_diag(*[A] * dim)
        B = linalg.block_diag(*[B] * dim)
        V = linalg.block_diag(*[V] * dim)

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

    dt_rw = 1.0 / 60.0

    params = {
        "process_noise": 1.0,
        "sigma_target": 6.0,
        "sigma_cursor": 6.0,
        "action_variability": 0.5,
        "action_cost": 0.01,
    }

    model = CorrelatedObservationModel(
        sigma=jnp.array([params["sigma_target"], params["sigma_target"]]),
        corr_chol=jnp.array([[1.0, 0.5], [0.0, jnp.sqrt(1 - 0.5**2)]]),
        process_noise=params["process_noise"],
        action_variability=jnp.array(
            [params["action_variability"], params["action_variability"]]
        ),
        action_cost=params["action_cost"],
        dt=dt_rw,
        T=500,
    )

    x = model.simulate(n=50, rng_key=random.PRNGKey(0))

    vels = jnp.diff(x, axis=-2)

    lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=100)

    plt.plot(lags[100:] * dt_rw, correls.mean(axis=0)[100:])
    plt.savefig("two_dim_correls.png")
