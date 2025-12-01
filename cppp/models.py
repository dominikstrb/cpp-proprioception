import jax.numpy as jnp
from jax.scipy import linalg
from lqg import Actor, System, Dynamics
from lqg.tracking.subjective import swap_dims


class CorrelatedRelativeObservationBoundedActor(System):
    def __init__(
        self,
        sigma,
        corr_chol,
        action_variability,
        action_cost,
        dim=2,
        process_noise=1.0,
        dt=1 / 60.0,
        T=1000,
    ):
        self.dim = dim
        self.process_noise = process_noise

        # dynamics model
        A = jnp.eye(4)
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0]])] * dim)

        # observation model
        F = linalg.block_diag(*[jnp.array([[1.0, -1.0]])] * dim)

        # noise model
        V = jnp.diag(
            jnp.concatenate(jnp.array([(process_noise, a) for a in action_variability]))
        )

        W = jnp.diag(sigma) @ corr_chol

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1.0, -1.0], [-1.0, 1.0]])] * dim)
        R = jnp.diag(action_cost)

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


class CorrelatedObservationBoundedActor(System):
    def __init__(
        self,
        sigma_target,
        sigma_cursor,
        corr_chol,
        action_variability,
        action_cost,
        dim=2,
        process_noise=1.0,
        dt=1.0 / 60.0,
        T=1000,
    ):
        self.dim = dim
        self.process_noise = process_noise

        # dimensionality
        d = 2 * dim

        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0]])] * dim)

        # observation model
        F = jnp.eye(d)

        # noise model
        V = jnp.diag(
            jnp.concatenate(jnp.array([(process_noise, a) for a in action_variability]))
        )

        W_target = jnp.diag(sigma_target) @ corr_chol
        W_cursor = jnp.diag(sigma_cursor)
        W = linalg.block_diag(W_target, W_cursor)

        dims = list(range(d)[::2]) + list(range(d)[1::2])
        W = W[dims, :][:, dims]

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1.0, -1.0], [-1.0, 1.0]])] * dim)
        R = jnp.diag(action_cost)

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


class CorrelatedObservationSubjectiveActor(System):
    def __init__(
        self,
        sigma_target,
        sigma_cursor,
        corr_chol,
        action_variability,
        action_cost,
        dim=2,
        process_noise=1.0,
        subj_noise=1.0,
        subj_vel_noise=0.1,
        dt=1.0 / 60.0,
        T=1000,
    ):
        self.dim = dim
        self.process_noise = process_noise

        # dimensionality
        d = 2 * dim

        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0]])] * dim)

        # observation model
        F = jnp.eye(d)

        # noise model
        V = jnp.diag(
            jnp.concatenate(jnp.array([(process_noise, a) for a in action_variability]))
        )

        W_target = jnp.diag(sigma_target) @ corr_chol
        W_cursor = jnp.diag(sigma_cursor)
        W = linalg.block_diag(W_target, W_cursor)

        dims = list(range(d)[::2]) + list(range(d)[1::2])
        W = W[dims, :][:, dims]

        dynamics = Dynamics(A=A, B=B, F=F, V=V, W=W, T=T)

        # subjective dynamics model
        A = linalg.block_diag(
            *[jnp.array([[1.0, 0.0, dt], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * dim
        )
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0], [0.0]])] * dim)
        F = linalg.block_diag(*[jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])] * dim)

        V = jnp.diag(
            jnp.concatenate(jnp.array([(subj_noise, a, subj_vel_noise) for a in action_variability]))
        )

        Q = linalg.block_diag(
            *[jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * dim
        )
        R = jnp.eye(B.shape[1]) * action_cost

        dims = swap_dims(A.shape[0], dim)

        A = A[dims, :][:, dims]
        B = B[dims, :]
        V = V[dims, :]
        F = F[:, dims]
        Q = Q[dims, :][:, dims]

        actor = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)
        super().__init__(actor=actor, dynamics=dynamics)


class CorrelatedObservationJerkBoundedActor(System):
    def __init__(
        self,
        sigma_target,
        sigma_cursor,
        corr_chol,
        action_variability,
        action_cost,
        jerk_cost=0.0,
        process_noise=1.0,
        dt=1.0 / 60.0,
        dim=2,  # TODO: currently only dim=2 is allowed
        T=1000,
    ):
        dim = 2
        # dimensionality
        d = 2 * dim

        m = 1.0

        self.process_noise = process_noise

        # dynamics model
        A = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt / m, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt / m],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        B = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [dt, 0.0],
                [0.0, dt],
            ]
        )

        # observation model
        F = jnp.eye(4, 8)

        # noise model
        V = jnp.array(
            [
                [process_noise, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, action_variability[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, process_noise, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, action_variability[1], 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        W_target = jnp.diag(sigma_target) @ corr_chol
        W_cursor = jnp.diag(sigma_cursor)
        W = linalg.block_diag(W_target, W_cursor)

        dims = list(range(d)[::2]) + list(range(d)[1::2])
        W = W[dims, :][:, dims]

        # cost function
        # state cost (target tracking) + action effort
        Q = jnp.array(
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action_cost[0], 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action_cost[1]],
            ]
        )

        # jerk cost
        R = jnp.eye(B.shape[1]) * jerk_cost * dt**2

        dynamics = Dynamics(A=A, B=B, F=F, V=V, W=W, T=T)

        actor = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(dynamics=dynamics, actor=actor)

if __name__ == "__main__":
    from lqg import xcorr
    from jax import random

    import matplotlib.pyplot as plt

    dt_rw = 1.0 / 60.0

    params = {"process_noise": 1.0, "sigma_target": 6.0, "sigma_cursor": 6.0, "action_variability": 0.5, "action_cost": 0.01}

    model = CorrelatedObservationSubjectiveActor(sigma_target=jnp.array([params["sigma_target"], params["sigma_target"]]),
                                                sigma_cursor=jnp.array([params["sigma_cursor"], params["sigma_cursor"]]),
                                                process_noise=params["process_noise"],
                                                action_variability=jnp.array([params["action_variability"], params["action_variability"]]),
                                                action_cost=params["action_cost"],
                                                corr_chol=jnp.array([[1.0, 0.5], [0.0, jnp.sqrt(1 - 0.5 ** 2)]]),
                                                dt=dt_rw,
                                                subj_noise=0.1,
                                                subj_vel_noise=100.,
                                                T=500)

    x = model.simulate(n=50, rng_key=random.PRNGKey(0))

    vels = jnp.diff(x, axis=-2)

    # plt.plot(x[0, :, 0])  # position
    # plt.plot(x[0, :, 1])  # response
    lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=100)

    plt.plot(lags[100:] * dt_rw, correls.mean(axis=0)[100:])
    plt.show()
