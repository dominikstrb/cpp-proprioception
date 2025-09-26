import jax.numpy as jnp
from jax.scipy import linalg
from lqg import Actor, System, Dynamics


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
                [0., 0.0],
                [0.0, 0.],
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