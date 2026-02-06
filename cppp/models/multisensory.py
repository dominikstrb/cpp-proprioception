import jax.numpy as jnp
from jax.scipy import linalg
from lqg import System
from lqg.utils import time_stack_spec


def multisensory_delay_system(A, B, V, Fs, Ws, Q, R, delays=[0, 1], T=500):

    d = A.shape[1]

    max_delay = max(delays)

    A = linalg.block_diag(A, jnp.diag(jnp.zeros(d * max_delay))) + jnp.diag(
        jnp.ones(d * max_delay), k=-d
    )

    B = jnp.vstack([B] + [jnp.zeros_like(B)] * max_delay)

    F = jnp.vstack(
        [
            jnp.hstack(
                [
                    jnp.zeros((F.shape[0], F.shape[1] * delay)),
                    F,
                    jnp.zeros((F.shape[0], F.shape[1] * (max_delay - delay))),
                ]
            )
            for F, delay in zip(Fs, delays)
        ]
    )

    V = linalg.block_diag(V, jnp.diag(jnp.zeros(d * max_delay)))
    W = linalg.block_diag(*Ws)

    Q = linalg.block_diag(Q, *[jnp.zeros_like(Q)] * max_delay)

    spec = time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

    return spec


class UnisensoryDelayModel(System):
    def __init__(
        self,
        process_noise=1.0,
        sigma=1.0,
        action_variability=0.5,
        action_cost=0.1,
        dt=0.075,
        delay=1,
        T=1000,
    ):

        A = jnp.eye(2)
        B = dt * jnp.array([[0.0], [1.0]])
        F = jnp.array([[1.0, -1.0]])
        V = jnp.diag(jnp.array([process_noise, action_variability]))
        Q = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
        R = jnp.array([[action_cost]])

        W_visual = jnp.diag(jnp.array([sigma]))

        spec = multisensory_delay_system(
            A, B, V, [F], [W_visual], Q, R, delays=[delay], T=T
        )
        super().__init__(actor=spec, dynamics=spec)


class MultisensoryDelayModel(System):
    def __init__(
        self,
        process_noise=1.0,
        sigmas=[1.0, 1.0],
        action_variability=0.5,
        action_cost=0.1,
        dt=0.075,
        delays=[1, 1],
        T=1000,
    ):

        A = jnp.eye(2)
        B = dt * jnp.array([[0.0], [1.0]])
        F = jnp.array([[1.0, -1.0]])
        V = jnp.diag(jnp.array([process_noise, action_variability]))
        Q = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
        R = jnp.array([[action_cost]])

        spec = multisensory_delay_system(
            A,
            B,
            V,
            [F for _ in sigmas],
            [jnp.diag(jnp.array([sigma])) for sigma in sigmas],
            Q,
            R,
            delays=delays,
            T=T,
        )
        super().__init__(actor=spec, dynamics=spec)
