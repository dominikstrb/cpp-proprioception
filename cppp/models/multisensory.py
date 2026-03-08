import jax.numpy as jnp
from jax.scipy import linalg
from lqg import System
from lqg.utils import time_stack_spec


def multisensory_delay_system(A, B, V, Fs, Ws, Q, R, delays=[0, 1], T=500):

    d = A.shape[1]

    # get the maximum delay across all sensory modalities to determine how many past states we need to include in the extended state vector
    max_delay = max(delays)

    # stack up the dynamics matrices to work with an extended state vector
    # the extended state vector contains the current state and the past states up to the maximum delay
    # this is described in more detail in Izawa & Shadmehr (2008), eqn (3)
    A = linalg.block_diag(A, jnp.diag(jnp.zeros(d * max_delay))) + jnp.diag(
        jnp.ones(d * max_delay), k=-d
    )

    # stack up the control gain matrix to work with the extended state vector
    B = jnp.vstack([B] + [jnp.zeros_like(B)] * max_delay)

    # stack up the sensory feedback matrices to work with the extended state vector
    # here we apply the appropriate delay to each sensory modality by padding with zeros as needed
    # this is described in more detail in Crevecoeur et al. (2016)
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

    # stack up the dynamics noise covariance factors to work with the extended state vector
    V = linalg.block_diag(V, jnp.diag(jnp.zeros(d * max_delay)))
    # the sensory noise covariance factors are block diagonal, with each block corresponding to a different sensory modality
    W = linalg.block_diag(*Ws)

    # stack up the state cost matrix to work with the extended state vector
    # the cost is only applied to the current state, not the past states,
    # which means that the resulting Q is block diagonal with the original Q in the first block and zeros in the remaining blocks
    Q = linalg.block_diag(Q, *[jnp.zeros_like(Q)] * max_delay)

    # create a time-stacked system specification that can be used with the lqg package to solve for the optimal control policy
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


class UnisensoryDelayModelForceDynamics(System):
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

        A = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        B = dt * jnp.array([[0.0], [0.0], [1.0]])
        F = jnp.array([[1.0, -1.0, 0.0]])
        # TODO: proper noise model
        V = jnp.diag(jnp.array([process_noise, 0.0, action_variability]))
        Q = jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        R = jnp.array([[action_cost]])

        W_visual = jnp.diag(jnp.array([sigma]))

        spec = multisensory_delay_system(
            A, B, V, [F], [W_visual], Q, R, delays=[delay], T=T
        )
        super().__init__(actor=spec, dynamics=spec)


class BiasMultisensoryDelayModel(System):
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

        # generative model that has a bias term
        A = jnp.eye(3)
        B = dt * jnp.array([[0.0], [1.0], [0.0]])

        # the first element of the sensory feedback is the proprioceptive feedback
        # the second element of the sensory feedback is the visual feedback, which is biased by the third element of the state vector
        Fs = [jnp.array([[1.0, -1.0, 0.0]]), jnp.array([[0.0, -1.0, 1.0]])]

        # here, we apply the same noise to the position and the biased visual position,
        # which means that the noise is shared across the two sensory modalities
        # note how the third column is all zeros, which means that we ignore one element of the noise vector
        V = jnp.array(
            [
                [process_noise, 0.0, 0.0],
                [0.0, action_variability, 0.0],
                [process_noise, 0.0, 0.0],
            ]
        )
        Q = jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        R = jnp.array([[action_cost]])
        spec = multisensory_delay_system(
            A=A,
            B=B,
            V=V,
            Fs=Fs,
            Ws=[jnp.diag(jnp.array([sigma])) for sigma in sigmas],
            Q=Q,
            R=R,
            delays=delays,
            T=T,
        )

        super().__init__(actor=spec, dynamics=spec)


class SubjectiveBiasMultisensoryDelayModel(System):
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

        # generative model that has a bias term
        A = jnp.eye(3)
        B = dt * jnp.array([[0.0], [1.0], [0.0]])

        # the first element of the sensory feedback is the proprioceptive feedback
        # the second element of the sensory feedback is the visual feedback, which is biased by the third element of the state vector
        Fs = [jnp.array([[1.0, -1.0, 0.0]]), jnp.array([[0.0, -1.0, 1.0]])]
        
        # here, we apply the same noise to the position and the biased visual position,
        # which means that the noise is shared across the two sensory modalities
        # note how the third column is all zeros, which means that we ignore one element of the noise vector
        V = jnp.array(
            [
                [process_noise, 0.0, 0.0],
                [0.0, action_variability, 0.0],
                [process_noise, 0.0, 0.0],
            ]
        )
        # these cost matrices are not actually used in this case, but we need to specify them to create the system specification
        Q = jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        R = jnp.array([[action_cost]])
        dynamics = multisensory_delay_system(
            A=A,
            B=B,
            V=V,
            Fs=Fs,
            Ws=[jnp.diag(jnp.array([sigma])) for sigma in sigmas],
            Q=Q,
            R=R,
            delays=delays,
            T=T,
        )

        # internal model that does not know about the bias
        A_subj = jnp.eye(2)
        B_subj = dt * jnp.array([[0.0], [1.0]])
        F_subj = jnp.array([[1.0, -1.0]])
        V_subj = jnp.array([[process_noise, 0.0], [0.0, action_variability]])
        Q_subj = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
        R_subj = jnp.array([[action_cost]])

        actor = multisensory_delay_system(
            A_subj,
            B_subj,
            V_subj,
            [F_subj for _ in sigmas],
            [jnp.diag(jnp.array([sigma])) for sigma in sigmas],
            Q_subj,
            R_subj,
            delays=delays,
            T=T,
        )

        super().__init__(actor=actor, dynamics=dynamics)
