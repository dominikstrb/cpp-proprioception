from cppp.models.utils import discretize_linear_system, make_psd, van_loan_discretization


import jax.numpy as jnp
from jax.scipy import linalg


def point_mass_dynamics_matrices(damping, m, tau, action_variability, dt):
    # continuous-time dynamics of a point mass with viscous damping and a first-order muscle activation dynamics
    A_c = jnp.array([[0.0, 1.0, 0.0], [0.0, -damping / m, 1.0 / m], [0.0, 0.0, -1.0 / tau]])
    B_c = jnp.array([[0.0], [0.0], [1.0 / tau]])

    # discretize dynamics
    A, B = discretize_linear_system(A_c, B_c, dt)
    # discretize noise model using van Loan's method, which accounts for the effect of the control input on the noise covariance (makes fitting more stable)
    V = linalg.cholesky(make_psd(van_loan_discretization(A_c, action_variability * B_c, dt)))

    return A, B, V