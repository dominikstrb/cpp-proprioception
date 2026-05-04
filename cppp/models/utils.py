import jax.numpy as jnp
from jax.scipy import linalg


def discretize_linear_system(A, B, dt):
    """
    Discretize continuous-time system x' = A x + B u
    using zero-order hold (exact method).

    Args:
        A: (n, n) array
        B: (n, m) array
        dt: scalar timestep

    Returns:
        Ad: (n, n) array
        Bd: (n, m) array
    """
    n = A.shape[0]
    m = B.shape[1]

    # Construct block matrix
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(A)
    M = M.at[:n, n:].set(B)

    # Matrix exponential
    M_exp = linalg.expm(M * dt)

    # Extract Ad and Bd
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:]

    return Ad, Bd


def van_loan_discretization(A, G, dt, Qc=None):
    """
    Compute discrete-time process noise covariance Qd.

    Args:
        A: (n, n)
        G: (n, r)
        Qc: (r, r) continuous noise covariance
        dt: scalar

    Returns:
        Qd: (n, n)
    """
    n = A.shape[0]

    if Qc is None:
        Qc = jnp.eye(G.shape[1])

    # Continuous noise covariance mapped into state space
    Q = G @ Qc @ G.T

    # Van Loan matrix
    M = jnp.block([[A, Q], [jnp.zeros_like(A), -A.T]])

    M_exp = linalg.expm(M * dt)

    Qd = M_exp[:n, n:]

    return Qd


def make_psd(M, eps=1e-6):
    """
    Make a symmetric matrix positive semi-definite by adding a small value to the diagonal.

    Args:
        M: (n, n) array
        eps: scalar, small value to add to the diagonal
    Returns:
        M_psd: (n, n) array, positive semi-definite version of M
    """
    M_sym = (M + M.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(M_sym)
    eigvals_clipped = jnp.clip(eigvals, a_min=eps)
    M_psd = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    return M_psd


if __name__ == "__main__":
    damping = 0.015
    m = 1.0
    tau = 0.066
    A_c = jnp.array([[0.0, 1.0, 0.0], [0.0, -damping / m, 1.0 / m], [0.0, 0.0, -1.0 / tau]])
    B = jnp.array([[0.0], [0.0], [1.0 / tau]])

    dt = 0.01
    A, B = discretize_linear_system(A_c, B, dt)
    Q = van_loan_discretization(A_c, 100.0 * B, dt)
    print(A.round(2))
    print(B)
    print(Q)

    A = jnp.array(
        [
            [1.0, dt, 0.0],
            [0.0, 1.0 - damping * dt / m, dt / m],
            [0.0, 0.0, 1.0 - dt / tau],
        ]
    )

    print(A.round(2))
