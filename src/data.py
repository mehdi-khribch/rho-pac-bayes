"""
Data-generating processes for contaminated exponential families.

Each function generates n i.i.d. observations from the epsilon-contaminated mixture

    P*_epsilon = (1 - epsilon) P_{theta_star} + epsilon Q,

where P_{theta_star} is the clean distribution and Q is the contaminating distribution.
The contamination models match those described in Section 3 of the paper.
"""

import numpy as np


# ====================================================================
# Gaussian location: P* = (1-epsilon)N(0,1) + epsilon*N(8,1)
# ====================================================================

def generate_contaminated_gaussian(
    n: int = 200,
    d: int = 1,
    epsilon: float = 0.10,
    seed: int | None = None,
) -> np.ndarray:
    r"""Contaminated Gaussian location data.

    Clean distribution: N(0, I_d).
    Contamination:      N(8*e_1, I_d), where e_1 is the first standard basis vector.

    Parameters
    ----------
    n : int
        Sample size.
    d : int
        Dimension.
    epsilon : float
        Contamination proportion.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    (n, d) ndarray
    """
    rng = np.random.default_rng(seed)
    n_clean = rng.binomial(n, 1 - epsilon)
    clean = rng.standard_normal(size=(n_clean, d))
    contam = rng.standard_normal(size=(n - n_clean, d))
    contam[:, 0] += 8.0
    data = np.vstack([clean, contam])
    rng.shuffle(data)
    return data


# ====================================================================
# Poisson intensity: P* = (1-epsilon)Pois(3) + epsilon*Pois(30)
# ====================================================================

def generate_contaminated_poisson(
    n: int = 200,
    lam0: float = 3.0,
    epsilon: float = 0.10,
    lam_out: float = 30.0,
    seed: int | None = None,
) -> np.ndarray:
    r"""Contaminated Poisson count data.

    Clean distribution:  Pois(lambda_0).
    Contamination:       Pois(lambda_out).

    Parameters
    ----------
    n : int
        Sample size.
    lam0 : float
        True intensity (default 3).
    epsilon : float
        Contamination proportion.
    lam_out : float
        Outlier intensity (default 30).
    seed : int, optional
        Random seed.

    Returns
    -------
    (n,) ndarray of floats
    """
    rng = np.random.default_rng(seed)
    is_out = rng.uniform(size=n) < epsilon
    x = rng.poisson(lam0, size=n)
    x[is_out] = rng.poisson(lam_out, size=is_out.sum())
    return x.astype(np.float64)


# ====================================================================
# Uniform scale: P* = (1-epsilon)U[0,1] + epsilon*U[101,102]
# ====================================================================

def generate_uniform_clean(
    n: int = 200,
    t0: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Clean Uniform[0, t_0] data."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, t0, size=n)


def generate_uniform_contaminated(
    n: int = 200,
    t0: float = 1.0,
    epsilon: float = 0.10,
    seed: int | None = None,
) -> np.ndarray:
    r"""Contaminated Uniform scale data.

    Clean distribution:  U[0, theta_star]  with theta_star = t_0.
    Contamination:       U[101, 102], placing outliers far from the support.

    Following the paper: P*_epsilon = (1-epsilon) U[0, theta_star] + epsilon U[theta_star+100, theta_star+100+1].

    Parameters
    ----------
    n : int
        Sample size.
    t0 : float
        True scale parameter theta_star (default 1).
    epsilon : float
        Contamination proportion.
    seed : int, optional
        Random seed.

    Returns
    -------
    (n,) ndarray
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, t0, size=n)
    is_out = rng.uniform(size=n) < epsilon
    x[is_out] = rng.uniform(t0 + 100.0, t0 + 101.0, size=is_out.sum())
    return x


# ====================================================================
# Regression: Fourier basis with Pareto contamination
# ====================================================================

def generate_fourier_regression(
    n: int = 200,
    K: int = 6,
    epsilon: float = 0.10,
    pareto_shift: float = 6.0,
    pareto_shape: float = 2.0,
    seed: int | None = None,
):
    r"""Fourier basis regression with contaminated noise.

    True function: f*(w) = sin(2*pi*w) + 0.3*cos(6*pi*w).
    Design: Fourier basis with K frequency components (p = 2K+1 features).
    Noise:  (1-epsilon)N(0,1) + epsilon*Pareto_two-sided(shift, shape).

    Parameters
    ----------
    n : int
        Sample size.
    K : int
        Number of Fourier frequency components.
    epsilon : float
        Contamination rate.
    pareto_shift, pareto_shape : float
        Parameters of the two-sided Pareto outlier distribution.
    seed : int, optional
        Random seed.

    Returns
    -------
    Phi : (n, p) ndarray
        Fourier design matrix.
    y : (n,) ndarray
        Response vector.
    beta_true : (p,) ndarray
        True coefficient vector.
    w : (n,) ndarray
        Design points in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Design points
    w = np.linspace(0, 1, n)

    # Fourier basis: [1, sin(2*pi*w), cos(2*pi*w), sin(4*pi*w), cos(4*pi*w), ...]
    p = 2 * K + 1
    Phi = np.ones((n, p))
    for k in range(1, K + 1):
        Phi[:, 2 * k - 1] = np.sin(2 * np.pi * k * w)
        Phi[:, 2 * k] = np.cos(2 * np.pi * k * w)

    # True coefficients: f*(w) = sin(2*pi*w) + 0.3*cos(6*pi*w)
    beta_true = np.zeros(p)
    beta_true[1] = 1.0     # sin(2*pi*w)
    beta_true[6] = 0.3     # cos(6*pi*w)

    # Signal
    f_true = Phi @ beta_true

    # Contaminated noise
    noise = rng.standard_normal(n)
    is_out = rng.uniform(size=n) < epsilon
    n_out = is_out.sum()
    if n_out > 0:
        signs = rng.choice([-1.0, 1.0], size=n_out)
        pareto_vals = (rng.pareto(pareto_shape, size=n_out) + 1) * pareto_shift
        noise[is_out] = signs * pareto_vals

    y = f_true + noise
    return Phi, y, beta_true, w


def generate_correlated_regression(
    n: int = 100,
    d: int = 10,
    rho_corr: float = 0.7,
    sparsity: int = 5,
    epsilon: float = 0.10,
    pareto_shift: float = 10.0,
    pareto_shape: float = 1.5,
    seed: int | None = None,
):
    r"""Correlated design regression with sparse parameters.

    Design: X ~ N(0, Sigma) with Toeplitz Sigma_{jk} = rho^|j-k|.
    True beta_star: first `sparsity` entries drawn from U[-3,3], rest zero.
    Noise: (1-epsilon)N(0,1) + epsilon*Pareto_two-sided(shift, shape).

    Parameters
    ----------
    n, d : int
        Sample size and number of features.
    rho_corr : float
        Toeplitz correlation parameter.
    sparsity : int
        Number of nonzero entries in beta_star.
    epsilon : float
        Contamination rate.
    seed : int, optional
        Random seed.

    Returns
    -------
    X : (n, d+1) ndarray
        Design matrix with intercept column.
    y : (n,) ndarray
        Response.
    beta_true : (d+1,) ndarray
        True coefficients (first entry = intercept = 0).
    """
    rng = np.random.default_rng(seed)

    # Toeplitz covariance
    indices = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    Sigma = rho_corr ** indices
    L = np.linalg.cholesky(Sigma)
    X_raw = rng.standard_normal((n, d)) @ L.T

    # Add intercept
    X = np.column_stack([np.ones(n), X_raw])

    # Sparse true beta
    beta_true = np.zeros(d + 1)
    beta_true[1 : sparsity + 1] = rng.uniform(-3, 3, size=sparsity)

    # Signal + contaminated noise
    y_clean = X @ beta_true
    noise = rng.standard_normal(n)
    is_out = rng.uniform(size=n) < epsilon
    n_out = is_out.sum()
    if n_out > 0:
        signs = rng.choice([-1.0, 1.0], size=n_out)
        pareto_vals = (rng.pareto(pareto_shape, size=n_out) + 1) * pareto_shift
        noise[is_out] = signs * pareto_vals

    y = y_clean + noise
    return X, y, beta_true
