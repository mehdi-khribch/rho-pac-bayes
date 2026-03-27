"""
Classical estimators: MLE and conjugate Bayes posteriors.

These serve as baselines for comparison with the variational
rho-posterior across all three exponential family models.
"""

import numpy as np
from typing import Tuple


# ====================================================================
# Gaussian location model
# ====================================================================

def mle_gaussian(data: np.ndarray) -> np.ndarray:
    r"""Maximum likelihood estimator for :math:`N(\theta, I_d)`.

    .. math::
        \hat{\theta}_{\mathrm{MLE}} = \bar{X}

    Parameters
    ----------
    data : (n, d) ndarray

    Returns
    -------
    (d,) ndarray
    """
    return np.mean(data, axis=0)


def bayes_gaussian(
    data: np.ndarray,
    prior_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Conjugate Bayes posterior mean and std for :math:`N(\theta, I_d)`.

    Prior: :math:`\pi(\theta) = N(0, \sigma_0^2 I_d)`.
    Posterior: :math:`N(m, s^2 I_d)` with

    .. math::
        m = \frac{n \bar{X}}{1/\sigma_0^2 + n},
        \quad
        s = \frac{1}{\sqrt{1/\sigma_0^2 + n}}.

    Parameters
    ----------
    data : (n, d) ndarray
    prior_std : float
        Prior standard deviation :math:`\sigma_0`.

    Returns
    -------
    mean : (d,) ndarray
    std : (d,) ndarray
    """
    n, d = data.shape
    precision_post = 1.0 / prior_std ** 2 + n
    mean = n * np.mean(data, axis=0) / precision_post
    std = np.ones(d) / np.sqrt(precision_post)
    return mean, std


# ====================================================================
# Poisson intensity model
# ====================================================================

def mle_poisson(x: np.ndarray) -> float:
    r"""MLE for Poisson: :math:`\hat{\lambda} = \bar{X}`.

    Parameters
    ----------
    x : (n,) ndarray

    Returns
    -------
    float
    """
    return float(np.mean(x))


def bayes_poisson(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
) -> float:
    r"""Conjugate Bayes posterior mean for Poisson with Gamma prior.

    Prior: :math:`\lambda \sim \mathrm{Gamma}(a, b)` (shape, rate).
    Posterior mean:

    .. math::
        \hat{\lambda}_B = \frac{a + \sum x_i}{b + n}.

    Parameters
    ----------
    x : (n,) ndarray
    a, b : float
        Gamma prior hyperparameters.

    Returns
    -------
    float
    """
    return (a + np.sum(x)) / (b + len(x))


# ====================================================================
# Uniform scale model
# ====================================================================

def mle_uniform(x: np.ndarray) -> float:
    r"""MLE for Uniform[0, theta]: :math:`\hat{\theta} = X_{(n)}`.

    Parameters
    ----------
    x : (n,) ndarray

    Returns
    -------
    float
    """
    return float(np.max(x))


def bayes_uniform(
    x: np.ndarray,
    a: float = 0.5,
    alpha: float = 2.0,
) -> float:
    r"""Bayes posterior mean for Uniform[0, theta] with Pareto prior.

    Prior: :math:`\pi(\theta) \propto \theta^{-\alpha} \mathbb{1}_{\theta \geq a}`.
    Posterior mean (exists when :math:`n + \alpha > 2`):

    .. math::
        \hat{\theta}_B
        = \frac{n + \alpha - 1}{n + \alpha - 2}
          \cdot \max(a, X_{(n)}).

    Parameters
    ----------
    x : (n,) ndarray
    a : float
        Prior lower bound.
    alpha : float
        Prior tail index.

    Returns
    -------
    float
    """
    n = len(x)
    xmax = float(np.max(x))
    t0 = max(a, xmax)
    if n + alpha - 2 <= 0:
        raise ValueError("Posterior mean does not exist: n + alpha <= 2.")
    return (n + alpha - 1) / (n + alpha - 2) * t0


# ====================================================================
# Regression baselines
# ====================================================================

def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Ordinary least squares: :math:`\hat{\beta} = (X^\top X)^{-1} X^\top y`.

    Parameters
    ----------
    X : (n, p) ndarray
    y : (n,) ndarray

    Returns
    -------
    (p,) ndarray
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def bayes_regression(
    X: np.ndarray,
    y: np.ndarray,
    prior_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Conjugate Bayes posterior for linear regression with Gaussian prior.

    Prior: :math:`\beta \sim N(0, \sigma_0^2 I_p)`.
    Posterior precision: :math:`\Lambda = X^\top X + (1/\sigma_0^2) I_p`.
    Posterior mean: :math:`\Lambda^{-1} X^\top y`.

    Parameters
    ----------
    X : (n, p) ndarray
    y : (n,) ndarray
    prior_std : float

    Returns
    -------
    mean : (p,) ndarray
    cov : (p, p) ndarray
    """
    p = X.shape[1]
    Lambda = X.T @ X + np.eye(p) / prior_std ** 2
    mean = np.linalg.solve(Lambda, X.T @ y)
    cov = np.linalg.inv(Lambda)
    return mean, cov
