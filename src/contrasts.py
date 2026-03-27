"""
Bounded contrast functions and likelihood ratio computations.

The Hellinger contrast psi(x) = (sqrt(x) - 1) / (sqrt(x) + 1) replaces the
unbounded log-likelihood with a bounded function mapping R+ to [-1, 1].
This module provides psi and the model-specific likelihood ratios needed to
evaluate the empirical psi-risk for each exponential family.

References
----------
Baraud, Y. & Birge, L. (2018). Rho-estimators revisited.
Baraud, Y. (2020). Robust Bayes-like estimation.
"""

import torch


# ====================================================================
# Hellinger contrast
# ====================================================================

def psi_hellinger(x: torch.Tensor) -> torch.Tensor:
    r"""Bounded Hellinger contrast.

    .. math::
        \psi(x) = \frac{\sqrt{x} - 1}{\sqrt{x} + 1},
        \qquad x \geq 0.

    Satisfies psi in [-1, 1], psi(1) = 0, and psi(1/x) = -psi(x).

    Parameters
    ----------
    x : torch.Tensor
        Non-negative likelihood ratios.

    Returns
    -------
    torch.Tensor
        Contrast values in [−1, 1].
    """
    x = torch.clamp(x, min=1e-12)
    s = torch.sqrt(x)
    return (s - 1.0) / (s + 1.0)


# ====================================================================
# Gaussian likelihood ratio
# ====================================================================

def gaussian_likelihood_ratio(
    x: torch.Tensor,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
) -> torch.Tensor:
    r"""Likelihood ratio p_{theta_2}(x) / p_{theta_1}(x) for Gaussian N(theta, I_d).

    Parameters
    ----------
    x : (n, d) tensor
        Observed data.
    theta1 : (k1, d) tensor
        First set of parameter values.
    theta2 : (k2, d) tensor
        Second set of parameter values (competitors).

    Returns
    -------
    (n, k1, k2) tensor
        Entry [i, j, l] = p_{theta_2^l}(x_i) / p_{theta_1^j}(x_i).
    """
    dist1 = torch.cdist(x, theta1).pow(2)  # (n, k1)
    dist2 = torch.cdist(x, theta2).pow(2)  # (n, k2)
    log_ratio = -0.5 * (dist2.unsqueeze(1) - dist1.unsqueeze(2))
    return torch.exp(torch.clamp(log_ratio, min=-40.0, max=40.0))


def empirical_psi_risk_gaussian(
    x: torch.Tensor,
    theta1: torch.Tensor,
    theta2: torch.Tensor,
) -> torch.Tensor:
    r"""Empirical psi-risk matrix for the Gaussian location model.

    .. math::
        \hat{R}_\psi(\theta_1, \theta_2)
        = \frac{1}{n} \sum_{i=1}^{n}
          \psi\!\left(\frac{p_{\theta_2}(x_i)}{p_{\theta_1}(x_i)}\right).

    Returns
    -------
    (k1, k2) tensor
    """
    ratios = gaussian_likelihood_ratio(x, theta1, theta2)
    return psi_hellinger(ratios).mean(dim=0)


# ====================================================================
# Poisson likelihood ratio
# ====================================================================

def poisson_likelihood_ratio(
    x: torch.Tensor,
    eta1: torch.Tensor,
    eta2: torch.Tensor,
) -> torch.Tensor:
    r"""Likelihood ratio for Poisson(exp(eta)) in the natural parameterisation.

    Parameters
    ----------
    x : (n,) tensor
        Count data.
    eta1 : (k1,) tensor
        Log-intensities (main).
    eta2 : (k2,) tensor
        Log-intensities (competitor).

    Returns
    -------
    (n, k1, k2) tensor
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    x = x.to(dtype=torch.float32)

    x_b = x.view(-1, 1, 1)
    eta1_b = eta1.view(1, -1, 1)
    eta2_b = eta2.view(1, 1, -1)

    lam1 = torch.exp(eta1_b)
    lam2 = torch.exp(eta2_b)

    log_ratio = (lam1 - lam2) + x_b * (torch.log(lam2) - torch.log(lam1))
    return torch.exp(torch.clamp(log_ratio, min=-40.0, max=40.0))


def empirical_psi_risk_poisson(
    x: torch.Tensor,
    eta1: torch.Tensor,
    eta2: torch.Tensor,
) -> torch.Tensor:
    """Empirical psi-risk matrix for the Poisson model."""
    ratios = poisson_likelihood_ratio(x, eta1, eta2)
    return psi_hellinger(ratios).mean(dim=0)


# ====================================================================
# Uniform likelihood ratio
# ====================================================================

def uniform_likelihood_ratio(
    x: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    r"""Likelihood ratio for Uniform[0, exp(u)].

    Reparametrises theta = exp(u) so that the variational family operates on u in R.

    Parameters
    ----------
    x : (n,) tensor
        Observations in [0, theta].
    u1 : (k1,) tensor
        Log-scale parameters (main).
    u2 : (k2,) tensor
        Log-scale parameters (competitor).

    Returns
    -------
    (n, k1, k2) tensor
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    x = x.to(dtype=torch.float32)

    x_b = x.view(-1, 1, 1)
    u1_b = u1.view(1, -1, 1)
    u2_b = u2.view(1, 1, -1)

    t1 = torch.exp(u1_b)
    t2 = torch.exp(u2_b)

    n, k1, k2 = x_b.shape[0], u1_b.shape[1], u2_b.shape[2]
    log_ratio = torch.zeros((n, k1, k2), device=x.device)

    mask1_exp = (x_b <= t1).expand(n, k1, k2)
    mask2_exp = (x_b <= t2).expand(n, k1, k2)
    t1_exp = t1.expand(n, k1, k2)
    t2_exp = t2.expand(n, k1, k2)

    # Both densities positive: ratio = t1 / t2
    both = mask1_exp & mask2_exp
    log_ratio[both] = (torch.log(t1_exp) - torch.log(t2_exp))[both]

    # Only f2 > 0: ratio -> +inf
    log_ratio[(~mask1_exp) & mask2_exp] = 40.0

    # Only f1 > 0: ratio -> 0
    log_ratio[mask1_exp & (~mask2_exp)] = -40.0

    return torch.exp(torch.clamp(log_ratio, min=-40.0, max=40.0))


def empirical_psi_risk_uniform(
    x: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """Empirical psi-risk matrix for the Uniform scale model."""
    ratios = uniform_likelihood_ratio(x, u1, u2)
    return psi_hellinger(ratios).mean(dim=0)
