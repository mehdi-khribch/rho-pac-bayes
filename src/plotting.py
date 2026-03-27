"""
Publication-quality figure generation (Python side).

Produces matplotlib figures matching the paper's style: posterior risk,
RMSE, and posterior density plots for each model. These are quick-check
figures; final publication figures are produced by the R scripts in R/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ====================================================================
# Style defaults
# ====================================================================

def set_paper_style():
    """Apply a clean matplotlib style suitable for academic papers."""
    plt.rcParams.update({
        "figure.figsize": (7, 4),
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ====================================================================
# Posterior risk plot
# ====================================================================

def plot_posterior_risk(
    summary_df: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """Posterior risk (mean squared error) vs contamination rate epsilon.

    Parameters
    ----------
    summary_df : DataFrame
        Must contain columns: epsilon, MLE_BayesRisk, Bayes_BayesRisk, Rho_BayesRisk.
    save_path : str or Path, optional
        If provided, save figure to this path.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()
    fig, ax = plt.subplots()

    eps_pct = summary_df["epsilon"] * 100

    ax.plot(eps_pct, summary_df["MLE_BayesRisk"],
            "o-", color="tab:red", label="MLE")
    ax.plot(eps_pct, summary_df["Bayes_BayesRisk"],
            "s--", color="tab:blue", label="Bayes")
    ax.plot(eps_pct, summary_df["Rho_BayesRisk"],
            "^-", color="tab:green", label="Rho-posterior")

    ax.set_xlabel(r"Contamination rate $\varepsilon$ (%)")
    ax.set_ylabel("Posterior risk")
    ax.legend(frameon=True, fancybox=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ====================================================================
# RMSE plot
# ====================================================================

def plot_rmse(
    summary_df: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """RMSE vs contamination rate epsilon.

    Parameters
    ----------
    summary_df : DataFrame
        Must contain columns: epsilon, MLE_RMSE, Bayes_RMSE, Rho_RMSE.
    """
    set_paper_style()
    fig, ax = plt.subplots()

    eps_pct = summary_df["epsilon"] * 100

    ax.plot(eps_pct, summary_df["MLE_RMSE"],
            "o-", color="tab:red", label="MLE")
    ax.plot(eps_pct, summary_df["Bayes_RMSE"],
            "s--", color="tab:blue", label="Bayes")
    ax.plot(eps_pct, summary_df["Rho_RMSE"],
            "^-", color="tab:green", label="Rho-posterior")

    ax.set_xlabel(r"Contamination rate $\varepsilon$ (%)")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=True, fancybox=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ====================================================================
# Density plot -- Gaussian
# ====================================================================

def plot_density_gaussian(
    trials_df: pd.DataFrame,
    epsilon: float = 0.10,
    prior_std: float = 2.0,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """Posterior density comparison for a single Gaussian dataset.

    Shows the prior, conjugate Bayes posterior, and variational rho-posterior
    densities, together with vertical lines at the true value and MLE.

    Parameters
    ----------
    trials_df : DataFrame
        Per-trial results; first row at the given epsilon is used.
    epsilon : float
        Contamination rate to select.
    prior_std : float
        Prior standard deviation for the density curve.
    """
    set_paper_style()
    row = trials_df[trials_df["epsilon"] == epsilon].iloc[0]

    theta_mle = row["theta_mle"]
    theta_bayes = row["theta_bayes_mean"]
    theta_rho = row["theta_rho_mean"]
    std_bayes = row["std_bayes"]
    std_rho = row["std_rho"]

    vals = np.array([theta_mle, theta_bayes, theta_rho, 0.0])
    grid = np.linspace(vals.min() - 2.0, vals.max() + 2.0, 400)

    prior_pdf = stats.norm.pdf(grid, loc=0.0, scale=prior_std)
    bayes_pdf = stats.norm.pdf(grid, loc=theta_bayes, scale=std_bayes)
    rho_pdf = stats.norm.pdf(grid, loc=theta_rho, scale=std_rho)

    fig, ax = plt.subplots()

    ax.plot(grid, prior_pdf, "--", color="gray", lw=1.5,
            label=f"Prior N(0, {prior_std}$^2$)")
    ax.plot(grid, bayes_pdf, "-.", color="tab:blue", lw=2,
            label="Bayes posterior")
    ax.plot(grid, rho_pdf, "-", color="tab:green", lw=2,
            label="Rho-posterior")

    ax.axvline(0.0, color="black", ls=":", lw=1.5,
               label=r"True $\theta^\star = 0$")
    ax.axvline(theta_mle, color="tab:red", ls="--", lw=2,
               label=f"MLE = {theta_mle:.3f}")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fancybox=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ====================================================================
# Density plot -- Poisson
# ====================================================================

def plot_density_poisson(
    trials_df: pd.DataFrame,
    epsilon: float = 0.10,
    lam0: float = 3.0,
    prior_std: float = 2.0,
    a_gamma: float = 1.0,
    b_gamma: float = 1.0,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """Posterior density comparison for a single Poisson dataset.

    Shows Gamma prior, conjugate Gamma posterior, and lognormal rho-posterior.
    """
    set_paper_style()
    row = trials_df[trials_df["epsilon"] == epsilon].iloc[0]

    lam_mle = row["lam_mle"]
    lam_bayes = row["lam_bayes_mean"]
    lam_rho = row["lam_rho_mean"]
    eta_std = row["eta_std_rho"]

    grid = np.linspace(0.01, max(lam_mle, lam_bayes, lam_rho, lam0) * 2.5, 400)

    # Gamma prior and posterior
    n_obs = int(row["n"])
    # Approximate: use lam_star * n_obs for sum
    a_post = a_gamma + lam_bayes * n_obs  # rough
    b_post = b_gamma + n_obs
    prior_pdf = stats.gamma(a=a_gamma, scale=1.0 / b_gamma).pdf(grid)
    bayes_pdf = stats.gamma(a=a_post, scale=1.0 / b_post).pdf(grid)

    # Rho-posterior as lognormal
    eta_mean = np.log(lam_rho)
    rho_pdf = stats.lognorm(s=eta_std, scale=np.exp(eta_mean)).pdf(grid)

    fig, ax = plt.subplots()

    ax.plot(grid, prior_pdf, "--", color="gray", lw=1.5,
            label=f"Prior Gamma({a_gamma}, {b_gamma})")
    ax.plot(grid, bayes_pdf, "-.", color="tab:blue", lw=2,
            label="Bayes posterior (Gamma)")
    ax.plot(grid, rho_pdf, "-", color="tab:green", lw=2,
            label="Rho-posterior (lognormal)")

    ax.axvline(lam0, color="black", ls=":", lw=1.5,
               label=r"True $\lambda^\star$")
    ax.axvline(lam_mle, color="tab:red", ls="--", lw=2,
               label=f"MLE = {lam_mle:.2f}")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fancybox=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ====================================================================
# Density plot -- Uniform
# ====================================================================

def plot_density_uniform(
    trials_df: pd.DataFrame,
    epsilon: float = 0.10,
    t0: float = 1.0,
    prior_std: float = 2.0,
    a_prior: float = 0.5,
    alpha_prior: float = 2.0,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """Posterior density comparison for a single Uniform dataset.

    Shows Pareto prior, exact Pareto posterior, and lognormal rho-posterior.
    """
    set_paper_style()
    row = trials_df[trials_df["epsilon"] == epsilon].iloc[0]

    t_mle = row["theta_mle"]
    t_bayes = row["theta_bayes_mean"]
    t_rho = row["theta_rho_mean"]
    u_std = row["std_rho"]

    t_max = max(t_mle, t_bayes, t_rho, t0) + 2.0
    grid = np.linspace(a_prior, min(t_max, 5.0), 400)

    # Pareto prior: pi(t) proportional to t^{-alpha} for t >= a
    prior_unnorm = grid ** (-alpha_prior)
    prior_pdf = prior_unnorm / np.trapezoid(prior_unnorm, grid)

    # Pareto posterior: g(t|X) proportional to t^{-(n + alpha)} for t >= max(a, X_(n))
    n_obs = int(row["n"]) if "n" in row.index else 200
    post_unnorm = grid ** (-(n_obs + alpha_prior))
    post_unnorm[grid < max(a_prior, t_mle)] = 0.0
    area = np.trapezoid(post_unnorm, grid)
    post_pdf = post_unnorm / area if area > 0 else post_unnorm

    # Rho-posterior: lognormal
    u_mean = np.log(t_rho)
    rho_pdf = stats.lognorm(s=max(u_std, 0.01), scale=np.exp(u_mean)).pdf(grid)

    fig, ax = plt.subplots()

    ax.plot(grid, prior_pdf, "--", color="gray", lw=1.5,
            label=r"Prior $\pi(\theta) \propto \theta^{-\alpha}$")
    ax.plot(grid, post_pdf, "-.", color="tab:blue", lw=2,
            label="Bayes posterior (exact)")
    ax.plot(grid, rho_pdf, "-", color="tab:green", lw=2,
            label="Rho-posterior (lognormal)")

    ax.axvline(t0, color="black", ls=":", lw=1.5,
               label=r"True $\theta^\star$")
    ax.axvline(t_mle, color="tab:red", ls="--", lw=2,
               label=f"MLE = {t_mle:.2f}")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fancybox=True)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


# ====================================================================
# Predicted vs fitted (correlated regression)
# ====================================================================

def plot_predicted_vs_true(
    pred_df: pd.DataFrame,
    epsilon: float = 0.10,
    save_path: str | Path | None = None,
    title: str = "",
) -> plt.Figure:
    """Predicted vs true values for the correlated regression setting."""
    set_paper_style()
    df = pred_df[pred_df["epsilon"] == epsilon]

    fig, ax = plt.subplots()
    ax.scatter(df["y_true"], df["y_pred_rho"], s=15, alpha=0.6, color="tab:green")
    lims = [
        min(df["y_true"].min(), df["y_pred_rho"].min()),
        max(df["y_true"].max(), df["y_pred_rho"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlabel(r"True $X \beta^\star$")
    ax.set_ylabel(r"Predicted $X \hat{\beta}_\rho$")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig
