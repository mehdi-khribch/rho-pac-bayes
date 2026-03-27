"""
Monte Carlo evaluation loops and summary statistics.

Each evaluate_* function runs T independent replications of the experiment,
collecting point estimates from MLE, conjugate Bayes, and the variational
rho-posterior. Results are returned as pandas DataFrames ready for plotting
and CSV export.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import (
    generate_contaminated_gaussian,
    generate_contaminated_poisson,
    generate_uniform_clean,
    generate_uniform_contaminated,
    generate_fourier_regression,
    generate_correlated_regression,
)
from .baselines import (
    mle_gaussian,
    bayes_gaussian,
    mle_poisson,
    bayes_poisson,
    mle_uniform,
    bayes_uniform,
    ols,
    bayes_regression,
)
from .optimizers import GaussianOptimizer, PoissonOptimizer, UniformOptimizer
from .regression import RegressionOptimizer


# ====================================================================
# Gaussian location
# ====================================================================

def evaluate_gaussian(
    n_samples: int = 200,
    d: int = 1,
    epsilon_values: List[float] | None = None,
    n_trials: int = 1000,
    tau: float = 0.5,
    prior_std: float = 2.0,
    n_iter_opt: int = 200,
    n_mc_opt: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo evaluation for the Gaussian location model.

    Parameters
    ----------
    n_samples : int
        Sample size n.
    d : int
        Dimension.
    epsilon_values : list of float
        Contamination rates to evaluate.
    n_trials : int
        Number of independent replications T.
    tau : float
        Temperature scaling: lambda = tau * n.
    prior_std : float
        Prior standard deviation.
    n_iter_opt : int
        Optimisation iterations per trial.
    n_mc_opt : int
        Monte Carlo samples per optimisation step.

    Returns
    -------
    summary_df : DataFrame
        Aggregated metrics (RMSE, Bayes risk, std) per epsilon.
    trials_df : DataFrame
        Per-trial estimates and errors.
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.08, 0.10]

    lambda_reg = tau * n_samples
    theta_star = np.zeros(d)
    trials_rows = []

    for eps in epsilon_values:
        desc = f"Gaussian eps={eps:.2f}"
        for trial in tqdm(range(n_trials), desc=desc, leave=False):
            data = generate_contaminated_gaussian(
                n=n_samples, d=d, epsilon=eps, seed=trial,
            )

            # MLE
            est_mle = mle_gaussian(data)
            err_mle = float(np.linalg.norm(est_mle - theta_star))

            # Bayes
            est_bayes, std_bayes = bayes_gaussian(data, prior_std=prior_std)
            err_bayes = float(np.linalg.norm(est_bayes - theta_star))

            # Rho-posterior
            opt = GaussianOptimizer(data, lambda_reg=lambda_reg, prior_std=prior_std)
            opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
            est_rho, std_rho = opt.get_estimate(use_polyak=True)
            err_rho = float(np.linalg.norm(est_rho - theta_star))

            trials_rows.append(dict(
                epsilon=eps, tau=tau, n=n_samples, lambda_reg=lambda_reg,
                trial=trial, seed=trial,
                theta_star=0.0,
                theta_mle=float(est_mle[0]),
                theta_bayes_mean=float(est_bayes[0]),
                theta_rho_mean=float(est_rho[0]),
                std_bayes=float(std_bayes[0]),
                std_rho=float(std_rho[0]),
                error_mle=err_mle, error_bayes_mean=err_bayes, error_rho_mean=err_rho,
                risk_mle=err_mle**2, risk_bayes=err_bayes**2, risk_rho=err_rho**2,
            ))

    trials_df = pd.DataFrame(trials_rows)
    summary_df = _summarise(trials_df)
    return summary_df, trials_df


# ====================================================================
# Poisson intensity
# ====================================================================

def evaluate_poisson(
    n_samples: int = 200,
    lam0: float = 3.0,
    lam_out: float = 30.0,
    epsilon_values: List[float] | None = None,
    n_trials: int = 1000,
    tau: float = 0.5,
    prior_std: float = 2.0,
    a_gamma: float = 1.0,
    b_gamma: float = 1.0,
    n_iter_opt: int = 400,
    n_mc_opt: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo evaluation for the Poisson intensity model.

    Returns
    -------
    summary_df, trials_df : DataFrames
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.10, 0.20]

    lambda_reg = tau * n_samples
    trials_rows = []

    for eps in epsilon_values:
        desc = f"Poisson eps={eps:.2f}"
        for trial in tqdm(range(n_trials), desc=desc, leave=False):
            x = generate_contaminated_poisson(
                n=n_samples, lam0=lam0, epsilon=eps,
                lam_out=lam_out, seed=trial,
            )

            est_mle = mle_poisson(x)
            est_bayes = bayes_poisson(x, a=a_gamma, b=b_gamma)

            opt = PoissonOptimizer(x, lambda_reg=lambda_reg, prior_std=prior_std)
            opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
            est_rho, eta_std = opt.get_estimate(use_polyak=True)

            trials_rows.append(dict(
                epsilon=eps, tau=tau, n=n_samples, lambda_reg=lambda_reg,
                trial=trial, seed=trial,
                lam_star=lam0,
                lam_mle=est_mle,
                lam_bayes_mean=est_bayes,
                lam_rho_mean=est_rho,
                eta_std_rho=eta_std,
                error_mle=abs(est_mle - lam0),
                error_bayes_mean=abs(est_bayes - lam0),
                error_rho_mean=abs(est_rho - lam0),
                risk_mle=(est_mle - lam0)**2,
                risk_bayes=(est_bayes - lam0)**2,
                risk_rho=(est_rho - lam0)**2,
            ))

    trials_df = pd.DataFrame(trials_rows)
    summary_df = _summarise(trials_df)
    return summary_df, trials_df


# ====================================================================
# Uniform scale
# ====================================================================

def evaluate_uniform(
    n_samples: int = 200,
    t0: float = 1.0,
    epsilon_values: List[float] | None = None,
    n_trials: int = 1000,
    tau: float = 0.5,
    prior_std: float = 2.0,
    a_prior: float = 0.5,
    alpha_prior: float = 2.0,
    n_iter_opt: int = 400,
    n_mc_opt: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo evaluation for the Uniform[0, theta] scale model.

    For each epsilon, data is generated from the contaminated mixture.

    Returns
    -------
    summary_df, trials_df : DataFrames
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.08, 0.10]

    lambda_reg = tau * n_samples
    trials_rows = []

    for eps in epsilon_values:
        desc = f"Uniform eps={eps:.2f}"
        for trial in tqdm(range(n_trials), desc=desc, leave=False):
            x = generate_uniform_contaminated(
                n=n_samples, t0=t0, epsilon=eps, seed=trial,
            )

            est_mle = mle_uniform(x)
            est_bayes = bayes_uniform(x, a=a_prior, alpha=alpha_prior)

            opt = UniformOptimizer(x, lambda_reg=lambda_reg, prior_std=prior_std)
            opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
            est_rho, u_std = opt.get_estimate(use_polyak=True)

            trials_rows.append(dict(
                epsilon=eps, tau=tau, n=n_samples, lambda_reg=lambda_reg,
                trial=trial, seed=trial,
                theta_star=t0,
                theta_mle=est_mle,
                theta_bayes_mean=est_bayes,
                theta_rho_mean=est_rho,
                std_rho=u_std,
                error_mle=abs(est_mle - t0),
                error_bayes_mean=abs(est_bayes - t0),
                error_rho_mean=abs(est_rho - t0),
                risk_mle=(est_mle - t0)**2,
                risk_bayes=(est_bayes - t0)**2,
                risk_rho=(est_rho - t0)**2,
            ))

    trials_df = pd.DataFrame(trials_rows)
    summary_df = _summarise(trials_df)
    return summary_df, trials_df


# ====================================================================
# Fourier basis regression
# ====================================================================

def evaluate_fourier_regression(
    n_samples: int = 200,
    K: int = 6,
    epsilon_values: List[float] | None = None,
    n_trials: int = 1000,
    tau: float = 0.5,
    prior_std: float = 2.0,
    n_iter_opt: int = 500,
    n_mc_opt: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo evaluation for Fourier basis regression.

    Returns
    -------
    summary_df, trials_df : DataFrames
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.08, 0.10]

    trials_rows = []

    for eps in epsilon_values:
        desc = f"Fourier eps={eps:.2f}"
        for trial in tqdm(range(n_trials), desc=desc, leave=False):
            Phi, y, beta_true, w = generate_fourier_regression(
                n=n_samples, K=K, epsilon=eps, seed=trial,
            )
            p = Phi.shape[1]
            lambda_reg = tau * n_samples

            # OLS (MLE)
            beta_ols = ols(Phi, y)
            err_ols = float(np.linalg.norm(beta_ols - beta_true))

            # Bayes
            beta_bayes, _ = bayes_regression(Phi, y, prior_std=prior_std)
            err_bayes = float(np.linalg.norm(beta_bayes - beta_true))

            # Rho-posterior
            opt = RegressionOptimizer(
                Phi, y, lambda_reg=lambda_reg, prior_std=prior_std,
            )
            opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
            beta_rho, _ = opt.get_estimate(use_polyak=True)
            err_rho = float(np.linalg.norm(beta_rho - beta_true))

            trials_rows.append(dict(
                epsilon=eps, tau=tau, n=n_samples, lambda_reg=lambda_reg,
                trial=trial, seed=trial,
                error_mle=err_ols,
                error_bayes_mean=err_bayes,
                error_rho_mean=err_rho,
                risk_mle=err_ols**2,
                risk_bayes=err_bayes**2,
                risk_rho=err_rho**2,
            ))

    trials_df = pd.DataFrame(trials_rows)
    summary_df = _summarise(trials_df)
    return summary_df, trials_df


# ====================================================================
# Correlated design regression
# ====================================================================

def evaluate_correlated_regression(
    n_samples: int = 100,
    d: int = 10,
    rho_corr: float = 0.7,
    sparsity: int = 5,
    epsilon_values: List[float] | None = None,
    n_trials: int = 1000,
    tau: float = 0.5,
    prior_std: float = 2.0,
    n_iter_opt: int = 500,
    n_mc_opt: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo evaluation for correlated design regression.

    Returns
    -------
    summary_df, trials_df : DataFrames with per-trial estimates,
        including predicted-vs-true for the last trial at each epsilon.
    """
    if epsilon_values is None:
        epsilon_values = [0.0, 0.05, 0.08, 0.10]

    trials_rows = []
    pred_rows = []

    for eps in epsilon_values:
        desc = f"Correlated eps={eps:.2f}"
        for trial in tqdm(range(n_trials), desc=desc, leave=False):
            X, y, beta_true = generate_correlated_regression(
                n=n_samples, d=d, rho_corr=rho_corr, sparsity=sparsity,
                epsilon=eps, seed=trial,
            )
            p = X.shape[1]
            lambda_reg = tau * n_samples

            beta_ols = ols(X, y)
            err_ols = float(np.linalg.norm(beta_ols - beta_true))

            beta_bayes, _ = bayes_regression(X, y, prior_std=prior_std)
            err_bayes = float(np.linalg.norm(beta_bayes - beta_true))

            opt = RegressionOptimizer(
                X, y, lambda_reg=lambda_reg, prior_std=prior_std,
            )
            opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
            beta_rho, _ = opt.get_estimate(use_polyak=True)
            err_rho = float(np.linalg.norm(beta_rho - beta_true))

            trials_rows.append(dict(
                epsilon=eps, tau=tau, n=n_samples, lambda_reg=lambda_reg,
                trial=trial, seed=trial,
                error_mle=err_ols,
                error_bayes_mean=err_bayes,
                error_rho_mean=err_rho,
                risk_mle=err_ols**2,
                risk_bayes=err_bayes**2,
                risk_rho=err_rho**2,
            ))

            # Store predictions for the last trial (for predicted-vs-true plot)
            if trial == n_trials - 1:
                y_true = X @ beta_true
                y_pred_rho = X @ beta_rho
                for i in range(len(y_true)):
                    pred_rows.append(dict(
                        epsilon=eps, i=i,
                        y_true=y_true[i], y_pred_rho=y_pred_rho[i],
                    ))

    trials_df = pd.DataFrame(trials_rows)
    summary_df = _summarise(trials_df)
    pred_df = pd.DataFrame(pred_rows) if pred_rows else pd.DataFrame()
    return summary_df, trials_df, pred_df


# ====================================================================
# Summary helper
# ====================================================================

def _summarise(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trial results into summary statistics.

    Computes RMSE, mean Bayes risk, and standard deviation of errors
    for each (epsilon, tau, n) configuration.
    """
    group_cols = [c for c in ["epsilon", "tau", "n", "lambda_reg"] if c in trials_df.columns]

    agg = trials_df.groupby(group_cols).agg(
        MLE_BayesRisk=("risk_mle", "mean"),
        Bayes_BayesRisk=("risk_bayes", "mean"),
        Rho_BayesRisk=("risk_rho", "mean"),
        MLE_RMSE=("error_mle", lambda x: np.sqrt(np.mean(x**2))),
        Bayes_RMSE=("error_bayes_mean", lambda x: np.sqrt(np.mean(x**2))),
        Rho_RMSE=("error_rho_mean", lambda x: np.sqrt(np.mean(x**2))),
        MLE_std=("error_mle", "std"),
        Bayes_std=("error_bayes_mean", "std"),
        Rho_std=("error_rho_mean", "std"),
    ).reset_index()

    return agg
