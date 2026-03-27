"""
Real-world dataset loading, contamination, and evaluation.

Implements the real-world regression experiments from Section 3 of the paper:
- Ames Housing (n=2930, 79 features, target = log SalePrice)
- Abalone (n=4177, 8 features, target = Rings)

Datasets are fetched from OpenML via scikit-learn. Training labels are
contaminated by adding +/- strength * MAD(y) to a random epsilon-fraction
of the training set. Test data remains clean.

Comparison: OLS, Huber regression, and the variational rho-posterior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ====================================================================
# Dataset loading
# ====================================================================

def load_ames_housing(data_dir: str | Path = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load Ames Housing dataset (OpenML id 42165).

    Returns X (n, p) and y = log(SalePrice) (n,).
    Downloads to data_dir/ on first call, then loads from cache.
    """
    data_dir = Path(data_dir)
    cache_x = data_dir / "ames_X.npy"
    cache_y = data_dir / "ames_y.npy"

    if cache_x.exists() and cache_y.exists():
        return np.load(cache_x), np.load(cache_y)

    from sklearn.datasets import fetch_openml

    housing = fetch_openml(name="house_prices", version=1, as_frame=True)
    df = housing.frame

    # Target: log sale price
    y = np.log(df["SalePrice"].values.astype(float))

    # Features: numeric columns only, drop target
    X_df = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.fillna(X_df.median())

    X = X_df.values.astype(float)

    # Standardize features
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Ames Housing: n={X.shape[0]}, p={X.shape[1]}")
    return X, y


def load_abalone(data_dir: str | Path = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load Abalone dataset (OpenML id 183).

    Returns X (n, p) and y = Rings (n,).
    Downloads to data_dir/ on first call, then loads from cache.
    """
    data_dir = Path(data_dir)
    cache_x = data_dir / "abalone_X.npy"
    cache_y = data_dir / "abalone_y.npy"

    if cache_x.exists() and cache_y.exists():
        return np.load(cache_x), np.load(cache_y)

    from sklearn.datasets import fetch_openml

    abalone = fetch_openml(name="abalone", version=1, as_frame=True)
    df = abalone.frame

    # Target
    y = df["Rings"].values.astype(float)

    # Features: one-hot encode Sex, keep numeric
    X_df = df.drop(columns=["Rings"])
    X_df = pd.get_dummies(X_df, drop_first=True)
    X_df = X_df.fillna(X_df.median())

    X = X_df.values.astype(float)

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)

    print(f"Abalone: n={X.shape[0]}, p={X.shape[1]}")
    return X, y


# ====================================================================
# Label contamination
# ====================================================================

def contaminate_labels(
    y_train: np.ndarray,
    epsilon: float = 0.10,
    strength: float = 15.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Contaminate training labels by adding +/- strength * MAD(y).

    A random epsilon-fraction of training labels are shifted by
    +/- strength * MAD(y_train), where MAD = median absolute deviation
    and the sign is chosen uniformly at random.

    Parameters
    ----------
    y_train : (n,) ndarray
        Clean training labels.
    epsilon : float
        Contamination fraction.
    strength : float
        Outlier magnitude in MAD units.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    (n,) ndarray
        Contaminated training labels.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_out = y_train.copy()
    n = len(y_train)
    mad = np.median(np.abs(y_train - np.median(y_train)))

    n_contam = int(np.ceil(epsilon * n))
    idx = rng.choice(n, size=n_contam, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_contam)
    y_out[idx] += signs * strength * mad

    return y_out


# ====================================================================
# Evaluation loop
# ====================================================================

def evaluate_realworld(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "Dataset",
    epsilon: float = 0.10,
    strength: float = 15.0,
    n_repeats: int = 1000,
    test_size: float = 0.2,
    tau: float = 0.1,
    prior_std: float = 1.0,
    huber_epsilon: float = 1.35,
    n_iter_opt: int = 500,
    n_mc_opt: int = 128,
    seed: int = 0,
) -> dict:
    """Run real-world experiment comparing OLS, Huber, and rho-posterior.

    For each repetition: split train/test, contaminate training labels,
    fit all three estimators, compute test residuals.

    Parameters
    ----------
    X : (n, p) ndarray
        Full design matrix (with intercept).
    y : (n,) ndarray
        Full response vector.
    dataset_name : str
        Name for display.
    epsilon : float
        Contamination rate.
    strength : float
        Outlier magnitude in MAD units.
    n_repeats : int
        Number of train/test splits.
    test_size : float
        Fraction held out for testing.
    tau : float
        Temperature scaling for rho-posterior (lambda = tau * n_train).
    prior_std : float
        Prior standard deviation.
    huber_epsilon : float
        Huber loss epsilon parameter.
    n_iter_opt : int
        Optimisation iterations for rho-posterior.
    n_mc_opt : int
        Monte Carlo samples per step.
    seed : int
        Base random seed.

    Returns
    -------
    dict with keys:
        mse_OLS, mse_Huber, mse_rho : (n_repeats,) arrays
        mae_OLS, mae_Huber, mae_rho : (n_repeats,) arrays
        res_OLS, res_Huber, res_rho : concatenated test residuals
        meta : dict of experiment parameters
    """
    from sklearn.linear_model import HuberRegressor
    from sklearn.model_selection import train_test_split
    from .regression import RegressionOptimizer
    from .baselines import ols

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_test = int(n * test_size)

    mse_ols_list, mse_huber_list, mse_rho_list = [], [], []
    mae_ols_list, mae_huber_list, mae_rho_list = [], [], []
    res_ols_all, res_huber_all, res_rho_all = [], [], []

    desc = f"{dataset_name} eps={epsilon:.0%}"
    for rep in tqdm(range(n_repeats), desc=desc, leave=False):
        # Train/test split
        idx = rng.permutation(n)
        idx_test = idx[:n_test]
        idx_train = idx[n_test:]

        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]

        # Contaminate training labels
        y_train_c = contaminate_labels(y_train, epsilon=epsilon,
                                        strength=strength, rng=rng)

        # 1) OLS
        beta_ols = ols(X_train, y_train_c)
        res_ols = y_test - X_test @ beta_ols
        mse_ols_list.append(float(np.mean(res_ols**2)))
        mae_ols_list.append(float(np.mean(np.abs(res_ols))))
        res_ols_all.append(res_ols)

        # 2) Huber regression
        huber = HuberRegressor(epsilon=huber_epsilon, max_iter=500)
        huber.fit(X_train[:, 1:], y_train_c)  # sklearn adds intercept internally
        y_pred_huber = huber.predict(X_test[:, 1:])
        res_huber = y_test - y_pred_huber
        mse_huber_list.append(float(np.mean(res_huber**2)))
        mae_huber_list.append(float(np.mean(np.abs(res_huber))))
        res_huber_all.append(res_huber)

        # 3) Rho-posterior
        lambda_reg = tau * len(y_train_c)
        opt = RegressionOptimizer(
            X_train, y_train_c,
            lambda_reg=lambda_reg, prior_std=prior_std,
        )
        opt.optimize(n_iter=n_iter_opt, n_mc=n_mc_opt, verbose=False)
        beta_rho, _ = opt.get_estimate(use_polyak=True)
        res_rho = y_test - X_test @ beta_rho
        mse_rho_list.append(float(np.mean(res_rho**2)))
        mae_rho_list.append(float(np.mean(np.abs(res_rho))))
        res_rho_all.append(res_rho)

    return dict(
        mse_OLS=np.array(mse_ols_list),
        mse_Huber=np.array(mse_huber_list),
        mse_rho=np.array(mse_rho_list),
        mae_OLS=np.array(mae_ols_list),
        mae_Huber=np.array(mae_huber_list),
        mae_rho=np.array(mae_rho_list),
        res_OLS=np.concatenate(res_ols_all),
        res_Huber=np.concatenate(res_huber_all),
        res_rho=np.concatenate(res_rho_all),
        meta=dict(
            dataset=dataset_name,
            epsilon=epsilon,
            strength=strength,
            n_repeats=n_repeats,
            test_size=test_size,
            tau=tau,
            prior_std=prior_std,
            huber_epsilon=huber_epsilon,
        ),
    )


def save_realworld_results(
    results: dict,
    output_dir: str | Path = "results",
    prefix: str = "housing",
) -> None:
    """Save real-world results to CSV and NPZ files.

    Saves:
        results/{prefix}_residuals.npz   -- all arrays
        results/{prefix}_metrics.csv     -- per-repeat MSE/MAE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.savez(
        output_dir / f"{prefix}_residuals.npz",
        res_OLS=results["res_OLS"],
        res_Huber=results["res_Huber"],
        res_rho=results["res_rho"],
    )

    # Save per-repeat metrics
    df = pd.DataFrame({
        "repeat": range(len(results["mse_OLS"])),
        "mse_OLS": results["mse_OLS"],
        "mse_Huber": results["mse_Huber"],
        "mse_rho": results["mse_rho"],
        "mae_OLS": results["mae_OLS"],
        "mae_Huber": results["mae_Huber"],
        "mae_rho": results["mae_rho"],
    })
    df.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)

    print(f"Saved {prefix} results to {output_dir}/")
