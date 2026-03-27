#!/usr/bin/env python3
"""
Quick end-to-end test: runs 5 trials of each experiment, generates all figures.

This verifies the full pipeline works before launching the long 1000-trial run.

Usage:
    cd ~/Desktop/pac-bayes-jasa
    .venv/bin/python scripts/test_quick.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_TRIALS = 5  # tiny number just to verify everything runs

# ==============================================================
# 1. Gaussian location
# ==============================================================
print("=" * 60)
print("[1/6] Gaussian location model")
print("=" * 60)

from src.evaluation import evaluate_gaussian

gs, gt = evaluate_gaussian(
    n_samples=200, d=1,
    epsilon_values=[0.0, 0.05, 0.08, 0.10],
    n_trials=N_TRIALS, tau=0.5, prior_std=2.0,
    n_iter_opt=100, n_mc_opt=64,
)
gs.to_csv(RESULTS / "gaussian_summary.csv", index=False)
gt.to_csv(RESULTS / "gaussian_trials.csv", index=False)
print(gs[["epsilon", "MLE_RMSE", "Bayes_RMSE", "Rho_RMSE"]].to_string(index=False))

# ==============================================================
# 2. Poisson intensity
# ==============================================================
print("\n" + "=" * 60)
print("[2/6] Poisson intensity model")
print("=" * 60)

from src.evaluation import evaluate_poisson

ps, pt = evaluate_poisson(
    n_samples=200, lam0=3.0, lam_out=30.0,
    epsilon_values=[0.0, 0.05, 0.10, 0.20],
    n_trials=N_TRIALS, tau=0.5, prior_std=2.0,
    n_iter_opt=200, n_mc_opt=64,
)
ps.to_csv(RESULTS / "poisson_summary.csv", index=False)
pt.to_csv(RESULTS / "poisson_trials.csv", index=False)
print(ps[["epsilon", "MLE_RMSE", "Bayes_RMSE", "Rho_RMSE"]].to_string(index=False))

# ==============================================================
# 3. Uniform scale
# ==============================================================
print("\n" + "=" * 60)
print("[3/6] Uniform scale model")
print("=" * 60)

from src.evaluation import evaluate_uniform

us, ut = evaluate_uniform(
    n_samples=200, t0=1.0,
    epsilon_values=[0.0, 0.05, 0.08, 0.10],
    n_trials=N_TRIALS, tau=0.5, prior_std=2.0,
    n_iter_opt=200, n_mc_opt=64,
)
us.to_csv(RESULTS / "uniform_summary.csv", index=False)
ut.to_csv(RESULTS / "uniform_trials.csv", index=False)
print(us[["epsilon", "MLE_RMSE", "Bayes_RMSE", "Rho_RMSE"]].to_string(index=False))

# ==============================================================
# 4. Fourier regression
# ==============================================================
print("\n" + "=" * 60)
print("[4/6] Fourier basis regression")
print("=" * 60)

from src.evaluation import evaluate_fourier_regression

fs, ft = evaluate_fourier_regression(
    n_samples=200, K=6,
    epsilon_values=[0.0, 0.05, 0.08, 0.10],
    n_trials=N_TRIALS, tau=0.5, prior_std=2.0,
    n_iter_opt=200, n_mc_opt=64,
)
fs.to_csv(RESULTS / "fourier_summary.csv", index=False)
ft.to_csv(RESULTS / "fourier_trials.csv", index=False)
print(fs[["epsilon", "MLE_RMSE", "Bayes_RMSE", "Rho_RMSE"]].to_string(index=False))

# ==============================================================
# 5. Correlated regression
# ==============================================================
print("\n" + "=" * 60)
print("[5/6] Correlated design regression")
print("=" * 60)

from src.evaluation import evaluate_correlated_regression

cs, ct, cp = evaluate_correlated_regression(
    n_samples=100, d=10, rho_corr=0.7, sparsity=5,
    epsilon_values=[0.0, 0.05, 0.08, 0.10],
    n_trials=N_TRIALS, tau=0.5, prior_std=2.0,
    n_iter_opt=200, n_mc_opt=64,
)
cs.to_csv(RESULTS / "correlated_summary.csv", index=False)
ct.to_csv(RESULTS / "correlated_trials.csv", index=False)
if not cp.empty:
    cp.to_csv(RESULTS / "correlated_predictions.csv", index=False)
print(cs[["epsilon", "MLE_RMSE", "Bayes_RMSE", "Rho_RMSE"]].to_string(index=False))

# ==============================================================
# 6. Python figures
# ==============================================================
print("\n" + "=" * 60)
print("[6/6] Generating Python figures")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")  # no display needed

from src.plotting import (
    set_paper_style,
    plot_posterior_risk,
    plot_rmse,
    plot_density_gaussian,
    plot_density_poisson,
    plot_density_uniform,
    plot_predicted_vs_true,
)

set_paper_style()

# Gaussian figures
plot_posterior_risk(gs, save_path=FIGURES / "posterior_risk_gaussian.pdf")
plot_rmse(gs, save_path=FIGURES / "rmse_gaussian.pdf")
plot_density_gaussian(gt, epsilon=0.10, save_path=FIGURES / "density_plot_gaussian.pdf")
print("  Gaussian figures saved.")

# Poisson figures
plot_posterior_risk(ps, save_path=FIGURES / "posterior_risk_pois.pdf")
plot_rmse(ps, save_path=FIGURES / "rmse_pois.pdf")
plot_density_poisson(pt, epsilon=0.10, save_path=FIGURES / "density_plot_pois.pdf")
print("  Poisson figures saved.")

# Uniform figures
plot_posterior_risk(us, save_path=FIGURES / "posterior_risk_uniform.pdf")
plot_rmse(us, save_path=FIGURES / "rmse_uniform.pdf")
plot_density_uniform(ut, epsilon=0.10, save_path=FIGURES / "density_plot_uniform.pdf")
print("  Uniform figures saved.")

# Fourier figures
plot_posterior_risk(fs, save_path=FIGURES / "fourier_bayes_risk.pdf")
plot_rmse(fs, save_path=FIGURES / "fourier_rmse.pdf")
print("  Fourier figures saved.")

# Correlated regression figures
plot_posterior_risk(cs, save_path=FIGURES / "posterior_risk_regression.pdf")
plot_rmse(cs, save_path=FIGURES / "rmseregression.pdf")
if not cp.empty:
    plot_predicted_vs_true(cp, epsilon=0.10, save_path=FIGURES / "predictedvsfitted.pdf")
print("  Correlated regression figures saved.")

import matplotlib.pyplot as plt
plt.close("all")

# ==============================================================
# Summary
# ==============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

n_csv = len(list(RESULTS.glob("*.csv")))
n_fig = len(list(FIGURES.glob("*.pdf")))
print(f"  {n_csv} CSV files in results/")
print(f"  {n_fig} PDF figures in figures/")
print()
print("To run full experiments (1000 trials), use:")
print("  make simulations")
print("  make realworld")
print("  make figures-r")
