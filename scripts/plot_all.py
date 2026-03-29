#!/usr/bin/env python3
"""
Generate all Python quick-check figures from saved CSV results.

Reads CSVs from results/ and writes PDF/PNG figures to figures/.
Run this after `make simulations`.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting import (
    plot_posterior_risk,
    plot_rmse,
    plot_density_gaussian,
    plot_density_poisson,
    plot_density_uniform,
    plot_predicted_vs_true,
    plot_residual_density,
)

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def main():
    # ----------------------------------------------------------
    # Gaussian
    # ----------------------------------------------------------
    gs = pd.read_csv(RESULTS / "gaussian_summary.csv")
    gt = pd.read_csv(RESULTS / "gaussian_trials.csv")
    plot_posterior_risk(gs, save_path=FIGURES / "posterior_risk_gaussian.pdf")
    plot_rmse(gs, save_path=FIGURES / "rmse_gaussian.pdf")
    plot_density_gaussian(gt, save_path=FIGURES / "density_plot_gaussian.pdf")
    print("Gaussian figures saved.")

    # ----------------------------------------------------------
    # Poisson
    # ----------------------------------------------------------
    ps = pd.read_csv(RESULTS / "poisson_summary.csv")
    pt = pd.read_csv(RESULTS / "poisson_trials.csv")
    plot_posterior_risk(ps, save_path=FIGURES / "posterior_risk_pois.pdf")
    plot_rmse(ps, save_path=FIGURES / "rmse_pois.pdf")
    plot_density_poisson(pt, save_path=FIGURES / "density_plot_pois.pdf")
    print("Poisson figures saved.")

    # ----------------------------------------------------------
    # Uniform
    # ----------------------------------------------------------
    us = pd.read_csv(RESULTS / "uniform_summary.csv")
    ut = pd.read_csv(RESULTS / "uniform_trials.csv")
    plot_posterior_risk(us, save_path=FIGURES / "posterior_risk_uniform.pdf")
    plot_rmse(us, save_path=FIGURES / "rmse_uniform.pdf")
    plot_density_uniform(ut, save_path=FIGURES / "density_plot_uniform.pdf")
    print("Uniform figures saved.")

    # ----------------------------------------------------------
    # Fourier regression
    # ----------------------------------------------------------
    fs = pd.read_csv(RESULTS / "fourier_summary.csv")
    plot_posterior_risk(fs, save_path=FIGURES / "fourier_bayes_risk.pdf")
    plot_rmse(fs, save_path=FIGURES / "fourier_rmse.pdf")
    print("Fourier regression figures saved.")

    # ----------------------------------------------------------
    # Correlated regression
    # ----------------------------------------------------------
    cs = pd.read_csv(RESULTS / "correlated_summary.csv")
    plot_posterior_risk(cs, save_path=FIGURES / "posterior_risk_regression.pdf")
    plot_rmse(cs, save_path=FIGURES / "rmseregression.pdf")

    cp = RESULTS / "correlated_predictions.csv"
    if cp.exists():
        pred = pd.read_csv(cp)
        plot_predicted_vs_true(pred, save_path=FIGURES / "predictedvsfitted.pdf")

    print("Correlated regression figures saved.")

    # ----------------------------------------------------------
    # Real-world: Housing
    # ----------------------------------------------------------
    hr = RESULTS / "housing_residuals.csv"
    if hr.exists():
        plot_residual_density(
            hr,
            save_path=FIGURES / "residuals_housing.pdf",
            title="Ames Housing -- test residuals",
            xlim=(-1.5, 1.5),
        )
        print("Housing residual density figure saved.")

    # ----------------------------------------------------------
    # Real-world: Abalone
    # ----------------------------------------------------------
    ar = RESULTS / "abalone_residuals.csv"
    if ar.exists():
        plot_residual_density(
            ar,
            save_path=FIGURES / "residuals_abalone.pdf",
            title="Abalone -- test residuals",
            xlim=(-10, 10),
        )
        print("Abalone residual density figure saved.")

    print(f"\nAll figures written to {FIGURES}/")


if __name__ == "__main__":
    main()
