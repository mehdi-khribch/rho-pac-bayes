#!/usr/bin/env python3
"""
Run real-world regression experiments (Ames Housing + Abalone).

Outputs:
    results/housing_metrics.csv
    results/housing_residuals.npz
    results/abalone_metrics.csv
    results/abalone_residuals.npz

Usage:  python scripts/run_realworld.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realworld import (
    load_ames_housing,
    load_abalone,
    evaluate_realworld,
    save_realworld_results,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- Ames Housing ----
    print("=" * 60)
    print("Ames Housing (log SalePrice)")
    print("=" * 60)

    X_h, y_h = load_ames_housing(data_dir=DATA_DIR)

    results_h = evaluate_realworld(
        X_h, y_h,
        dataset_name="Ames Housing",
        epsilon=0.10,
        strength=15.0,
        n_repeats=1000,
        test_size=0.2,
        tau=0.9,
        prior_std=5.0,
        huber_epsilon=1.35,
        n_iter_opt=150,
        n_mc_opt=32,
        seed=0,
    )

    save_realworld_results(results_h, output_dir=RESULTS_DIR, prefix="housing")

    print(f"  OLS  test MSE: {results_h['mse_OLS'].mean():.4f}")
    print(f"  Huber test MSE: {results_h['mse_Huber'].mean():.4f}")
    print(f"  Rho   test MSE: {results_h['mse_rho'].mean():.4f}")

    # ---- Abalone ----
    print("\n" + "=" * 60)
    print("Abalone (Rings)")
    print("=" * 60)

    X_a, y_a = load_abalone(data_dir=DATA_DIR)

    results_a = evaluate_realworld(
        X_a, y_a,
        dataset_name="Abalone",
        epsilon=0.10,
        strength=15.0,
        n_repeats=1000,
        test_size=0.2,
        tau=0.9,
        prior_std=5.0,
        huber_epsilon=1.35,
        n_iter_opt=150,
        n_mc_opt=32,
        seed=0,
    )

    save_realworld_results(results_a, output_dir=RESULTS_DIR, prefix="abalone")

    print(f"  OLS  test MSE: {results_a['mse_OLS'].mean():.4f}")
    print(f"  Huber test MSE: {results_a['mse_Huber'].mean():.4f}")
    print(f"  Rho   test MSE: {results_a['mse_rho'].mean():.4f}")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
