#!/usr/bin/env python3
"""
Run the correlated design regression experiment.

Outputs:
    results/correlated_summary.csv
    results/correlated_trials.csv
    results/correlated_predictions.csv
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_correlated_regression

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("Correlated design regression")
    print("=" * 60)

    summary_df, trials_df, pred_df = evaluate_correlated_regression(
        n_samples=100,
        d=10,
        rho_corr=0.7,
        sparsity=5,
        epsilon_values=[0.0, 0.05, 0.08, 0.10],
        n_trials=1000,
        tau=0.5,
        prior_std=2.0,
        n_iter_opt=500,
        n_mc_opt=128,
    )

    summary_df.to_csv(RESULTS_DIR / "correlated_summary.csv", index=False)
    trials_df.to_csv(RESULTS_DIR / "correlated_trials.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(RESULTS_DIR / "correlated_predictions.csv", index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/correlated_*.csv")


if __name__ == "__main__":
    main()
