#!/usr/bin/env python3
"""
Run the Fourier basis regression experiment.

Outputs:
    results/fourier_summary.csv
    results/fourier_trials.csv
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_fourier_regression

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("Fourier basis regression")
    print("=" * 60)

    summary_df, trials_df = evaluate_fourier_regression(
        n_samples=200,
        K=6,
        epsilon_values=[0.0, 0.05, 0.08, 0.10],
        n_trials=1000,
        tau=0.9,
        prior_std=5.0,
        n_iter_opt=200,
        n_mc_opt=64,
    )

    summary_df.to_csv(RESULTS_DIR / "fourier_summary.csv", index=False)
    trials_df.to_csv(RESULTS_DIR / "fourier_trials.csv", index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/fourier_*.csv")


if __name__ == "__main__":
    main()
