#!/usr/bin/env python3
"""
Run the Gaussian location model experiment.

Outputs:
    results/gaussian_summary.csv
    results/gaussian_trials.csv
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_gaussian

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("Gaussian location model")
    print("=" * 60)

    summary_df, trials_df = evaluate_gaussian(
        n_samples=200,
        d=1,
        epsilon_values=[0.0, 0.05, 0.08, 0.10],
        n_trials=1000,
        tau=0.5,
        prior_std=2.0,
        n_iter_opt=200,
        n_mc_opt=128,
    )

    summary_df.to_csv(RESULTS_DIR / "gaussian_summary.csv", index=False)
    trials_df.to_csv(RESULTS_DIR / "gaussian_trials.csv", index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/gaussian_*.csv")


if __name__ == "__main__":
    main()
