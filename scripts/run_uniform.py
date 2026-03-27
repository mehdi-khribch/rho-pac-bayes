#!/usr/bin/env python3
"""
Run the Uniform scale model experiment.

Outputs:
    results/uniform_summary.csv
    results/uniform_trials.csv
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import evaluate_uniform

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("Uniform scale model")
    print("=" * 60)

    summary_df, trials_df = evaluate_uniform(
        n_samples=200,
        t0=1.0,
        epsilon_values=[0.0, 0.05, 0.08, 0.10],
        n_trials=1000,
        tau=0.5,
        prior_std=2.0,
        a_prior=0.5,
        alpha_prior=2.0,
        n_iter_opt=400,
        n_mc_opt=128,
    )

    summary_df.to_csv(RESULTS_DIR / "uniform_summary.csv", index=False)
    trials_df.to_csv(RESULTS_DIR / "uniform_trials.csv", index=False)

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/uniform_*.csv")


if __name__ == "__main__":
    main()
