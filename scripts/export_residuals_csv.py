#!/usr/bin/env python3
"""
Convert NPZ residual files to CSV for R plotting.

Reads results/{housing,abalone}_residuals.npz and writes
results/{housing,abalone}_residuals.csv with columns OLS, Huber, Rho.

Usage:  python scripts/export_residuals_csv.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path("results")


def convert(prefix: str):
    npz_path = RESULTS / f"{prefix}_residuals.npz"
    csv_path = RESULTS / f"{prefix}_residuals.csv"

    if not npz_path.exists():
        print(f"Skipping {prefix}: {npz_path} not found.")
        return

    data = np.load(npz_path)
    df = pd.DataFrame({
        "OLS": data["res_OLS"],
        "Huber": data["res_Huber"],
        "Rho": data["res_rho"],
    })
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")


def main():
    convert("housing")
    convert("abalone")


if __name__ == "__main__":
    main()
