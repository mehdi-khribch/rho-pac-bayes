#!/usr/bin/env python3
"""
Download real-world datasets from OpenML and cache locally.

Fetches Ames Housing and Abalone datasets via scikit-learn,
preprocesses them (standardize, add intercept), and saves
NumPy arrays to data/.

Usage:  python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realworld import load_ames_housing, load_abalone


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading Ames Housing...")
    X_h, y_h = load_ames_housing(data_dir=data_dir)
    print(f"  Saved: X {X_h.shape}, y {y_h.shape}")

    print("Downloading Abalone...")
    X_a, y_a = load_abalone(data_dir=data_dir)
    print(f"  Saved: X {X_a.shape}, y {y_a.shape}")

    print(f"\nAll datasets cached in {data_dir}/")


if __name__ == "__main__":
    main()
