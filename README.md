# Variational Approximations for Robust Bayesian Inference via Rho-Posteriors

[![arXiv](https://img.shields.io/badge/arXiv-2601.07325-b31b1b.svg)](https://arxiv.org/abs/2601.07325)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code to reproduce all numerical experiments and figures in:

> **EL Mahdi Khribch & Pierre Alquier** (2026).
> *Variational Approximations for Robust Bayesian Inference via Rho-Posteriors.*
> Submitted to the Journal of the American Statistical Association (JASA).
> [arXiv:2601.07325](https://arxiv.org/abs/2601.07325)

## About the paper

Standard Bayesian inference can break down when observed data deviates
from model assumptions -- even a small fraction of outliers can pull
posterior estimates far from the truth. The **rho-posterior** framework
addresses this by replacing the log-likelihood with a bounded loss
(the Hellinger contrast), producing posteriors that are provably robust
to epsilon-contamination while retaining optimal convergence rates.

This paper develops **tractable variational approximations** to
rho-posteriors using a PAC-Bayesian saddle-point formulation. The key
idea is to jointly optimise a mean-field Gaussian variational family
for the parameter of interest and a worst-case competitor distribution,
recovering the minimax robustness guarantees of exact rho-posteriors at
a fraction of the computational cost. We derive finite-sample oracle
inequalities with explicit rates and demonstrate that our method
achieves theoretical contamination breakdown points in practice.

## Experiments overview

The repository contains six experiments that cover four model families,
each testing robustness under increasing contamination. Every parametric
experiment runs **T = 1000 independent Monte Carlo replications** with
sample size n, comparing four estimators:

- **MLE** -- Maximum likelihood estimator (no robustness)
- **Bayes** -- Conjugate Bayesian posterior mean (no robustness)
- **Rho-posterior** -- Our variational rho-tilde-posterior (robust)
- **Huber** (regression only) -- Huber robust regression baseline

### Experiment 1: Gaussian location estimation

Estimate the mean theta of a Gaussian N(theta, 1) from n = 200
observations under epsilon-contamination (outliers drawn from
N(theta + 10, 1)). Contamination levels: 0%, 5%, 8%, 10%.
Produces posterior risk curves, RMSE bar plots, and density overlays
showing how the rho-posterior stays concentrated near the true theta
while the standard Bayes posterior is pulled toward the outliers.

### Experiment 2: Poisson intensity estimation

Estimate the rate lambda of a Poisson distribution from n = 200
observations. Outliers are drawn from Pois(10 * lambda).
Contamination levels: 0%, 5%, 10%, 20%. Uses a log-reparametrisation
(eta = log(lambda)) for unconstrained variational optimisation.
Demonstrates robustness in a discrete exponential-family setting.

### Experiment 3: Uniform scale estimation

Estimate the upper bound theta of a Uniform[0, theta] distribution
from n = 200 observations contaminated with U[0, 5 * theta] outliers.
Contamination levels: 0%, 5%, 8%, 10%. Uses a log-reparametrisation
(u = log(theta)) and a Pareto conjugate prior. This is a non-regular
model (the support depends on the parameter), making it a challenging
test case for robust inference.

### Experiment 4: Fourier regression

Fixed-design linear regression with a Fourier basis (5 sine/cosine
features) on n = 200 observations. Label contamination adds
+/- strength * std(y) to an epsilon-fraction of responses.
Contamination levels: 0%, 5%, 8%, 10%. Produces predicted-vs-true
curves and RMSE comparisons against OLS and conjugate Bayes.

### Experiment 5: Correlated-design regression

Linear regression with correlated Gaussian covariates (5 features,
Toeplitz correlation rho = 0.5) on n = 100 observations. Same label
contamination scheme. Contamination levels: 0%, 5%, 8%, 10%.
Includes predicted-vs-fitted scatter plots showing how the
rho-posterior maintains accurate predictions despite outliers.

### Experiment 6: Real-world regression

Applies the method to two benchmark datasets from OpenML:

- **Ames Housing** -- Predicting log(SalePrice) from 10 numeric
  features (2930 observations). Labels are artificially contaminated
  at varying rates to measure robustness on real covariate structure.
- **Abalone** -- Predicting the number of rings from 8 physical
  measurements (4177 observations). Same contamination protocol.

Each dataset is evaluated over 1000 random train/test splits, comparing
OLS, Huber regression, and the variational rho-posterior. Test residual
density plots reveal how the rho-posterior produces tighter residual
distributions than OLS under contamination.

## Quick start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/pac-bayes-jasa.git
cd pac-bayes-jasa
make venv
```

This creates a `.venv/` virtual environment and installs the package
with all dependencies (NumPy, SciPy, Pandas, PyTorch, Matplotlib, tqdm).

### 2. Quick test (2-5 minutes)

Verify the full pipeline works with 5 trials per experiment:

```bash
.venv/bin/python scripts/test_quick.py
```

### 3. Reproduce all results

```bash
make all
```

This runs, in order: `venv` -> `data` (downloads real-world datasets) ->
`simulations` (5 parametric experiments, ~2 hours) -> `realworld`
(housing + abalone, ~1 hour) -> `figures-py` -> `figures-r`.

Results are saved to `results/` and figures to `figures/`.

### 4. Run individual experiments

```bash
source .venv/bin/activate
python scripts/run_gaussian.py        # Gaussian location
python scripts/run_poisson.py         # Poisson intensity
python scripts/run_uniform.py         # Uniform scale
python scripts/run_fourier.py         # Fourier regression
python scripts/run_correlated.py      # Correlated design
python scripts/run_realworld.py       # Ames Housing + Abalone
```

### 5. Generate figures

Python figures:

```bash
.venv/bin/python scripts/plot_all.py
```

R publication figures (requires R with `tidyverse`, `ggthemes`,
`latex2exp`, `pracma`):

```bash
Rscript R/plot_gaussian.R
Rscript R/plot_poisson.R
Rscript R/plot_uniform.R
Rscript R/plot_regression.R
Rscript R/plot_realworld.R
```

### 6. Interactive notebook

```bash
jupyter notebook notebooks/main.ipynb
```

## Project structure

```
pac-bayes-jasa/
|-- src/                        # Python package
|   |-- __init__.py
|   |-- contrasts.py            # Hellinger contrast and likelihood ratios
|   |-- variational.py          # Mean-field Gaussian variational family
|   |-- optimizers.py           # Saddle-point optimisers (Gaussian, Poisson, Uniform)
|   |-- regression.py           # Regression saddle-point optimiser
|   |-- data.py                 # Contaminated data generators
|   |-- baselines.py            # MLE, conjugate Bayes, Huber baselines
|   |-- evaluation.py           # Monte Carlo evaluation loops
|   |-- plotting.py             # Python figure generation
|   |-- realworld.py            # Real-world dataset loading and evaluation
|
|-- scripts/                    # Standalone experiment runners
|   |-- run_gaussian.py
|   |-- run_poisson.py
|   |-- run_uniform.py
|   |-- run_fourier.py
|   |-- run_correlated.py
|   |-- run_realworld.py
|   |-- download_data.py        # Downloads and caches OpenML datasets
|   |-- export_residuals_csv.py # Converts NPZ residuals to CSV for R
|   |-- plot_all.py             # Generates all Python figures
|   |-- test_quick.py           # Fast end-to-end verification (5 trials)
|
|-- R/                          # R scripts for publication figures
|   |-- theme.R                 # Shared Tufte-style ggplot theme
|   |-- plot_gaussian.R
|   |-- plot_poisson.R
|   |-- plot_uniform.R
|   |-- plot_regression.R
|   |-- plot_realworld.R
|
|-- notebooks/
|   |-- main.ipynb              # Interactive notebook with all experiments
|
|-- results/                    # Generated CSV/NPZ files (git-ignored)
|-- figures/                    # Generated PDF figures (git-ignored)
|-- data/                       # Cached datasets (git-ignored)
|-- pyproject.toml
|-- Makefile
|-- .gitignore
|-- README.md
```

## Summary of experiments

| Experiment | Model | n | Contamination | Figures |
|---|---|---|---|---|
| Gaussian location | N(theta, 1) | 200 | 0, 5, 8, 10% | posterior risk, RMSE, density |
| Poisson intensity | Pois(lambda) | 200 | 0, 5, 10, 20% | posterior risk, RMSE, density |
| Uniform scale | U[0, theta] | 200 | 0, 5, 8, 10% | posterior risk, RMSE, density |
| Fourier regression | Y = Phi * beta + noise | 200 | 0, 5, 8, 10% | predicted-vs-true, RMSE |
| Correlated design | Y = X * beta + noise | 100 | 0, 5, 8, 10% | predicted-vs-fitted, RMSE, risk |
| Ames Housing | real-world regression | 2930 | 0, 5, 10, 20% | test residual density |
| Abalone | real-world regression | 4177 | 0, 5, 10, 20% | test residual density |

## Dependencies

**Python** (>= 3.10): numpy, scipy, pandas, torch, scikit-learn,
matplotlib, tqdm, jupyter.

**R** (optional, for publication figures): tidyverse, ggthemes,
latex2exp, pracma.

## Citation

```bibtex
@article{khribch2026variational,
  title   = {Variational Approximations for Robust {B}ayesian Inference
             via Rho-Posteriors},
  author  = {Khribch, EL Mahdi and Alquier, Pierre},
  journal = {Submitted to JASA},
  year    = {2026},
  note    = {arXiv:2601.07325}
}
```

## License

MIT
