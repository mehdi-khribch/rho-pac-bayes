# Variational Approximations of Generalised Rho-Posteriors

Code to reproduce the numerical experiments in:

> **M. Khribch & P. Alquier** (2025).
> *Robust Bayesian Inference via Variational Approximations of Generalised Rho-Posteriors.*
> Submitted to the Journal of the American Statistical Association (JASA).

## Overview

This repository implements the variational rho-tilde-posterior framework
for robust Bayesian inference under epsilon-contamination.  The package
provides:

- A modular Python library (`src/`) implementing the bounded Hellinger
  contrast, mean-field Gaussian variational families, and PAC-Bayesian
  saddle-point optimisers for four model classes.
- Standalone scripts (`scripts/`) that run each Monte Carlo experiment
  and save results as CSV files.
- R plotting scripts (`R/`) that produce publication-quality ggplot
  figures from the saved CSVs.
- A Jupyter notebook (`notebooks/`) that runs the full pipeline
  interactively and displays all figures inline.

## Quick start

### 1. Install

```bash
git clone https://github.com/<your-username>/pac-bayes-jasa.git
cd pac-bayes-jasa
make venv
```

This creates a `.venv/` virtual environment and installs the package
with all dependencies (numpy, torch, scipy, pandas, matplotlib, tqdm).

### 2. Reproduce everything

```bash
make all
```

This runs `make venv`, `make simulations`, `make figures-py`, and
`make figures-r` in sequence.  Results are saved to `results/` and
figures to `figures/`.

### 3. Run individual experiments

Each experiment has its own script. The Makefile uses the venv
automatically, but you can also activate it manually:

```bash
source .venv/bin/activate
python scripts/run_gaussian.py      # Gaussian location (Fig. 1)
python scripts/run_poisson.py       # Poisson intensity  (Fig. 2)
python scripts/run_uniform.py       # Uniform scale      (Fig. 3)
python scripts/run_fourier.py       # Fourier regression  (Fig. 4)
python scripts/run_correlated.py    # Correlated design   (Fig. 5)
```

### 4. Generate figures

Python quick-check figures:

```bash
python scripts/plot_all.py
```

R publication figures (requires R with `tidyverse`, `ggthemes`,
`latex2exp`, and `pracma`):

```bash
Rscript R/plot_gaussian.R
Rscript R/plot_poisson.R
Rscript R/plot_uniform.R
Rscript R/plot_regression.R
```

### 5. Interactive notebook

```bash
jupyter notebook notebooks/main.ipynb
```

## Project structure

```
pac-bayes-jasa/
|-- src/                     # Python package
|   |-- __init__.py
|   |-- contrasts.py         # Hellinger contrast and likelihood ratios
|   |-- variational.py       # Mean-field Gaussian variational family
|   |-- optimizers.py        # Saddle-point optimisers (Gaussian, Poisson, Uniform)
|   |-- regression.py        # Saddle-point optimiser for linear regression
|   |-- data.py              # Contaminated data generators
|   |-- baselines.py         # MLE and conjugate Bayes estimators
|   |-- evaluation.py        # Monte Carlo evaluation loops
|   |-- plotting.py          # Python figure generation
|
|-- scripts/                 # Standalone experiment scripts
|   |-- run_gaussian.py
|   |-- run_poisson.py
|   |-- run_uniform.py
|   |-- run_fourier.py
|   |-- run_correlated.py
|   |-- plot_all.py
|
|-- R/                       # R scripts for publication figures
|   |-- theme.R              # Shared ggplot theme
|   |-- plot_gaussian.R
|   |-- plot_poisson.R
|   |-- plot_uniform.R
|   |-- plot_regression.R
|
|-- notebooks/               # Jupyter notebook
|   |-- main.ipynb
|
|-- results/                 # Generated CSV files (not tracked by git)
|-- figures/                 # Generated PDF figures (not tracked by git)
|-- pyproject.toml           # Package metadata and dependencies
|-- Makefile                 # Automation
|-- .gitignore
|-- README.md
```

## Experiments

All experiments use T = 1000 independent replications unless otherwise noted.

| Experiment | Model | n | epsilon values | Figures |
|---|---|---|---|---|
| Gaussian location | N(theta, 1) | 200 | 0, 5, 8, 10% | 1a-c |
| Poisson intensity | Pois(lambda) | 200 | 0, 5, 10, 20% | 2a-c |
| Uniform scale | U[0, theta] | 200 | 0, 5, 8, 10% | 3a-c |
| Fourier regression | Y = Phi beta + noise | 200 | 0, 5, 8, 10% | 4a-b |
| Correlated design | Y = X beta + noise | 100 | 0, 5, 8, 10% | 5a-c |

## Dependencies

**Python** (>= 3.10):
numpy, scipy, pandas, torch, matplotlib, tqdm, jupyter.

**R** (optional, for publication figures):
tidyverse, ggthemes, latex2exp, pracma.

## Citation

```bibtex
@article{khribch2025robust,
  title   = {Robust {B}ayesian Inference via Variational Approximations
             of Generalised Rho-Posteriors},
  author  = {Khribch, Mehdi and Alquier, Pierre},
  journal = {Submitted to JASA},
  year    = {2025}
}
```

## License

MIT
