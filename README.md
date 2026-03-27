# Variational Approximations for Robust Bayesian Inference via $\rho$-Posteriors

[![arXiv](https://img.shields.io/badge/arXiv-2601.07325-b31b1b.svg)](https://arxiv.org/abs/2601.07325)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for:

> **EL Mahdi Khribch & Pierre Alquier** (2026).
> *Variational Approximations for Robust Bayesian Inference via $\rho$-Posteriors.*
> [arXiv:2601.07325](https://arxiv.org/abs/2601.07325)

## Experiments

All parametric experiments use $T = 1{,}000$ independent Monte Carlo replications. We compare four estimators throughout: the MLE, the conjugate Bayesian posterior mean, our variational $\tilde{\rho}$-posterior, and (for regression) the Huber M-estimator.

### Experiment 1 &mdash; Gaussian location

Observations $X_1, \dots, X_n \overset{\text{iid}}{\sim} (1-\varepsilon)\,\mathcal{N}(\theta_0, 1) + \varepsilon\,\mathcal{N}(\theta_0 + 10, 1)$ with $n = 200$ and $\varepsilon \in \{0, 0.05, 0.08, 0.10\}$. We report the posterior Bayes risk $r(\tilde{\rho}, \theta_0)$, the RMSE of each point estimator, and overlay the posterior densities $\pi(\theta \mid X)$ versus $\tilde{\rho}(\theta \mid X)$ to visualise how contamination shifts the standard posterior while the $\rho$-posterior remains concentrated at $\theta_0$.

### Experiment 2 &mdash; Poisson intensity

Observations $X_i \overset{\text{iid}}{\sim} (1-\varepsilon)\,\mathrm{Pois}(\lambda_0) + \varepsilon\,\mathrm{Pois}(10\lambda_0)$ with $n = 200$ and $\varepsilon \in \{0, 0.05, 0.10, 0.20\}$. The variational optimisation uses the log-reparametrisation $\eta = \log \lambda$ for unconstrained gradient descent. Demonstrates robustness in a discrete exponential-family setting where the MLE is highly sensitive to inflated counts.

### Experiment 3 &mdash; Uniform scale

Observations from $(1-\varepsilon)\,\mathcal{U}[0, \theta_0] + \varepsilon\,\mathcal{U}[0, 5\theta_0]$ with $n = 200$ and $\varepsilon \in \{0, 0.05, 0.08, 0.10\}$. Uses $u = \log \theta$ reparametrisation and a Pareto conjugate prior. This is a non-regular model whose support depends on the parameter, making it a particularly challenging test case: the MLE $\hat{\theta} = X_{(n)}$ is pulled directly by outlying observations.

### Experiment 4 &mdash; Fourier regression

Fixed-design regression $Y = \Phi\beta + \sigma\epsilon$ with a Fourier basis of $p = 5$ sine/cosine features, $n = 200$. An $\varepsilon$-fraction of labels is contaminated by additive shifts of magnitude $\pm c \cdot \mathrm{std}(Y)$. Contamination levels: $\varepsilon \in \{0, 0.05, 0.08, 0.10\}$. Produces predicted-versus-true function curves and RMSE comparisons against OLS and conjugate Bayes.

### Experiment 5 &mdash; Correlated-design regression

Random-design regression $Y = X\beta + \sigma\epsilon$ with $p = 5$ Gaussian covariates following a Toeplitz correlation structure ($\rho = 0.5$), $n = 100$, and the same label-contamination scheme. Includes predicted-versus-fitted scatter plots demonstrating that the variational $\tilde{\rho}$-posterior maintains accurate predictions under contamination where OLS deteriorates substantially.

### Experiment 6 &mdash; Real-world benchmarks

We apply the method to two OpenML regression datasets with artificially contaminated labels ($\varepsilon \in \{0, 0.05, 0.10, 0.20\}$, shift magnitude scaled by the median absolute deviation):

- **Ames Housing** &mdash; predicting $\log(\text{SalePrice})$ from 10 numeric features ($n = 2{,}930$).
- **Abalone** &mdash; predicting the number of rings from 8 physical measurements ($n = 4{,}177$).

Each configuration is evaluated over $1{,}000$ random 80/20 train-test splits, comparing OLS, Huber regression, and the variational $\tilde{\rho}$-posterior. Test-residual density plots show that the $\rho$-posterior yields tighter, more symmetric residual distributions than OLS under contamination.

## Reproducing the results

```bash
git clone https://github.com/mehdi-khribch/rho-pac-bayes.git
cd pac-bayes-jasa
make venv          # create virtual environment and install dependencies
make all           # run all experiments and generate all figures
```

To run a quick end-to-end check (5 trials, ~3 minutes):

```bash
.venv/bin/python scripts/test_quick.py
```

Individual experiments can be launched separately:

```bash
source .venv/bin/activate
python scripts/run_gaussian.py
python scripts/run_poisson.py
python scripts/run_uniform.py
python scripts/run_fourier.py
python scripts/run_correlated.py
python scripts/run_realworld.py
```

Publication-quality figures (R with `tidyverse`, `ggthemes`, `latex2exp`, `pracma`):

```bash
Rscript R/plot_gaussian.R
Rscript R/plot_poisson.R
Rscript R/plot_uniform.R
Rscript R/plot_regression.R
Rscript R/plot_realworld.R
```

## Requirements

**Python** $\geq$ 3.10: `numpy`, `scipy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`, `tqdm`, `jupyter`.

**R** (optional, for publication figures): `tidyverse`, `ggthemes`, `latex2exp`, `pracma`.

## Citation

```bibtex
@article{khribch2026variational,
  title   = {Variational Approximations for Robust {B}ayesian Inference
             via $\rho$-Posteriors},
  author  = {Khribch, EL Mahdi and Alquier, Pierre},
  year    = {2026},
  eprint  = {2601.07325},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME}
}
```

## License

MIT
