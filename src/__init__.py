"""
Variational Approximations of Generalized Rho-Posteriors
=========================================================

A Python package implementing the variational rho-tilde-posterior framework
for robust Bayesian inference, as described in:

    Khribch & Alquier (2025). "Robust Bayesian Inference via Variational
    Approximations of Generalized Rho-Posteriors." Submitted to JASA.

Modules
-------
contrasts
    Bounded contrast functions (Hellinger psi) and likelihood ratio computations.
variational
    Mean-field Gaussian variational family with KL divergence.
optimizers
    PAC-Bayesian saddle-point optimizers for Gaussian, Poisson, and Uniform models.
regression
    PAC-Bayesian saddle-point optimizer for linear regression.
data
    Data-generating processes for contaminated exponential families.
baselines
    Classical estimators: MLE and conjugate Bayes posteriors.
evaluation
    Monte Carlo evaluation loops and summary statistics.
plotting
    Publication-quality figure generation (Python side).
"""

__version__ = "1.0.0"
