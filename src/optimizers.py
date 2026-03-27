"""
PAC-Bayesian saddle-point optimisers.

Each optimiser solves the variational objective

    L_n(rho, rho') = E_{theta~rho, theta'~rho'}[ R-hat_psi(theta, theta') ]
                 + (KL(rho || pi) - KL(rho' || pi)) / lambda

by ascending in rho' (competitor) and descending in rho (main posterior),
using Adam with gradient flipping for the competitor.  Polyak averaging
of the posterior mean reduces Monte Carlo variance.

Model-specific subclasses differ only in the empirical psi-risk computation,
the parameter interpretation, and the mapping to the natural scale.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.utils as nn_utils

from .variational import VariationalGaussian
from .contrasts import (
    empirical_psi_risk_gaussian,
    empirical_psi_risk_poisson,
    empirical_psi_risk_uniform,
)


# ====================================================================
# Base class
# ====================================================================

class _PACBayesBase:
    """Shared logic for all PAC-Bayes saddle-point optimisers."""

    # Subclasses set these
    _risk_fn = None
    _param_label: str = "param"

    def __init__(
        self,
        data: np.ndarray,
        lambda_reg: float,
        prior_std: float = 2.0,
        d: int = 1,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.n = self.data.shape[0]
        self.d = d
        self.lambda_reg = lambda_reg

        # Gaussian prior on the (possibly reparametrised) parameter
        self.prior_mean = torch.zeros(self.d, device=self.device)
        self.prior_std = torch.ones(self.d, device=self.device) * prior_std

        # Variational distributions
        self.rho_main = VariationalGaussian(self.d).to(self.device)
        self.rho_comp = VariationalGaussian(self.d).to(self.device)

        # Adam optimisers
        self.opt_main = torch.optim.Adam(self.rho_main.parameters(), lr=1e-2)
        self.opt_comp = torch.optim.Adam(self.rho_comp.parameters(), lr=1e-2)

        # Training history
        self.history: dict[str, list] = dict(
            objective=[], risk=[], kl_main=[], kl_comp=[],
            param_mean=[], param_std=[],
        )

        # Polyak running average
        self._polyak_mean: np.ndarray | None = None
        self._polyak_count: int = 0

    # ----------------------------------------------------------------
    # Objective
    # ----------------------------------------------------------------

    def _compute_risk(self, theta_main, theta_comp) -> torch.Tensor:
        """Model-specific empirical psi-risk. Overridden by subclasses."""
        raise NotImplementedError

    def compute_objective(self, n_mc: int = 128):
        """Monte Carlo estimate of the saddle-point objective.

        Returns
        -------
        obj, risk, kl_main, kl_comp : torch.Tensor (scalar each)
        """
        theta_main = self.rho_main.sample(n_mc)
        theta_comp = self.rho_comp.sample(n_mc)

        risk = self._compute_risk(theta_main, theta_comp).mean()
        kl_main = self.rho_main.kl_divergence(self.prior_mean, self.prior_std)
        kl_comp = self.rho_comp.kl_divergence(self.prior_mean, self.prior_std)

        obj = risk + (kl_main - kl_comp) / self.lambda_reg
        return obj, risk, kl_main, kl_comp

    # ----------------------------------------------------------------
    # Optimisation step
    # ----------------------------------------------------------------

    def step(
        self,
        n_mc: int = 128,
        clip_grad: float = 5.0,
        lr_main: float | None = None,
        lr_comp: float | None = None,
    ) -> Tuple[float, float, float, float]:
        """One gradient step: descend in rho, ascend in rho'."""
        if lr_main is not None:
            for pg in self.opt_main.param_groups:
                pg["lr"] = lr_main
        if lr_comp is not None:
            for pg in self.opt_comp.param_groups:
                pg["lr"] = lr_comp

        self.opt_main.zero_grad()
        self.opt_comp.zero_grad()

        obj, risk, kl_main, kl_comp = self.compute_objective(n_mc=n_mc)
        obj.backward()

        # Flip gradients for the competitor (ascend)
        for p in self.rho_comp.parameters():
            if p.grad is not None:
                p.grad.mul_(-1.0)

        if clip_grad is not None:
            nn_utils.clip_grad_norm_(self.rho_main.parameters(), max_norm=clip_grad)
            nn_utils.clip_grad_norm_(self.rho_comp.parameters(), max_norm=clip_grad)

        self.opt_main.step()
        self.opt_comp.step()

        # Record
        with torch.no_grad():
            mean_np = self.rho_main.mean.detach().cpu().numpy().copy()
            std_np = self.rho_main.std.detach().cpu().numpy().copy()

        self.history["objective"].append(obj.item())
        self.history["risk"].append(risk.item())
        self.history["kl_main"].append(kl_main.item())
        self.history["kl_comp"].append(kl_comp.item())
        self.history["param_mean"].append(mean_np)
        self.history["param_std"].append(std_np)

        # Polyak averaging
        if self._polyak_mean is None:
            self._polyak_mean = mean_np
        else:
            c = self._polyak_count
            self._polyak_mean = (self._polyak_mean * c + mean_np) / (c + 1)
        self._polyak_count += 1

        return obj.item(), risk.item(), kl_main.item(), kl_comp.item()

    def optimize(
        self,
        n_iter: int = 500,
        n_mc: int = 128,
        lr_main: float = 1e-2,
        lr_comp: float = 1e-2,
        clip_grad: float = 5.0,
        verbose: bool = False,
        log_every: int = 50,
    ) -> None:
        """Run the full optimisation loop.

        Parameters
        ----------
        n_iter : int
            Number of gradient steps.
        n_mc : int
            Monte Carlo samples per step.
        lr_main, lr_comp : float
            Learning rates for main and competitor.
        clip_grad : float
            Gradient norm clipping threshold.
        verbose : bool
            Print progress every ``log_every`` iterations.
        """
        for t in range(1, n_iter + 1):
            obj, risk, kl_m, kl_c = self.step(
                n_mc=n_mc, clip_grad=clip_grad,
                lr_main=lr_main, lr_comp=lr_comp,
            )
            if verbose and (t == 1 or t % log_every == 0):
                m0 = self.rho_main.mean[0].item()
                print(
                    f"[{self._param_label} iter {t:4d}] "
                    f"obj={obj:.4f}  risk={risk:.4f}  "
                    f"KL_main={kl_m:.3f}  KL_comp={kl_c:.3f}  "
                    f"mean[0]={m0:.4f}"
                )


# ====================================================================
# Gaussian location model  theta in R^d
# ====================================================================

class GaussianOptimizer(_PACBayesBase):
    r"""PAC-Bayes optimiser for N(theta, I_d).

    Parameters
    ----------
    data : (n, d) array
        Observed samples.
    lambda_reg : float
        Temperature lambda = tau * n.
    prior_std : float
        Prior standard deviation (default 2).
    """

    _param_label = "Gaussian"

    def __init__(self, data, lambda_reg, prior_std=2.0, device=None):
        data = np.atleast_2d(data)
        super().__init__(data, lambda_reg, prior_std, d=data.shape[1], device=device)

    def _compute_risk(self, theta_main, theta_comp):
        return empirical_psi_risk_gaussian(self.data, theta_main, theta_comp)

    def get_estimate(self, use_polyak: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return (posterior_mean, posterior_std) in theta-space."""
        mean = self.rho_main.mean.detach().cpu().numpy()
        std = self.rho_main.std.detach().cpu().numpy()
        if use_polyak and self._polyak_mean is not None:
            mean = np.array(self._polyak_mean)
        return mean, std


# ====================================================================
# Poisson intensity model  eta = log lambda
# ====================================================================

class PoissonOptimizer(_PACBayesBase):
    r"""PAC-Bayes optimiser for Poisson(exp(eta)), eta = log lambda.

    Parameters
    ----------
    data : (n,) array
        Count observations.
    lambda_reg : float
        Temperature lambda = tau * n.
    prior_std : float
        Prior standard deviation on eta (default 2).
    """

    _param_label = "Poisson"

    def __init__(self, data, lambda_reg, prior_std=2.0, device=None):
        data = np.asarray(data, dtype=np.float32).ravel()
        super().__init__(data, lambda_reg, prior_std, d=1, device=device)

    def _compute_risk(self, eta_main, eta_comp):
        return empirical_psi_risk_poisson(self.data, eta_main, eta_comp)

    def get_estimate(self, use_polyak: bool = True) -> Tuple[float, float]:
        r"""Return (lambda-hat, sigma-eta) where lambda-hat = exp(E[eta])."""
        eta_mean = self.rho_main.mean.detach().cpu().numpy()[0]
        eta_std = self.rho_main.std.detach().cpu().numpy()[0]
        if use_polyak and self._polyak_mean is not None:
            eta_mean = float(self._polyak_mean[0])
        return float(np.exp(eta_mean)), float(eta_std)


# ====================================================================
# Uniform scale model  u = log theta
# ====================================================================

class UniformOptimizer(_PACBayesBase):
    r"""PAC-Bayes optimiser for Uniform[0, exp(u)], u = log theta.

    Parameters
    ----------
    data : (n,) array
        Observations in [0, theta].
    lambda_reg : float
        Temperature lambda = tau * n.
    prior_std : float
        Prior standard deviation on u (default 2).
    """

    _param_label = "Uniform"

    def __init__(self, data, lambda_reg, prior_std=2.0, device=None):
        data = np.asarray(data, dtype=np.float32).ravel()
        super().__init__(data, lambda_reg, prior_std, d=1, device=device)

    def _compute_risk(self, u_main, u_comp):
        return empirical_psi_risk_uniform(self.data, u_main, u_comp)

    def get_estimate(self, use_polyak: bool = True) -> Tuple[float, float]:
        r"""Return (theta-hat, sigma-u) where theta-hat = exp(E[u])."""
        u_mean = self.rho_main.mean.detach().cpu().numpy()[0]
        u_std = self.rho_main.std.detach().cpu().numpy()[0]
        if use_polyak and self._polyak_mean is not None:
            u_mean = float(self._polyak_mean[0])
        return float(np.exp(u_mean)), float(u_std)
