"""
PAC-Bayesian saddle-point optimiser for linear regression.

The regression model is :math:`Y = X \\beta + \\xi` with fixed design
:math:`X \\in \\mathbb{R}^{n \\times p}` and Gaussian errors.
The likelihood ratio involves the regression residuals rather than
raw density ratios.

Following Section 3 of the paper, the variational rho-posterior for
regression uses a mean-field Gaussian :math:`q(\\beta) = N(m, \\mathrm{diag}(s^2))`
and minimizes the saddle-point objective with temperature :math:`\\lambda = \\tau n`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.utils as nn_utils

from .variational import VariationalGaussian
from .contrasts import psi_hellinger


class RegressionOptimizer:
    r"""PAC-Bayes optimiser for fixed-design Gaussian linear regression.

    The empirical psi-risk for regression is

    .. math::
        \hat{R}_\psi(\beta_1, \beta_2)
        = \frac{1}{n} \sum_{i=1}^{n}
          \psi\!\left(
            \exp\!\left(-\tfrac{1}{2}\left[
              (y_i - x_i^\top \beta_2)^2 - (y_i - x_i^\top \beta_1)^2
            \right]\right)
          \right).

    Parameters
    ----------
    X : (n, p) ndarray
        Design matrix.
    y : (n,) ndarray
        Response vector.
    lambda_reg : float
        Temperature parameter :math:`\lambda = \tau n`.
    prior_std : float
        Prior standard deviation on each coordinate of :math:`\beta`.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_reg: float,
        prior_std: float = 2.0,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.n, self.p = X.shape
        self.lambda_reg = lambda_reg

        self.prior_mean = torch.zeros(self.p, device=self.device)
        self.prior_std = torch.ones(self.p, device=self.device) * prior_std

        self.rho_main = VariationalGaussian(self.p).to(self.device)
        self.rho_comp = VariationalGaussian(self.p).to(self.device)

        self.opt_main = torch.optim.Adam(self.rho_main.parameters(), lr=1e-2)
        self.opt_comp = torch.optim.Adam(self.rho_comp.parameters(), lr=1e-2)

        self.history: dict[str, list] = dict(
            objective=[], risk=[], kl_main=[], kl_comp=[],
        )
        self._polyak_mean: np.ndarray | None = None
        self._polyak_count: int = 0

    def _regression_likelihood_ratio(
        self,
        beta1: torch.Tensor,
        beta2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute likelihood ratios for all data points and MC samples.

        Parameters
        ----------
        beta1 : (k1, p) tensor
        beta2 : (k2, p) tensor

        Returns
        -------
        (n, k1, k2) tensor
        """
        # Residuals: (n, k)
        resid1 = self.y.unsqueeze(1) - self.X @ beta1.T  # (n, k1)
        resid2 = self.y.unsqueeze(1) - self.X @ beta2.T  # (n, k2)

        sq1 = resid1.pow(2).unsqueeze(2)  # (n, k1, 1)
        sq2 = resid2.pow(2).unsqueeze(1)  # (n, 1, k2)

        log_ratio = -0.5 * (sq2 - sq1)  # (n, k1, k2)
        return torch.exp(torch.clamp(log_ratio, min=-40.0, max=40.0))

    def compute_objective(self, n_mc: int = 128):
        """Monte Carlo estimate of the saddle-point objective."""
        beta_main = self.rho_main.sample(n_mc)
        beta_comp = self.rho_comp.sample(n_mc)

        ratios = self._regression_likelihood_ratio(beta_main, beta_comp)
        risk = psi_hellinger(ratios).mean()

        kl_main = self.rho_main.kl_divergence(self.prior_mean, self.prior_std)
        kl_comp = self.rho_comp.kl_divergence(self.prior_mean, self.prior_std)

        obj = risk + (kl_main - kl_comp) / self.lambda_reg
        return obj, risk, kl_main, kl_comp

    def step(
        self,
        n_mc: int = 128,
        clip_grad: float = 5.0,
        lr_main: float | None = None,
        lr_comp: float | None = None,
    ) -> Tuple[float, float, float, float]:
        """One gradient step."""
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

        for p in self.rho_comp.parameters():
            if p.grad is not None:
                p.grad.mul_(-1.0)

        if clip_grad is not None:
            nn_utils.clip_grad_norm_(self.rho_main.parameters(), max_norm=clip_grad)
            nn_utils.clip_grad_norm_(self.rho_comp.parameters(), max_norm=clip_grad)

        self.opt_main.step()
        self.opt_comp.step()

        with torch.no_grad():
            mean_np = self.rho_main.mean.detach().cpu().numpy().copy()

        self.history["objective"].append(obj.item())
        self.history["risk"].append(risk.item())
        self.history["kl_main"].append(kl_main.item())
        self.history["kl_comp"].append(kl_comp.item())

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
        """Run the full optimization loop."""
        for t in range(1, n_iter + 1):
            obj, risk, kl_m, kl_c = self.step(
                n_mc=n_mc, clip_grad=clip_grad,
                lr_main=lr_main, lr_comp=lr_comp,
            )
            if verbose and (t == 1 or t % log_every == 0):
                print(
                    f"[Regression iter {t:4d}] "
                    f"obj={obj:.4f}  risk={risk:.4f}  "
                    f"KL_main={kl_m:.3f}  KL_comp={kl_c:.3f}"
                )

    def get_estimate(self, use_polyak: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return (posterior mean, posterior std) for :math:`\beta`.

        Parameters
        ----------
        use_polyak : bool
            Use Polyak-averaged mean.

        Returns
        -------
        mean : (p,) ndarray
        std : (p,) ndarray
        """
        mean = self.rho_main.mean.detach().cpu().numpy()
        std = self.rho_main.std.detach().cpu().numpy()
        if use_polyak and self._polyak_mean is not None:
            mean = np.array(self._polyak_mean)
        return mean, std
