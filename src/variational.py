"""
Mean-field Gaussian variational family.

The variational distributions q_phi(theta) = N(m, diag(s^2)) are used for both
the main posterior rho and the competitor rho' in the saddle-point formulation.
KL divergences are computed in closed form against the Gaussian prior.
"""

import torch
import torch.nn as nn


class VariationalGaussian(nn.Module):
    r"""Diagonal Gaussian variational distribution q(theta) = N(m, diag(s^2)).

    Parameters
    ----------
    d : int
        Dimension of the parameter space.

    Attributes
    ----------
    mean : nn.Parameter
        Variational mean m in R^d.
    log_std : nn.Parameter
        Log standard deviation log(s) in R^d, ensuring s > 0.
    """

    def __init__(self, d: int):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(d))
        self.log_std = nn.Parameter(torch.zeros(d))

    @property
    def std(self) -> torch.Tensor:
        """Current standard deviation s = exp(log_std)."""
        return torch.exp(self.log_std)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw n_samples from q via the reparametrisation trick.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        (n_samples, d) tensor
        """
        eps = torch.randn(n_samples, self.mean.shape[0], device=self.mean.device)
        return self.mean + self.std * eps

    def kl_divergence(
        self,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
    ) -> torch.Tensor:
        r"""KL(q || pi) for Gaussian q and Gaussian prior pi.

        .. math::
            \mathrm{KL}(q \| \pi)
            = \frac{1}{2} \sum_{j=1}^{d} \left[
                \frac{(m_j - \mu_j)^2}{\sigma_j^2}
                + \frac{s_j^2}{\sigma_j^2}
                - 1
                + 2\log\frac{\sigma_j}{s_j}
            \right].

        Parameters
        ----------
        prior_mean : (d,) tensor
            Prior mean mu.
        prior_std : (d,) tensor
            Prior standard deviation sigma (per coordinate).

        Returns
        -------
        scalar tensor
        """
        std = self.std
        return 0.5 * torch.sum(
            (self.mean - prior_mean) ** 2 / prior_std ** 2
            + std ** 2 / prior_std ** 2
            - 1.0
            + 2.0 * torch.log(prior_std)
            - 2.0 * self.log_std
        )
