import math

import torch

from .unconstrained_kernel import UnconstrainedKernel


def postprocess_matern(dist_mat: torch.Tensor, nu: float) -> torch.Tensor:
    """
    Apply the closed-form Matérn function for special values of nu.
    Supported: nu = 0.5, 1.5, 2.5

    Notes
    -----
    This expects `dist_mat` to already be the (scaled) Euclidean distance r.
    """
    if nu == 0.5:
        return torch.exp(-dist_mat)
    elif nu == 1.5:
        sqrt3 = math.sqrt(3.0)
        return (1.0 + sqrt3 * dist_mat) * torch.exp(-sqrt3 * dist_mat)
    elif nu == 2.5:
        sqrt5 = math.sqrt(5.0)
        return (1.0 + sqrt5 * dist_mat + 5.0 / 3.0 * dist_mat**2) * torch.exp(-sqrt5 * dist_mat)
    else:
        raise NotImplementedError("Only nu=0.5, 1.5, 2.5 are supported")


class MaternKernel(UnconstrainedKernel):
    """
    Matérn kernel where the learnable lengthscale parameter is in base-10 log space.

    Implementation detail: this repo parameterizes lengthscales in base-10 log space in a way
    that matches `GaussianKernel` / `PowerExponentialKernel`: inputs are scaled by
    ``10**(lengthscale / 2)`` (for Euclidean distances), so that squared distances scale by
    ``10**lengthscale``. Equivalently, `lengthscale` behaves like a log10 inverse *squared*
    lengthscale.
    """

    has_lengthscale = True
    is_stationary = True

    def __init__(self, nu: float = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.nu = float(nu)

    def forward(self, x1, x2, diag: bool = False, **params):
        power = 2.0  # Euclidean norm

        # Match repo convention (see `GaussianKernel`): scale coordinates by 10**(w/2)
        # so that squared distances scale by 10**w.
        scaling_factors = torch.pow(10.0, self.lengthscale / power)
        x1_ = x1.mul(scaling_factors)
        x2_ = x2.mul(scaling_factors)

        # covar_dist returns (cdist)**power; take sqrt to recover Euclidean distance
        dist = self.covar_dist(x1_, x2_, ord=power, diag=diag, **params).sqrt()
        return postprocess_matern(dist, self.nu)


