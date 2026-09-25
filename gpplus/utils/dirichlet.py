"""Dirichlet-to-Gaussian target transform (Milios et al., NeurIPS 2018).

Script to execute the Dirichlet classification transform (Eq. 6):
given integer class labels, compute the per-class heteroscedastic noise and the
transformed regression targets.

Zero-mean centering (``ymean``) is intentionally not applied here: it is a
modeling choice made by the user. ``DirichletGaussianLikelihood`` applies it;
the legacy fallback uses ``ymean = 0``.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def prepare_dirichlet_targets(
    targets: Tensor,
    alpha_epsilon: float = 0.01,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, int]:
    """
    Compute Dirichlet alpha, per-class sigma^2 and transformed regression targets.

    Args:
        targets (Tensor): Long tensor of class labels, shape (N,).
        alpha_epsilon (float, optional): Small prior concentration mass added to all classes.
            Default: 0.01.
        dtype (torch.dtype, optional): dtype for output tensors. Defaults to the current
            default dtype if None.

    Returns:
        sigma2_labels: Tensor of shape (num_classes, N) suitable for FixedNoise style noise (transposed)
        transformed_targets: Tensor of shape (N, num_classes)
        num_classes: int
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    targets = targets.long().detach()
    N = targets.shape[0]
    num_classes = int(targets.max().item()) + 1

    alpha = alpha_epsilon * torch.ones((N, num_classes), device=targets.device, dtype=dtype)
    alpha[torch.arange(N, device=targets.device), targets] += 1.0

    # sigma^2 = log(1 / alpha + 1)
    sigma2 = torch.log(alpha.reciprocal() + 1.0)  # shape (N, num_classes)

    # Transform targets: y = log(alpha) - 0.5 * sigma^2
    transformed_targets = alpha.log() - 0.5 * sigma2  # (N, num_classes)

    # Return sigma2 transposed to (num_classes, N) and transformed targets (N, num_classes)
    return sigma2.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), num_classes
