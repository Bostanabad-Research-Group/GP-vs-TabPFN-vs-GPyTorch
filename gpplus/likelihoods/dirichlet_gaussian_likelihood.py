"""Heteroscedastic Gaussian likelihood for Dirichlet-based GP classification.

Based on Milios et al., NeurIPS 2018: https://arxiv.org/abs/1805.10915

This module provides DirichletGaussianLikelihood — the likelihood for
GPC as outlined by Milios et al. It replaces LogGaussianLikelihood
(homoskedastic, single learned scalar per class) with a fixed heteroscedastic
noise model where each training point gets a different noise variance per class:

    sigma2_ci = log(1 / alpha_ci + 1)

where alpha_ci = 1 + alpha_epsilon if y_i == c (true class), else alpha_epsilon.

The noise is fixed and fully determined by
the labels and alpha_epsilon.

The likelihood also computes and stores:
    - sigma2_CxN  : (C, N) per-point per-class noise matrix
    - ymean       : (C,)   zero-mean centering offset (added back at prediction)
    - transformed_targets : (N, C) Y_tilde - ymean  (zero-centered, for training)
"""

from typing import Optional, Tuple

import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from ..config import logger
from ..utils.dirichlet import prepare_dirichlet_targets


class DirichletGaussianLikelihood(FixedNoiseGaussianLikelihood):
    """Heteroscedastic fixed-noise Gaussian likelihood for Dirichlet GP classification.

    Implements the noise model from Milios et al. (2018) exactly:

        alpha_ci = 1 + alpha_epsilon   if y_i == c   (true class)
        alpha_ci = alpha_epsilon        otherwise     (false class)

        sigma2_ci = log(1 / alpha_ci + 1)             (Eq. 6)
        y_tilde_ci = log(alpha_ci) - 0.5 * sigma2_ci  (Eq. 6)

    Targets are zero-mean centered before training (ymean subtracted),
    and ymean is added back to GP predictions at inference time.

    The noise sigma2_CxN is fixed.

    Args:
        train_y (torch.Tensor): Class labels, shape (N,), dtype long.
        alpha_epsilon (float): Dirichlet prior concentration. Default: 0.01.
        dtype (torch.dtype, optional): Dtype for computations. Defaults to float64.

    Attributes:
        sigma2_CxN (torch.Tensor): Fixed noise, shape (C, N).
        ymean (torch.Tensor): Zero-mean centering offset, shape (C,).
        transformed_targets (torch.Tensor): Zero-centered Y_tilde, shape (N, C).
        num_classes (int): Number of classes C.
        alpha_epsilon (float): Dirichlet concentration parameter.
    """

    def __init__(
        self,
        train_y: torch.Tensor,
        alpha_epsilon: float = 0.01,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if dtype is None:
            dtype = torch.float64

        if not isinstance(train_y, torch.Tensor):
            raise TypeError("train_y must be a torch.Tensor.")
        if train_y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(train_y.shape)}")

        train_y = train_y.long()
        N = train_y.shape[0]
        num_classes = int(train_y.max().item()) + 1

        sigma2_CxN, transformed_targets, ymean = self._compute_targets(train_y, alpha_epsilon, num_classes, dtype)

        # FixedNoiseGaussianLikelihood expects noise shape (C, N) for batched GPs
        super().__init__(
            noise=sigma2_CxN,
            learn_additional_noise=False,
            batch_shape=torch.Size([num_classes]),
        )

        # Store for access by GPC and at prediction time
        self.register_buffer("sigma2_CxN", sigma2_CxN)
        self.register_buffer("ymean", ymean)
        self.register_buffer("transformed_targets", transformed_targets)

        self.num_classes = num_classes
        self.alpha_epsilon = float(alpha_epsilon)

        logger.info(
            "DirichletGaussianLikelihood: %d classes, %d points, alpha_epsilon=%.4f",
            num_classes,
            N,
            alpha_epsilon,
        )

    @staticmethod
    def _compute_targets(
        train_y: torch.Tensor,
        alpha_epsilon: float,
        num_classes: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sigma2_CxN, zero-centered transformed targets, and ymean.

        Args:
            train_y (torch.Tensor): Class labels, shape (N,), dtype long.
            alpha_epsilon (float): Dirichlet prior concentration.
            num_classes (int): Number of classes C.
            dtype (torch.dtype): Dtype for computations.

        Returns:
            sigma2_CxN          : (C, N)
            transformed_targets : (N, C)  zero-centered Y_tilde
            ymean               : (C,)
        """
        N = train_y.shape[0]
        device = train_y.device

        # Core Dirichlet transform (Eq. 6) — single source of truth in utils.dirichlet.
        # Returns sigma2_CxN (C, N) and the UNCENTERED transformed targets (N, C).
        sigma2_CxN, Y_tilde, _ = prepare_dirichlet_targets(train_y, alpha_epsilon=alpha_epsilon, dtype=dtype)

        # One-hot encode (needed only for the centering offset): Y01 (N, C)
        Y01 = torch.zeros(N, num_classes, dtype=dtype, device=device)
        Y01.scatter_(1, train_y.unsqueeze(1), 1.0)

        # Zero-mean centering — matches Milios script:
        # ymean = log(Y01.mean(0)) + mean(Y_tilde - log(Y01.mean(0)), dim=0)
        log_mean_Y01 = Y01.mean(0).log()  # (C,)
        ymean = log_mean_Y01 + (Y_tilde - log_mean_Y01).mean(0)  # (C,)
        Y_tilde = (Y_tilde - ymean.unsqueeze(0)).contiguous()  # (N, C)

        return sigma2_CxN, Y_tilde, ymean

    def set_dirichlet_targets(
        self,
        train_y: torch.Tensor,
        alpha_epsilon: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Recompute Dirichlet targets and fixed noise from new labels.

        Useful when updating training data between restarts or folds. Named
        ``set_dirichlet_targets`` (not ``set_train_data``) to avoid shadowing
        gpytorch's ``ExactGP.set_train_data`` and to match the hook that
        ``GPC`` probes for on the likelihood.

        Args:
            train_y: New class labels, shape (N,).
            alpha_epsilon: Override alpha_epsilon. Uses stored value if None.
            dtype: Override dtype. Uses float64 if None.
        """
        if alpha_epsilon is None:
            alpha_epsilon = self.alpha_epsilon
        if dtype is None:
            dtype = torch.float64

        train_y = train_y.long()
        num_classes = int(train_y.max().item()) + 1

        sigma2_CxN, transformed_targets, ymean = self._compute_targets(train_y, alpha_epsilon, num_classes, dtype)

        self.sigma2_CxN = sigma2_CxN
        self.ymean = ymean
        self.transformed_targets = transformed_targets
        self.num_classes = num_classes
        self.alpha_epsilon = float(alpha_epsilon)

        # Update the fixed noise in the parent FixedNoiseGaussianLikelihood
        self._set_noise(sigma2_CxN)

    def _set_noise(self, noise: torch.Tensor) -> None:
        """Update the fixed noise buffer in the parent class."""
        noise_covar = self.noise_covar
        if hasattr(noise_covar, "noise"):
            noise_covar.noise = noise
        elif hasattr(noise_covar, "_noise"):
            noise_covar._noise = noise
