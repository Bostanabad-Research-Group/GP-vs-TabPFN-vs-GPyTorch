"""Gaussian Process Classifier using Dirichlet likelihood for multi-class classification.

Based on Milios et al., NeurIPS 2018: https://arxiv.org/abs/1805.10915

This module provides a Gaussian Process Classifier that treats multi-class classification
as a regression problem with carefully chosen heteroscedastic noise, enabling analytic
inference and calibrated uncertainty estimates.
"""

import os

import gpytorch
import torch
from gpytorch.means import ConstantMean

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods.dirichlet_gaussian_likelihood import DirichletGaussianLikelihood
from ..utils.dirichlet import prepare_dirichlet_targets


class GPC(gpytorch.models.ExactGP):
    """Multi-class Gaussian Process Classifier using one GP per class.

    The GPC encapsulates:
      - A mean module (defaults to ConstantMean with batch_shape=(num_classes,) if None).
      - A kernel module (defaults to LogScaleKernel(GaussianKernel) with ARD and batch_shape if None).
      - A likelihood module (defaults to DirichletGaussianLikelihood if None).

    The default DirichletGaussianLikelihood correctly implements the heteroscedastic
    fixed noise from Milios et al. (2018), matching SGPRh(sn2=s2_tilde) in the
    original codebase. Each training point gets a different noise variance per class:
        sigma2_ci = log(1 / alpha_ci + 1)
    The noise is fixed (not learned). Targets are zero-mean centered via ymean,
    which must be added back to GP predictions at inference time.

    Predictive probabilities are computed either via:
      1) softmax(mean) — fast deterministic path.
      2) MC integration using marginal variances — calibrated probabilistic path (Eq. 8).

    Attributes:
        num_classes (int): Number of classification classes.
        mean_module (gpytorch.means.Mean): The mean function of the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance (kernel) function.
        ymean (torch.Tensor): Zero-mean centering offset, shape (num_classes,).
            Must be added back to GP posterior mean at prediction time.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        mean_module=None,
        covar_module=None,
        likelihood=None,
        alpha_epsilon: float = 0.01,
    ):
        """Initializes GPC.

        Args:
            train_x (torch.Tensor): Training data features.
            train_y (torch.Tensor): Training class labels (0 to num_classes-1). Shape: (n_samples,).
            mean_module (gpytorch.means.Mean, optional): Mean function. Defaults to
                ConstantMean with batch_shape=(num_classes,) if None.
            covar_module (gpytorch.kernels.Kernel, optional): Covariance kernel. Defaults to
                LogScaleKernel(GaussianKernel) with ARD and batch_shape=(num_classes,) if None.
            likelihood (gpytorch.likelihoods.Likelihood, optional): Likelihood function.
                Defaults to DirichletGaussianLikelihood (heteroscedastic fixed noise,
                as in Milios et al.). Pass a LogGaussianLikelihood instance to use
                the old homoskedastic behavior.
            alpha_epsilon (float, optional): Dirichlet prior concentration parameter. Default: 0.01.

        Raises:
            TypeError: If train_x or train_y are not torch.Tensor instances.
            ValueError: If train_y is not 1D, is empty, contains negative labels, or
                labels are not contiguous integers starting at 0.
        """
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        if train_y.ndim != 1:
            logger.error("train_y must be 1D, got shape %s.", tuple(train_y.shape))
            raise ValueError(f"train_y must be 1D, got shape {tuple(train_y.shape)}")

        if not torch.is_floating_point(train_x):
            train_x = train_x.float()

        if train_y.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            train_y = train_y.long()

        if train_y.numel() == 0:
            logger.error("train_y must not be empty.")
            raise ValueError("train_y must not be empty.")

        if torch.any(train_y < 0):
            logger.error("Class labels must be non-negative.")
            raise ValueError("Class labels must be non-negative.")

        unique_labels = torch.unique(train_y).sort().values
        expected = torch.arange(unique_labels.numel(), device=train_y.device)

        if not torch.equal(unique_labels, expected):
            logger.error("Labels must be contiguous starting at 0. Got %s.", unique_labels.tolist())
            raise ValueError(f"Labels must be contiguous starting at 0. Got {unique_labels.tolist()}")

        num_classes = int(unique_labels.numel())
        input_dim = int(train_x.shape[-1])

        logger.debug("train_x shape: %s, train_y shape: %s", train_x.shape, train_y.shape)

        # -------------------------
        # Likelihood + targets
        # -------------------------

        if likelihood is None:
            # Default: DirichletGaussianLikelihood — heteroscedastic fixed noise
            # matching Milios et al. SGPRh(sn2=s2_tilde). Targets and ymean are
            # computed and stored inside the likelihood.
            likelihood = DirichletGaussianLikelihood(
                train_y=train_y,
                alpha_epsilon=alpha_epsilon,
                dtype=train_x.dtype,
            )
            logger.info(
                "No likelihood provided. Using DirichletGaussianLikelihood "
                "(heteroscedastic fixed noise, Milios et al.)."
            )

        # -------------------------
        # Regression targets (C, N)
        # -------------------------

        if isinstance(likelihood, DirichletGaussianLikelihood):
            # Targets and ymean come from the likelihood — single source of truth.
            # transformed_targets shape: (N, C) -> transpose to (C, N)
            transformed_targets = likelihood.transformed_targets.t().contiguous()
            ymean = likelihood.ymean
        else:
            # Fallback: legacy behavior for LogGaussianLikelihood or custom likelihoods.
            # Use prepare_dirichlet_targets (no ymean centering).
            logger.warning(
                "Using legacy target computation for non-DirichletGaussianLikelihood. "
                "Heteroscedastic noise structure will not be correctly applied."
            )
            _, transformed_targets_nxc, inferred_classes = prepare_dirichlet_targets(
                train_y,
                alpha_epsilon=alpha_epsilon,
                dtype=train_x.dtype,
            )
            if inferred_classes != num_classes:
                raise ValueError("Mismatch in number of classes.")
            transformed_targets = transformed_targets_nxc.t().contiguous()
            ymean = torch.zeros(num_classes, dtype=train_x.dtype, device=train_x.device)

            # Update likelihood buffers if supported (legacy MultitaskDirichlet etc.)
            if hasattr(likelihood, "set_dirichlet_targets"):
                likelihood.set_dirichlet_targets(
                    train_y,
                    alpha_epsilon=alpha_epsilon,
                    dtype=train_x.dtype,
                )

        super().__init__(train_x, transformed_targets, likelihood)
        self.dtype = train_x.dtype

        # Store ymean — must be added back to GP posterior mean at prediction time
        self.register_buffer("ymean", ymean.to(dtype=self.dtype))

        # -------------------------
        # Mean
        # -------------------------

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
            logger.warning(
                "No mean_module provided. Using ConstantMean with batch_shape=(%d,) as default.",
                num_classes,
            )

        self.mean_module = mean_module

        # -------------------------
        # Kernel
        # -------------------------

        if covar_module is None:
            covar_module = LogScaleKernel(
                GaussianKernel(
                    batch_shape=torch.Size((num_classes,)),
                    ard_num_dims=input_dim,
                ),
                batch_shape=torch.Size((num_classes,)),
            )
            logger.warning(
                "No covar_module provided. Using LogScaleKernel(GaussianKernel) with ARD "
                "(ard_num_dims=%d, num_classes=%d) as default.",
                input_dim,
                num_classes,
            )

        self.covar_module = covar_module
        self.num_classes = num_classes

        # Ensure all components use the same dtype as the input data
        self.mean_module = self.mean_module.to(dtype=self.dtype)
        self.covar_module = self.covar_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)

        logger.info(
            "GPC initialized with %d classes (%d samples, %d features).",
            num_classes,
            train_x.shape[0],
            train_x.shape[1],
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Runs the forward pass of the batched Gaussian Process model.

        Args:
            x (torch.Tensor): Test data features for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal:
                Batched multivariate normal distribution with one GP per class,
                containing the mean and covariance of the predictions.

        Raises:
            TypeError: If x is not a torch.Tensor.
        """
        if not isinstance(x, torch.Tensor):
            logger.error("Input x must be a torch.Tensor instance.")
            raise TypeError("Input x must be a torch.Tensor.")

        if not torch.is_floating_point(x):
            x = x.float()

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # A kernel with no batch shares one set of hyperparameters across classes.
        # Repeat it so each class keeps its own latent GP.
        if covar_x.batch_shape == torch.Size() and self.num_classes > 1:
            covar_x = covar_x.unsqueeze(0).expand(
                self.num_classes, covar_x.shape[-2], covar_x.shape[-1]
            )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(
        self,
        x: torch.Tensor,
        return_std: bool = True,
        num_mc_samples: int = 256,
        return_entropy: bool = True,
    ) -> dict:
        """Predict class probabilities for input data.

        The GP posterior mean is in the zero-centered Dirichlet space. ymean is
        added back before computing probabilities, matching the Milios script:
            fmu = fmu + ymean

        Args:
            x (torch.Tensor): Input features. Shape: (n_samples, n_features).
            return_std (bool, optional): If True, includes per-class latent standard
                deviation in the output. Default: True.
            num_mc_samples (int, optional): Number of Monte Carlo samples for probability
                estimation following Eq. (8) of Milios et al. (2018).
                If 0, uses fast softmax(mean). If > 0, samples from marginal GP
                posteriors (class-independent) and averages softmax. Default: 256.
            return_entropy (bool, optional): If True, includes predictive entropy in
                the output. Default: True.

        Returns:
            dict: A dictionary containing:
                - **probs** (torch.Tensor): Class probabilities. Shape: (n_samples, num_classes).
                - **classes** (torch.Tensor): Predicted class indices. Shape: (n_samples,).
                - **logits** (torch.Tensor): Raw latent means (after ymean). Shape: (num_classes, n_samples).
                - **latent_std** (torch.Tensor, optional): Per-class latent standard deviation.
                  Shape: (n_samples, num_classes). Only present if return_std is True.
                - **entropy** (torch.Tensor, optional): Predictive entropy per sample.
                  Shape: (n_samples,). Only present if return_entropy is True.
        """
        self.eval()
        self.likelihood.eval()

        if not torch.is_floating_point(x):
            x = x.float()

        with torch.no_grad():
            # -------------------------------------
            # Fast deterministic probabilities
            # softmax(posterior mean + ymean)
            # -------------------------------------

            if num_mc_samples <= 0:
                with gpytorch.settings.fast_pred_var():
                    latent_dist = self(x)
                    logits = latent_dist.mean  # (C, N)
                    variances = latent_dist.variance.clamp_min(1e-12)
                    latent_std = torch.sqrt(variances)

                # Add ymean back — matches Milios: fmu = fmu + ymean
                # ymean shape (C,) -> unsqueeze to (C, 1) for broadcasting
                logits = logits + self.ymean.unsqueeze(1)

                probs = torch.softmax(logits.t(), dim=-1)

            # -------------------------------------
            # MC integration of Eq. (8) — Milios et al. (2018)
            # Sample from marginal GP posteriors per class (class-independent
            # by Section 4 Remark), apply softmax, average over samples.
            # -------------------------------------

            else:
                with gpytorch.settings.fast_pred_var():
                    latent_dist = self(x)
                    logits = latent_dist.mean  # (C, N)
                    variances = latent_dist.variance.clamp_min(1e-12)
                    latent_std = torch.sqrt(variances)

                # Add ymean back before sampling
                logits = logits + self.ymean.unsqueeze(1)

                # Posterior variance diagnostic
                var_mean = variances.mean().item()
                var_max = variances.max().item()
                var_min = variances.min().item()
                logger.debug(
                    "Posterior var diagnostic: mean=%.6f  max=%.6f  min=%.6f%s",
                    var_mean,
                    var_max,
                    var_min,
                    "  <- near-zero: MC approx softmax(mean)" if var_max < 0.01 else "",
                )

                # Sample from each class marginal posterior: (S, C, N)
                eps = torch.randn(
                    num_mc_samples,
                    *logits.shape,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                latent_samples = logits.unsqueeze(0) + eps * latent_std.unsqueeze(0)

                # Permute to (S, N, C), softmax over class dim, average over S
                probs_samples = torch.softmax(
                    latent_samples.permute(0, 2, 1),
                    dim=-1,
                )
                probs = probs_samples.mean(dim=0)  # (N, C)

            predicted_classes = torch.argmax(probs, dim=-1)

            results = {
                "probs": probs,
                "classes": predicted_classes,
                "logits": logits,
            }

            if return_std:
                results["latent_std"] = latent_std.transpose(0, 1)

            if return_entropy:
                p = probs.clamp_min(1e-12)
                entropy = -(p * p.log()).sum(dim=-1)
                results["entropy"] = entropy

            return results

    def save(self, filepath: str = "model_weights.pth") -> None:
        """Saves this model's state dictionary to the specified file.

        Args:
            filepath (str, optional): Path to save the state dictionary file.
                Defaults to 'model_weights.pth' in the current directory.
        """
        logger.info("Saving model state dict to %s", filepath)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "model_weights.pth") -> None:
        """Loads this model's state dictionary from the specified file.

        Args:
            filepath (str, optional): Path to the file containing the saved state dict.
                Defaults to 'model_weights.pth' in the current directory.

        Raises:
            FileNotFoundError: If no file is found at filepath.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")

        logger.info("Loading model state dict from %s", filepath)
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)
