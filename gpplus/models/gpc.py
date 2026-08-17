# """Gaussian Process Classifier using Dirichlet likelihood for multi-class classification.

# Based on Milios et al., NeurIPS 2018: https://arxiv.org/abs/1805.10915

# This module provides a Gaussian Process Classifier that treats multi-class classification
# as a regression problem with carefully chosen heteroscedastic noise, enabling analytic
# inference and calibrated uncertainty estimates.
# """

# import torch
# import gpytorch
# from gpytorch.means import ConstantMean
# from gpytorch.kernels import ScaleKernel, RBFKernel

# from ..likelihoods import LogGaussianLikelihood
# from ..config import logger


# class GPClassifier(gpytorch.models.ExactGP):
#     """
#     Multi-class Gaussian Process Classifier.
    
#     Implements classification using the Dirichlet likelihood approach from Milios et al.
#     Treats classification as regression with fixed heteroscedastic noise, enabling
#     analytic inference and calibrated uncertainty for K-class predictions.
    
#     The model maintains one GP per class, with latent function values transformed to
#     class probabilities via the Dirichlet parameterization.
    
#     Attributes:
#         num_classes (int): Number of classification classes.
#         mean_module (gpytorch.means.Mean): Mean function for GP.
#         covar_module (gpytorch.kernels.Kernel): Covariance kernel for GP.
#         likelihood (DirichletClassificationLikelihood): Dirichlet classification likelihood.
    
#     Example:
#         >>> train_x = torch.randn(100, 5)
#         >>> train_y = torch.randint(0, 3, (100,))
#         >>> model = GPClassifier(train_x, train_y)
#         >>> model.train()
#         >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#         >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
#         >>> 
#         >>> for epoch in range(100):
#         ...     optimizer.zero_grad()
#         ...     output = model(train_x)
#         ...     loss = -mll(output, model.likelihood.transformed_targets.t())
#         ...     loss.backward()
#         ...     optimizer.step()
#     """

#     def __init__(
#         self,
#         train_x: torch.Tensor,
#         train_y: torch.Tensor,
#         mean_module=None,
#         covar_module=None,
#         likelihood=None,
#         alpha_epsilon: float = 0.01,
#     ):
#         """
#         Initialize Gaussian Process Classifier.

#         Args:
#             train_x (torch.Tensor): Training input features. Shape: (n_samples, n_features)
#             train_y (torch.Tensor): Training class labels (0 to num_classes-1). Shape: (n_samples,)
#             mean_module (gpytorch.means.Mean, optional): Mean function. Defaults to ConstantMean.
#             covar_module (gpytorch.kernels.Kernel, optional): Covariance kernel. 
#                 Defaults to ScaleKernel(RBFKernel).
#             likelihood (DirichletClassificationLikelihood, optional): Likelihood.
#                 Defaults to DirichletClassificationLikelihood.
#             alpha_epsilon (float, optional): Dirichlet prior parameter. Default: 0.01
            
#         Raises:
#             ValueError: If train_y contains invalid class indices.
#         """
#         # Validate labels
#         if train_y.min() < 0 or train_y.max() >= 1000:  # sanity check
#             num_classes = int(train_y.max().item()) + 1
#             if train_y.min() < 0:
#                 raise ValueError(f"Class labels must be non-negative, got min={train_y.min()}")
#         else:
#             num_classes = int(train_y.max().item()) + 1
        
#         # Create likelihood (which transforms targets to regression targets)
#         if likelihood is None:
#             # Use MultiLikelihood with Dirichlet-style fixed per-sample noise
#             # encoded_cols=0 is a placeholder; set_fidelity_indices will infer fidelities from training data
#             likelihood = LogGaussianLikelihood(
#                 # batch_shape=torch.Size([num_classes]),
#                 use_dirichlet=True,
#                 dirichlet_targets=train_y,
#                 dirichlet_alpha_epsilon=alpha_epsilon,
#             )

#         # If the likelihood provided transformed targets buffer, use it.
#         if hasattr(likelihood, "dirichlet_transformed_targets") and likelihood.dirichlet_transformed_targets.numel() > 0:
#             # MultiLikelihood stores transformed targets as (N, num_classes);
#             # ExactGP expects training targets matching batch shape (num_classes, N), so transpose.
#             transformed_targets = likelihood.dirichlet_transformed_targets.t()
#         else:
#             # Fall back to raw labels if no transformed targets are available
#             transformed_targets = train_y

#         super().__init__(train_x, transformed_targets, likelihood)
        
#         self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))

        
#         self.covar_module = ScaleKernel(
#             RBFKernel(batch_shape=torch.Size((num_classes,))),
#             batch_shape=torch.Size((num_classes,)),
#         )
#         self.num_classes = num_classes
        
#         logger.info(f"GPClassifier initialized with {num_classes} classes. "
#                    f"Train set: {train_x.shape[0]} samples, {train_x.shape[1]} features. "
#                    f"Likelihood: DirichletClassificationLikelihood(alpha_epsilon={alpha_epsilon})")

#     def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
#         """
#         Forward pass for the GP with batch dimension for each class.
        
#         Args:
#             x (torch.Tensor): Input features. Shape: (batch_size, n_features)
            
#         Returns:
#             gpytorch.distributions.MultivariateNormal: GP distribution with batch shape (num_classes, batch_size).
#         """
#         mean_x = self.mean_module(x)  # Shape: (batch_size,)
#         covar_x = self.covar_module(x)  # Shape: (batch_size, batch_size)

        
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#     def predict(
#         self,
#         x: torch.Tensor,
#         return_std: bool = True,
#     ) -> dict:
#         """
#         Make class predictions with calibrated probabilities.
        
#         Args:
#             x (torch.Tensor): Input features. Shape: (batch_size, n_features)
#             return_std (bool, optional): Return predictive std dev. Default: True
            
#         Returns:
#             dict: Prediction results with keys:
#                 - 'predictions': Class probabilities. Shape: (batch_size, num_classes)
#                 - 'classes': Predicted class indices. Shape: (batch_size,)
#                 - 'std' (if return_std): Predictive std. Shape: (batch_size,)
#         """
#         self.eval()
#         self.likelihood.eval()
        
#         with torch.no_grad():
#             # Get GP predictions for latent functions (batch_size=num_classes)
#             preds = self(x)  # MultivariateNormal with batch shape (num_classes, test_size)

#             # preds.mean shape: (num_classes, test_size) -> logits per class
#             logits = preds.mean  # (num_classes, test_size)

#             # For convenience compute class probabilities *from logits* (these are estimates)
#             probs = torch.softmax(logits.t(), dim=-1)  # (test_size, num_classes)

#             predicted_classes = torch.argmax(probs, dim=-1)  # (test_size,)

#             results = {
#                 'logits': logits,                # shape (num_classes, test_size)
#                 'classes': predicted_classes,    # shape (test_size,)
#             }

#             if not return_std:
#                 # Return only logits and classes
#                 return results

#             # If std requested, return per-sample aggregated std (average across classes)
#             var = preds.variance  # (num_classes, test_size)
#             std_per_sample = torch.sqrt(var).mean(dim=0)  # (test_size,)
#             results['std'] = std_per_sample
        
#         results = {
#             "probs": probs,                 # (test_size, num_classes)
#             "logits": logits,               # (num_classes, test_size)
#             "classes": predicted_classes,   # (test_size,)
#         }

"""Gaussian Process Classifier using Dirichlet likelihood for multi-class classification.

Based on Milios et al., NeurIPS 2018: https://arxiv.org/abs/1805.10915

This module provides a Gaussian Process Classifier that treats multi-class classification
as a regression problem with carefully chosen heteroscedastic noise, enabling analytic
inference and calibrated uncertainty estimates.
"""

import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from ..likelihoods import LogGaussianLikelihood
from ..likelihoods.dirichlet_utils import prepare_dirichlet_targets
from ..config import logger


class GPClassifier(gpytorch.models.ExactGP):
    """
    Multi-class Gaussian Process Classifier using a batched ExactGP,
    one latent GP per class.

    Key ideas:
        • Dirichlet classification as regression
        • One GP per class
        • Batched ExactGP

    Predictive probabilities are computed either via:

        1) softmax(mean) (fast)
        2) MC using marginal variance (safe)
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

        # -------------------------
        # Validate labels
        # -------------------------

        if train_y.ndim != 1:
            raise ValueError(f"train_y must be 1D, got shape {tuple(train_y.shape)}")

        if not torch.is_floating_point(train_x):
            train_x = train_x.float()

        if train_y.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            train_y = train_y.long()

        if train_y.numel() == 0:
            raise ValueError("train_y must not be empty")

        if torch.any(train_y < 0):
            raise ValueError("Class labels must be non-negative")

        unique_labels = torch.unique(train_y).sort().values
        expected = torch.arange(unique_labels.numel(), device=train_y.device)

        if not torch.equal(unique_labels, expected):
            raise ValueError(
                f"Labels must be contiguous starting at 0. Got {unique_labels.tolist()}"
            )

        num_classes = int(unique_labels.numel())
        input_dim = int(train_x.shape[-1])

        # -------------------------
        # Likelihood
        # -------------------------

        if likelihood is None:
            likelihood = LogGaussianLikelihood(
                use_dirichlet=True,
                dirichlet_targets=train_y,
                dirichlet_alpha_epsilon=alpha_epsilon,
            )

        # -------------------------
        # Dirichlet regression targets
        # -------------------------

        _, transformed_targets_nxc, inferred_classes = prepare_dirichlet_targets(
            train_y,
            alpha_epsilon=alpha_epsilon,
            dtype=train_x.dtype,
        )

        if inferred_classes != num_classes:
            raise ValueError("Mismatch in number of classes")

        transformed_targets = transformed_targets_nxc.t().contiguous()

        # Update likelihood buffers if supported
        if hasattr(likelihood, "set_dirichlet_targets"):
            likelihood.set_dirichlet_targets(
                train_y,
                alpha_epsilon=alpha_epsilon,
                dtype=train_x.dtype,
            )

        super().__init__(train_x, transformed_targets, likelihood)

        # -------------------------
        # Mean
        # -------------------------

        self.mean_module = (
            mean_module
            if mean_module is not None
            else ConstantMean(batch_shape=torch.Size((num_classes,)))
        )

        # -------------------------
        # Kernel
        # -------------------------

        self.covar_module = (
            covar_module
            if covar_module is not None
            else ScaleKernel(
                RBFKernel(
                    batch_shape=torch.Size((num_classes,)),
                    ard_num_dims=input_dim,
                ),
                batch_shape=torch.Size((num_classes,)),
            )
        )

        self.num_classes = num_classes

        logger.info(
            f"GPClassifier initialized with {num_classes} classes "
            f"({train_x.shape[0]} samples, {train_x.shape[1]} features)"
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):

        if not torch.is_floating_point(x):
            x = x.float()

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # ------------------------------------------------------------------

    def predict(
        self,
        x: torch.Tensor,
        return_std: bool = True,
        num_mc_samples: int = 0,
        return_entropy: bool = True,
    ):
        """
        Predict class probabilities.

        num_mc_samples options:

        0 → softmax(mean) (fastest)

        >0 → MC integration using marginal Gaussian approx
              (no full joint sampling)
        """

        self.eval()
        self.likelihood.eval()

        if not torch.is_floating_point(x):
            x = x.float()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            latent_dist = self(x)

            logits = latent_dist.mean      # (C,N)
            variances = latent_dist.variance.clamp_min(1e-12)
            latent_std = torch.sqrt(variances)

            # -------------------------------------
            # Fast deterministic probabilities
            # -------------------------------------

            if num_mc_samples <= 0:

                probs = torch.softmax(logits.t(), dim=-1)

            # -------------------------------------
            # MC approximation using marginal vars
            # -------------------------------------

            else:

                eps = torch.randn(
                    num_mc_samples,
                    *logits.shape,
                    device=logits.device,
                    dtype=logits.dtype,
                )

                latent_samples = logits.unsqueeze(0) + eps * latent_std.unsqueeze(0)

                probs_samples = torch.softmax(
                    latent_samples.permute(0, 2, 1),
                    dim=-1,
                )

                probs = probs_samples.mean(dim=0)

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