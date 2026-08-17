# from typing import Any, Optional, Union

# import torch
# from linear_operator import to_linear_operator
# from linear_operator.operators import (
#     ConstantDiagLinearOperator,
#     DiagLinearOperator,
#     KroneckerProductDiagLinearOperator,
#     KroneckerProductLinearOperator,
#     LinearOperator,
#     RootLinearOperator,
# )
# from torch import Tensor
# from torch.distributions import Normal

# from gpytorch.constraints import GreaterThan, Interval
# from gpytorch.distributions import base_distributions, MultitaskMultivariateNormal
# from gpytorch.lazy import LazyEvaluatedKernelTensor
# from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood
# from gpytorch.priors import Prior
# from gpytorch.likelihoods.noise_models import FixedGaussianNoise, Noise
# from gpytorch.distributions import MultivariateNormal
# from .dirichlet_utils import prepare_dirichlet_targets

# class _MultitaskGaussianLikelihoodBase(_GaussianLikelihoodBase):
#     r"""
#     Base class for multi-task Gaussian Likelihoods, supporting general heteroskedastic noise models.

#     :param num_tasks: Number of tasks.
#     :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
#         that is to be fitted on the observed measurement errors.
#     :param rank: The rank of the task noise covariance matrix to fit. If `rank`
#         is set to 0, then a diagonal covariance matrix is fit.
#     :param task_correlation_prior: Prior to use over the task noise correlation
#         matrix. Only used when :math:`\text{rank} > 0`.
#     :param batch_shape: Number of batches.
#     """

#     def __init__(
#         self,
#         num_tasks: int,
#         noise_covar: Union[Noise, FixedGaussianNoise],
#         rank: int = 0,
#         task_correlation_prior: Optional[Prior] = None,
#         batch_shape: torch.Size = torch.Size(),
#     ) -> None:
#         super().__init__(noise_covar=noise_covar)
#         if rank != 0:
#             if rank > num_tasks:
#                 raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
#             tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
#             self.tidcs: Tensor = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
#             task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
#             self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
#             if task_correlation_prior is not None:
#                 self.register_prior(
#                     "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda m: m._eval_corr_matrix
#                 )
#         elif task_correlation_prior is not None:
#             raise ValueError("Can only specify task_correlation_prior if rank>0")
#         self.num_tasks = num_tasks
#         self.rank = rank

#     def _eval_corr_matrix(self) -> Tensor:
#         tnc = self.task_noise_corr
#         fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
#         Cfac = torch.diag_embed(fac_diag)
#         Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
#         # squared rows must sum to one for this to be a correlation matrix
#         C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
#         return C @ C.transpose(-1, -2)

#     def marginal(
#         self, function_dist: MultitaskMultivariateNormal, *params: Any, **kwargs: Any
#     ) -> MultitaskMultivariateNormal:  # pyre-ignore[14]
#         r"""
#         If :math:`\text{rank} = 0`, adds the task noises to the diagonal of the
#         covariance matrix of the supplied
#         :obj:`~gpytorch.distributions.MultivariateNormal` or
#         :obj:`~gpytorch.distributions.MultitaskMultivariateNormal`.  Otherwise,
#         adds a rank `rank` covariance matrix to it.

#         To accomplish this, we form a new
#         :obj:`~linear_operator.operators.KroneckerProductLinearOperator`
#         between :math:`I_{n}`, an identity matrix with size equal to the data
#         and a (not necessarily diagonal) matrix containing the task noises
#         :math:`D_{t}`.

#         We also incorporate a shared `noise` parameter from the base
#         :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

#         The final covariance matrix after this method is then
#         :math:`\mathbf K + \mathbf D_{t} \otimes \mathbf I_{n} + \sigma^{2} \mathbf I_{nt}`.

#         :param function_dist: Random variable whose covariance
#             matrix is a :obj:`~linear_operator.operators.LinearOperator` we intend to augment.
#         :rtype: `gpytorch.distributions.MultitaskMultivariateNormal`:
#         :return: A new random variable whose covariance matrix is a
#             :obj:`~linear_operator.operators.LinearOperator` with
#             :math:`\mathbf D_{t} \otimes \mathbf I_{n}` and :math:`\sigma^{2} \mathbf I_{nt}` added.
#         """
#         mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

#         # ensure that sumKroneckerLT is actually called
#         if isinstance(covar, LazyEvaluatedKernelTensor):
#             covar = covar.evaluate_kernel()

#         covar_kron_lt = self._shaped_noise_covar(
#             mean.shape, add_noise=self.has_global_noise, interleaved=function_dist._interleaved
#         )
#         covar = covar + covar_kron_lt

#         return function_dist.__class__(mean, covar, interleaved=function_dist._interleaved)

#     def _shaped_noise_covar(
#         self, shape: torch.Size, add_noise: Optional[bool] = True, interleaved: bool = True, *params: Any, **kwargs: Any
#     ) -> LinearOperator:
#         if not self.has_task_noise:
#             noise = ConstantDiagLinearOperator(self.noise, diag_shape=shape[-2] * self.num_tasks)
#             return noise

#         if self.rank == 0:
#             task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
#             task_var_lt = DiagLinearOperator(task_noises)
#             dtype, device = task_noises.dtype, task_noises.device
#             ckl_init = KroneckerProductDiagLinearOperator
#         else:
#             task_noise_covar_factor = self.task_noise_covar_factor
#             task_var_lt = RootLinearOperator(task_noise_covar_factor)
#             dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device
#             ckl_init = KroneckerProductLinearOperator

#         eye_lt = ConstantDiagLinearOperator(
#             torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2]
#         )
#         task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)  # pyre-ignore[6]

#         # to add the latent noise we exploit the fact that
#         # I \kron D_T + \sigma^2 I_{NT} = I \kron (D_T + \sigma^2 I)
#         # which allows us to move the latent noise inside the task dependent noise
#         # thereby allowing exploitation of Kronecker structure in this likelihood.
#         if add_noise and self.has_global_noise:
#             noise = ConstantDiagLinearOperator(self.noise, diag_shape=task_var_lt.shape[-1])
#             task_var_lt = task_var_lt + noise

#         if interleaved:
#             covar_kron_lt = ckl_init(eye_lt, task_var_lt)
#         else:
#             covar_kron_lt = ckl_init(task_var_lt, eye_lt)

#         return covar_kron_lt

#     def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
#         noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
#         noise = noise.reshape(*noise.shape[:-1], *function_samples.shape[-2:])
#         return base_distributions.Independent(base_distributions.Normal(function_samples, noise.sqrt()), 1)


# class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
#     r"""
#     A convenient extension of the :class:`~gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
#     for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
#     If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
#     allows for a different `noise` parameter for each task.)

#     Like the Gaussian likelihood, this object can be used with exact inference.

#     .. note::
#         At least one of :attr:`has_global_noise` or :attr:`has_task_noise` should be specified.

#     .. note::
#         MultittaskGaussianLikelihood has an analytic marginal distribution.

#     :param num_tasks: Number of tasks.
#     :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
#         that is to be fitted on the observed measurement errors.
#     :param rank: The rank of the task noise covariance matrix to fit. If `rank`
#         is set to 0, then a diagonal covariance matrix is fit.
#     :param task_prior: Prior to use over the task noise correlation
#         matrix. Only used when :math:`\text{rank} > 0`.
#     :param batch_shape: Number of batches.
#     :param has_global_noise: Whether to include a :math:`\sigma^2 \mathbf I_{nt}` term in the noise model.
#     :param has_task_noise: Whether to include task-specific noise terms, which add
#         :math:`\mathbf I_n \otimes \mathbf D_T` into the noise model.

#     :ivar torch.Tensor task_noise_covar: The inter-task noise covariance matrix
#     :ivar torch.Tensor task_noises: (Optional) task specific noise variances (added onto the `task_noise_covar`)
#     :ivar torch.Tensor noise: (Optional) global noise variance (added onto the `task_noise_covar`)
#     """

#     def __init__(
#         self,
#         num_tasks: int,
#         rank: int = 0,
#         batch_shape: torch.Size = torch.Size(),
#         task_prior: Optional[Prior] = None,
#         noise_prior: Optional[Prior] = None,
#         noise_constraint: Optional[Interval] = None,
#         has_global_noise: bool = True,
#         has_task_noise: bool = True,
#         use_dirichlet: bool = False,
#         dirichlet_targets: Optional[Tensor] = None,
#         dirichlet_alpha_epsilon: float = 0.01,
#     ) -> None:
#         super(Likelihood, self).__init__()  # pyre-ignore[20]
#         if noise_constraint is None:
#             noise_constraint = GreaterThan(1e-4)

#         if not has_task_noise and not has_global_noise:
#             raise ValueError(
#                 "At least one of has_task_noise or has_global_noise must be specified. "
#                 "Attempting to specify a likelihood that has no noise terms."
#             )

#         if has_task_noise:
#             if rank == 0:
#                 self.register_parameter(
#                     name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
#                 )
#                 self.register_constraint("raw_task_noises", noise_constraint)
#                 if noise_prior is not None:
#                     self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
#                 if task_prior is not None:
#                     raise RuntimeError("Cannot set a `task_prior` if rank=0")
#             else:
#                 self.register_parameter(
#                     name="task_noise_covar_factor",
#                     parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
#                 )
#                 if task_prior is not None:
#                     self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
#         self.num_tasks = num_tasks
#         self.rank = rank

#         if has_global_noise:
#             self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
#             self.register_constraint("raw_noise", noise_constraint)
#             if noise_prior is not None:
#                 self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

#         self.has_global_noise = has_global_noise
#         self.has_task_noise = has_task_noise
        
        
#         # Optional Dirichlet buffers: per-sample additive variance and transformed targets
#         # `dirichlet_diag` is a 1D tensor of shape (N,) representing per-sample additive variance
#         # `dirichlet_transformed_targets` is (N, num_classes)
#         self.register_buffer("dirichlet_diag", torch.empty(0), persistent=True)
#         self.register_buffer("dirichlet_transformed_targets", torch.empty(0), persistent=True)

#         # Dirichlet control flags and hyperparameter
#         self.use_dirichlet = bool(use_dirichlet)
#         self.dirichlet_alpha_epsilon = float(dirichlet_alpha_epsilon)

#         # If targets are provided at construction time and dirichlet is enabled, set them
#         if self.use_dirichlet and dirichlet_targets is not None:
#             # try:
#                 # set_dirichlet_targets will register the buffers
#             self.set_dirichlet_targets(dirichlet_targets, alpha_epsilon=self.dirichlet_alpha_epsilon)
#             # except Exception:
#             #     # ignore malformed inputs; user can call setter later
#             #     pass

#     @property
#     def noise(self) -> Optional[Tensor]:
#         return self.raw_noise_constraint.transform(self.raw_noise)

#     @noise.setter
#     def noise(self, value: Union[float, Tensor]) -> None:
#         self._set_noise(value)

#     @property
#     def task_noises(self) -> Optional[Tensor]:
#         if self.rank == 0:
#             return self.raw_task_noises_constraint.transform(self.raw_task_noises)
#         else:
#             raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

#     @task_noises.setter
#     def task_noises(self, value: Union[float, Tensor]) -> None:
#         if self.rank == 0:
#             self._set_task_noises(value)
#         else:
#             raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

#     def _set_noise(self, value: Union[float, Tensor]) -> None:
#         self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

#     def _set_task_noises(self, value: Union[float, Tensor]) -> None:
#         self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

#     @property
#     def task_noise_covar(self) -> Tensor:
#         if self.rank > 0:
#             return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2))
#         else:
#             raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

#     @task_noise_covar.setter
#     def task_noise_covar(self, value: Tensor) -> None:
#         # internally uses a pivoted cholesky decomposition to construct a low rank
#         # approximation of the covariance
#         if self.rank > 0:
#             with torch.no_grad():
#                 self.task_noise_covar_factor.data = to_linear_operator(value).pivoted_cholesky(rank=self.rank)
#         else:
#             raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

#     # def _eval_covar_matrix(self) -> Tensor:
#     #     covar_factor = self.task_noise_covar_factor
#     #     noise = self.noise
#     #     D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
#     #     return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D
    
#     def _eval_covar_matrix(self) -> Tensor:
#         covar_factor = self.task_noise_covar_factor
#         noise = self.noise
#         D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
#         temp= covar_factor.matmul(covar_factor.transpose(-1, -2)) + D
#         dirichlet_diag = self.dirichlet_diag
#         if dirichlet_diag is not None and dirichlet_diag.numel() > 0:
#             temp = temp + torch.diag(dirichlet_diag)
#         return temp
        

#     def marginal(
#         self, function_dist: MultitaskMultivariateNormal, *args: Any, **kwargs: Any
#     ) -> MultitaskMultivariateNormal:
#         r"""
#         :return: Analytic marginal :math:`p(\mathbf y)`.
#         """
#         return super().marginal(function_dist, *args, **kwargs)
        
#     def set_dirichlet_targets(self, targets: Tensor, alpha_epsilon: float = 0.01, dtype: torch.dtype = None) -> int:
#         """
#         Compute and register Dirichlet-transformed targets and per-sample variance.

#         Args:
#             targets: Long tensor of class labels shape (N,)
#             alpha_epsilon: small prior mass for Dirichlet
#             dtype: dtype for computations (defaults to default dtype)

#         Returns:
#             num_classes (int)
#         """
#         # Delegate math to shared helper to ensure single source of truth
#         sigma2, transformed_targets, num_classes = prepare_dirichlet_targets(targets, alpha_epsilon=alpha_epsilon, dtype=dtype)

#         # Per-sample diag is sigma2 for the observed class; sigma2 returned is (K, N)
#         sigma2_per_sample = sigma2.transpose(-2, -1)[torch.arange(targets.shape[0], device=targets.device), targets]

#         # Register buffers
#         if self.dirichlet_diag.numel() == 0 or self.dirichlet_diag.shape[0] != sigma2_per_sample.shape[0]:
#             self.register_buffer("dirichlet_diag", sigma2_per_sample.detach(), persistent=True)
#         else:
#             self.dirichlet_diag.copy_(sigma2_per_sample.detach())

#         if self.dirichlet_transformed_targets.numel() == 0 or self.dirichlet_transformed_targets.shape != transformed_targets.shape:
#             self.register_buffer("dirichlet_transformed_targets", transformed_targets.detach(), persistent=True)
#         else:
#             self.dirichlet_transformed_targets.copy_(transformed_targets.detach())

#         return num_classes


from typing import Any, Optional, Union

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
)
from torch import Tensor
from torch.distributions import Normal

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.distributions import base_distributions, MultitaskMultivariateNormal
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.likelihoods import _GaussianLikelihoodBase, Likelihood
from gpytorch.priors import Prior
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, Noise
from gpytorch.distributions import MultivariateNormal

from .dirichlet_utils import prepare_dirichlet_targets


class _MultitaskGaussianLikelihoodBase(_GaussianLikelihoodBase):
    r"""
    Base class for multi-task Gaussian Likelihoods, supporting general heteroskedastic noise models.

    :param num_tasks: Number of tasks.
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_correlation_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    """

    def __init__(
        self,
        num_tasks: int,
        noise_covar: Union[Noise, FixedGaussianNoise],
        rank: int = 0,
        task_correlation_prior: Optional[Prior] = None,
        batch_shape: torch.Size = torch.Size(),
    ) -> None:
        super().__init__(noise_covar=noise_covar)

        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs: Tensor = tidcs[:, 1:]
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior",
                    task_correlation_prior,
                    lambda m: m._eval_corr_matrix,
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")

        self.num_tasks = num_tasks
        self.rank = rank

    def _eval_corr_matrix(self) -> Tensor:
        tnc = self.task_noise_corr
        fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
        Cfac = torch.diag_embed(fac_diag)
        Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
        C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return C @ C.transpose(-1, -2)

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> MultitaskMultivariateNormal:
        r"""
        Adds task/global observation noise to the covariance of the supplied
        multitask distribution.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        covar_kron_lt = self._shaped_noise_covar(
            mean.shape,
            add_noise=self.has_global_noise,
            interleaved=function_dist._interleaved,
        )
        covar = covar + covar_kron_lt

        return function_dist.__class__(mean, covar, interleaved=function_dist._interleaved)

    def _shaped_noise_covar(
        self,
        shape: torch.Size,
        add_noise: Optional[bool] = True,
        interleaved: bool = True,
        *params: Any,
        **kwargs: Any,
    ) -> LinearOperator:
        """
        Build the multitask observation-noise operator.

        Generic Dirichlet per-sample heteroscedastic noise is added here only when a
        dedicated multitask Dirichlet subclass is NOT already handling the full
        multitask Dirichlet covariance inside marginal().
        """
        n = shape[-2]

        if not self.has_task_noise:
            base_noise = ConstantDiagLinearOperator(self.noise, diag_shape=n * self.num_tasks)
        else:
            if self.rank == 0:
                task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
                task_var_lt = DiagLinearOperator(task_noises)
                dtype, device = task_noises.dtype, task_noises.device
                ckl_init = KroneckerProductDiagLinearOperator
            else:
                task_noise_covar_factor = self.task_noise_covar_factor
                task_var_lt = RootLinearOperator(task_noise_covar_factor)
                dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device
                ckl_init = KroneckerProductLinearOperator

            eye_lt = ConstantDiagLinearOperator(
                torch.ones(*shape[:-2], 1, dtype=dtype, device=device),
                diag_shape=n,
            )
            task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)

            if add_noise and self.has_global_noise:
                noise = ConstantDiagLinearOperator(self.noise, diag_shape=task_var_lt.shape[-1])
                task_var_lt = task_var_lt + noise

            if interleaved:
                base_noise = ckl_init(eye_lt, task_var_lt)
            else:
                base_noise = ckl_init(task_var_lt, eye_lt)

        # Generic Dirichlet support:
        # - if a subclass already exposes dirichlet_sigma2_full, let that subclass
        #   handle the full multitask diagonal in marginal()
        if getattr(self, "use_dirichlet", False):
            dirichlet_sigma2_full = getattr(self, "dirichlet_sigma2_full", None)
            has_full_multitask_dirichlet = (
                dirichlet_sigma2_full is not None and dirichlet_sigma2_full.numel() > 0
            )

            if not has_full_multitask_dirichlet:
                dirichlet_diag = getattr(self, "dirichlet_diag", None)
                if dirichlet_diag is not None and dirichlet_diag.numel() > 0:
                    if dirichlet_diag.shape[0] == n:
                        if interleaved:
                            full_diag = dirichlet_diag.repeat_interleave(self.num_tasks)
                        else:
                            full_diag = dirichlet_diag.repeat(self.num_tasks)
                    elif dirichlet_diag.shape[0] == n * self.num_tasks:
                        full_diag = dirichlet_diag
                    else:
                        raise RuntimeError(
                            f"dirichlet_diag has length {dirichlet_diag.shape[0]}, "
                            f"expected either {n} or {n * self.num_tasks}"
                        )

                    base_noise = base_noise + DiagLinearOperator(full_diag)

        return base_noise

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(
            dim1=-1, dim2=-2
        )
        noise = noise.reshape(*noise.shape[:-1], *function_samples.shape[-2:])
        return base_distributions.Independent(
            base_distributions.Normal(function_samples, noise.sqrt()),
            1,
        )


class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    r"""
    Extension of GaussianLikelihood to the multitask setting.

    Allows for a full cross-task covariance structure in the noise when rank > 0.
    If rank=0, uses diagonal task noise.

    This version also supports optional Dirichlet-style transformed targets and
    per-sample heteroscedastic noise for classification-via-regression.
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
        use_dirichlet: bool = False,
        dirichlet_targets: Optional[Tensor] = None,
        dirichlet_alpha_epsilon: float = 0.01,
    ) -> None:
        # Preserving your original initialization style to minimize breakage.
        super(Likelihood, self).__init__()  # pyre-ignore[20]

        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises",
                    parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks)),
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior(
                        "MultitaskErrorCovariancePrior",
                        task_prior,
                        lambda m: m._eval_covar_matrix,
                    )

        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(
                name="raw_noise",
                parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)),
            )
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

        self.register_buffer("dirichlet_diag", torch.empty(0), persistent=True)
        self.register_buffer("dirichlet_transformed_targets", torch.empty(0), persistent=True)

        self.use_dirichlet = bool(use_dirichlet)
        self.dirichlet_alpha_epsilon = float(dirichlet_alpha_epsilon)

        if self.use_dirichlet and dirichlet_targets is not None:
            self.set_dirichlet_targets(
                dirichlet_targets,
                alpha_epsilon=self.dirichlet_alpha_epsilon,
            )

    @property
    def noise(self) -> Optional[Tensor]:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Union[float, Tensor]) -> None:
        self._set_noise(value)

    @property
    def task_noises(self) -> Optional[Tensor]:
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        raise AttributeError("Cannot set diagonal task noises when covariance has rank > 0")

    @task_noises.setter
    def task_noises(self, value: Union[float, Tensor]) -> None:
        if self.rank == 0:
            self._set_task_noises(value)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has rank > 0")

    def _set_noise(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

    @property
    def task_noise_covar(self) -> Tensor:
        if self.rank > 0:
            return self.task_noise_covar_factor.matmul(
                self.task_noise_covar_factor.transpose(-1, -2)
            )
        raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value: Tensor) -> None:
        if self.rank > 0:
            with torch.no_grad():
                self.task_noise_covar_factor.data = to_linear_operator(value).pivoted_cholesky(
                    rank=self.rank
                )
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self) -> Tensor:
        """
        Evaluate task-space covariance matrix.

        Per-sample Dirichlet variance does not belong here; it belongs in the
        data-space shaped noise operator or a dedicated Dirichlet subclass.
        """
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(
            self.num_tasks,
            dtype=noise.dtype,
            device=noise.device,
        )
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> MultitaskMultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)

    def set_dirichlet_targets(
        self,
        targets: Tensor,
        alpha_epsilon: float = 0.01,
        dtype: Optional[torch.dtype] = None,
    ) -> int:
        """
        Compute and store Dirichlet-transformed targets and per-sample variance.

        Args:
            targets: Long tensor of class labels, shape (N,)
            alpha_epsilon: Small prior mass for Dirichlet
            dtype: Optional dtype for computations

        Returns:
            num_classes (int)
        """
        sigma2, transformed_targets, num_classes = prepare_dirichlet_targets(
            targets,
            alpha_epsilon=alpha_epsilon,
            dtype=dtype,
        )

        sigma2_per_sample = sigma2.transpose(-2, -1)[
            torch.arange(targets.shape[0], device=targets.device),
            targets,
        ]

        sigma2_per_sample = sigma2_per_sample.detach()
        transformed_targets = transformed_targets.detach()

        if self.dirichlet_diag.numel() == 0 or self.dirichlet_diag.shape != sigma2_per_sample.shape:
            self.dirichlet_diag = sigma2_per_sample
        else:
            self.dirichlet_diag.copy_(sigma2_per_sample)

        if (
            self.dirichlet_transformed_targets.numel() == 0
            or self.dirichlet_transformed_targets.shape != transformed_targets.shape
        ):
            self.dirichlet_transformed_targets = transformed_targets
        else:
            self.dirichlet_transformed_targets.copy_(transformed_targets)

        return num_classes