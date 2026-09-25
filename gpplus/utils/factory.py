"""
Kernel construction helpers (optional convenience utilities).

This module provides a single convenience function ``build_scaled_kernel``
for composing the package's stationary ARD base kernels into the
``LogScaleKernel``-wrapped form used across the GP models.

**This function is entirely optional.** You can construct kernels directly
without it — see the example below. The factory only covers the Gaussian
and PowerExponential kernels. For Matérn and any other kernel, construct
directly:

Example (direct construction, no factory needed)::

    import torch
    from gpytorch.kernels import MaternKernel
    from gpplus.kernels.log_scale_kernel import LogScaleKernel
    from gpplus.kernels.gaussian_kernel import GaussianKernel

    num_classes = 3
    input_dim   = 10
    batch_shape = torch.Size([num_classes])

    # Gaussian (equivalent to build_scaled_kernel("gaussian", ...))
    kernel = LogScaleKernel(
        GaussianKernel(batch_shape=batch_shape, ard_num_dims=input_dim),
        batch_shape=batch_shape,
    )

    # Matérn ν=0.5 (not in factory — construct directly)
    kernel = LogScaleKernel(
        MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=input_dim),
        batch_shape=batch_shape,
    )

The key things to get right when constructing kernels manually:
    - Pass ``batch_shape=torch.Size([num_classes])`` to both the base kernel
      and the ``LogScaleKernel`` wrapper for multi-class GPs.
    - Pass ``ard_num_dims=input_dim`` to enable one lengthscale per dimension.
    - Always wrap in ``LogScaleKernel`` to use the log-scale reparameterization.
"""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    # Type-checking-only import: never executed at runtime, so it does not
    # reintroduce the circular import that forces the deferred imports below.
    from gpplus.kernels.log_scale_kernel import LogScaleKernel

VALID_KINDS = ("gaussian", "powerexp")


def build_scaled_kernel(
    kind: str = "gaussian",
    batch_shape: Optional[torch.Size] = None,
    ard_num_dims: Optional[int] = None,
    fixed_power: Optional[float] = None,
    **kwargs,
) -> "LogScaleKernel":
    """Builds a ``LogScaleKernel``-wrapped base kernel of the given ``kind``.

    This is an optional convenience function covering the Gaussian and
    PowerExponential kernels only. For Matérn or any other kernel, construct
    directly — see the module docstring for examples.

    Kernel imports are deferred to call time to avoid circular imports at
    package initialization.

    Args:
        kind (str): Base kernel type. One of ``"gaussian"`` (squared-exponential,
            power fixed at 2) or ``"powerexp"`` (power-exponential).
        batch_shape (torch.Size, optional): Batch shape for independent per-batch
            hyperparameters. Defaults to ``torch.Size()`` (a single, unbatched
            kernel). For multi-class GPs pass ``torch.Size([num_classes])``.
        ard_num_dims (int, optional): Number of input dimensions for ARD (one
            lengthscale per dimension). ``None`` uses a single shared lengthscale.
        fixed_power (float, optional): Only used when ``kind="powerexp"``. When
            provided, builds a ``PowerExponentialKernelFixed`` holding this power
            constant (must lie in ``[1, 2]``); otherwise ``PowerExponentialKernel``
            learns the power. Ignored for other kinds.
        **kwargs: Forwarded to the base kernel constructor.

    Returns:
        LogScaleKernel: The base kernel wrapped with a log output scale,
        sharing ``batch_shape``.

    Raises:
        ValueError: If ``kind`` is not one of the recognized types
            (``"gaussian"``, ``"powerexp"``). For other kernels, construct
            directly without this factory.
    """
    # Deferred imports to avoid circular import at package initialization:
    # gpplus.kernels.__init__ -> advanced_kernels -> gpplus.utils.__init__
    # -> factory -> gpplus.kernels (partially initialized)
    from gpplus.kernels.gaussian_kernel import GaussianKernel
    from gpplus.kernels.log_scale_kernel import LogScaleKernel
    from gpplus.kernels.power_exponential_kernel import (
        PowerExponentialKernel,
        PowerExponentialKernelFixed,
    )

    resolved_batch_shape = batch_shape if batch_shape is not None else torch.Size()

    if kind == "gaussian":
        base_kernel = GaussianKernel(
            batch_shape=resolved_batch_shape,
            ard_num_dims=ard_num_dims,
            **kwargs,
        )
    elif kind == "powerexp":
        if fixed_power is not None:
            base_kernel = PowerExponentialKernelFixed(
                power=fixed_power,
                batch_shape=resolved_batch_shape,
                ard_num_dims=ard_num_dims,
                **kwargs,
            )
        else:
            base_kernel = PowerExponentialKernel(
                batch_shape=resolved_batch_shape,
                ard_num_dims=ard_num_dims,
                **kwargs,
            )
    else:
        raise ValueError(
            f"Unknown kernel kind: {kind!r}. Expected one of {VALID_KINDS}. "
            f"For other kernels (e.g. Matérn), construct directly — "
            f"see gpplus.utils.factory module docstring for examples."
        )

    return LogScaleKernel(base_kernel, batch_shape=resolved_batch_shape)
