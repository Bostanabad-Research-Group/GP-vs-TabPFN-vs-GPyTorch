# import torch
# from gpplus.constraints import SoftClamp
# from gpytorch.constraints import Interval
# # from gpytorch.kernels import Kernel
# from .unconstrained_kernel import UnconstrainedKernel


# def postprocess_(dist_mat):
#     return torch.exp(-dist_mat)


# class PowerExponentialKernelFixed(UnconstrainedKernel):
#     has_lengthscale = True  # Enable lengthscale functionality
#     is_stationary = True  # The kernel is stationary

#     def __init__(self, power=2.0, **kwargs):
#         # Validate that power is between 1 and 2
#         if not (1 <= power <= 2):
#             raise ValueError("The 'power' parameter must be between 1 and 2.")
#         # power is the exponent parameter (κ)
#         super().__init__(**kwargs)
#         self.power = power

#     def forward(self, x1, x2, diag=False, **params):
#         scaling_factors = torch.pow(10, self.lengthscale / self.power)

#         x1_ = x1.mul(scaling_factors)
#         x2_ = x2.mul(scaling_factors)

#         return postprocess_(self.covar_dist(x1_, x2_, ord=self.power, diag=diag, **params))


# class PowerExponentialKernel(UnconstrainedKernel):
#     has_lengthscale = True  # Enable lengthscale functionality
#     is_stationary = True  # The kernel is stationary

#     def __init__(self, **kwargs):
#         # power is the exponent parameter (κ)
#         super().__init__(**kwargs)
#         # Register the power parameter (as a raw parameter to allow constraints)
#         self.register_parameter(
#             name="raw_power",
#             parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
#         )

#         # Constrain raw_power to be in the interval [1, 2]
#         # self.register_constraint("raw_power", SoftClamp(lower_bound=1.0, upper_bound=2.0, margin=1e-2))
#         self.register_constraint("raw_power", Interval(lower_bound=1.0, upper_bound=2.0))

        

#     @property
#     def power(self):
#         # Transform raw_power via the constraint's transformation.
#         return self.raw_power_constraint.transform(self.raw_power)

#     @power.setter
#     def power(self, value):
#         self._set_power(value)

#     def _set_power(self, value):
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_power)
#         # Use the inverse transform to set the raw parameter value
#         self.initialize(raw_power=self.raw_power_constraint.inverse_transform(value))

#     def forward(self, x1, x2, diag=False, **params):
#         scaling_factors = torch.pow(10, self.lengthscale / self.power)

#         x1_ = x1.mul(scaling_factors)
#         x2_ = x2.mul(scaling_factors)

#         # Get power as tensor (maintains gradients)
#         power = self.power
#         # Squeeze trailing singleton dimensions for cleaner broadcasting
#         # Shape: (*batch_shape, 1, 1) -> (*batch_shape,) or scalar
#         if power.dim() >= 2:
#             power = power.squeeze(-1).squeeze(-1)
#         elif power.dim() == 1 and power.shape[0] == 1:
#             power = power.squeeze(0)

#         # Compute ||x1 - x2||_p^p = sum(|x1 - x2|^p) over feature dimension
#         if diag:
#             # Diagonal case: x1 and x2 should be the same
#             diff = x1_ - x2_  # Shape: (... x N x D)
#             dist_p = diff.abs().pow(power).sum(dim=-1)  # Shape: (... x N)
#         else:
#             # Non-diagonal case: compute pairwise distances
#             # x1_: (... x N x D), x2_: (... x M x D)
#             # Expand for broadcasting: x1_ -> (... x N x 1 x D), x2_ -> (... x 1 x M x D)
#             x1_expanded = x1_.unsqueeze(-2)  # (... x N x 1 x D)
#             x2_expanded = x2_.unsqueeze(-3)  # (... x 1 x M x D)
#             diff = x1_expanded - x2_expanded  # (... x N x M x D)
#             dist_p = diff.abs().pow(power).sum(dim=-1)  # Shape: (... x N x M)
#             # Clamp to avoid numerical issues (matching covar_dist behavior)
#             dist_p = dist_p.clamp_min(1e-15)

#         return postprocess_(dist_p)
    
    
    
    
    
    
    
import torch
from gpplus.constraints import SoftClamp
from gpytorch.constraints import Interval


@@ -71,19 +176,23 @@ class PowerExponentialKernel(UnconstrainedKernel):
        x2_ = x2.mul(scaling_factors)

        # Get power as tensor (maintains gradients)
        power = self.power
        # power = self.power
        # Squeeze trailing singleton dimensions for cleaner broadcasting
        # Shape: (*batch_shape, 1, 1) -> (*batch_shape,) or scalar
        if power.dim() >= 2:
            power = power.squeeze(-1).squeeze(-1)
        elif power.dim() == 1 and power.shape[0] == 1:
            power = power.squeeze(0)
        # if power.dim() >= 2:
            # power = power.squeeze(-1).squeeze(-1)
        # elif power.dim() == 1 and power.shape[0] == 1:
            # power = power.squeeze(0)

        # Compute ||x1 - x2||_p^p = sum(|x1 - x2|^p) over feature dimension
        if diag:
            # Diagonal case: x1 and x2 should be the same
            power = self.power
            diff = x1_ - x2_  # Shape: (... x N x D)
            dist_p = diff.abs().pow(power).sum(dim=-1)  # Shape: (... x N)
            while power.dim() < diff.dim():
                power = power.unsqueeze(-1)
                dist_p = diff.abs().pow(power).sum(dim=-1)
        else:
            # Non-diagonal case: compute pairwise distances
            # x1_: (... x N x D), x2_: (... x M x D)

@@ -91,8 +200,12 @@ class PowerExponentialKernel(UnconstrainedKernel):
            x1_expanded = x1_.unsqueeze(-2)  # (... x N x 1 x D)
            x2_expanded = x2_.unsqueeze(-3)  # (... x 1 x M x D)
            diff = x1_expanded - x2_expanded  # (... x N x M x D)
            dist_p = diff.abs().pow(power).sum(dim=-1)  # Shape: (... x N x M)
            # Clamp to avoid numerical issues (matching covar_dist behavior)
            
            power = self.power
            while power.dim() < diff.dim():
                power = power.unsqueeze(-1)
            dist_p = diff.abs().pow(power).sum(dim=-1)  # Shape: (... x N x M)            # Clamp to avoid numerical issues (matching covar_dist behavior)
            dist_p = dist_p.clamp_min(1e-15)

        return postprocess_(dist_p)