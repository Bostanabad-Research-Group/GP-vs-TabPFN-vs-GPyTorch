import torch
import gpytorch
from gpytorch.kernels import Kernel
from .mvmf_kernel import MVMFKernel
from .advanced_kernels import NeuralScaleKernel, ExponentialKernel, SinhKernel, CoshKernel
from .log_scale_kernel import LogScaleKernel
from ..utils.encoders import NeuralEncoder


class SEEKKernel(Kernel):
    """
    Stacked Ensemble of Kernels (SEEK)
    
    This kernel allows users to specify multiple continuous kernels and combines them
    using neural scaling weights. Each base kernel (MVMFKernel or direct kernel) is multiplied
    by a NeuralScaleKernel weight, and optionally adds a bias term.
    
    Args:
        cont_cols: Indices of continuous features
        cat_cols: Categorical feature groups (same format as MVMFKernel.cat_cols)
        source_cols: Indices of source/fidelity features
        continuous_kernels: List of continuous kernels to use for each base kernel
        cat_kernel: Kernel for categorical features (shared across all base kernels)
        source_kernel: Kernel for source features (shared across all base kernels)
        cat_encoder: Encoder function for categorical features
        source_encoder: Encoder function for source features
        z_dim: Dimension of the latent space for the encoders
        fix_lengthscale_cat: Whether to fix the lengthscale of the categorical features
        fix_lengthscale_source: Whether to fix the lengthscale of the source features
        weight_layer_config: Layer configuration for neural scaling weights
        bias_layer_config: Layer configuration for neural bias (optional)
        use_bias: Whether to include a bias term
        use_mvmf: Whether to wrap kernels in MVMFKernel (True) or use kernels directly (False).
                  When False, kernels are used directly without MVMFKernel wrapper. This is useful
                  when you only have continuous features and don't need categorical/source handling.
                  Default: True (backward compatible).
        activation: Activation kernel class to apply to pre-activation z(x, x').
                    Must be one of: ExponentialKernel, SinhKernel, or CoshKernel from gpplus.
                    Default: ExponentialKernel. According to SEEK paper, these preserve kernel validity.
        **kwargs: Additional kernel arguments
    """
    
    def __init__(
        self,
        cont_cols: list = None,
        cat_cols: list = None,
        source_cols: list = None,
        continuous_kernels: list = None,
        cat_kernel=None,
        source_kernel=None,
        cat_encoder=None,
        source_encoder=None,
        # Pass-through options supported by MVMFKernel
        z_dim: int = 2,
        fix_lengthscale_cat: bool = False,
        fix_lengthscale_source: bool = False,
        # Backwards-compat (older CombinedKernel API). MVMFKernel is multiplicative,
        # so these are currently ignored but accepted to avoid breaking old scripts.
        cat_combination_method=None,
        source_combination_method=None,
        weight_layer_config: dict = None,
        bias_layer_config: dict = None,
        use_bias: bool = False,
        use_mvmf: bool = True,
        activation = None,  # None = no activation, or ExponentialKernel/SinhKernel/CoshKernel
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.cont_cols = cont_cols or []
        self.cat_cols = cat_cols or []
        self.source_cols = source_cols or []
        self.continuous_kernels = continuous_kernels
        self.cat_kernel = cat_kernel
        self.source_kernel = source_kernel
        self.cat_encoder = cat_encoder
        self.source_encoder = source_encoder
        self.use_bias = use_bias
        self.use_mvmf = use_mvmf
        self.z_dim = z_dim
        self.fix_lengthscale_cat = fix_lengthscale_cat
        self.fix_lengthscale_source = fix_lengthscale_source
        
        # Cache continuous column indices for efficiency (when not using MVMFKernel)
        if not self.use_mvmf and self.cont_cols:
            # Will be set to proper device/dtype on first forward pass
            self._cached_cont_idx = None
        else:
            self._cached_cont_idx = None
        
        # Validate inputs
        if not isinstance(continuous_kernels, list) or len(continuous_kernels) == 0:
            raise ValueError("continuous_kernels must be a non-empty list")
        
        # Validate use_mvmf configuration
        if not use_mvmf:
            # When not using MVMFKernel, we can only use continuous features
            if cat_cols or source_cols:
                raise ValueError(
                    "When use_mvmf=False, cat_cols and source_cols must be empty. "
                    "Use use_mvmf=True if you need categorical or source features."
                )
            if not cont_cols:
                raise ValueError(
                    "When use_mvmf=False, cont_cols must be specified and non-empty."
                )
        
        # Set up activation function according to SEEK paper (Section 2.4)
        # Allow None to disable activation, or gpplus kernel classes: ExponentialKernel, SinhKernel, CoshKernel
        valid_activation_classes = {ExponentialKernel, SinhKernel, CoshKernel}
        
        if activation is None:
            # No activation - return kernel values directly
            self.activation = None
            self.activation_fn = None
            self.activation_kernel_module = None
        else:
            if not (isinstance(activation, type) and issubclass(activation, Kernel)):
                raise ValueError(
                    f"activation must be None or a kernel class (ExponentialKernel, SinhKernel, or CoshKernel), "
                    f"got {type(activation).__name__}"
                )
            
            if activation not in valid_activation_classes:
                raise ValueError(
                    f"activation must be None or one of {[cls.__name__ for cls in valid_activation_classes]}, "
                    f"got {activation.__name__}"
                )
            
            # User provided a kernel class (e.g., ExponentialKernel)
            self.activation = activation
            
            # Map kernel class to corresponding torch function for efficiency
            # We apply activation directly in forward() for efficiency and thread-safety
            if activation == ExponentialKernel:
                self.activation_fn = torch.exp
            elif activation == SinhKernel:
                self.activation_fn = torch.sinh
            elif activation == CoshKernel:
                self.activation_fn = torch.cosh
            else:
                # This shouldn't happen due to validation above, but handle it just in case
                self.activation_fn = None
            
            # No dummy kernel - activation is applied directly via torch functions
            self.activation_kernel_module = None
        
        # Set default weight layer configuration if not provided
        if weight_layer_config is None:
            weight_layer_config = {
                0: {"dims": 32, "activation": torch.nn.ReLU},
                1: {"dims": 16, "activation": torch.nn.ReLU},
                2: {"dims": 1, "activation": torch.nn.Identity},  # Output between 0 and 1
            }
        
        # Set default bias layer configuration if not provided
        if bias_layer_config is None:
            bias_layer_config = {
                0: {"dims": 32, "activation": torch.nn.ReLU},
                1: {"dims": 16, "activation": torch.nn.ReLU},
                2: {"dims": 1, "activation": torch.nn.Identity},  # Output between -1 and 1
            }
        
        # Create base kernels and wrap each with LogScaleKernel
        self.base_kernels = torch.nn.ModuleList()
        if self.use_mvmf:
            # Use MVMFKernel wrapper (original behavior)
            for i, cont_kernel in enumerate(continuous_kernels):
                base_kernel = MVMFKernel(
                    cont_cols=self.cont_cols,
                    cat_cols=self.cat_cols,
                    source_cols=self.source_cols,
                    cont_kernel=cont_kernel,
                    cat_kernel=cat_kernel,
                    source_kernel=source_kernel,
                    cat_encoder=cat_encoder,
                    source_encoder=source_encoder,
                    z_dim=self.z_dim,
                    fix_lengthscale_cat=self.fix_lengthscale_cat,
                    fix_lengthscale_source=self.fix_lengthscale_source,
                )
                # Wrap each base kernel with LogScaleKernel
                wrapped_base_kernel = LogScaleKernel(base_kernel)
                self.base_kernels.append(wrapped_base_kernel)
        else:
            # Use kernels directly without MVMFKernel wrapper
            for i, cont_kernel in enumerate(continuous_kernels):
                # Wrap each kernel directly with LogScaleKernel
                wrapped_base_kernel = LogScaleKernel(cont_kernel)
                self.base_kernels.append(wrapped_base_kernel)

        # Calculate the input dimension for the neural weight/bias networks
        if self.use_mvmf:
            # After constructing an MVMFKernel so we can read the actual encoder z_dims
            # (handles string encoders like "matrix"/"nn" and lists of encoders).
            total_input_dim = self._weight_input_dim_from_base(self.base_kernels[0])
        else:
            # When using kernels directly, input dimension is just the continuous columns
            total_input_dim = len(self.cont_cols) if self.cont_cols else 1
        
        # Create neural scaling weights for each base kernel
        self.weight_kernels = torch.nn.ModuleList()
        for i in range(len(continuous_kernels)):
            weight_kernel = NeuralScaleKernel(
                input_dim=total_input_dim,
                layer_config=weight_layer_config
            )
            self.weight_kernels.append(weight_kernel)
        
        # Create bias kernel if requested
        if self.use_bias:
            self.bias_kernel = NeuralScaleKernel(
                input_dim=total_input_dim,
                layer_config=bias_layer_config
            )
        else:
            self.bias_kernel = None

    def _weight_input_dim_from_base(self, base_kernel) -> int:
        """
        Determine the input dimensionality of the features fed into NeuralScaleKernel.

        We mirror MVMFKernel's internal encoding so the weight network sees the same
        representation (continuous + encoded categorical + encoded source).
        
        Note: base_kernel may be wrapped in LogScaleKernel, so we unwrap it if needed.
        """
        # Unwrap LogScaleKernel if present to get the underlying MVMFKernel
        if hasattr(base_kernel, 'base_kernel') and isinstance(base_kernel, LogScaleKernel):
            mvmf_kernel = base_kernel.base_kernel
        else:
            mvmf_kernel = base_kernel
        
        total = 0

        # Continuous dims are always the raw continuous columns
        total += len(self.cont_cols or [])

        # Categorical latent dims: sum z_dim over each categorical encoder
        if getattr(mvmf_kernel, "cat_kernel", None) is not None and getattr(mvmf_kernel, "cat_encoder", None) is not None:
            encoders = (
                list(mvmf_kernel.cat_encoder)
                if isinstance(mvmf_kernel.cat_encoder, (list, tuple))
                else [mvmf_kernel.cat_encoder]
            )
            if len(encoders) == 1 and isinstance(mvmf_kernel.cat_cols, list) and len(mvmf_kernel.cat_cols) > 0 and not hasattr(mvmf_kernel.cat_cols[0], "__iter__"):
                # Single categorical group specified as a flat list of indices
                cat_groups = [mvmf_kernel.cat_cols]
            else:
                cat_groups = list(mvmf_kernel.cat_cols) if mvmf_kernel.cat_cols is not None else []

            # Prefer encoder-provided z_dim; fallback to the raw one-hot/group size
            for enc, group in zip(encoders, cat_groups):
                if hasattr(enc, "z_dim"):
                    total += int(enc.z_dim)
                else:
                    total += len(group)

        # Source latent dims: encoder z_dim (MatrixEncoder / NeuralEncoder)
        if getattr(mvmf_kernel, "source_kernel", None) is not None and getattr(mvmf_kernel, "source_encoder", None) is not None:
            if hasattr(mvmf_kernel.source_encoder, "z_dim"):
                total += int(mvmf_kernel.source_encoder.z_dim)
            else:
                total += len(self.source_cols or [])

        # If there are truly no features, fall back to "raw input dim at runtime"
        # (handled in forward by using x directly). But NeuralScaleKernel needs a
        # positive input_dim, so keep it at least 1.
        return max(1, total)

    def _encode_for_weights(self, x: torch.Tensor, base_kernel, epsilon=None):
        """
        Build the encoded feature representation used for NeuralScaleKernel.

        Returns (x_encoded, epsilon) where epsilon is reused for the source encoder
        when it is probabilistic, ensuring consistent encoding between x1 and x2.
        
        Note: base_kernel may be wrapped in LogScaleKernel, so we unwrap it if needed.
        """
        # When not using MVMFKernel, just return continuous columns directly
        if not self.use_mvmf:
            if self.cont_cols:
                device = x.device
                cont_idx = torch.as_tensor(self.cont_cols, dtype=torch.long, device=device)
                return x.index_select(-1, cont_idx), epsilon
            else:
                return x, epsilon
        
        # Unwrap LogScaleKernel if present to get the underlying MVMFKernel
        if hasattr(base_kernel, 'base_kernel') and isinstance(base_kernel, LogScaleKernel):
            mvmf_kernel = base_kernel.base_kernel
        else:
            mvmf_kernel = base_kernel
        
        device = x.device
        parts = []

        # Continuous
        if self.cont_cols:
            cont_idx = torch.as_tensor(self.cont_cols, dtype=torch.long, device=device)
            parts.append(x.index_select(-1, cont_idx))

        # Categorical (mirror MVMFKernel.forward)
        if getattr(mvmf_kernel, "cat_kernel", None) is not None and getattr(mvmf_kernel, "cat_encoder", None) is not None:
            cat_cols = getattr(mvmf_kernel, "cat_cols", []) or []
            if len(cat_cols) > 0 and not hasattr(cat_cols[0], "__iter__"):
                cat_groups = [cat_cols]
            else:
                cat_groups = list(cat_cols)

            encoders = (
                list(mvmf_kernel.cat_encoder)
                if isinstance(mvmf_kernel.cat_encoder, (list, tuple))
                else [mvmf_kernel.cat_encoder]
            )

            z_list = []
            for encoder, col_group in zip(encoders, cat_groups):
                cat_idx = torch.as_tensor(col_group, dtype=torch.long, device=device)
                x_group = x.index_select(-1, cat_idx)

                # Match dtype for MatrixEncoder / any encoder with projection_matrix
                if hasattr(encoder, "projection_matrix"):
                    x_group = x_group.to(dtype=encoder.projection_matrix.dtype)

                z_list.append(encoder(x_group))

            if z_list:
                parts.append(torch.cat(z_list, dim=-1))

        # Source (mirror MVMFKernel.forward)
        if getattr(mvmf_kernel, "source_kernel", None) is not None and getattr(mvmf_kernel, "source_encoder", None) is not None:
            if self.source_cols:
                source_idx = torch.as_tensor(self.source_cols, dtype=torch.long, device=device)
                x_source = x.index_select(-1, source_idx)

                use_eps = isinstance(mvmf_kernel.source_encoder, NeuralEncoder) and getattr(
                    mvmf_kernel.source_encoder, "is_probabilistic", True
                )
                if use_eps:
                    if epsilon is None:
                        epsilon = torch.normal(
                            mean=0,
                            std=1,
                            size=[len(self.source_cols), 2],
                            device=x.device,
                            dtype=x.dtype,
                        )
                    z_source = mvmf_kernel.source_encoder(x_source, epsilon=epsilon)
                else:
                    z_source = mvmf_kernel.source_encoder(x_source)

                parts.append(z_source)

        if not parts:
            return x, epsilon
        return torch.cat(parts, dim=-1), epsilon
    
    def forward(self, x1, x2=None, diag=False, use_encoded_inputs=True, **kwargs):
        """
        Forward pass of the MF-SEEK kernel.
        
        Optimized version that:
        1. Encodes inputs once and reuses them
        2. Pre-computes neural network transformations for weight kernels
        3. Avoids redundant encoding in base kernels when possible
        
        Args:
            x1: First input tensor
            x2: Second input tensor (if None, uses x1)
            diag: Whether to return only diagonal elements
            use_encoded_inputs: Whether to use encoded inputs for weight computation
            **kwargs: Additional arguments
            
        Returns:
            Combined kernel matrix
        """
        if x2 is None:
            x2 = x1

        # Get encoded representations from the first base kernel
        # This ensures we use the exact same encodings as the base kernels
        if use_encoded_inputs and len(self.base_kernels) > 0:
            # Use the first base kernel to get encoded representations
            base_kernel = self.base_kernels[0]
            epsilon = None
            x1_encoded, epsilon = self._encode_for_weights(x1, base_kernel, epsilon=epsilon)
            x2_encoded, _ = self._encode_for_weights(x2, base_kernel, epsilon=epsilon)
        else:
            # Use original inputs for weight computation
            x1_encoded = x1
            x2_encoded = x2
        
        # Pre-compute neural network transformations for all weight kernels
        # This avoids redundant forward passes when multiple base kernels share the same encoded inputs
        weight_transforms_x1 = []
        weight_transforms_x2 = []
        for weight_kernel in self.weight_kernels:
            # Apply neural network transformation once per input
            f_x1 = weight_kernel.input_transform(x1_encoded)
            f_x2 = weight_kernel.input_transform(x2_encoded)
            weight_transforms_x1.append(f_x1)
            weight_transforms_x2.append(f_x2)
        
        # Pre-compute bias transformation if needed
        if self.use_bias and self.bias_kernel is not None:
            bias_f_x1 = self.bias_kernel.input_transform(x1_encoded)
            bias_f_x2 = self.bias_kernel.input_transform(x2_encoded)
        
        # Compute weighted sum of base kernels
        result = None
        for i, (base_kernel, weight_kernel) in enumerate(zip(self.base_kernels, self.weight_kernels)):
            # Compute base kernel
            if self.use_mvmf:
                # MVMFKernel handles column selection internally
                k_base = base_kernel(x1, x2, diag=diag, **kwargs)
            else:
                # Direct kernels need continuous columns to be selected
                if self.cont_cols:
                    device = x1.device
                    # Cache the index tensor to avoid recreating it every forward pass
                    if self._cached_cont_idx is None or self._cached_cont_idx.device != device:
                        self._cached_cont_idx = torch.as_tensor(self.cont_cols, dtype=torch.long, device=device)
                    x1_cont = x1.index_select(-1, self._cached_cont_idx)
                    x2_cont = x2.index_select(-1, self._cached_cont_idx)
                    k_base = base_kernel(x1_cont, x2_cont, diag=diag, **kwargs)
                else:
                    k_base = base_kernel(x1, x2, diag=diag, **kwargs)
            
            # Compute weight kernel using pre-computed transformations
            f_x1 = weight_transforms_x1[i]
            f_x2 = weight_transforms_x2[i]
            
            if diag:
                # Diagonal: element-wise product and sum
                k_weight = (f_x1 * f_x2).sum(dim=-1)
            else:
                # Full matrix: matrix multiplication
                k_weight = torch.mm(f_x1, f_x2.t())
            
            # Multiply base kernel by weight kernel
            weighted_kernel = k_base * k_weight
            
            # Add to result
            result = weighted_kernel if result is None else (result + weighted_kernel)
        
        # Add bias if requested
        if self.use_bias and self.bias_kernel is not None:
            if diag:
                k_bias = (bias_f_x1 * bias_f_x2).sum(dim=-1)
            else:
                k_bias = torch.mm(bias_f_x1, bias_f_x2.t())
            result = k_bias if result is None else (result + k_bias)
        
        # Apply activation function φ to pre-activation z(x, x') according to SEEK paper (Equation 5)
        # c(x, x') = φ(z(x, x'))
        # This ensures kernel validity as discussed in Section 2.3 and 2.4
        
        # Convert result to dense if needed
        if diag:
            result_tensor = result if isinstance(result, torch.Tensor) else result
        else:
            if hasattr(result, 'to_dense'):
                result_tensor = result.to_dense()
            else:
                result_tensor = result
        
        # Apply activation function φ directly to pre-activation z(x, x') for efficiency and thread-safety
        # This avoids the threading issues with PreActivationKernel wrapper approach
        # If activation is None, return the result directly without any transformation
        if self.activation_fn is not None:
            # Use direct torch function for efficiency (thread-safe)
            return self.activation_fn(result_tensor)
        elif self.activation is None:
            # No activation - return kernel values directly
            return result_tensor
        else:
            # Fallback: should not happen with validation, but handle gracefully
            # Create a temporary kernel wrapper for the activation
            class TempPreActivationKernel(Kernel):
                def __init__(self, pre_activation_matrix):
                    super().__init__()
                    self.pre_activation_matrix = pre_activation_matrix
                
                def forward(self, x1, x2=None, diag=False, **params):
                    if diag:
                        return torch.diag(self.pre_activation_matrix) if len(self.pre_activation_matrix.shape) == 2 else self.pre_activation_matrix
                    return self.pre_activation_matrix
            
            temp_kernel = TempPreActivationKernel(result_tensor)
            activation_kernel = self.activation(temp_kernel)
            return activation_kernel(x1, x2, diag=diag, **kwargs)
    
    def num_outputs_per_input(self, x1, x2):
        """Return the number of outputs for each input."""
        return 1 