import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import Kernel
from .mvmf_kernel import MVMFKernel
from .advanced_kernels import CompositeScaleKernel, ExponentialKernel
from .log_scale_kernel import LogScaleKernel
from ..utils.encoders import NeuralEncoder
from ..utils import InputTransformNet
from ..utils.trunk_head_net import TrunkHeadNet


# OPTIMIZATION: Define PreActivationKernel at module level to avoid recreating class each time
class _PreActivationKernel(Kernel):
    """Helper kernel class for exponential wrapper that holds a pre-computed matrix."""
    def __init__(self, pre_activation_matrix):
        super().__init__()
        self.pre_activation_matrix = pre_activation_matrix
    
    def forward(self, x1, x2=None, diag=False, **params):
        if diag:
            return torch.diag(self.pre_activation_matrix) if len(self.pre_activation_matrix.shape) == 2 else self.pre_activation_matrix
        return self.pre_activation_matrix


class SEEKKernelTrunkHead(Kernel):
    """
    Stacked Ensemble of Kernels (SEEK) with Trunk-Head Architecture
    
    This kernel uses a shared trunk network with separate head networks for each weight component
    and bias. This is more efficient than SEEKKernel because the trunk is computed once and shared.
    
    Architecture:
    - Shared trunk network (feature extractor)
    - Separate head networks for each base kernel weight
    - Optional separate head network for bias
    
    Computational Complexity:
    - Time complexity: O(K × N²) where K = number of base kernels, N = number of data points
      - Each base kernel computation: O(N²) for full covariance matrix
      - Weight kernel computation: O(N²) for matrix multiplication (trunk computed once, shared)
      - Element-wise operations: O(N²) per kernel
    - Space complexity: O(N²) for full covariance matrices
    - Performance notes:
      * Trunk network is computed once and cached, then reused for all weight kernel heads
      * Adding more kernels scales linearly in computation time (each kernel adds O(N²) work)
      * Adding more layers increases neural network forward/backward pass time
      * For small numbers of kernels (≤2), the overhead is manageable
      * For large N (e.g., N > 1000), consider using diagonal-only computations when possible
      * **Bias kernel adds significant overhead**: When use_bias=True, a separate bias trunk network
        computes another full N×N matrix. This adds ~33-50% more computation (depending on number
        of base kernels). Consider matching bias_trunk_layer_config size to trunk_layer_config,
        or disable bias if not needed for model performance.
    
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
        trunk_layer_config: Layer configuration dict for weight trunk network (InputTransformNet).
                          Format: {0: {"dims": int, "activation": nn.Module}, ...}
        bias_trunk_layer_config: Layer configuration dict for bias trunk network (InputTransformNet).
                                Format: {0: {"dims": int, "activation": nn.Module}, ...}
                                If None, uses same config as trunk_layer_config.
        weight_head_configs: List of head configuration dicts, one per base kernel.
                           Format: [{"dims": int, "activation": nn.Module}, ...]
                           If None, creates one head per base kernel with default config.
                           If single dict, uses same config for all weight heads.
        bias_head_config: Head configuration dict for bias network.
                         Format: {"dims": int, "activation": nn.Module}
                         Default: {"dims": 1, "activation": nn.Identity}
        use_bias: Whether to include a bias term
        use_mvmf: Whether to wrap kernels in MVMFKernel (True) or use kernels directly (False).
                  When False, kernels are used directly without MVMFKernel wrapper. This is useful
                  when you only have continuous features and don't need categorical/source handling.
                  Default: False.
        use_log_scale_kernel: Whether to wrap base kernels in LogScaleKernel.
                              Default: True.
        use_exponential_wrapper: Whether to wrap the final result in ExponentialKernel.
                                 This is independent of use_log_scale_kernel.
        normalize: Whether to apply L2 normalization in TrunkHeadNet (default: True).
                   Note: Keeping normalization ON is recommended for numerical stability and faster
                   optimizer convergence. Disabling normalization may cause slower training due to
                   numerical issues and harder optimization landscapes.
        share_bias_trunk: Whether to reuse the weight trunk network for bias (default: False).
                          When True, the bias head uses the same trunk as weight heads, saving
                          computation and memory. When False, a separate bias trunk is created.
                          Only applies when use_bias=True.
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
        act: nn.Module = nn.Tanh(),
        # Pass-through options supported by MVMFKernel
        z_dim: int = 2,
        fix_lengthscale_cat: bool = False,
        fix_lengthscale_source: bool = False,
        # Backwards-compat (older CombinedKernel API). MVMFKernel is multiplicative,
        # so these are currently ignored but accepted to avoid breaking old scripts.
        cat_combination_method=None,
        source_combination_method=None,
        # Trunk-head architecture configs
        trunk_layer_config: dict = None,
        bias_trunk_layer_config: dict = None,
        weight_head_configs: list | dict | None = None,
        bias_head_config: dict = None,
        use_bias: bool = False,
        use_mvmf: bool = False,
        use_log_scale_kernel: bool = True,
        use_exponential_wrapper: bool = False,
        normalize: bool = True,  # Whether to apply L2 normalization in TrunkHeadNet
        share_bias_trunk: bool = False,  # Whether to reuse weight trunk for bias
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.cont_cols = cont_cols or []
        self.cat_cols = cat_cols or []
        self.source_cols = source_cols or []
        self.continuous_kernels = continuous_kernels
        self.act = act
        self.cat_kernel = cat_kernel
        self.source_kernel = source_kernel
        self.cat_encoder = cat_encoder
        self.source_encoder = source_encoder
        self.use_bias = use_bias
        self.use_mvmf = use_mvmf
        self.use_log_scale_kernel = use_log_scale_kernel
        self.use_exponential_wrapper = use_exponential_wrapper
        self.normalize = normalize
        self.share_bias_trunk = share_bias_trunk
        self.z_dim = z_dim
        self.fix_lengthscale_cat = fix_lengthscale_cat
        self.fix_lengthscale_source = fix_lengthscale_source
        
        # Cache continuous column indices for efficiency (when not using MVMFKernel)
        if not self.use_mvmf and self.cont_cols:
            # Will be set to proper device/dtype on first forward pass
            self._cached_cont_idx = None
        else:
            self._cached_cont_idx = None
        
        # Cache for device/dtype checks to avoid repeated parameter access
        self._cached_device = None
        self._cached_dtype = None
        
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
        
        # Set default weight trunk layer configuration if not provided
        if trunk_layer_config is None:
            trunk_layer_config = {
                0: {"dims": 32, "activation": act},
                1: {"dims": 16, "activation": act},
            }
        
        # Set default bias trunk layer configuration if not provided (only needed if use_bias=True and not sharing)
        if self.use_bias and not self.share_bias_trunk and bias_trunk_layer_config is None:
            bias_trunk_layer_config = {
                0: {"dims": 32, "activation": act},
                1: {"dims": 16, "activation": act},
            }
        
        # Create base kernels with optional LogScaleKernel wrapping
        # LogScaleKernel wrapping is controlled independently by use_log_scale_kernel
        self.base_kernels = nn.ModuleList()
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
                # Wrap with LogScaleKernel if requested (independent of use_exponential_wrapper)
                if self.use_log_scale_kernel:
                    wrapped_base_kernel = LogScaleKernel(base_kernel)
                else:
                    wrapped_base_kernel = base_kernel
                self.base_kernels.append(wrapped_base_kernel)
        else:
            # Use kernels directly without MVMFKernel wrapper
            for i, cont_kernel in enumerate(continuous_kernels):
                # Wrap with LogScaleKernel if requested (independent of use_exponential_wrapper)
                if self.use_log_scale_kernel:
                    wrapped_base_kernel = LogScaleKernel(cont_kernel)
                else:
                    wrapped_base_kernel = cont_kernel
                self.base_kernels.append(wrapped_base_kernel)

        # Calculate the input dimension for the neural networks
        if self.use_mvmf:
            # After constructing an MVMFKernel so we can read the actual encoder z_dims
            total_input_dim = self._weight_input_dim_from_base(self.base_kernels[0])
        else:
            # When using kernels directly, input dimension is just the continuous columns
            total_input_dim = len(self.cont_cols) if self.cont_cols else 1

        # Create weight trunk network (shared by all weight heads)
        self.weight_trunk = InputTransformNet(
            input_dim=total_input_dim,
            layer_config=trunk_layer_config,
        )
        
        # Determine weight trunk output dimension (last layer's output dim)
        last_weight_trunk_idx = max(trunk_layer_config.keys())
        weight_trunk_output_dim = trunk_layer_config[last_weight_trunk_idx]["dims"]
        
        # Create bias trunk network (separate from weight trunk, only if use_bias=True and not sharing)
        if self.use_bias:
            if self.share_bias_trunk:
                # Reuse weight trunk for bias (more efficient)
                self.bias_trunk = self.weight_trunk
                bias_trunk_output_dim = weight_trunk_output_dim
            else:
                # Create separate bias trunk network
                self.bias_trunk = InputTransformNet(
                    input_dim=total_input_dim,
                    layer_config=bias_trunk_layer_config,
                )
                
                # Determine bias trunk output dimension (last layer's output dim)
                last_bias_trunk_idx = max(bias_trunk_layer_config.keys())
                bias_trunk_output_dim = bias_trunk_layer_config[last_bias_trunk_idx]["dims"]
        else:
            self.bias_trunk = None
            bias_trunk_output_dim = None
        
        # Process weight_head_configs
        num_base_kernels = len(continuous_kernels)
        if weight_head_configs is None:
            # Default: create one head per base kernel with default config
            weight_head_configs = [
                {"dims": 1, "activation": nn.Identity}
                for _ in range(num_base_kernels)
            ]
        elif isinstance(weight_head_configs, dict):
            # Single dict: use same config for all weight heads
            weight_head_configs = [weight_head_configs.copy() for _ in range(num_base_kernels)]
        elif isinstance(weight_head_configs, list):
            # List of dicts: one per base kernel
            if len(weight_head_configs) != num_base_kernels:
                raise ValueError(
                    f"weight_head_configs must have length {num_base_kernels} (one per base kernel), "
                    f"got {len(weight_head_configs)}"
                )
        else:
            raise TypeError(
                f"weight_head_configs must be None, dict, or list, got {type(weight_head_configs)}"
            )
        
        # Validate weight head configs
        for i, head_config in enumerate(weight_head_configs):
            if not isinstance(head_config, dict):
                raise TypeError(f"weight_head_configs[{i}] must be a dict, got {type(head_config)}")
            if "dims" not in head_config:
                raise ValueError(f"weight_head_configs[{i}] must have 'dims' key")
            if "activation" not in head_config:
                raise ValueError(f"weight_head_configs[{i}] must have 'activation' key")
        
        # Create weight head networks (one per base kernel)
        self.weight_kernels = nn.ModuleList()
        for i, head_config in enumerate(weight_head_configs):
            # Create TrunkHeadNet with weight trunk and individual head
            weight_network = TrunkHeadNet(
                trunk=self.weight_trunk,
                input_head_dim=weight_trunk_output_dim,
                head_config=head_config,
                normalize=self.normalize,
            )
            # Wrap in CompositeScaleKernel
            weight_kernel = CompositeScaleKernel(input_transform=weight_network)
            self.weight_kernels.append(weight_kernel)
        
        # Process bias_head_config
        if self.use_bias:
            if bias_head_config is None:
                bias_head_config = {"dims": 1, "activation": nn.Identity}
            elif not isinstance(bias_head_config, dict):
                raise TypeError(f"bias_head_config must be a dict, got {type(bias_head_config)}")
            elif "dims" not in bias_head_config:
                raise ValueError("bias_head_config must have 'dims' key")
            elif "activation" not in bias_head_config:
                raise ValueError("bias_head_config must have 'activation' key")
            
            # Create bias head network with bias trunk (may be shared with weight trunk if share_bias_trunk=True)
            bias_network = TrunkHeadNet(
                trunk=self.bias_trunk,
                input_head_dim=bias_trunk_output_dim,
                head_config=bias_head_config,
                normalize=self.normalize,
            )
            # Wrap in CompositeScaleKernel
            self.bias_kernel = CompositeScaleKernel(input_transform=bias_network)
        else:
            self.bias_kernel = None

    def _weight_input_dim_from_base(self, base_kernel) -> int:
        """
        Determine the input dimensionality of the features fed into the neural networks.

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
        return max(1, total)

    def _encode_for_weights(self, x: torch.Tensor, base_kernel, epsilon=None):
        """
        Build the encoded feature representation used for the neural networks.

        Returns (x_encoded, epsilon) where epsilon is reused for the source encoder
        when it is probabilistic, ensuring consistent encoding between x1 and x2.
        
        Note: base_kernel may be wrapped in LogScaleKernel, so we unwrap it if needed.
        """
        # When not using MVMFKernel, just return continuous columns directly
        if not self.use_mvmf:
            if self.cont_cols:
                # OPTIMIZATION: Use cached index tensor if available, otherwise create it
                device = x.device
                if self._cached_cont_idx is None or self._cached_cont_idx.device != device:
                    self._cached_cont_idx = torch.as_tensor(self.cont_cols, dtype=torch.long, device=device)
                return x.index_select(-1, self._cached_cont_idx), epsilon
        
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
        Forward pass of the SEEK kernel with trunk-head architecture (optimized version).
        
        Args:
            x1: First input tensor
            x2: Second input tensor (if None, uses x1)
            diag: Whether to return only diagonal elements
            use_encoded_inputs: Whether to use encoded inputs for weight computation
            **kwargs: Additional arguments
            
        Returns:
            Combined kernel matrix wrapped in ExponentialKernel
        """
        if x2 is None:
            x2 = x1

        # OPTIMIZATION: When diag=True, x1 and x2 are the same (or should be)
        # So we only need to encode once and reuse it
        if use_encoded_inputs and len(self.base_kernels) > 0:
            # Use the first base kernel to get encoded representations
            base_kernel = self.base_kernels[0]
            epsilon = None
            if diag:
                # OPTIMIZATION: For diagonal, x1 == x2, so encode once
                x1_encoded, epsilon = self._encode_for_weights(x1, base_kernel, epsilon=epsilon)
                x2_encoded = x1_encoded  # Reuse the same encoding
            else:
                x1_encoded, epsilon = self._encode_for_weights(x1, base_kernel, epsilon=epsilon)
                x2_encoded, _ = self._encode_for_weights(x2, base_kernel, epsilon=epsilon)
        else:
            # Use original inputs for weight computation
            x1_encoded = x1
            x2_encoded = x2 if not diag else x1  # Reuse x1 when diag=True

        # OPTIMIZATION: Check device/dtype once per forward pass and cache
        # This avoids repeated parameter access in the loop
        target_device = x1_encoded.device
        target_dtype = x1_encoded.dtype
        
        # Only update cache if device/dtype changed
        if self._cached_device != target_device or self._cached_dtype != target_dtype:
            self._cached_device = target_device
            self._cached_dtype = target_dtype
            # Move all neural networks to the correct device/dtype once
            self.weight_trunk.to(device=target_device, dtype=target_dtype)
            # Only move bias_trunk if it's separate (not shared with weight_trunk)
            if self.use_bias and self.bias_trunk is not None and not self.share_bias_trunk:
                self.bias_trunk.to(device=target_device, dtype=target_dtype)
            for weight_kernel in self.weight_kernels:
                weight_kernel.input_transform.to(device=target_device, dtype=target_dtype)
            if self.use_bias and self.bias_kernel is not None:
                self.bias_kernel.input_transform.to(device=target_device, dtype=target_dtype)
        
        # OPTIMIZATION: Pre-compute continuous column indices if needed (avoid repeated tensor creation)
        if not self.use_mvmf and self.cont_cols:
            device = x1.device
            if self._cached_cont_idx is None or self._cached_cont_idx.device != device:
                self._cached_cont_idx = torch.as_tensor(self.cont_cols, dtype=torch.long, device=device)
        
        # OPTIMIZATION: For diagonal computation, we can optimize further
        # When diag=True, all operations are element-wise, so we can avoid matrix operations
        if diag:
            # Compute weighted sum: sum of (base_kernel * CompositeScaleKernel weight) for each base kernel
            result = None
            for base_kernel, weight_kernel in zip(self.base_kernels, self.weight_kernels):
                # Compute base kernel (diagonal)
                if self.use_mvmf:
                    k_base = base_kernel(x1, x2, diag=diag, **kwargs)
                else:
                    if self.cont_cols:
                        # OPTIMIZATION: Use cached index tensor
                        x1_cont = x1.index_select(-1, self._cached_cont_idx)
                        # When diag=True, x1 == x2, so reuse x1_cont
                        x2_cont = x1_cont
                        k_base = base_kernel(x1_cont, x2_cont, diag=diag, **kwargs)
                    else:
                        k_base = base_kernel(x1, x2, diag=diag, **kwargs)
                
                # Compute weight kernel (diagonal) - already optimized in CompositeScaleKernel
                k_weight = weight_kernel(x1_encoded, x2_encoded, diag=diag, **kwargs)
                
                # Element-wise multiplication for diagonal (both are vectors)
                if result is None:
                    result = k_base * k_weight
                else:
                    result = result + (k_base * k_weight)
            
            # Add bias if requested (diagonal)
            if self.use_bias and self.bias_kernel is not None:
                k_bias = self.bias_kernel(x1_encoded, x2_encoded, diag=diag, **kwargs)
                if result is None:
                    result = k_bias
                else:
                    result = result + k_bias
        else:
            # Full matrix computation (non-diagonal)
            result = None
            for base_kernel, weight_kernel in zip(self.base_kernels, self.weight_kernels):
                # Compute base kernel
                if self.use_mvmf:
                    k_base = base_kernel(x1, x2, diag=diag, **kwargs)
                else:
                    if self.cont_cols:
                        # OPTIMIZATION: Use cached index tensor
                        x1_cont = x1.index_select(-1, self._cached_cont_idx)
                        x2_cont = x2.index_select(-1, self._cached_cont_idx)
                        k_base = base_kernel(x1_cont, x2_cont, diag=diag, **kwargs)
                    else:
                        k_base = base_kernel(x1, x2, diag=diag, **kwargs)
                
                # Compute weight kernel (CompositeScaleKernel) - use raw dot products directly
                k_weight = weight_kernel(x1_encoded, x2_encoded, diag=diag, **kwargs)
                
                # OPTIMIZATION: Convert both to dense before element-wise operations
                # Element-wise multiplication with LazyTensor is extremely slow (500x+ slower)
                # Converting to dense first is much faster
                if hasattr(k_base, 'to_dense'):
                    k_base = k_base.to_dense()
                # k_weight is already a Tensor from CompositeScaleKernel (torch.mm result), no conversion needed
                
                # Multiply base kernel by weight kernel and accumulate
                if result is None:
                    result = k_base * k_weight
                else:
                    result = result + (k_base * k_weight)
            
            # Add bias if requested
            if self.use_bias and self.bias_kernel is not None:
                k_bias = self.bias_kernel(x1_encoded, x2_encoded, diag=diag, **kwargs)
                # OPTIMIZATION: Convert to dense before addition
                # Element-wise operations with LazyTensor are extremely slow
                if hasattr(k_bias, 'to_dense'):
                    k_bias = k_bias.to_dense()
                if result is None:
                    result = k_bias
                else:
                    result = result + k_bias
        
        # Apply exponential wrapper if requested
        if self.use_exponential_wrapper:
            # OPTIMIZATION: Only convert to dense if necessary
            if diag:
                result_tensor = result if isinstance(result, torch.Tensor) else result
            else:
                if hasattr(result, 'to_dense'):
                    result_tensor = result.to_dense()
                else:
                    result_tensor = result
            
            # OPTIMIZATION: Use module-level PreActivationKernel class (defined at top of file)
            temp_kernel = _PreActivationKernel(result_tensor)
            exponential_kernel = ExponentialKernel(base_kernel=temp_kernel)
            return exponential_kernel(x1, x2, diag=diag, **kwargs)
        else:
            # Return the combined kernel directly
            return result
    
    def num_outputs_per_input(self, x1, x2):
        """Return the number of outputs for each input."""
        return 1
