import torch
import torch.nn.functional as F

from .input_transform_net import InputTransformNet


class TrunkHeadNet(torch.nn.Module):
    """
    A neural network module that combines a trunk (feature extractor) and a head (final layer).
    
    The architecture is: trunk → head (Flatten → Linear → activation) → [optional L2 normalize]
    
    Args:
        trunk: An nn.Module mapping inputs to some feature tensor (e.g., InputTransformNet)
        input_head_dim: The input dimension for the head's Linear layer (output dim of trunk)
        head_config: Dictionary with keys:
            - "dims": int - output dimension of the final Linear layer
            - "activation": nn.Module class - activation function class (e.g., torch.nn.ReLU, torch.nn.Identity)
        normalize: bool - whether to apply L2 normalization (default: True for backward compatibility)
    """
    
    def __init__(self, trunk: torch.nn.Module, input_head_dim: int, head_config: dict, normalize: bool = True):
        """
        Args:
            trunk: an nn.Module mapping inputs to some feature tensor
            input_head_dim: The input dimension for the head's Linear layer
            head_config: {
                "dims": int,              # output dimension of the final Linear
                "activation": nn.Module   # class of activation, e.g. nn.ReLU or nn.Tanh
            }
            normalize: bool - whether to apply L2 normalization (default: True)
        """
        super().__init__()
        self.trunk = trunk
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),               # flatten (batch, …) → (batch, features)
            torch.nn.Linear(input_head_dim, head_config["dims"]),
            head_config["activation"]()            # e.g. nn.Tanh()
        )
        self.normalize = normalize

    @classmethod
    def from_configs(cls, *, input_dim: int, trunk_layer_config: dict, head_config: dict, normalize: bool = True) -> "TrunkHeadNet":
        """
        Convenience constructor to build trunk+head in one place (no separate init at callsite).

        Args:
            input_dim: Input dimensionality to the trunk network.
            trunk_layer_config: Layer config for InputTransformNet.
            head_config: Head config dict: {"dims": int, "activation": nn.Module class}
            normalize: bool - whether to apply L2 normalization (default: True)

        Returns:
            TrunkHeadNet: a module with trunk=InputTransformNet(...) and a Linear head.
        """
        if not isinstance(trunk_layer_config, dict) or len(trunk_layer_config) == 0:
            raise ValueError("trunk_layer_config must be a non-empty dict")

        trunk = InputTransformNet(input_dim=input_dim, layer_config=trunk_layer_config)
        last_idx = max(trunk_layer_config.keys())
        input_head_dim = int(trunk_layer_config[last_idx]["dims"])
        return cls(trunk=trunk, input_head_dim=input_head_dim, head_config=head_config, normalize=normalize)

    def forward(self, x):
        """
        Forward pass: trunk → head → [optional L2 normalize]
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (L2-normalized if normalize=True)
        """
        x = self.trunk(x)
        x = self.head(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1, eps=1e-6)
        return x
