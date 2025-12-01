"""
LoRA (Low-Rank Adaptation) implementation for LeRobot.
This implementation is adapted from OpenPI's JAX/Flax LoRA to PyTorch,
designed to work with LeRobot's PI0.5 model architecture.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.
    Args:
        rank: The rank of the low-rank matrices (typically 8-64).
        alpha: The scaling factor for LoRA updates (typically = rank or 2*rank).
        dropout: Dropout rate for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        init_std: Standard deviation for LoRA weight initialization.
        rslora: Whether to use rank-stabilized LoRA (https://arxiv.org/abs/2312.03732).
    """
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = None
    init_std: float = 0.01
    rslora: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for transformer models
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    @property
    def scaling_value(self) -> float:
        """Get the scaling factor for LoRA updates."""
        if self.rslora:
            return self.alpha / math.sqrt(self.rank)
        else:
            return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.
    This replaces a standard nn.Linear layer with a frozen base layer
    plus trainable low-rank adaptation matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        base_layer: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Base layer (frozen)
        if base_layer is not None:
            self.base_layer = base_layer
        else:
            self.base_layer = nn.Linear(in_features, out_features, bias=False)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(config.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.rank))

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        self.reset_parameters()

        # Scaling factor
        self.scaling = config.scaling_value

        # Enable/disable LoRA
        self.enabled = True

    @property
    def weight(self):
        """Return the weight of the base layer for compatibility."""
        return self.base_layer.weight

    @property
    def bias(self):
        """Return the bias of the base layer for compatibility."""
        return self.base_layer.bias if hasattr(self.base_layer, 'bias') else None

    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with normal distribution
        nn.init.normal_(self.lora_A, mean=0.0, std=self.config.init_std)
        # Initialize B with zeros (common practice)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base forward pass
        result = self.base_layer(x)

        # Add LoRA adaptation if enabled
        if self.enabled and self.training:
            x_dropped = self.dropout(x)
            # Compute low-rank adaptation: x @ A^T @ B^T * scaling
            lora_out = x_dropped @ self.lora_A.t() @ self.lora_B.t()
            result = result + lora_out * self.scaling
        elif self.enabled and not self.training:
            # No dropout during inference
            lora_out = x @ self.lora_A.t() @ self.lora_B.t()
            result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into base layer for deployment."""
        if self.enabled:
            # Compute merged weight: W + BA * scaling
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_weight
            self.enabled = False

    def unmerge_weights(self):
        """Unmerge LoRA weights from base layer."""
        if not self.enabled:
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_weight
            self.enabled = True


class LoRAAdapter:
    """Utility class to inject LoRA into existing models."""

    @staticmethod
    def inject_lora(
        model: nn.Module,
        config: LoRAConfig,
        prefix: str = "",
    ) -> Dict[str, LoRALinear]:
        """Inject LoRA layers into a model.
        Args:
            model: The model to inject LoRA into.
            config: LoRA configuration.
            prefix: Prefix for module names (for nested models).
        Returns:
            Dictionary mapping module names to LoRA layers.
        """
        lora_modules = {}

        for name, module in model.named_modules():
            # Check if this module should have LoRA
            if not any(target in name for target in config.target_modules):
                continue

            # Only apply to Linear layers
            if not isinstance(module, nn.Linear):
                continue

            # Get parent module and attribute name
            *parent_names, attr_name = name.split(".")
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)

            # Create LoRA layer
            full_name = f"{prefix}.{name}" if prefix else name
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                config,
                base_layer=module
            )

            # Replace module with LoRA layer
            setattr(parent, attr_name, lora_layer)
            lora_modules[full_name] = lora_layer

        return lora_modules

    @staticmethod
    def mark_only_lora_as_trainable(
        model: nn.Module,
        bias_training: bool = False
    ):
        """Mark only LoRA parameters as trainable.
        Args:
            model: Model with LoRA layers.
            bias_training: Whether to also train bias parameters.
        """
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Then, unfreeze LoRA parameters
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            elif bias_training and "bias" in name:
                param.requires_grad = True

    @staticmethod
    def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
        """Get all LoRA parameters from a model.
        Args:
            model: Model with LoRA layers.
        Returns:
            List of LoRA parameters.
        """
        lora_params = []
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_params.append(param)
        return lora_params

    @staticmethod
    def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA parameters.
        Args:
            model: Model with LoRA layers.
        Returns:
            State dict with only LoRA parameters.
        """
        state_dict = {}
        for name, param in model.state_dict().items():
            if "lora_" in name:
                state_dict[name] = param
        return state_dict

    @staticmethod
    def load_lora_weights(
        model: nn.Module,
        lora_state_dict: Dict[str, torch.Tensor],
        strict: bool = True
    ):
        """Load LoRA weights into a model.
        Args:
            model: Model with LoRA layers.
            lora_state_dict: State dict with LoRA weights.
            strict: Whether to enforce strict loading.
        """
        # Filter current model state dict to get LoRA keys
        model_lora_keys = set(k for k in model.state_dict().keys() if "lora_" in k)
        lora_keys = set(lora_state_dict.keys())

        if strict:
            missing = model_lora_keys - lora_keys
            unexpected = lora_keys - model_lora_keys

            if missing:
                raise RuntimeError(f"Missing LoRA keys: {missing}")
            if unexpected:
                raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")

        # Load the weights
        model.load_state_dict(lora_state_dict, strict=False)

    @staticmethod
    def print_trainable_parameters(model: nn.Module):
        """Print the number of trainable parameters.
        Args:
            model: The model to analyze.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_param if all_param > 0 else 0
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {percentage:.2f}%"
        )