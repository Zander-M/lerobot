"""
Configuration for PI0.5 with LoRA adaptation.
This extends the base PI0.5 configuration with LoRA-specific settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_lora")
@dataclass
class PI05LoRAConfig(PI05Config):
    """Configuration for PI0.5 with LoRA adaptation."""

    # LoRA configuration
    use_lora: bool = True  # Enable/disable LoRA
    freeze_vlm: bool = False  # Freeze VLM (PaliGemma) entirely when not using LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_init_std: float = 0.01
    lora_rslora: bool = False

    # Training configuration adjustments for LoRA
    lora_learning_rate_scale: float = 1.0  # Scale LR for LoRA params (set to 1.0 for frozen VLM + IMLE)
    action_expert_learning_rate_scale: float = 1.0  # Scale LR for action expert

    # Whether to apply LoRA to vision tower (usually False)
    lora_vision_tower: bool = False

    # IMLE-specific configuration (replacing flow matching)
    use_imle: bool = True
    imle_num_samples: int = 64
    imle_noise_scale: float = 1.0  # Standard normal distribution (match original IMLE Policy)
    imle_epsilon: float = 0.03  # Minimum distance threshold for valid samples (prevents mode collapse)
    imle_temperature: float = 0.1  # Reserved for future soft-IMLE variants (not used in hard IMLE)

    # IMLE network architecture settings
    imle_network_type: str = "unet"  # "unet" or "simple"
    imle_noise_dim: int = 32  # Dimension of noise input for generator
    imle_hidden_dim: int = 256  # Hidden dimension for simple generator
    imle_down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])  # For UNet
    imle_kernel_size: int = 5  # Conv kernel size for UNet
    imle_n_groups: int = 8  # Number of groups for GroupNorm

    # Optimizer settings optimized for LoRA + IMLE
    optimizer_lr: float = 1e-4  # Match original IMLE Policy implementation (peak LR for scheduler)
    optimizer_weight_decay: float = 1e-6  # Match original IMLE Policy implementation
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0  # Match original IMLE Policy implementation

    # Scheduler settings
    scheduler_warmup_steps: int = 500  # Match original IMLE Policy implementation (was 1000)
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-5  # End LR after decay (10x lower than peak LR)

    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()

        # Validate LoRA configuration
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")

        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")

        if not self.lora_target_modules:
            raise ValueError("lora_target_modules cannot be empty")

        # Validate IMLE configuration
        if self.use_imle:
            if self.imle_num_samples <= 0:
                raise ValueError(f"imle_num_samples must be positive, got {self.imle_num_samples}")

            if self.imle_noise_scale <= 0:
                raise ValueError(f"imle_noise_scale must be positive, got {self.imle_noise_scale}")

            if self.imle_epsilon <= 0:
                raise ValueError(f"imle_epsilon must be positive, got {self.imle_epsilon}")

            if self.imle_temperature <= 0:
                raise ValueError(f"imle_temperature must be positive, got {self.imle_temperature}")

            if self.imle_network_type not in ["unet", "simple"]:
                raise ValueError(f"imle_network_type must be 'unet' or 'simple', got {self.imle_network_type}")

            if self.imle_noise_dim <= 0:
                raise ValueError(f"imle_noise_dim must be positive, got {self.imle_noise_dim}")

    def to_lora_config(self):
        """Convert to LoRAConfig object."""
        from lerobot.common.policies.lora import LoRAConfig

        return LoRAConfig(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            init_std=self.lora_init_std,
            rslora=self.lora_rslora
        )


# Preset configurations for different scenarios
PI05_LORA_LIBERO_CONFIG = PI05LoRAConfig(
    # Model architecture (same as base PI0.5)
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",

    # LoRA settings optimized for LIBERO
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # IMLE settings
    use_imle=True,
    imle_num_samples=64,
    imle_noise_scale=0.3,

    # Training settings
    optimizer_lr=2.5e-5,
    optimizer_grad_clip_norm=1.0,

    # Scheduler
    scheduler_warmup_steps=1000,
    scheduler_decay_steps=30000,
)


PI05_LORA_LARGE_CONFIG = PI05LoRAConfig(
    # Larger LoRA rank for more capacity
    lora_rank=32,
    lora_alpha=64.0,
    lora_dropout=0.05,

    # More IMLE samples for complex distributions
    imle_num_samples=128,
    imle_noise_scale=0.3,

    # Adjusted training settings
    optimizer_lr=1e-4,
    optimizer_grad_clip_norm=1.0,
)


PI05_LORA_EFFICIENT_CONFIG = PI05LoRAConfig(
    # Smaller LoRA rank for efficiency
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.0,  # No dropout for small rank

    # Fewer IMLE samples for speed
    imle_num_samples=32,
    imle_noise_scale=0.3,

    # Faster training settings
    optimizer_lr=5e-5,
    optimizer_grad_clip_norm=1.0,
)