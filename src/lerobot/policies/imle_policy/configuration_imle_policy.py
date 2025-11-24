"""
IMLE Policy Config tuned to match the typical LIBERO setup.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("imle_policy")
@dataclass
class IMLEConfig(PreTrainedConfig):
    """
    Configuration for the standalone IMLE policy.

    Defaults mirror the LIBERO dataset:
    - Two RGB cameras at 256x256.
    - 8D robot state.
    - 7D action.
    - Short temporal horizons following the original IMLE policy recipe.
    """

    # Input/output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Model related.
    vision_feature_dim: int = 512  # resnet18 output dimension
    lowdim_obs_dim: int = 8
    obs_dim: int = 1032  # 2 * vision_feature_dim + lowdim_obs_dim for LIBERO default
    action_dim: int = 7
    noise_dim: int = 32
    obs_horizon: int = 2
    dataset_path: str = "HuggingFaceVLA/libero"
    num_diffusion_iters: int = 100
    device: str | None = None  # Allow auto device selection
    n_samples_per_condition: int = 20
    epsilon: float = 0.03
    max_steps: int = 520  # Adjust according to task
    seed_start: int = 0
    num_trails: int = 50
    num_epochs: int = 1000
    timestep_integer_scaler: int = 100
    num_cameras: int = 2
    batch_size: int = 128
    num_flow_iters: int = 1

    # Optimizer preset.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def validate_features(self) -> None:
        # TODO: perform necessary checks here.
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.n_action_steps))

    @property
    def reward_delta_indices(self) -> None:
        return None
