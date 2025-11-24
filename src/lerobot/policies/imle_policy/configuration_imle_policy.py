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
    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision input
    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )

    # Action output
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )

    # Model related.
    vision_feature_dim: int = 512  # resnet18 output dimension
    lowdim_obs_dim: int = 8
    obs_dim: int = 1032  # 2 * vision_feature_dim + lowdim_obs_dim for LIBERO default
    action_dim: int = 7
    noise_dim: int = 32
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
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

        # Keep feature dicts in sync if a user overrides shapes via kwargs.
        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.lowdim_obs_dim,)
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
