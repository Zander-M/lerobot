#!/usr/bin/env python

# Minimal IMLE policy wrapper for LeRobot.
# This wires the existing IMLE UNet and loss into the PreTrainedPolicy
# interface so it can be trained and evaluated with the common pipeline.

from collections import deque

import torch
from torch import nn, Tensor

from lerobot.policies.imle_policy.configuration_imle_policy import IMLEConfig
from lerobot.policies.imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from lerobot.policies.imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from lerobot.policies.imle_policy.utils.losses import rs_imle_loss
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

class IMLEPolicy(PreTrainedPolicy):
    """
        Lightweight IMLE policy: encodes camera images with ResNet18, concatenates
        low-dim observations, and learns an action generator with IMLE loss.
    """

    config_class = IMLEConfig
    name = "imle_policy"

    def __init__(self, config: IMLEConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.device = torch.device(config.device) if config.device is not None else torch.device("cpu")
        self._queues = None

        # Build vision encoders for all visual inputs defined in the config.
        self._vision_key_map = {}
        self.vision_encoders = nn.ModuleDict()
        for key, feature in config.input_features.items():
            if feature.type == FeatureType.VISUAL:
                clean = key.replace(".", "_") # Remove "." from key
                self._vision_key_map[clean] = key
                encoder = get_resnet("resnet18")
                encoder = replace_bn_with_gn(encoder)
                self.vision_encoders[clean] = encoder

        # IMLE generator network.
        self.policy_net = GeneratorConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.obs_dim * config.obs_horizon,
        )
        self.to(self.device)
        self.reset()

    def get_optim_params(self):
        # Single parameter group; customize if you need differential lrs.
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        return

    def _encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """
            Encode observations into a flattened conditioning vector of shape [B, obs_dim * obs_horizon].

            Expects image tensors under keys from config.input_features (B, T, C, H, W)
            and a low-dim state under "observation.state" (B, T, D). Only the first
            `obs_horizon` steps are used.
        """
        obs_h = self.config.obs_horizon
        pieces = []

        # Encode images.
        for clean_key, encoder in self.vision_encoders.items():
            batch_key = self._vision_key_map[clean_key]
            imgs = batch[batch_key].to(self.device)
            imgs = imgs[:, :obs_h]  # [B, T, C, H, W]
            b, t, c, h, w = imgs.shape
            feats = encoder(imgs.reshape(b * t, c, h, w))  # [B*T, feat_dim]
            feats = feats.reshape(b, t, -1)
            pieces.append(feats)

        # Low-dim observations.
        if "observation.state" not in batch:
            raise KeyError("Expected 'observation.state' in batch for IMLEPolicy.")
        state = batch["observation.state"].to(self.device)
        state = state[:, :obs_h]  # [B, T, D]
        pieces.append(state)

        # Concatenate features along channel dim, then flatten time.
        cond = torch.cat(pieces, dim=-1)  # [B, T, obs_dim]
        cond = cond.flatten(start_dim=1)  # [B, obs_dim * obs_horizon]
        expected = self.config.obs_dim * self.config.obs_horizon
        if cond.shape[1] != expected:
            raise ValueError(
                f"Conditioning dimension mismatch: got {cond.shape[1]}, expected {expected}. "
                "Check obs_dim/obs_horizon or the encoder feature sizes."
            )
        return cond

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """
            Compute IMLE loss. Expects:
            - actions under "action": [B, T_action, action_dim]
            - images and state as in `_encode_observations`.
        """
        actions = batch["action"][:, : self.config.n_action_steps].to(self.device)
        global_cond = self._encode_observations(batch).to(self.device)

        b = actions.shape[0]
        n_samples = self.config.n_samples_per_condition
        noise = torch.randn(
            b * n_samples, self.config.n_action_steps, self.config.action_dim, device=self.device
        )

        repeated_cond = global_cond.repeat_interleave(n_samples, dim=0)
        preds = self.policy_net(repeated_cond, noise)
        preds = preds.view(b, n_samples, self.config.n_action_steps, self.config.action_dim)

        loss, logs = rs_imle_loss(actions, preds, epsilon=self.config.epsilon)
        return loss, logs

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
            Sample one action chunk by drawing a single noise sample.
        """
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        global_cond = self._encode_observations(batch).to(self.device)
        actions_shape = (
            global_cond.shape[0],
            self.config.n_action_steps,
            self.config.action_dim,
        )
        noise = torch.randn(actions_shape, device=self.device)
        preds = self.policy_net(global_cond, noise)
        return preds.detach()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
            Return the first action in the predicted chunk.
        """
        chunk = self.predict_action_chunk(batch)
        return chunk[:, 0]
