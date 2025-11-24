#!/usr/bin/env python

# Minimal IMLE policy wrapper for LeRobot.
# This wires the existing IMLE UNet and loss into the PreTrainedPolicy
# interface so it can be trained and evaluated with the common pipeline.
# Implementation based on diffusion policy

from collections import deque

import torch
from torch import nn, Tensor

from lerobot.policies.imle_policy.configuration_imle_policy import IMLEConfig
from lerobot.policies.imle_policy.models.rs_imle_network import GeneratorConditionalUnet1D
from lerobot.policies.imle_policy.models.vision_network import get_resnet, replace_bn_with_gn
from lerobot.policies.imle_policy.utils.losses import rs_imle_loss
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    populate_queues,
)

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

class IMLEPolicy(PreTrainedPolicy):
    """
        Lightweight IMLE policy: encodes camera images with ResNet18, concatenates
        low-dim observations, and learns an action generator with IMLE loss.
        Implementation based on IMLE policy
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
        self._vision_keys: list[str] = []
        self.vision_encoders = nn.ModuleList()
        for key in config.image_features:
            self._vision_keys.append(key)
            encoder = get_resnet("resnet18")
            encoder = replace_bn_with_gn(encoder)
            self.vision_encoders.append(encoder)

        # IMLE generator network.
        self.policy_net = GeneratorConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.obs_dim * config.obs_horizon,
        )
        self.to(self.device)
        self.reset()

    def get_optim_params(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self):
        """
            Clear observation and action queues. Should be called on `env.reset()`
        """
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
        if self.vision_encoders:
            if OBS_IMAGES in batch:
                imgs = batch[OBS_IMAGES].to(self.device)
                imgs = imgs[:, :obs_h]  # [B, T, N, C, H, W]
                b, t, n, c, h, w = imgs.shape
                if n != len(self.vision_encoders):
                    raise ValueError(
                        f"Number of image streams ({n}) does not match number of vision encoders "
                        f"({len(self.vision_encoders)})."
                    )
                for idx, encoder in enumerate(self.vision_encoders):
                    cam_imgs = imgs[:, :, idx]
                    feats = encoder(cam_imgs.reshape(b * t, c, h, w))  # [B*T, feat_dim]
                    feats = feats.reshape(b, t, -1)
                    pieces.append(feats)
            else:
                for vision_key, encoder in zip(self._vision_keys, self.vision_encoders, strict=True):
                    imgs = batch[vision_key].to(self.device)
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
        preds = self.policy_net(global_cond, noise) # predicted action sequences
        return preds.detach()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
            Return the first action in the predicted chunk.
        """
        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action
