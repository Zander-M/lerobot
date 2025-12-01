"""
PI0.5 model with LoRA adaptation for VLM component and IMLE action generation.
This variant of PI0.5 applies LoRA to the VLM (PaliGemma) component
while replacing the action expert with an IMLE-based generator for
efficient finetuning with implicit maximum likelihood estimation.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from lerobot.common.policies.lora import LoRAAdapter, LoRAConfig, LoRALinear
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05_lora.configuration_pi05_lora import PI05LoRAConfig
from lerobot.policies.pi05_lora.imle.model.rs_imle_network import (
    GeneratorConditionalUnet1D,
    SimpleActionGenerator,
)


class PI05LoRAPytorch(PI05Pytorch, PyTorchModelHubMixin):
    """PI0.5 model with LoRA adaptation for VLM and optional IMLE action generator."""

    def __init__(self, config: PI05LoRAConfig, lora_config: Optional[LoRAConfig] = None):
        """Initialize PI0.5 with LoRA and optional IMLE.
        Args:
            config: PI0.5 LoRA model configuration.
            lora_config: LoRA configuration. If None, uses config settings.
        """
        super().__init__(config)

        # Use config settings for LoRA if not explicitly provided
        if lora_config is None and hasattr(config, 'lora_rank'):
            lora_config = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                init_std=config.lora_init_std,
                rslora=config.lora_rslora
            )
        elif lora_config is None:
            # Default LoRA configuration
            lora_config = LoRAConfig(
                rank=16,
                alpha=32.0,
                dropout=0.05,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                init_std=0.01,
                rslora=False
            )

        self.lora_config = lora_config
        self.lora_modules = {}
        self.use_lora = getattr(config, 'use_lora', True)

        # Apply LoRA to VLM component if enabled
        if self.use_lora:
            self._apply_lora()

        # Initialize IMLE action generator if enabled
        if hasattr(config, 'use_imle') and config.use_imle:
            self._initialize_imle_generator(config)

        # Freeze VLM base weights, keep LoRA and action expert/IMLE trainable
        self._setup_trainable_parameters()

    def _initialize_imle_generator(self, config: PI05LoRAConfig):
        """Initialize IMLE action generator network.
        Args:
            config: PI0.5 LoRA configuration with IMLE settings.
        """
        # Get conditioning dimension from VLM output
        # This should match the hidden size of the language model
        paligemma = self.paligemma_with_expert.paligemma
        cond_dim = paligemma.config.text_config.hidden_size

        # Get action dimension from config
        action_dim = config.max_action_dim
        horizon = config.chunk_size

        if config.imle_network_type == "unet":
            self.imle_generator = GeneratorConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=cond_dim,
                down_dims=config.imle_down_dims,
                kernel_size=config.imle_kernel_size,
                n_groups=config.imle_n_groups
            )
        elif config.imle_network_type == "simple":
            self.imle_generator = SimpleActionGenerator(
                state_dim=cond_dim,
                action_dim=action_dim,
                noise_dim=config.imle_noise_dim,
                hidden_dim=config.imle_hidden_dim,
                horizon=horizon,
                n_groups=config.imle_n_groups
            )
        else:
            raise ValueError(f"Unknown IMLE network type: {config.imle_network_type}")

        print(f"\nInitialized IMLE {config.imle_network_type} generator:")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Horizon: {horizon}")
        print(f"  - Conditioning dim: {cond_dim}")
        print(f"  - Noise dim: {config.imle_noise_dim}")

        # Store IMLE config
        self.imle_config = {
            'num_samples': config.imle_num_samples,
            'noise_scale': config.imle_noise_scale,
            'noise_dim': config.imle_noise_dim,
            # Note: temperature not used in hard IMLE (kept in config for future soft-IMLE variants)
        }

    def _apply_lora(self):
        """Apply LoRA to the VLM component by replacing Linear layers."""
        # Access the PaliGemma model
        paligemma = self.paligemma_with_expert.paligemma

        # Replace Linear layers with LoRALinear in the language model
        # NOTE: Use paligemma.model.language_model (not paligemma.language_model property)
        # because state_dict keys use the full path through .model
        self._replace_with_lora(paligemma.model.language_model, "paligemma.model.language_model")

        print(f"Applied LoRA with rank={self.lora_config.rank}")
        print(f"Target modules: {self.lora_config.target_modules}")

    def _replace_with_lora(self, module: nn.Module, prefix: str = ""):
        """Recursively replace Linear layers with LoRALinear.
        Args:
            module: Module to process
            prefix: Current module path prefix
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this is a Linear layer we want to replace
            if isinstance(child, nn.Linear):
                # Check if any target module pattern matches
                should_replace = any(
                    target in name for target in self.lora_config.target_modules
                )

                if should_replace:
                    # Create LoRA layer
                    lora_layer = LoRALinear(
                        child.in_features,
                        child.out_features,
                        self.lora_config,
                        base_layer=child
                    )

                    # Replace the module
                    setattr(module, name, lora_layer)
                    self.lora_modules[full_name] = lora_layer
                    print(f"Replaced {full_name} with LoRA (in={child.in_features}, out={child.out_features})")
            else:
                # Recurse into child modules
                self._replace_with_lora(child, full_name)

    def _setup_trainable_parameters(self):
        """Setup which parameters are trainable."""
        # Count parameters by category
        lora_param_count = 0
        expert_param_count = 0
        imle_param_count = 0
        vlm_param_count = 0

        # Check if we're using LoRA, IMLE, or VLM freezing (selective training mode)
        freeze_vlm = getattr(self.config, 'freeze_vlm', False)
        using_selective_training = self.use_lora or hasattr(self, 'imle_generator') or freeze_vlm

        if using_selective_training:
            # Only freeze parameters if we're doing selective training
            # First, freeze all parameters
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze LoRA parameters in VLM (if LoRA is enabled)
            if self.use_lora:
                for name, param in self.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
                        lora_param_count += param.numel()
            else:
                # If not using LoRA, optionally unfreeze entire VLM (PaliGemma)
                freeze_vlm = getattr(self.config, 'freeze_vlm', False)
                if not freeze_vlm:
                    for name, param in self.named_parameters():
                        if "paligemma" in name:
                            param.requires_grad = True
                            vlm_param_count += param.numel()

            # Unfreeze IMLE generator if present
            if hasattr(self, 'imle_generator'):
                for param in self.imle_generator.parameters():
                    param.requires_grad = True
                    imle_param_count += param.numel()
            else:
                # Unfreeze entire action expert (gemma_expert) if not using IMLE
                for name, param in self.named_parameters():
                    if "gemma_expert" in name:
                        param.requires_grad = True
                        expert_param_count += param.numel()
        else:
            # No selective training - behave like vanilla PI05
            # All parameters remain trainable (default PyTorch behavior)
            for name, param in self.named_parameters():
                if "paligemma" in name:
                    vlm_param_count += param.numel()
                elif "gemma_expert" in name:
                    expert_param_count += param.numel()

        # Print detailed summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*50)
        print("Parameter Summary:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        if using_selective_training:
            # VLM status
            if self.use_lora:
                print(f"  - LoRA parameters: {lora_param_count:,}")
            else:
                freeze_vlm = getattr(self.config, 'freeze_vlm', False)
                if freeze_vlm:
                    print(f"  - VLM (PaliGemma): FROZEN")
                else:
                    print(f"  - VLM (PaliGemma) parameters: {vlm_param_count:,}")

            # Action generation status
            if imle_param_count > 0:
                print(f"  - IMLE generator parameters: {imle_param_count:,}")
            else:
                print(f"  - Action expert parameters: {expert_param_count:,}")
        else:
            # Vanilla PI05 mode
            print("  - Mode: Full Fine-tuning (like vanilla PI05)")
            print(f"  - VLM (PaliGemma) parameters: {vlm_param_count:,}")
            print(f"  - Action expert parameters: {expert_param_count:,}")

        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        print("="*50 + "\n")

    def get_optim_params(self) -> List[Dict]:
        """Get optimizer parameter groups with different learning rates.
        Returns:
            List of parameter groups for optimizer.
        """
        # Separate LoRA, IMLE, and action expert parameters
        lora_params = []
        imle_params = []
        action_expert_params = []

        # Collect IMLE parameters separately
        if hasattr(self, 'imle_generator'):
            imle_params = list(self.imle_generator.parameters())

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_" in name:
                lora_params.append(param)
            elif "gemma_expert" in name and not hasattr(self, 'imle_generator'):
                action_expert_params.append(param)

        # Return parameter groups with potentially different LRs
        param_groups = []

        if lora_params:
            param_groups.append({
                "params": lora_params,
                "name": "lora",
                "lr_scale": 0.5,  # Can use lower LR for LoRA
            })

        if imle_params:
            param_groups.append({
                "params": imle_params,
                "name": "imle_generator",
                "lr_scale": 1.0,  # Full LR for IMLE
            })

        if action_expert_params:
            param_groups.append({
                "params": action_expert_params,
                "name": "action_expert",
                "lr_scale": 1.0,  # Full LR for action expert
            })

        return param_groups if param_groups else list(self.parameters())

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None):
        """Forward pass - uses IMLE if enabled, otherwise uses flow matching."""
        if hasattr(self, 'imle_generator'):
            return self._forward_imle(images, img_masks, tokens, masks, actions)
        else:
            # Use parent's flow matching forward
            return super().forward(images, img_masks, tokens, masks, actions, noise, time)

    def _forward_imle(self, images, img_masks, tokens, masks, actions):
        """IMLE-based forward pass for training.
        Uses implicit maximum likelihood estimation to train the action generator.
        """
        # Get VLM conditioning
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # Process prefix through VLM to get conditioning features
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Forward through VLM (paligemma) to get conditioning
        (prefix_output, _), _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        # Use mean pooling over sequence for conditioning
        cond_features = prefix_output.mean(dim=1)  # [B, hidden_dim]

        # Sample multiple noise vectors for IMLE
        batch_size = actions.shape[0]
        num_samples = self.imle_config['num_samples']

        # Repeat conditioning for all samples: [B * num_samples, hidden_dim]
        cond_features_repeated = cond_features.repeat_interleave(num_samples, dim=0)

        # Generate actions using IMLE generator
        # NOTE: Following original IMLE Policy - noise is standard normal N(0,1), no scaling
        if isinstance(self.imle_generator, SimpleActionGenerator):
            # Simple generator: forward(state, noise) -> actions
            noise_dim = self.imle_config['noise_dim']
            noise = torch.randn(
                batch_size * num_samples,
                noise_dim,
                device=actions.device,
                dtype=torch.float32
            )
            predicted_actions = self.imle_generator(cond_features_repeated, noise)
        else:
            # UNet generator: forward(global_cond, sample) -> actions
            # For UNet, noise is action-shaped (B*num_samples, horizon, action_dim)
            noise = torch.randn(
                batch_size * num_samples,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=actions.device,
                dtype=torch.float32
            )
            predicted_actions = self.imle_generator(cond_features_repeated, noise)

        # Reshape: [B, num_samples, horizon, action_dim]
        predicted_actions = predicted_actions.reshape(
            batch_size, num_samples, self.config.chunk_size, -1
        )

        # IMLE loss: Hard nearest-neighbor matching with epsilon filtering
        # Following the original IMLE Policy implementation (RSS 2025)
        # https://github.com/krishanrana/imle_policy

        # Expand ground truth: [B, 1, horizon, action_dim]
        actions_expanded = actions.unsqueeze(1)

        # Debug: Print action statistics (first 5 steps, then every 100 steps)
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1

        should_print = self._debug_step <= 5 or self._debug_step % 100 == 0
        if should_print:
            print(f"[Step {self._debug_step}] Action stats - min: {actions.min().item():.4f}, max: {actions.max().item():.4f}, "
                  f"mean: {actions.mean().item():.4f}, std: {actions.std().item():.4f}")
            print(f"[Step {self._debug_step}] Predicted action stats - min: {predicted_actions.min().item():.4f}, "
                  f"max: {predicted_actions.max().item():.4f}")

        # Flatten for distance computation: [B, 1, horizon*action_dim] and [B, num_samples, horizon*action_dim]
        real_flat = actions.reshape(batch_size, 1, -1)
        fake_flat = predicted_actions.reshape(batch_size, num_samples, -1)

        # Compute Euclidean distances using cdist: [B, num_samples]
        distances = torch.cdist(real_flat, fake_flat).squeeze(1)

        # Debug: Print distance statistics
        if should_print:
            print(f"[Step {self._debug_step}] Distance stats - min: {distances.min().item():.4f}, max: {distances.max().item():.4f}, "
                  f"mean: {distances.mean().item():.4f}")

        # Epsilon filtering: only consider samples farther than epsilon from ground truth
        # This prevents mode collapse and maintains diversity in the generator
        epsilon = getattr(self.config, 'imle_epsilon', 0.03)
        valid_samples = (distances > epsilon).float()

        # Add penalty to invalid samples (closer than epsilon) so they won't be selected
        # Use distances.max() as a large penalty value
        penalized_distances = distances + (1 - valid_samples) * distances.max()

        # Hard selection: find closest valid sample for each batch item
        min_distances, _ = penalized_distances.min(dim=1)

        # Only compute loss for samples that have at least one valid sample
        valid_real_samples = (min_distances < distances.max()).float()

        if valid_real_samples.sum() > 0:
            loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
        else:
            # All samples are within epsilon - no gradient needed
            loss = torch.tensor(0.0, device=actions.device)

        # Return scalar loss (NOT expanded - that would multiply the loss by chunk_size * action_dim!)
        return loss

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
        """Sample actions - uses IMLE if enabled, otherwise uses flow matching."""
        if hasattr(self, 'imle_generator'):
            return self._sample_actions_imle(images, img_masks, tokens, masks)
        else:
            # Use parent's flow matching sampling
            return super().sample_actions(images, img_masks, tokens, masks, noise, num_steps, **kwargs)

    @torch.no_grad()
    def _sample_actions_imle(self, images, img_masks, tokens, masks):
        """IMLE-based action sampling for inference."""
        # Get VLM conditioning
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        # Process prefix through VLM
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Forward through VLM to get conditioning
        (prefix_output, _), _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        # Use mean pooling for conditioning
        cond_features = prefix_output.mean(dim=1)

        # Sample noise (standard normal, matching training)
        batch_size = tokens.shape[0]

        # Generate actions using IMLE generator
        # NOTE: Following original IMLE Policy - noise is standard normal N(0,1), no scaling
        if isinstance(self.imle_generator, SimpleActionGenerator):
            # Simple generator: forward(state, noise) -> actions
            noise_dim = self.imle_config['noise_dim']
            noise = torch.randn(batch_size, noise_dim, device=tokens.device, dtype=torch.float32)
            actions = self.imle_generator(cond_features, noise)
        else:
            # UNet generator: forward(global_cond, sample) -> actions
            # For UNet, noise is action-shaped (B, horizon, action_dim)
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=tokens.device,
                dtype=torch.float32
            )
            actions = self.imle_generator(cond_features, noise)

        return actions

    def save_lora_weights(self, save_path: Path):
        """Save only the LoRA weights.
        Args:
            save_path: Path to save LoRA weights.
        """
        lora_state_dict = {}
        for name, param in self.state_dict().items():
            if "lora_" in name:
                lora_state_dict[name] = param

        torch.save(lora_state_dict, save_path)
        print(f"Saved {len(lora_state_dict)} LoRA weights to {save_path}")

    def load_lora_weights(self, load_path: Path, strict: bool = False):
        """Load LoRA weights.
        Args:
            load_path: Path to LoRA weights.
            strict: Whether to enforce strict loading.
        """
        lora_state_dict = torch.load(load_path, map_location="cpu")

        # Load only LoRA weights
        model_state = self.state_dict()
        for key, value in lora_state_dict.items():
            if key in model_state and "lora_" in key:
                model_state[key] = value

        self.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(lora_state_dict)} LoRA weights from {load_path}")

    def merge_and_save(self, save_path: Path):
        """Merge LoRA weights into base model and save.
        This creates a standard PI0.5 model without LoRA overhead.
        Args:
            save_path: Path to save merged model.
        """
        # Merge LoRA weights into base layers
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()

        # Save the entire model
        torch.save(self.state_dict(), save_path)
        print(f"Saved merged model to {save_path}")

        # Unmerge for continued training if needed
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        lora_config: Optional[LoRAConfig] = None,
        **kwargs
    ):
        """Load a pretrained PI0.5 model and add LoRA.
        Args:
            pretrained_model_name_or_path: Model identifier or path.
            lora_config: LoRA configuration.
            **kwargs: Additional arguments for model loading.
        Returns:
            PI0.5 model with LoRA.
        """
        # First load the base PI0.5 model
        config = PI05Config.from_pretrained(pretrained_model_name_or_path)

        # Create model with LoRA
        model = cls(config, lora_config)

        # Load pretrained weights
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(
                Path(pretrained_model_name_or_path) / "model.safetensors",
                map_location="cpu"
            )
        else:
            # Load from HuggingFace hub
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model.safetensors"
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load weights (will ignore missing LoRA weights)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        # Filter expected missing keys (LoRA weights)
        lora_missing = [k for k in missing_keys if "lora_" in k]
        other_missing = [k for k in missing_keys if "lora_" not in k]

        print(f"DEBUG: Total missing keys: {len(missing_keys)}")
        print(f"DEBUG: LoRA missing: {len(lora_missing)}")
        print(f"DEBUG: Other missing: {len(other_missing)}")

        if lora_missing:
            print(f"Initialized {len(lora_missing)} new LoRA parameters")
        if other_missing:
            # Check if IMLE generator weights are missing
            imle_missing = [k for k in other_missing if "imle_generator" in k]
            print(f"DEBUG: IMLE missing keys: {len(imle_missing)}")
            print(f"DEBUG: Has imle_generator: {hasattr(model, 'imle_generator')}")
            if hasattr(model, 'imle_generator'):
                print(f"DEBUG: Has _initialize_weights: {hasattr(model.imle_generator, '_initialize_weights')}")

            if imle_missing:
                print(f"IMLE generator not found in checkpoint ({len(imle_missing)} keys)")
                # Re-initialize IMLE generator with proper weights
                if hasattr(model, 'imle_generator') and hasattr(model.imle_generator, '_initialize_weights'):
                    print("Re-initializing IMLE generator with Xavier initialization")
                    model.imle_generator._initialize_weights()
                    print("Initialization complete!")
            if other_missing:
                print(f"Warning: Missing non-LoRA keys: {other_missing[:5]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:5]}")

        # Re-setup trainable parameters after loading
        model._setup_trainable_parameters()

        return model

    def get_action_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about action predictions."""
        # This is useful for debugging IMLE training
        stats = {}

        if hasattr(self, "_last_action_pred"):
            stats["action_mean"] = self._last_action_pred.mean()
            stats["action_std"] = self._last_action_pred.std()
            stats["action_min"] = self._last_action_pred.min()
            stats["action_max"] = self._last_action_pred.max()

        return stats