"""PI0.5 LoRA Policy for LeRobot training system."""

import builtins
from pathlib import Path
from typing import Dict, Optional

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pi05_lora.configuration_pi05_lora import PI05LoRAConfig
from lerobot.policies.pi05_lora.modeling_pi05_lora import PI05LoRAPytorch
from lerobot.policies.pi05_lora.lora import LoRAAdapter, LoRAConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION


T = builtins.type["PI05LoRAPolicy"]


class PI05LoRAPolicy(PI05Policy):
    """PI0.5 Policy with LoRA adaptation for VLM component."""

    config_class = PI05LoRAConfig
    name = "pi05_lora"

    def __init__(
        self,
        config: PI05LoRAConfig,
        dataset_stats: Optional[Dict] = None,
    ):
        """
        Initialize PI0.5 LoRA policy.
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics for normalization.
        """
        # Don't call parent __init__ as we need to replace the model
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        # Store dataset stats if provided
        self.dataset_stats = dataset_stats if dataset_stats is not None else {}

        # Initialize the RTC processor (needed for PI05)
        self.init_rtc_processor()

        # Create LoRA config from the policy config
        lora_config = config.to_lora_config()

        # Initialize the PI05 model with LoRA
        self.model = PI05LoRAPytorch(config, lora_config)

        # Set up RTC processor for the model
        self.model.rtc_processor = self.rtc_processor

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

        # Print LoRA information
        print(f"Initialized PI0.5 with LoRA (rank={lora_config.rank})")
        LoRAAdapter.print_trainable_parameters(self.model)

    def forward(self, batch: Dict) -> tuple[torch.Tensor, Dict]:
        """Forward pass through the model - same as PI05Policy."""
        # Use parent class's preprocessing methods
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)

        # Call the underlying model's forward with proper arguments
        loss = self.model.forward(images, img_masks, tokens, masks, actions)

        # For IMLE, loss is already a scalar. For flow matching, it would be per-element
        if loss.ndim == 0:  # Scalar loss (IMLE)
            loss_dict = {
                "loss": loss.item(),
            }
        else:  # Per-element loss (flow matching)
            # Truncate losses to actual action dimensions
            original_action_dim = self.config.output_features[ACTION].shape[0]
            loss = loss[:, :, :original_action_dim]

            loss_dict = {
                "loss": loss.mean().item(),
                "loss_per_dim": loss.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            }
            loss = loss.mean()

        return loss, loss_dict

    def get_optim_params(self) -> Dict:
        """Get optimizer parameters with proper grouping for LoRA."""
        return self.model.get_optim_params()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,  # Set to False for LoRA
        **kwargs,
    ) -> T:
        """Load a pretrained PI0.5 model and add LoRA."""
        print(
            "Loading PI0.5 with LoRA adaptation.\n"
            "VLM will use LoRA, action expert will be fully trainable.\n"
        )

        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            # Try to load config from the pretrained path
            try:
                config = PI05LoRAConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
            except:
                # If no LoRA config exists, create one from base PI05 config
                base_config = PI05Config.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
                # Convert to LoRA config
                config = PI05LoRAConfig(**base_config.__dict__)

        # Initialize model with LoRA
        model = cls(config, **kwargs)

        # Load pretrained weights (base model only, LoRA will be initialized)
        try:
            # Check if it's a local path first
            pretrained_path = Path(pretrained_name_or_path)
            if pretrained_path.exists():
                # Local path - try safetensors first, then pytorch
                safetensors_path = pretrained_path / "model.safetensors"
                pytorch_path = pretrained_path / "pytorch_model.bin"

                if safetensors_path.exists():
                    from safetensors.torch import load_file
                    original_state_dict = load_file(str(safetensors_path))
                    print(f"✓ Loaded state dict from {safetensors_path}")
                elif pytorch_path.exists():
                    original_state_dict = torch.load(str(pytorch_path), map_location="cpu")
                    print(f"✓ Loaded state dict from {pytorch_path}")
                else:
                    raise FileNotFoundError(f"No model weights found in {pretrained_path}")
            else:
                # HuggingFace Hub path
                from transformers.utils import cached_file

                # Try safetensors first
                try:
                    resolved_file = cached_file(
                        pretrained_name_or_path,
                        "model.safetensors",
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    from safetensors.torch import load_file
                    original_state_dict = load_file(resolved_file)
                    print("✓ Loaded state dict from model.safetensors")
                except:
                    # Try pytorch format
                    resolved_file = cached_file(
                        pretrained_name_or_path,
                        "pytorch_model.bin",
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    original_state_dict = torch.load(resolved_file, map_location="cpu")
                    print("✓ Loaded state dict from pytorch_model.bin")

            # Fix and remap keys
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Add "model." prefix for keys that don't have it
            remapped_state_dict = {}
            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                else:
                    remapped_state_dict[key] = value

            # Debug: print sample keys from state_dict and model
            print(f"DEBUG: Sample ORIGINAL state_dict keys (first 5):")
            for i, k in enumerate(list(original_state_dict.keys())[:5]):
                print(f"  Original[{i}]: {k}")

            # Check if checkpoint has language_model keys
            has_language_model = any("language_model" in k for k in original_state_dict.keys())
            has_vision_tower = any("vision_tower" in k for k in original_state_dict.keys())
            print(f"DEBUG: Checkpoint has language_model keys: {has_language_model}")
            print(f"DEBUG: Checkpoint has vision_tower keys: {has_vision_tower}")

            print(f"DEBUG: Sample state_dict keys (first 3 with q_proj):")
            q_proj_keys = [k for k in remapped_state_dict.keys() if "q_proj" in k][:3]
            for k in q_proj_keys:
                print(f"  State dict: {k}")

            print(f"DEBUG: Sample model keys (first 3 with q_proj):")
            model_q_proj_keys = [k for k in model.state_dict().keys() if "q_proj" in k][:3]
            for k in model_q_proj_keys:
                print(f"  Model: {k}")

            # Remap keys for LoRA modules: q_proj.weight -> q_proj.base_layer.weight
            # This is needed because LoRALinear wraps the original Linear in base_layer
            # IMPORTANT: Only remap keys in paligemma.model.language_model (where LoRA is applied)
            # NOT in vision_tower or gemma_expert
            if model.model.use_lora:
                lora_remapped = {}
                lora_target_modules = model.model.lora_config.target_modules
                for key, value in remapped_state_dict.items():
                    new_key = key
                    # Only apply base_layer remapping for language_model keys (where LoRA is applied)
                    if "paligemma.model.language_model" in key:
                        for target in lora_target_modules:
                            # Match patterns like "q_proj.weight" or "q_proj.bias"
                            if f".{target}.weight" in key or f".{target}.bias" in key:
                                # Insert "base_layer" before "weight" or "bias"
                                if ".weight" in key:
                                    new_key = key.replace(f".{target}.weight", f".{target}.base_layer.weight")
                                elif ".bias" in key:
                                    new_key = key.replace(f".{target}.bias", f".{target}.base_layer.bias")
                                break
                    lora_remapped[new_key] = value
                remapped_state_dict = lora_remapped
                print(f"DEBUG: Remapped keys for LoRA base_layer (language_model only)")
                print(f"DEBUG: Sample remapped keys (first 3 with language_model q_proj):")
                remapped_q_proj = [k for k in remapped_state_dict.keys() if "language_model" in k and "q_proj" in k][:3]
                for k in remapped_q_proj:
                    print(f"  Remapped: {k}")

            # Handle shared tensors: embed_tokens and lm_head share weights in original checkpoint
            # But they are stored as separate keys in our model. Copy lm_head to embed_tokens if missing.
            embed_tokens_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
            if embed_tokens_key not in remapped_state_dict and lm_head_key in remapped_state_dict:
                print(f"DEBUG: Copying lm_head.weight to embed_tokens.weight (shared tensor)")
                remapped_state_dict[embed_tokens_key] = remapped_state_dict[lm_head_key]

            # Load with strict=False to ignore missing LoRA weights
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            # Filter out expected missing keys (LoRA weights)
            lora_missing = [k for k in missing_keys if "lora_" in k]
            other_missing = [k for k in missing_keys if "lora_" not in k]

            print(f"DEBUG: Total missing keys: {len(missing_keys)}")
            print(f"DEBUG: LoRA missing: {len(lora_missing)}")
            print(f"DEBUG: Other missing: {len(other_missing)}")

            if lora_missing:
                print(f"Initialized {len(lora_missing)} LoRA parameters")
            if other_missing:
                # Check if IMLE generator weights are missing
                imle_missing = [k for k in other_missing if "imle_generator" in k]
                print(f"DEBUG: IMLE missing keys: {len(imle_missing)}")
                print(f"DEBUG: Has model.model: {hasattr(model, 'model')}")

                # IMLE generator is in model.model (PI05LoRAPytorch), not model (PI05LoRAPolicy)
                actual_model = model.model if hasattr(model, 'model') else model
                print(f"DEBUG: Has imle_generator: {hasattr(actual_model, 'imle_generator')}")
                if hasattr(actual_model, 'imle_generator'):
                    print(f"DEBUG: Has _initialize_weights: {hasattr(actual_model.imle_generator, '_initialize_weights')}")

                if imle_missing:
                    print(f"IMLE generator not found in checkpoint ({len(imle_missing)} keys)")
                    # Re-initialize IMLE generator with proper weights
                    if hasattr(actual_model, 'imle_generator') and hasattr(actual_model.imle_generator, '_initialize_weights'):
                        print("Re-initializing IMLE generator with Xavier initialization")

                        # Check weights BEFORE initialization
                        final_layer = actual_model.imle_generator.trajectory_generator[-1]
                        before_mean = final_layer.weight.abs().mean().item()
                        print(f"BEFORE init - Final layer weight mean: {before_mean:.6f}")

                        actual_model.imle_generator._initialize_weights()

                        # Check weights AFTER initialization
                        after_mean = final_layer.weight.abs().mean().item()
                        print(f"AFTER init - Final layer weight mean: {after_mean:.6f}")
                        print(f"Expected: ~0.001, Got: {after_mean:.6f}")

                        if after_mean > 0.01:
                            print("WARNING: Initialization may have failed! Weight magnitude too large")
                        else:
                            print("Initialization verified successful!")
                    else:
                        print("Cannot re-initialize: imle_generator or _initialize_weights not found")
                print(f"Warning: Missing non-LoRA keys: {other_missing[:5]}")

            print("Model loaded successfully with LoRA adaptation!")

            # Debug: Check LoRA B initialization (should be zeros)
            print("\n" + "="*50)
            print("DEBUG: LoRA B matrix initialization check")
            print("="*50)
            lora_b_count = 0
            lora_b_nonzero = 0
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    lora_b_count += 1
                    mean_val = param.abs().mean().item()
                    if mean_val > 1e-8:
                        lora_b_nonzero += 1
                        print(f"  WARNING: {name} has non-zero values! mean={mean_val:.6f}")
                    if lora_b_count <= 3:  # Print first 3
                        print(f"  {name}: mean={mean_val:.10f}")
            print(f"Total LoRA B matrices: {lora_b_count}, Non-zero: {lora_b_nonzero}")
            if lora_b_nonzero == 0:
                print("✓ All LoRA B matrices are zero (correct initialization)")
            else:
                print("✗ Some LoRA B matrices are non-zero (incorrect!)")
            print("="*50 + "\n")

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
            print("Using randomly initialized model")

        return model

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """Save the model with LoRA weights."""
        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self.config.save_pretrained(save_directory)

        # Get state dict and add "model." prefix to match original checkpoint format
        # This is necessary because from_pretrained expects keys like "model.xxx"
        # Clone tensors to avoid shared tensor issues (e.g., embed_tokens and lm_head)
        state_dict = self.model.state_dict()
        prefixed_state_dict = {f"model.{k}": v.clone() for k, v in state_dict.items()}

        # Save full model state in safetensors format with correct key prefix
        save_file(prefixed_state_dict, save_directory / "model.safetensors")

        # Also save in pytorch format for compatibility (with prefix)
        torch.save(prefixed_state_dict, save_directory / "pytorch_model.bin")

        # Also save LoRA weights separately
        self.model.save_lora_weights(save_directory / "lora_weights.pt")

        print(f"Model saved to {save_directory}")

        if push_to_hub:
            # TODO: Implement hub upload
            print("Hub upload not yet implemented for LoRA models")