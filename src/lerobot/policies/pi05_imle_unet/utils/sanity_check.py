#!/usr/bin/env python
"""
Quick sanity checker for the PI05 IMLE LoRA policy.

It tries to:
1) Load a checkpoint with the provided config and report missing/unexpected keys.
2) Optionally run a single dummy forward pass to catch obvious runtime issues.

Example:
python sanity_check.py \\
    --checkpoint /path/to/pi05_imle_checkpoint \\
    --config-from-checkpoint \\
    --device cuda \\
    --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure we can import lerobot when run from arbitrary cwd
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lerobot.policies.pi05_imle_lora.configuration_pi05_imle_lora import PI05IMLELoRAConfig
from lerobot.policies.pi05_imle_lora.modeling_pi05_imle_lora import PI05IMLELoRAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check PI05 IMLE LoRA loading and a dummy forward.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint directory or file (model.safetensors/pytorch_model.bin).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config.json compatible with the checkpoint. If omitted and the checkpoint is a "
        "directory containing config.json, that one is used.",
    )
    parser.add_argument(
        "--config-from-checkpoint",
        action="store_true",
        help="If set, prefer config.json found next to the checkpoint file/directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the dummy forward on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy forward.",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Run a dummy forward pass (otherwise only load weights).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "bfloat16", None],
        help="Override dtype in config (default: use config value).",
    )
    return parser.parse_args()


def load_config(config_path: Path, device: str, dtype_override: str | None) -> PI05IMLELoRAConfig:
    cfg = PI05IMLELoRAConfig.from_pretrained(config_path)
    cfg.device = device
    if dtype_override is not None:
        cfg.dtype = dtype_override
    return cfg


def dummy_forward(policy: PI05IMLELoRAPolicy, cfg: PI05IMLELoRAConfig, batch_size: int, device: str) -> None:
    """Run a minimal forward pass with synthetic data to catch shape/dtype issues."""
    policy.eval()

    # Tokens: simple zeros; masks mark all as valid.
    tokens = torch.zeros(batch_size, cfg.tokenizer_max_length, dtype=torch.long, device=device)
    masks = torch.ones(batch_size, cfg.tokenizer_max_length, dtype=torch.bool, device=device)

    # No images provided to keep it lightweight.
    images: list[torch.Tensor] = []
    img_masks: list[torch.Tensor] = []

    # Random actions shaped to chunk_size x max_action_dim.
    actions = torch.randn(batch_size, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        loss = policy.model.forward(images, img_masks, tokens, masks, actions)
    print(f"✓ Dummy forward ok; loss tensor shape={tuple(loss.shape)}, mean={loss.mean().item():.6f}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Resolve config: prefer checkpoint directory if requested/available
    resolved_config: Path | None = None
    if args.config_from_checkpoint or args.config is None:
        candidate = checkpoint_path / "config.json" if checkpoint_path.is_dir() else checkpoint_path.parent / "config.json"
        if candidate.exists():
            resolved_config = candidate
    if resolved_config is None:
        if args.config is None:
            # Fall back to known default inside repo if user provided nothing
            resolved_config = REPO_ROOT / "models/pi05_imle_libero_fintuned/config.json"
        else:
            resolved_config = Path(args.config)

    if not resolved_config.exists():
        raise FileNotFoundError(f"Config not found: {resolved_config}")

    print(f"Using config: {resolved_config}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")

    cfg = load_config(resolved_config, args.device, args.dtype)

    # Load policy; missing/unexpected keys are printed by from_pretrained.
    print("Loading policy...")
    policy = PI05IMLELoRAPolicy.from_pretrained(
        pretrained_name_or_path=checkpoint_path,
        config=cfg,
        local_files_only=True,
        strict=False,  # allow missing LoRA keys if checkpoint doesn't have them
    )
    policy.to(args.device)

    if args.forward:
        dummy_forward(policy, cfg, args.batch_size, args.device)
    else:
        print("✓ Loaded policy. Skipping forward (use --forward to run a dummy pass).")


if __name__ == "__main__":
    main()
