#!/usr/bin/env python
"""
Compare two policy checkpoints (pi05 vs pi05_imle_lora) on a synthetic input.

This is intended to spot large regressions after converting pi05 -> pi05_imle_lora.
It loads each checkpoint with its own config.json (or user-provided paths), runs
`sample_actions` with identical noise and inputs, and reports L1/L2 diffs.

Example:
python compare_policies.py \
  --ckpt-a /localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_libero \
  --ckpt-b /localhome/zma40/Desktop/project/generative_models_course_project/models/pi05_imle_libero_fintuned \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_imle_lora.modeling_pi05_imle_lora import PI05IMLELoRAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_imle_lora.utils.lora_config import disable_lora as disable_lora_util


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two policy checkpoints on a dummy input.")
    p.add_argument("--ckpt-a", type=str, help="Path to first checkpoint (e.g., pi05_libero).")
    p.add_argument("--ckpt-b", type=str, help="Path to second checkpoint (e.g., pi05_imle_libero_fintuned).")
    p.add_argument("--config-a", type=str, default=None, help="Optional config path for ckpt A (defaults to ckpt_a/config.json).")
    p.add_argument("--config-b", type=str, default=None, help="Optional config path for ckpt B (defaults to ckpt_b/config.json).")
    p.add_argument("--single-ckpt", action="store_true", help="Compare a single checkpoint with LoRA enabled vs disabled.")
    p.add_argument("--ckpt", type=str, help="Checkpoint path when using --single-ckpt.")
    p.add_argument("--config", type=str, default=None, help="Config path when using --single-ckpt (defaults to ckpt/config.json).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for first policy.")
    p.add_argument("--second-device", type=str, default="cpu", help="Device for second policy (use cpu to save VRAM).")
    p.add_argument("--zero-lora-a", action="store_true", help="Zero/disable LoRA adapters for policy A.")
    p.add_argument("--zero-lora-b", action="store_true", help="Zero/disable LoRA adapters for policy B.")
    p.add_argument("--paligemma-only", action="store_true", help="Compare only the PaLI-Gemma language stack (no action head).")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return p.parse_args()


def load_policy(ckpt: Path, config_path: Path, device: str):
    with open(config_path) as f:
        raw_cfg = json.load(f)
    model_type = raw_cfg.get("type", "")
    cfg = PreTrainedConfig.from_pretrained(config_path.parent, local_files_only=True)
    # Disable compilation and gradient checkpointing for lightweight compare and to avoid Triton autotune chatter
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    if hasattr(cfg, "gradient_checkpointing"):
        cfg.gradient_checkpointing = False
    cfg.device = device

    if model_type == "pi05_imle_lora":
        policy_cls = PI05IMLELoRAPolicy
    elif model_type == "pi05":
        policy_cls = PI05Policy
    else:
        raise ValueError(f"Unsupported policy type '{model_type}' in config {config_path}")

    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=ckpt,
        config=cfg,
        local_files_only=True,
        strict=False,  # allow minor key drift
    )
    if model_type == "pi05_imle_lora":
        cfg.use_lora = False

    policy.to(device)
    policy.eval()
    return policy, cfg, model_type


def make_dummy_inputs(cfg, batch_size: int, device: str):
    tokens = torch.zeros(batch_size, cfg.tokenizer_max_length, dtype=torch.long, device=device)
    masks = torch.ones(batch_size, cfg.tokenizer_max_length, dtype=torch.bool, device=device)
    images: list[torch.Tensor] = []
    img_masks: list[torch.Tensor] = []
    return images, img_masks, tokens, masks


def describe_action_space(label: str, cfg):
    print(
        f"{label} action space: chunk_size={getattr(cfg, 'chunk_size', 'n/a')}, "
        f"max_action_dim={getattr(cfg, 'max_action_dim', 'n/a')}"
    )


@torch.no_grad()
def run_paligemma_only(policy, cfg, images, img_masks, tokens, masks):
    # Reuse embed_prefix to build embeddings/attention masks but skip suffix/action head.
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(images, img_masks, tokens, masks)
    att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    att_2d_masks_4d = policy.model._prepare_attention_masks_4d(att_2d_masks)
    outputs, _ = policy.model.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    # outputs is [prefix_out, suffix_out]; suffix is None in this branch
    prefix_out = outputs[0]
    return prefix_out


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.single_ckpt:
        if not args.ckpt:
            raise ValueError("--ckpt is required when --single-ckpt is set")
        ckpt_a = Path(args.ckpt)
        config_a = Path(args.config) if args.config else ckpt_a / "config.json"
        if not config_a.exists():
            raise FileNotFoundError(f"Config not found at {config_a}")

        print(f"Loading (LoRA-enabled) from {ckpt_a} with {config_a}")
        policy_a, cfg_a, model_type_a = load_policy(ckpt_a, config_a, args.device)
        if model_type_a != "pi05_imle_lora":
            raise ValueError("Single-ckpt comparison only supports pi05_imle_lora policies.")

        print(f"Loading (LoRA-disabled) from {ckpt_a} with {config_a} on {args.second_device}")
        policy_b, cfg_b, _ = load_policy(ckpt_a, config_a, args.second_device)
        # Disable LoRA adapters on the second copy
        if hasattr(policy_b, "_freeze_lora_adapters"):
            policy_b._freeze_lora_adapters()
        else:
            raise ValueError("Policy B does not support disabling LoRA adapters.")
    else:
        if not args.ckpt_a or not args.ckpt_b:
            raise ValueError("Provide --ckpt-a and --ckpt-b, or use --single-ckpt.")
        ckpt_a = Path(args.ckpt_a)
        ckpt_b = Path(args.ckpt_b)
        config_a = Path(args.config_a) if args.config_a else ckpt_a / "config.json"
        config_b = Path(args.config_b) if args.config_b else ckpt_b / "config.json"

        if not config_a.exists():
            raise FileNotFoundError(f"Config A not found at {config_a}")
        if not config_b.exists():
            raise FileNotFoundError(f"Config B not found at {config_b}")

        print(f"Loading A from {ckpt_a} with {config_a}")
        policy_a, cfg_a, _ = load_policy(ckpt_a, config_a, args.device)
        print(f"Loading B from {ckpt_b} with {config_b}")
        policy_b, cfg_b, _ = load_policy(ckpt_b, config_b, args.second_device)

    describe_action_space("Policy A", cfg_a)
    describe_action_space("Policy B", cfg_b)

    # Optionally zero/disable LoRA adapters
    if args.zero_lora_a:
        if hasattr(policy_a, "_freeze_lora_adapters"):
            policy_a._freeze_lora_adapters()
        # Extra: call global disable to ensure adapters are turned off at runtime
        disable_lora_util(policy_a)
    if args.zero_lora_b:
        if hasattr(policy_b, "_freeze_lora_adapters"):
            policy_b._freeze_lora_adapters()
        disable_lora_util(policy_b)

    if cfg_a.chunk_size != cfg_b.chunk_size or cfg_a.max_action_dim != cfg_b.max_action_dim:
        print(
            f"Warning: chunk/action dims differ (A chunk {cfg_a.chunk_size}, action {cfg_a.max_action_dim}; "
            f"B chunk {cfg_b.chunk_size}, action {cfg_b.max_action_dim}). Comparison may be meaningless."
        )

    images_a, img_masks_a, tokens_a, masks_a = make_dummy_inputs(cfg_a, args.batch_size, args.device)
    images_b, img_masks_b, tokens_b, masks_b = make_dummy_inputs(cfg_b, args.batch_size, args.second_device)

    if args.paligemma_only:
        print("Running PaLI-Gemma prefix forward on both policies...")
        acts_a = run_paligemma_only(policy_a, cfg_a, images_a, img_masks_a, tokens_a, masks_a)
        acts_b = run_paligemma_only(policy_b, cfg_b, images_b, img_masks_b, tokens_b, masks_b)
    else:
        # Consistent noise for comparability
        noise_cpu = torch.randn(
            args.batch_size,
            min(cfg_a.chunk_size, cfg_b.chunk_size),
            min(cfg_a.max_action_dim, cfg_b.max_action_dim),
            device="cpu",
            dtype=torch.float32,
        )
        noise_a = noise_cpu.to(args.device)
        noise_b = noise_cpu.to(args.second_device)

        print("Running sample_actions on both policies...")
        acts_a = policy_a.model.sample_actions(images_a, img_masks_a, tokens_a, masks_a, noise=noise_a)
        acts_b = policy_b.model.sample_actions(images_b, img_masks_b, tokens_b, masks_b, noise=noise_b)

    # Align shapes if different (only for B x L x D tensors)
    if acts_a.dim() == 3 and acts_b.dim() == 3:
        min_chunk = min(acts_a.shape[1], acts_b.shape[1])
        min_dim = min(acts_a.shape[2], acts_b.shape[2])
        acts_a = acts_a[:, :min_chunk, :min_dim]
        acts_b = acts_b[:, :min_chunk, :min_dim]

    acts_a = acts_a.to("cpu")
    acts_b = acts_b.to("cpu")

    l1 = (acts_a - acts_b).abs().mean().item()
    l2 = torch.sqrt(((acts_a - acts_b) ** 2).mean()).item()
    print(f"L1 diff: {l1:.6f}")
    print(f"L2 diff: {l2:.6f}")
    print(f"A stats: mean {acts_a.mean().item():.6f}, std {acts_a.std().item():.6f}")
    print(f"B stats: mean {acts_b.mean().item():.6f}, std {acts_b.std().item():.6f}")
    print(f"A range: min {acts_a.min().item():.6f}, max {acts_a.max().item():.6f}")
    print(f"B range: min {acts_b.min().item():.6f}, max {acts_b.max().item():.6f}")


if __name__ == "__main__":
    main()
