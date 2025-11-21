"""
    Building skeleton model and save as checkpoint. Used to load pretrained 
    PaliGemma parameters in the new model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import asdict
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_imle_lora import PI05IMLELoRAConfig, PI05IMLELoRAPolicy

def build_checkpoint(pretrained_path: str, output_path: str, lora_config_path: str) -> None:
    pretrained_ckpt = Path(pretrained_path)  # non-LoRA checkpoint
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = PreTrainedConfig.from_pretrained(pretrained_ckpt)
    cfg_dict = asdict(base_config)
    cfg_dict.pop("type", None)
    cfg_dict.pop("pretrained_path", None)

    config = PI05IMLELoRAConfig(**cfg_dict)

    with open(lora_config_path, "r") as f:
        lora_config = json.load(f)
        config.use_lora = lora_config.get("use_lora", True)
        config.lora_r = lora_config.get("lora_r", 16)              
        config.lora_alpha = lora_config.get("lora_alpha", 32)
        config.lora_dropout = lora_config.get("lora_dropout", 0.05)
        config.pretrained_path = None   # keepskeleton self-contained
        config.device = "cuda"

    policy = PI05IMLELoRAPolicy.from_pretrained(pretrained_ckpt, 
                                                config=config, 
                                                strict=not config.use_lora)
    policy.save_pretrained(output_dir)
    print(f"LoRA-ready checkpoint saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PI05IMLELoRA checkpoint.")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora_config_path", type=str, required=True)
    args = parser.parse_args()
    build_checkpoint(
       pretrained_path=args.pretrained_path,
       output_path=args.output_path, 
       lora_config_path=args.lora_config_path
    )

