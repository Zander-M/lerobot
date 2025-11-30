"""
    Building skeleton model and save as checkpoint. 
    Used to load pretrained PaliGemma parameters in the new model,
    and initialize the IMLE model to empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataclasses import asdict
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_imle_unet import PI05IMLEUnetConfig, PI05IMLEUnetPolicy

def build_checkpoint(pretrained_path: str, output_path: str) -> None:
    pretrained_ckpt = Path(pretrained_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = PreTrainedConfig.from_pretrained(pretrained_ckpt)
    cfg_dict = asdict(base_config)
    cfg_dict.pop("type", None)
    cfg_dict.pop("pretrained_path", None)

    config = PI05IMLEUnetConfig(**cfg_dict)

    policy = PI05IMLEUnetPolicy.from_pi05(pretrained_ckpt, 
                                                config=config, 
                                               )
    policy.save_pretrained(output_dir)
    print(f"IMLE checkpoint saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PI05IMLE checkpoint.")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    build_checkpoint(
       pretrained_path=args.pretrained_path,
       output_path=args.output_path, 
    )

