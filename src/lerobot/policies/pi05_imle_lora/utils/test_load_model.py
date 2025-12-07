"""
    Test model load
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import asdict
from lerobot.policies.pi05_imle_lora import PI05IMLELoRAPolicy

def build_checkpoint(pretrained_path: str) -> None:
    pretrained_ckpt = Path(pretrained_path)  # non-LoRA checkpoint
    PI05IMLELoRAPolicy.from_pretrained(pretrained_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PI05IMLELoRA checkpoint.")
    parser.add_argument("--pretrained_path", type=str, required=True)
    args = parser.parse_args()
    build_checkpoint(
       pretrained_path=args.pretrained_path,
    )

