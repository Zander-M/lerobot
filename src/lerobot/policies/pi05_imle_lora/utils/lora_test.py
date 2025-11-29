"""
    Verify if LoRA adapter is working correctly
"""

#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


# LoRA related
from peft import LoraConfig, get_peft_model

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from lerobot.policies.pi05_imle_lora.modeling_pi05_imle_lora import get_gemma_config 
from lerobot.policies.pi05_imle_lora.configuration_pi05_imle_lora import PI05IMLELoRAConfig

## Util Functions

def add_lora_to_gemma(paligemma: PaliGemmaForConditionalGeneration, config: dict) -> None:
    """
        Wrapping Gemma model with LoRA. We only finetune the text Gemma model with LoRA
    """

    # Freeze vision tower; we only finetune the language stack.
    for p in paligemma.vision_tower.parameters():
        p.requires_grad = False
    paligemma.vision_tower.eval()

    # Finding LoRA modules in language model
    # Keep it consistent with the _fix_pytorch_state_dict_keys
    attn_proj = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_proj = ["gate_proj", "up_proj", "down_proj"]
    proj_names = attn_proj + mlp_proj

    def find_llm_modules(model, proj):
        return [
            name
            for name, module in model.named_modules()
            if name.split(".")[-1] == proj and "language_model" in name
        ]

    target_modules = [leaf 
                      for proj in proj_names
                      for leaf in find_llm_modules(paligemma, proj)]
 
    lora_cfg = LoraConfig(
        r=config[ "lora_r" ],
        lora_alpha=config["lora_alpha" ],
        lora_dropout=config["lora_dropout" ],
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    paligemma = get_peft_model(paligemma, lora_cfg)
    paligemma.enable_input_require_grads()
    print_lora_parameter_stats(paligemma)
    return paligemma


def print_lora_parameter_stats(model: nn.Module):
    """
        Print LoRA parameters
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
            if "lora_" in name:
                lora_params += n
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    print("\n==================== LoRA Parameter Report ====================")
    print(f"Total parameters:            {total_params:,}")
    print(f"Trainable parameters:        {trainable_params:,}")
    print(f"   of which LoRA params:     {lora_params:,}")
    print(f"Percentage trainable:        {pct:.6f}%")
    print("===============================================================\n")

def init_paligemma(config, use_adarms=[False, False]):

    vlm_config_hf = CONFIG_MAPPING["paligemma"]()
    vlm_config_hf._vocab_size = 257152  # noqa: SLF001
    vlm_config_hf.image_token_index = 257152
    vlm_config_hf.text_config.hidden_size = config.width
    vlm_config_hf.text_config.intermediate_size = config.mlp_dim
    vlm_config_hf.text_config.num_attention_heads = config.num_heads
    vlm_config_hf.text_config.head_dim = config.head_dim
    vlm_config_hf.text_config.num_hidden_layers = config.depth
    vlm_config_hf.text_config.num_key_value_heads = config.num_kv_heads
    vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
    vlm_config_hf.text_config.torch_dtype = "float32"
    vlm_config_hf.text_config.vocab_size = 257152
    vlm_config_hf.text_config.use_adarms = use_adarms[0]
    vlm_config_hf.text_config.adarms_cond_dim = config.width if use_adarms[0] else None
    vlm_config_hf.vision_config.intermediate_size = 4304
    vlm_config_hf.vision_config.projection_dim = 2048
    vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
    vlm_config_hf.vision_config.torch_dtype = "float32"

    return PaliGemmaForConditionalGeneration(config=vlm_config_hf)

### Test script

def _dummy_batch(model, batch_size=1, seq_len=8):
    device = next(model.parameters()).device
    vocab = model.config.text_config.vocab_size
    image_token = model.config.image_token_index
    image_size = model.config.vision_config.image_size

    # SigLIP-specific: compute #patches manually
    patch = model.config.vision_config.patch_size
    num_image_tokens = (image_size // patch) ** 2

    # 1) Create required block of <image> tokens
    image_token_block = torch.full(
        (batch_size, num_image_tokens),
        fill_value=image_token,
        dtype=torch.long,
        device=device,
    )

    # 2) Add some random normal text tokens
    random_tokens = torch.randint(
        0, vocab, (batch_size, seq_len), device=device
    )

    input_ids = torch.cat([image_token_block, random_tokens], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # 3) Dummy image
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }




@torch.no_grad()
def test_lora():
    """
    Test LoRA adapter behavior:

    1. Wrapping with LoRA does not change outputs initially.
    2. After perturbing LoRA weights, outputs change when adapters are enabled.
    3. Disabling adapters restores the original base behavior.
    """
    torch.manual_seed(0)

    config = PI05IMLELoRAConfig()
    lora_config = {
        "use_lora": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    gemma_config = get_gemma_config(config.paligemma_variant)

    # --- Build identical base + LoRA models ---
    base_model = init_paligemma(gemma_config).eval()
    lora_model = init_paligemma(gemma_config).eval()
    lora_model.load_state_dict(base_model.state_dict(), strict=True)

    # Wrap with LoRA
    lora_model = add_lora_to_gemma(lora_model, lora_config).eval()

    # Ensure both on same device
    device = next(base_model.parameters()).device
    lora_model.to(device)
    base_model.to(device)

    batch = _dummy_batch(base_model)

    # 1) Initial outputs should match (LoRA has zero effect initially)
    base_logits = base_model(**batch).logits
    lora_logits_initial = lora_model(**batch).logits
    max_diff_initial = (base_logits - lora_logits_initial).abs().max().item()
    print(f"[Step 1] Max diff (base vs LoRA-initial): {max_diff_initial}")
    assert torch.allclose(
        base_logits, lora_logits_initial, atol=1e-6
    ), "Initial outputs differ even though LoRA should be zero-effect."

    # 2) Manually perturb LoRA weights to force a non-zero effect
    delta_scale = 1e-3
    for name, param in lora_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # small but non-zero modification
            param.add_(delta_scale * torch.randn_like(param))

    lora_logits_perturbed = lora_model(**batch).logits
    max_diff_perturbed = (base_logits - lora_logits_perturbed).abs().max().item()
    print(f"[Step 2] Max diff (base vs LoRA-perturbed): {max_diff_perturbed}")
    assert max_diff_perturbed > 1e-5, "LoRA perturbation did not change outputs."

    # 3) Disable adapters and make sure we revert to base behavior
    lora_model.disable_adapter_layers()
    lora_logits_disabled = lora_model(**batch).logits
    max_diff_disabled = (base_logits - lora_logits_disabled).abs().max().item()
    print(f"[Step 3] Max diff (base vs LoRA-disabled): {max_diff_disabled}")
    assert torch.allclose(
        base_logits, lora_logits_disabled, atol=1e-6
    ), "Outputs differ when adapters are disabled."

if __name__ == "__main__":
    test_lora()
