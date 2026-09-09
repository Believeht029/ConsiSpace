# Adapted from QwenLM/Qwen3-VL, qwen-vl-finetune/qwenvl/train/train_qwen.py.
# Adopted from https://github.com/lm-sys/FastChat and tatsu-lab/stanford_alpaca.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modifications: select Qwen3-VL Dense only; keep its text decoder and LM head;
# restrict LoRA to language layers; expose native DeepStack input arguments.
"""Minimal Qwen3-VL upstream adaptation; no demos or Qwen2/MoE branches."""
from torch import nn


class QwenTextBackbone(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.decoder = pretrained.model.language_model
        self.lm_head = pretrained.lm_head
        self.config = pretrained.config.text_config

    def forward(self, **kwargs):
        return self.decoder(**kwargs)


def load_qwen_policy(config, attention, dtype):
    from transformers import Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    pretrained = Qwen3VLForConditionalGeneration.from_pretrained(
        config.backbone, dtype=dtype, attn_implementation=attention)
    backbone = QwenTextBackbone(pretrained)
    del pretrained
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    # Full names prevent accidentally adapting vision projections or the LM head.
    targets = [name for name, module in backbone.named_modules()
               if name.startswith("decoder.layers.") and isinstance(module, nn.Linear)
               and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"}]
    lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                             lora_dropout=config.lora_dropout, target_modules=targets, bias="none")
    # Generic PeftModel preserves the decoder's native DeepStack arguments.
    return get_peft_model(backbone, lora_config)
