"""Qwen3-VL text decoder + native DeepStack injection, without its unused ViT."""
from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F

from .config import ConsiSpaceConfig
from .memory import GeometryConsistentMemory
from .rewards import consistency_loss
from .sampling import sample_views
from .qwen import load_qwen_policy


class ConsiSpace(nn.Module):
    def __init__(self, config: ConsiSpaceConfig, attention="sdpa", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoTokenizer
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone, padding_side="right")
        self.policy = load_qwen_policy(config, attention, dtype)
        backbone = self.backbone
        self.memory = GeometryConsistentMemory(config)
        self.aligner = nn.Linear(config.memory_dim, backbone.config.hidden_size)
        self.stage = "uninitialized"
        self.bridge_ready = False
        self.sft_ready = False
        self.injection_layers = list(range(config.inject_start - 1, backbone.config.num_hidden_layers,
                                           config.inject_every))
        if not self.injection_layers:
            raise ValueError("Injection start is beyond the decoder's last layer")
        self.memory.to(dtype=dtype)
        self.aligner.to(dtype=dtype)

    @property
    def backbone(self):
        return self.policy.get_base_model()

    def set_stage(self, stage):
        if stage not in {"align", "sft", "ssrl", "predict"}:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in {"sft", "ssrl", "predict"} and not self.bridge_ready:
            raise ValueError("Load a trained bridge checkpoint first; frozen random aligners are not usable")
        if stage in {"ssrl", "predict"} and not self.sft_ready:
            raise ValueError("UC-SSRL/prediction requires a completed SFT checkpoint")
        self.requires_grad_(False)
        if stage == "align":
            self.sft_ready = False  # Changing the bridge invalidates prior SFT readiness.
            self.memory.requires_grad_(True)
            self.aligner.requires_grad_(True)
        elif stage in {"sft", "ssrl"}:
            for name, parameter in self.policy.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
        self.stage = stage
        self.train(stage != "predict")

    def enable_gradient_checkpointing(self):
        # Native decoder's checkpointing passes DeepStack tensors explicitly through
        # the graph: no mutable hook state shared between the paired forwards.
        self.backbone.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def read_scene(self, scene):
        return self.memory(scene.visual, scene.spatial, scene.poses, scene.depth,
                           scene.timestamps, scene.query, scene.frame_ids)

    def _decode(self, prompt_ids, answer_ids, read_tokens):
        """[completed user message, M memory slots, assistant prefix + answer]."""
        device = self.aligner.weight.device
        prompt_ids = torch.as_tensor(prompt_ids, device=device, dtype=torch.long)
        answer_ids = torch.as_tensor(answer_ids, device=device, dtype=torch.long)
        if prompt_ids.ndim != 1 or answer_ids.ndim != 1 or not answer_ids.numel():
            raise ValueError("Expected 1D prompt and nonempty continuation")
        total = prompt_ids.numel() + self.config.memory_tokens + answer_ids.numel()
        if total > self.config.max_length:
            raise ValueError(f"Sequence length {total} exceeds {self.config.max_length}; truncate during data preparation")
        embedding = self.backbone.decoder.get_input_embeddings()
        read_tokens = read_tokens.to(device=device, dtype=self.aligner.weight.dtype)
        projected = self.aligner(read_tokens).to(embedding.weight.dtype)
        inputs = torch.cat((embedding(prompt_ids), projected, embedding(answer_ids)))[None]
        mask = torch.zeros((1, total), device=device, dtype=torch.bool)
        mask[:, len(prompt_ids):len(prompt_ids) + len(projected)] = True
        # DeepStack adds features after each decoder layer. Zeros leave other layers unchanged.
        deepstack = [projected if i in self.injection_layers else torch.zeros_like(projected)
                     for i in range(self.injection_layers[-1] + 1)]
        positions = torch.arange(total, device=device)[None, None].expand(3, 1, -1)
        outputs = self.policy(inputs_embeds=inputs, attention_mask=torch.ones((1, total), device=device, dtype=torch.long),
                              position_ids=positions, visual_pos_masks=mask,
                              deepstack_visual_embeds=deepstack, use_cache=False, return_dict=True)
        # Only materialize continuation logits, not 64 memory slots x vocabulary.
        start = len(prompt_ids) + len(projected) - 1
        hidden = outputs.last_hidden_state[:, start:-1]
        return self.backbone.lm_head(hidden).float()[0]

    def answer_score(self, prompt_ids, answer_ids, answer_start, memory):
        logits = self._decode(prompt_ids, answer_ids, memory)
        target = torch.as_tensor(answer_ids, device=logits.device, dtype=torch.long)
        if not 0 <= answer_start < len(target):
            raise ValueError("No supervised answer tokens remain")
        log_probs = F.log_softmax(logits[answer_start:], dim=-1)
        selected = log_probs.gather(-1, target[answer_start:, None]).squeeze(-1)
        return selected.sum(), -selected.mean()

    def forward(self, scene, prompt_ids, answer_ids=None, answer_start=0,
                candidate_ids=None, metric_values=None, relation_groups=None,
                anchor_frame_ids=(), view_invariant=False):
        if self.stage == "ssrl":
            if not view_invariant:
                raise ValueError("UC-SSRL records must declare view_invariant=true and preserve the question's reference frame")
            if not candidate_ids or len(candidate_ids) < 2:
                raise ValueError("UC-SSRL requires shared candidate answers")
            first, second = sample_views(scene, self.config, anchor_frame_ids)
            memories = [self.read_scene(view)[0] for view in (first, second)]
            scores = [torch.stack([self.answer_score(prompt_ids, ids, answer_start, memory)[0]
                                   for ids in candidate_ids]) for memory in memories]
            device = scores[0].device
            values = None if metric_values is None else torch.as_tensor(metric_values, device=device)
            groups = None if relation_groups is None else torch.as_tensor(relation_groups, device=device, dtype=torch.long)
            return consistency_loss(*scores, self.config, metric_values=values, relation_groups=groups)
        if self.stage not in {"align", "sft"}:
            raise RuntimeError("Select align or sft stage before supervised forward")
        memory, metadata = self.read_scene(scene)
        if answer_ids is None:
            raise ValueError("SFT requires an answer")
        _, loss = self.answer_score(prompt_ids, answer_ids, answer_start, memory)
        return {"loss": loss, "retrieval": metadata}

    @torch.no_grad()
    def generate_answer(self, scene, prompt_ids, assistant_prefix, max_new_tokens=128, temperature=0.0):
        if self.stage != "predict":
            raise RuntimeError("Load a completed checkpoint and select predict stage")
        if max_new_tokens <= 0 or temperature < 0:
            raise ValueError("Invalid decoding parameters")
        memory, metadata = self.read_scene(scene)
        continuation = list(assistant_prefix)
        if not continuation:
            raise ValueError("Assistant prefix must not be empty")
        limit = self.config.max_length - len(prompt_ids) - self.config.memory_tokens - len(continuation)
        if limit < 1:
            raise ValueError("Prompt leaves no room for generation")
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        generated = []
        # Full-context decoding is conservative: native KV-cache paths do not carry
        # arbitrary memory masks. No unsupported custom generate kwargs or hooks.
        for _ in range(min(max_new_tokens, limit)):
            # Append a dummy target solely to expose the logit following the prefix.
            logits = self._decode(prompt_ids, continuation + [self.tokenizer.eos_token_id], memory)[-1]
            if temperature > 0:
                token = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
            else:
                token = int(logits.argmax())
            if token in stop_ids:
                break
            generated.append(token)
            continuation.append(token)
        return self.tokenizer.decode(generated, skip_special_tokens=True), metadata

    def save_checkpoint(self, directory, gathered_state=None):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.state_dict() if gathered_state is None else gathered_state
        policy_state = {k[len("policy."):]: v for k, v in state.items() if k.startswith("policy.")}
        lora = get_peft_model_state_dict(self.policy, state_dict=policy_state)
        bridge = {k: v for k, v in state.items() if k.startswith(("memory.", "aligner."))}
        save_file({k: v.detach().cpu().contiguous() for k, v in bridge.items()}, directory / "bridge.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in lora.items()}, directory / "lora.safetensors")
        self.config.save(directory / "config.json")
        self.tokenizer.save_pretrained(directory / "tokenizer")
        (directory / "stage.json").write_text(json.dumps({
            "stage": self.stage, "bridge_ready": self.bridge_ready or self.stage == "align",
            "sft_ready": self.sft_ready or self.stage in {"sft", "ssrl"}}, indent=2) + "\n")

    def load_checkpoint(self, directory):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        directory = Path(directory)
        saved_config = ConsiSpaceConfig.load(directory / "config.json")
        if saved_config != self.config:
            raise ValueError("Checkpoint configuration differs; construct the model with its config.json")
        bridge = load_file(directory / "bridge.safetensors")
        for name, module in (("memory", self.memory), ("aligner", self.aligner)):
            module.load_state_dict({k[len(name) + 1:]: v for k, v in bridge.items() if k.startswith(name + ".")}, strict=True)
        lora = load_file(directory / "lora.safetensors")
        result = set_peft_model_state_dict(self.policy, lora)
        missing_adapters = [key for key in result.missing_keys if "lora_" in key]
        if missing_adapters or result.unexpected_keys:
            raise ValueError(f"Invalid LoRA checkpoint: missing={missing_adapters}, unexpected={result.unexpected_keys}")
        stage = json.loads((directory / "stage.json").read_text())
        self.bridge_ready, self.sft_ready = stage["bridge_ready"], stage["sft_ready"]
