from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ConsiSpaceConfig:
    backbone: str = "Qwen/Qwen3-VL-8B-Instruct"
    siglip: str = "google/siglip2-base-patch16-512"
    vggt: str = "facebook/VGGT-1B"
    visual_dim: int = 768
    spatial_dim: int = 2048
    text_dim: int = 768
    memory_dim: int = 512
    memory_tokens: int = 64
    chunks: int = 4
    chunk_tokens: int = 8
    frame_tokens: int = 4
    top_chunks: int = 8
    top_frames: int = 8
    heads: int = 8
    inject_start: int = 8  # One-based layer numbering, injection after this layer.
    inject_every: int = 4
    # Tables 10-11 specify 0.15m / 15deg and 0.40m / 30deg.
    write_translation: float = 0.15
    write_angle_deg: float = 15.0
    fusion_radius: float = 0.40
    fusion_angle_deg: float = 30.0
    semantic_weight: float = 1.0
    direction_weight: float = 0.2
    normalize_scene_scale: bool = False
    use_faiss: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_length: int = 5120
    answer_reward_weight: float = 1.0
    metric_reward_weight: float = 1.0
    topology_reward_weight: float = 1.0
    ssrl_weight: float = 0.01
    view_fraction: float = 0.8
    evidence_keep: float = 0.9
    seed: int = 42

    def __post_init__(self):
        for name in ("memory_tokens", "chunks", "chunk_tokens", "frame_tokens", "top_chunks",
                     "top_frames", "heads", "inject_start", "inject_every", "lora_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.memory_dim % self.heads:
            raise ValueError("memory_dim must be divisible by heads")
        if self.max_length <= self.memory_tokens + 8:
            raise ValueError("max_length leaves no room for text")
        if not 0 < self.view_fraction <= 1 or not 0 < self.evidence_keep <= 1:
            raise ValueError("View sampling fractions must be in (0, 1]")
        for name in ("write_translation", "write_angle_deg", "fusion_radius", "fusion_angle_deg",
                     "answer_reward_weight", "metric_reward_weight", "topology_reward_weight", "ssrl_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
