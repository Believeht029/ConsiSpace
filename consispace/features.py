"""Frozen dual encoders and a versioned, weights-only feature cache."""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor, nn

from .config import ConsiSpaceConfig
from .geometry import camera_to_world, normalize_geometry


@dataclass
class SceneFeatures:
    visual: Tensor           # [T,Nv,Dv]
    spatial: Tensor          # [T,Ns,Ds]
    poses: Tensor            # [T,4,4], camera-to-world
    depth: Tensor            # [T,H,W]
    timestamps: Tensor       # [T], seconds
    frame_ids: Tensor        # [T], original observation indices
    query: Tensor            # [Dt], question-specific frozen text embedding

    def validate(self, config):
        length = self.visual.shape[0]
        if length == 0 or self.visual.ndim != 3 or self.spatial.ndim != 3:
            raise ValueError("visual/spatial must contain nonempty [T,N,D] tokens")
        if self.visual.shape[-1] != config.visual_dim or self.spatial.shape[-1] != config.spatial_dim:
            raise ValueError("Cached feature dimensions do not match the configuration")
        if self.query.shape != (config.text_dim,):
            raise ValueError("Query embedding dimension does not match configuration")
        if self.poses.shape != (length, 4, 4) or self.depth.ndim != 3:
            raise ValueError("Expected camera-to-world [T,4,4] poses and [T,H,W] depths")
        for name in ("spatial", "poses", "depth", "timestamps", "frame_ids"):
            if getattr(self, name).shape[0] != length:
                raise ValueError(f"Misaligned frame count in {name}")
        if self.timestamps.ndim != 1 or self.frame_ids.ndim != 1:
            raise ValueError("timestamps and frame_ids must be one-dimensional")
        if torch.any(self.timestamps[1:] < self.timestamps[:-1]):
            raise ValueError("Frames must be chronological")
        for name in ("visual", "spatial", "poses", "query", "timestamps"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"Non-finite {name}")
        if not torch.allclose(self.poses[:, 3], self.poses.new_tensor([0, 0, 0, 1]).expand(length, -1)):
            raise ValueError("Invalid homogeneous camera poses")
        valid_depth = torch.isfinite(self.depth) & (self.depth > 0)
        if not valid_depth.flatten(1).any(1).all():
            raise ValueError("Every frame needs at least one valid depth pixel")
        return self

    def to(self, device, dtype=torch.bfloat16):
        return SceneFeatures(**{name: value.to(device=device, dtype=(dtype if name in ("visual", "spatial", "query")
                                                                     else value.dtype))
                                for name, value in vars(self).items()})

    def subset(self, indices):
        return SceneFeatures(**{name: value if name == "query" else value[indices]
                                for name, value in vars(self).items()})

    def save(self, path, question: str, config: ConsiSpaceConfig, geometry_unit="native"):
        torch.save({"format_version": 1, "question": question, "siglip": config.siglip,
                    "vggt": config.vggt, "pose_convention": "camera_to_world", "geometry_unit": geometry_unit,
                    "features": {name: tensor.detach().cpu() for name, tensor in vars(self).items()}}, path)

    @classmethod
    def load(cls, path, question: str, config: ConsiSpaceConfig):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("pose_convention") != "camera_to_world":
            raise ValueError("Unsupported feature cache format or pose convention")
        if payload.get("question") != question:
            raise ValueError("Feature cache's query embedding belongs to a different question")
        if payload.get("siglip") != config.siglip or payload.get("vggt") != config.vggt:
            raise ValueError("Feature cache encoder identities differ from the configuration")
        if not config.normalize_scene_scale and payload.get("geometry_unit") != "meters":
            raise ValueError("Paper thresholds use meters: supply calibrated meters_per_unit during extraction, "
                             "or explicitly choose normalize_scene_scale=true for a scale-relative variant")
        return cls(**payload["features"]).validate(config)

    def normalized(self):
        poses, depth, _ = normalize_geometry(self.poses, self.depth)
        return SceneFeatures(self.visual, self.spatial, poses, depth, self.timestamps, self.frame_ids, self.query)


class FrozenDualEncoder(nn.Module):
    def __init__(self, config: ConsiSpaceConfig, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        from vggt.models.vggt import VGGT
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.siglip)
        self.semantic = AutoModel.from_pretrained(config.siglip, dtype=dtype).to(device)
        self.geometry_encoder = VGGT.from_pretrained(config.vggt).to(device)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent .train() must not turn frozen towers' dropout back on.
        return super().train(False)

    @torch.no_grad()
    def forward(self, frame_paths: list[str], question: str, timestamps=None):
        from PIL import Image
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        if not frame_paths:
            raise ValueError("At least one frame is required")
        parameter = next(self.semantic.parameters())
        images = []
        for path in frame_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(parameter.device, dtype=parameter.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        visual = self.semantic.vision_model(**inputs).last_hidden_state
        text = self.processor(text=[question], padding="max_length", truncation=True, return_tensors="pt")
        text = {k: v.to(parameter.device) for k, v in text.items()}
        query = self.semantic.get_text_features(**text)[0]
        frames = load_and_preprocess_images([str(Path(p)) for p in frame_paths]).to(parameter.device)[None]
        # Extract the whole scene jointly to maintain one VGGT gauge across paired views.
        with torch.autocast(device_type=parameter.device.type, dtype=parameter.dtype,
                            enabled=parameter.device.type == "cuda"):
            tokens, patch_start = self.geometry_encoder.aggregator(frames)
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            float_tokens = [token.float() for token in tokens]
            pose_encoding = self.geometry_encoder.camera_head(float_tokens)[-1]
            depth, _ = self.geometry_encoder.depth_head(float_tokens, images=frames.float(), patch_start_idx=patch_start)
            extrinsics, _ = pose_encoding_to_extri_intri(pose_encoding, frames.shape[-2:])
        length = len(frame_paths)
        if timestamps is None:
            # These are indices unless caller supplies seconds; metadata records this convention.
            timestamps = torch.arange(length, device=parameter.device, dtype=torch.float32)
        else:
            timestamps = torch.as_tensor(timestamps, device=parameter.device, dtype=torch.float32)
        return SceneFeatures(visual, tokens[-1][0, :, patch_start:].to(visual),
                             camera_to_world(extrinsics[0]), depth[0, ..., 0], timestamps,
                             torch.arange(length, device=parameter.device), query).validate(self.config)
