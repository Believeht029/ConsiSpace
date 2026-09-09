"""Pose convention: camera-to-world, +Z forward; calculations always in fp32."""
import torch
from torch import Tensor
import torch.nn.functional as F


def camera_to_world(extrinsics: Tensor) -> Tensor:
    """Invert VGGT's world-to-camera [T,3,4] extrinsics."""
    rotation, translation = extrinsics.float()[..., :3], extrinsics.float()[..., 3]
    result = torch.eye(4, device=extrinsics.device).expand(*extrinsics.shape[:-2], 4, 4).clone()
    result[..., :3, :3] = rotation.transpose(-1, -2)
    result[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    return result


def position_direction(pose: Tensor):
    return pose[..., :3, 3].float(), F.normalize(pose[..., :3, 2].float(), dim=-1)


def angle_degrees(a: Tensor, b: Tensor) -> Tensor:
    cosine = (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1)
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def normalize_geometry(poses: Tensor, depth: Tensor):
    """Normalize one *whole scene* before sampling windows, never each view separately."""
    valid = depth[torch.isfinite(depth) & (depth > 0)].float()
    if valid.numel() == 0:
        raise ValueError("Scene contains no valid positive depth")
    scale = valid.median().clamp_min(1e-6)
    poses = poses.float().clone()
    poses[..., :3, 3] = (poses[..., :3, 3] - poses[0, :3, 3].clone()) / scale
    return poses, depth.float() / scale, scale


def geometry_descriptor(poses: Tensor, depth: Tensor) -> Tensor:
    """12 pose values + 4 depth statistics, invalid pixels excluded (16 channels)."""
    descriptors = []
    for pose, frame_depth in zip(poses, depth):
        valid = frame_depth[torch.isfinite(frame_depth) & (frame_depth > 0)].float()
        if not valid.numel():
            raise ValueError("Memory entry has no valid depth")
        stats = torch.stack((valid.mean(), valid.std(unbiased=False), valid.amin(), valid.amax()))
        descriptors.append(torch.cat((pose[:3].float().reshape(-1), stats)))
    return torch.stack(descriptors)
