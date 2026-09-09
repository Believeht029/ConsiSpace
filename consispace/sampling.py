"""Paired temporal/retrieval contexts from one jointly reconstructed scene."""
import math
import torch


def sample_views(scene, config, anchor_frame_ids=(), generator=None):
    length = len(scene.visual)
    if length < 2:
        raise ValueError("UC-SSRL needs at least two frames")
    anchors = set(int(value) for value in anchor_frame_ids)
    known = set(scene.frame_ids.tolist())
    if not anchors <= known:
        raise ValueError("Unknown anchor frame IDs")
    width = max(2, math.ceil(length * config.view_fraction))
    width = min(width, length)

    def sample():
        start = int(torch.randint(length - width + 1, (), generator=generator))
        indices = torch.arange(start, start + width)
        keep = torch.rand(width, generator=generator) < config.evidence_keep
        keep[0] = True
        selected = set(indices[keep].tolist())
        # Required evidence anchors survive both perturbations.
        selected.update(i for i, value in enumerate(scene.frame_ids.tolist()) if value in anchors)
        return sorted(selected)

    first, second = sample(), sample()
    for _ in range(16):
        if first != second:
            break
        second = sample()
    if first == second:
        removable = [i for i in second if int(scene.frame_ids[i]) not in anchors]
        if len(second) > 1 and removable:
            second.remove(removable[-1])
        else:
            additions = [i for i in range(length) if i not in second]
            if not additions:
                raise ValueError("Sampling constraints forbid two distinct evidence sets")
            second.append(additions[0])
            second.sort()
    device = scene.visual.device
    return (scene.subset(torch.tensor(first, device=device, dtype=torch.long)),
            scene.subset(torch.tensor(second, device=device, dtype=torch.long)))
