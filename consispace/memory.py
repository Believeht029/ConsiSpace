"""Equations (3)-(12): scene-local dual memory, writing, fusion and retrieval."""
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ConsiSpaceConfig
from .geometry import angle_degrees, geometry_descriptor, position_direction


def pool_tokens(tokens: Tensor, count: int) -> Tensor:
    return F.adaptive_avg_pool1d(tokens.transpose(-1, -2), count).transpose(-1, -2)


class CrossAttention(nn.Module):
    """Explicit Linear calls allow ZeRO-3 to gather each projection's parameters."""
    def __init__(self, dimension, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dimension, dimension)
        self.k_proj = nn.Linear(dimension, dimension)
        self.v_proj = nn.Linear(dimension, dimension)
        self.out_proj = nn.Linear(dimension, dimension)

    def forward(self, query, key, value, need_weights=False):
        def split(tensor):
            return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, -1).transpose(1, 2)
        result = F.scaled_dot_product_attention(split(self.q_proj(query)), split(self.k_proj(key)),
                                                split(self.v_proj(value)), dropout_p=0.0)
        result = result.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(result), None


@dataclass
class MemoryEntry:
    evidence: Tensor
    pose: Tensor
    depth: Tensor
    timestamp: float
    count: int = 1
    source_ids: list[int] = field(default_factory=list)


@dataclass
class MemoryState:
    """A state belongs to one scene/view; never shared between DataLoader samples."""
    entries: list[MemoryEntry] = field(default_factory=list)
    previous_pose: Tensor | None = None
    cached_keys: Tensor | None = None
    writes: int = 0
    fusions: int = 0
    skipped: int = 0


class GeometryConsistentMemory(nn.Module):
    def __init__(self, config: ConsiSpaceConfig):
        super().__init__()
        self.config = config
        d = config.memory_dim
        self.visual_projection = nn.Linear(config.visual_dim, d)
        self.spatial_projection = nn.Linear(config.spatial_dim, d)
        self.pack_projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
        self.key = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(config.text_dim, d)
        self.topology_query = nn.Linear(d, d, bias=False)
        self.metric_query = nn.Linear(d, d, bias=False)
        self.direction = nn.Linear(3, d, bias=False)
        self.geometry = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.LayerNorm(d))
        self.topology_attention = CrossAttention(d, config.heads)
        self.metric_attention = CrossAttention(d, config.heads)
        self.read_queries = nn.Parameter(torch.randn(1, config.memory_tokens, d) * 0.02)
        self.read_attention = CrossAttention(d, config.heads)
        self.read_norm = nn.LayerNorm(d)

    def forward(self, visual, spatial, poses, depth, timestamps, query, frame_ids=None):
        # Enter through nn.Module.__call__ so distributed hooks gather read_queries.
        state = self.build(visual, spatial, poses, depth, timestamps, frame_ids)
        return self.read(state, query)

    def pack(self, visual: Tensor, spatial: Tensor) -> Tensor:
        # Patch counts need not match: concatenate along token axis after projecting.
        tokens = torch.cat((self.visual_projection(visual), self.spatial_projection(spatial)), dim=-2)
        return self.pack_projection(pool_tokens(tokens, self.config.frame_tokens))

    def write(self, state: MemoryState, evidence: Tensor, pose: Tensor, depth: Tensor,
              timestamp: float, frame_id: int):
        config = self.config
        position, direction = position_direction(pose)
        previous = state.previous_pose
        # Equation (4) uses the preceding OBSERVED frame, including skipped writes.
        state.previous_pose = pose.detach()
        if previous is not None:
            prev_position, prev_direction = position_direction(previous)
            moved = torch.linalg.vector_norm(position - prev_position) > config.write_translation
            rotated = angle_degrees(direction, prev_direction) > config.write_angle_deg
            if not bool(moved | rotated):
                state.skipped += 1
                return
        state.writes += 1
        neighbors = []
        for i, entry in enumerate(state.entries):
            other_position, other_direction = position_direction(entry.pose)
            if bool((torch.linalg.vector_norm(position - other_position) < config.fusion_radius)
                    & (angle_degrees(direction, other_direction) < config.fusion_angle_deg)):
                neighbors.append(i)
        if neighbors:
            with torch.no_grad():
                incoming = self.key(evidence.mean(0))
                keys = torch.stack([self.key(state.entries[i].evidence.mean(0)) for i in neighbors])
                index = neighbors[int(F.cosine_similarity(incoming[None], keys, dim=-1).argmax())]
            target = state.entries[index]
            # Count-weighted fusion is an explicit implementation choice, not specified in Eq. (7).
            target.evidence = (target.evidence * target.count + evidence) / (target.count + 1)
            target.count += 1
            target.source_ids.append(frame_id)
            # Keep one observed pose/depth pair; never average unregistered depth maps.
            state.fusions += 1
        else:
            state.entries.append(MemoryEntry(evidence, pose.detach(), depth.detach(), timestamp,
                                             source_ids=[frame_id]))
        state.cached_keys = None

    def build(self, visual: Tensor, spatial: Tensor, poses: Tensor, depth: Tensor,
              timestamps: Tensor, frame_ids: Tensor | None = None) -> MemoryState:
        if not len(visual):
            raise ValueError("Cannot construct memory from an empty view")
        state = MemoryState()
        evidence = self.pack(visual, spatial)
        ids = list(range(len(visual))) if frame_ids is None else frame_ids.tolist()
        for i in range(len(visual)):
            self.write(state, evidence[i], poses[i], depth[i], float(timestamps[i]), ids[i])
        return state

    def _topk(self, keys: Tensor, query: Tensor, k: int) -> Tensor:
        k = min(k, len(keys))
        if self.config.use_faiss:
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("use_faiss requires installing consispace[retrieval]") from exc
            index = faiss.IndexFlatIP(keys.shape[-1])
            index.add(F.normalize(keys.detach().float(), dim=-1).cpu().numpy())
            _, indices = index.search(F.normalize(query.detach().float(), dim=-1)[None].cpu().numpy(), k)
            return torch.as_tensor(indices[0], device=keys.device)
        return F.cosine_similarity(query[None], keys, dim=-1).topk(k).indices

    def read(self, state: MemoryState, text_embedding: Tensor):
        """Return [64,d] read tokens plus auditable retrieval provenance."""
        if not state.entries:
            raise ValueError("Cannot retrieve from empty memory")
        config = self.config
        entries = state.entries
        evidence = torch.stack([entry.evidence for entry in entries])
        if state.cached_keys is None or torch.is_grad_enabled():
            frame_keys = self.key(evidence.mean(1))
            if not torch.is_grad_enabled():
                state.cached_keys = frame_keys
        else:
            frame_keys = state.cached_keys
        query = self.query(text_embedding.reshape(-1))
        topology_query, metric_query = self.topology_query(query), self.metric_query(query)
        chunks = [chunk for chunk in torch.tensor_split(
            torch.arange(len(entries), device=evidence.device), min(config.chunks, len(entries))) if chunk.numel()]
        chunk_keys = torch.stack([frame_keys[chunk].mean(0) for chunk in chunks])
        chunk_indices = self._topk(chunk_keys, query, config.top_chunks)
        candidates = torch.cat([chunks[int(index)] for index in chunk_indices])
        poses = torch.stack([entry.pose for entry in entries])
        directions = position_direction(poses)[1].to(evidence)
        direction_keys = self.direction(directions)
        # Equation (9): soft geometric down-weighting, not an undocumented hard mask.
        semantic_scores = F.cosine_similarity(query[None], frame_keys[candidates], dim=-1)
        geometry_scores = F.cosine_similarity(topology_query[None], direction_keys[candidates], dim=-1)
        scores = config.semantic_weight * semantic_scores + config.direction_weight * geometry_scores
        selected = candidates[scores.topk(min(config.top_frames, len(candidates))).indices]
        # Retain chronology after selection so presentation order is stable.
        selected = selected.sort().values
        descriptors = geometry_descriptor(poses, torch.stack([entry.depth for entry in entries]))
        geometry_tokens = self.geometry(descriptors.to(evidence))
        topo, _ = self.topology_attention(topology_query[None, None], evidence.flatten(0, 1)[None],
                                           evidence.flatten(0, 1)[None], need_weights=False)
        metric, _ = self.metric_attention(metric_query[None, None], geometry_tokens[None],
                                          geometry_tokens[None], need_weights=False)
        # Soft weights keep scoring projections learnable during optional alignment;
        # hard top-k alone would give key/direction projections no gradient.
        chunk_scores = F.cosine_similarity(query[None], chunk_keys[chunk_indices], dim=-1)
        chunk_weights = torch.softmax(chunk_scores.float(), dim=0).to(evidence) * len(chunk_indices)
        chunk_values = torch.cat([pool_tokens(evidence[chunks[int(i)]].flatten(0, 1), config.chunk_tokens) * weight
                                  for i, weight in zip(chunk_indices, chunk_weights)])
        selected_scores = (config.semantic_weight * F.cosine_similarity(query[None], frame_keys[selected], dim=-1)
                           + config.direction_weight * F.cosine_similarity(topology_query[None], direction_keys[selected], dim=-1))
        selected_weights = torch.softmax(selected_scores.float(), dim=0).to(evidence) * len(selected)
        selected_evidence = evidence[selected] * selected_weights[:, None, None]
        context = torch.cat((query[None], topo[0], metric[0], chunk_values,
                             selected_evidence.flatten(0, 1), geometry_tokens[selected] * selected_weights[:, None]))
        # Eq. (12)'s variable-length context is resampled to Appendix B's fixed 64-token budget.
        result, _ = self.read_attention(self.read_queries + query[None, None], context[None],
                                       context[None], need_weights=False)
        result = self.read_norm(result + self.read_queries)[0]
        metadata = {
            "entries": len(entries), "writes": state.writes, "skipped": state.skipped,
            "fusions": state.fusions, "selected_entries": selected.tolist(),
            "source_frame_ids": [entries[int(i)].source_ids for i in selected],
            "timestamps": [entries[int(i)].timestamp for i in selected],
            "selected_chunks": chunk_indices.tolist(),
        }
        return result, metadata
