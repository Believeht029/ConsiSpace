"""UC-SSRL equations (14)-(17), evaluated on a shared finite answer support."""
import torch
from torch import Tensor
import torch.nn.functional as F

from .config import ConsiSpaceConfig


def kl(log_p: Tensor, log_q: Tensor) -> Tensor:
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def relation_distribution(log_probs: Tensor, groups: Tensor) -> Tensor:
    if groups.shape != log_probs.shape or groups.dtype != torch.long or bool((groups < 0).any()):
        raise ValueError("Each candidate needs one nonnegative integer relation group")
    _, groups = torch.unique(groups, sorted=True, return_inverse=True)
    # logsumexp avoids underflow for uncommon relations in bf16-trained policies.
    return torch.stack([torch.logsumexp(log_probs[groups == group], dim=0)
                        for group in range(int(groups.max()) + 1)])


def consistency_loss(scores_i: Tensor, scores_j: Tensor, config: ConsiSpaceConfig,
                     metric_values: Tensor | None = None, relation_groups: Tensor | None = None):
    """Exact expectation on candidate support; no ground-truth answers are read.

    pi(y|o,q) is normalized over the SAME candidate strings in both views.
    For metric candidates, sum pi_i(a) pi_j(b) |d(a)-d(b)| is the exact
    policy-gradient expectation of Eq. (15), without Monte Carlo variance.
    The differentiable KL terms retain their distribution-dependent gradients.
    This finite-support policy is an explicit engineering choice; the paper does
    not specify a rollout algorithm, answer-distribution estimator, or baseline.
    """
    if scores_i.ndim != 1 or scores_i.shape != scores_j.shape or scores_i.numel() < 2:
        raise ValueError("UC-SSRL requires at least two shared answer candidates")
    log_i, log_j = F.log_softmax(scores_i.float(), dim=-1), F.log_softmax(scores_j.float(), dim=-1)
    answer = -config.answer_reward_weight * kl(log_i, log_j)
    metric = answer.new_zeros(())
    topology = answer.new_zeros(())
    if metric_values is not None:
        if metric_values.shape != log_i.shape or not torch.isfinite(metric_values).all():
            raise ValueError("Each metric candidate needs a finite value in the SAME unit")
        distances = (metric_values.float()[:, None] - metric_values.float()[None, :]).abs()
        metric = -config.metric_reward_weight * (log_i.exp()[:, None] * log_j.exp()[None, :] * distances).sum()
    if relation_groups is not None:
        topo_i = relation_distribution(log_i, relation_groups)
        topo_j = relation_distribution(log_j, relation_groups)
        topology = -config.topology_reward_weight * (kl(topo_i, topo_j) + kl(topo_j, topo_i))
    reward = answer + metric + topology
    return {"loss": -config.ssrl_weight * reward, "answer_reward": answer.detach(),
            "metric_reward": metric.detach(), "topology_reward": topology.detach(),
            "reward": reward.detach()}
