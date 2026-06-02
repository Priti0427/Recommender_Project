# L_sparse: soft-budget penalty on the EXPECTED number of active aspects.
from __future__ import annotations

import torch
import torch.nn.functional as F


def sparsity_loss(
    aspect_logits: torch.Tensor,
    target_min: int = 3,
    target_max: int = 5,
) -> torch.Tensor:
    """Drive the expected number of active aspects into ``[target_min, target_max]``.

    The selection probability for aspect i is ``sigmoid(logit_i)`` (the
    Gumbel-Sigmoid "on" probability against a 0 "off" logit). The expected
    number active per row is ``sum_i sigmoid(logit_i)``. We penalize the
    expected count when it goes ABOVE target_max (too dense -> explanation not
    sparse) and when it goes BELOW target_min (too terse -> collapses to 1 and
    the explanation under-describes the recommendation). The two-sided budget
    keeps the gate at ~3-5 active aspects.

    Why not the old ``mean(|logits|)``? That drove logits toward 0, i.e.
    sigmoid -> 0.5, i.e. ~HALF the aspects active regardless of lambda_sparse.
    Why two-sided (vs the earlier single ``relu(active - 5)``)? That allowed the
    gate to collapse to 1 aspect for free; the lower bound forces a richer 3-5.

    Args:
        aspect_logits: [B, A] raw (pre-Gumbel) selector logits.
        target_min: minimum desired active aspects per row.
        target_max: maximum desired active aspects per row.

    Returns:
        Scalar: mean per-row penalty for being outside [target_min, target_max].
    """
    probs = torch.sigmoid(aspect_logits)          # [B, A] = P(select)
    expected_active = probs.sum(dim=-1)           # [B]
    over = F.relu(expected_active - float(target_max))
    under = F.relu(float(target_min) - expected_active)
    return (over + under).mean()
