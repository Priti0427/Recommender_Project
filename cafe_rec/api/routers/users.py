# Lightweight endpoint that returns a sample of trained user ids so the demo
# UI can populate a dropdown instead of forcing the visitor to remember 28-char
# Amazon ASINs.
from __future__ import annotations

import random

from fastapi import APIRouter, Query, Request

router = APIRouter(tags=["users"])


@router.get("/users/sample")
def sample_users(
    request: Request,
    n: int = Query(default=100, ge=1, le=200),
    seed: int = Query(default=0, ge=0),
):
    """Return up to `n` (user_idx, user_id, n_interactions) entries the model
    actually has embeddings for. Drawn from the top-200 most-active users in
    the in-range training set, shuffled deterministically with `seed`."""
    svc = request.app.state.service
    pool = list(getattr(svc, "_sample_users", []))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return {"users": pool[:n]}
