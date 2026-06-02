from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & meta
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    device: str
    num_users: int
    num_items: int
    num_aspects: int
    checkpoint: str


class AspectsResponse(BaseModel):
    num_aspects: int
    aspects: List[str]


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------
class RecommendItem(BaseModel):
    item_id: str = Field(..., description="Original item id (e.g. Amazon ASIN).")
    item_idx: int = Field(..., description="Internal integer index.")
    score: float


class RecommendResponse(BaseModel):
    user_id: str
    user_idx: int
    k: int
    items: List[RecommendItem]


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    user_id: str
    item_id: str
    max_length: Optional[int] = None
    num_beams: Optional[int] = None


class ExplainResponse(BaseModel):
    user_id: str
    item_id: str
    score: float
    selected_aspects: List[str]
    aspect_weights: List[float]
    explanation: str


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------
class CounterfactualRequest(BaseModel):
    user_id: str
    item_id: str
    num_candidates: Optional[int] = Field(
        default=None,
        description="How many sampled negatives to use for the rank-flip test. "
                    "Defaults to settings.cf_num_negatives.",
    )
    top_k: int = 10


class CounterfactualResponse(BaseModel):
    user_id: str
    item_id: str
    selected_aspects: List[str]
    factual_score: float
    counterfactual_score: float
    delta: float
    sufficiency: float
    comprehensiveness: float
    factual_rank: int
    counterfactual_rank: int
    flipped_out_of_top_k: bool
    top_k: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str
