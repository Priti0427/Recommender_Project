from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import RecommendItem, RecommendResponse

router = APIRouter(tags=["recommend"])


@router.get("/users/{user_id}/recommend", response_model=RecommendResponse)
def recommend(
    request: Request,
    user_id: str,
    k: int = Query(default=None, ge=1),
) -> RecommendResponse:
    svc = request.app.state.service
    if k is None:
        k = svc.settings.default_top_k
    if k > svc.settings.max_top_k:
        raise HTTPException(
            status_code=400,
            detail=f"k={k} exceeds max_top_k={svc.settings.max_top_k}",
        )
    try:
        u_idx = svc.user_to_idx(user_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pairs = svc.recommend(user_id, k)
    items = [
        RecommendItem(
            item_id=svc.idx_to_item(i),
            item_idx=int(i),
            score=float(s),
        )
        for i, s in pairs
    ]
    return RecommendResponse(
        user_id=user_id, user_idx=u_idx, k=k, items=items,
    )
