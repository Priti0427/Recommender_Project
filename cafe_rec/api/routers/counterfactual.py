from fastapi import APIRouter, HTTPException, Request

from ..schemas import CounterfactualRequest, CounterfactualResponse

router = APIRouter(tags=["explain"])


@router.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual(
    request: Request, body: CounterfactualRequest,
) -> CounterfactualResponse:
    svc = request.app.state.service
    n = body.num_candidates or svc.settings.cf_num_negatives
    try:
        result = svc.counterfactual(
            user_id=body.user_id,
            item_id=body.item_id,
            num_candidates=int(n),
            top_k=int(body.top_k),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return CounterfactualResponse(
        user_id=body.user_id,
        item_id=body.item_id,
        **result,
    )
