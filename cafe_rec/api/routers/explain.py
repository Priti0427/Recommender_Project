from fastapi import APIRouter, HTTPException, Request

from ..schemas import ExplainRequest, ExplainResponse

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
def explain(request: Request, body: ExplainRequest) -> ExplainResponse:
    svc = request.app.state.service
    try:
        result = svc.explain(
            user_id=body.user_id,
            item_id=body.item_id,
            max_length=body.max_length,
            num_beams=body.num_beams,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ExplainResponse(
        user_id=body.user_id,
        item_id=body.item_id,
        score=result["score"],
        selected_aspects=result["selected_aspects"],
        aspect_weights=result["aspect_weights"],
        explanation=result["explanation"],
    )
