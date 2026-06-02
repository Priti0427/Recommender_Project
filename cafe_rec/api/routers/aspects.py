from fastapi import APIRouter, Request

from ..schemas import AspectsResponse

router = APIRouter(tags=["meta"])


@router.get("/aspects", response_model=AspectsResponse)
def list_aspects(request: Request) -> AspectsResponse:
    svc = request.app.state.service
    return AspectsResponse(num_aspects=svc.num_aspects, aspects=svc.aspects())
