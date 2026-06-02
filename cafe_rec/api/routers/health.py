from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    svc = request.app.state.service
    return HealthResponse(
        status="ok",
        device=str(svc.device),
        num_users=svc.num_users,
        num_items=svc.num_items,
        num_aspects=svc.num_aspects,
        checkpoint=svc.settings.checkpoint_path,
    )
