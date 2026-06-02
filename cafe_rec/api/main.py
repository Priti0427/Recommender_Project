"""FastAPI entrypoint for CaFE-Rec.

Run from repo root:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routers import aspects, counterfactual, explain, health, recommend, users
from .service import CaFERecService
from .settings import get_settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger("cafe_rec.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("loading CaFE-Rec service: ckpt=%s", settings.checkpoint_path)
    service = CaFERecService(settings)
    logger.info(
        "service ready on %s | %d users, %d items, %d aspects",
        service.device, service.num_users, service.num_items, service.num_aspects,
    )
    app.state.service = service
    try:
        yield
    finally:
        app.state.service = None


app = FastAPI(
    title="CaFE-Rec API",
    description=(
        "Counterfactually Faithful Explanation Recommender — REST API. "
        "Endpoints: top-K recommendations, natural-language explanations, "
        "and per-pair counterfactual fidelity diagnostics."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(aspects.router)
app.include_router(users.router)
app.include_router(recommend.router)
app.include_router(explain.router)
app.include_router(counterfactual.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["meta"], include_in_schema=False)
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"name": "CaFE-Rec API", "docs": "/docs", "health": "/health"}
