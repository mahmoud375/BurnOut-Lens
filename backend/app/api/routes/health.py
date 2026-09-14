"""GET /health — liveness and readiness check."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core import model_loader

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health() -> HealthResponse:
    """Return service liveness status.

    ``model_loaded`` is ``true`` only if the XGBoost model *and* the SHAP
    explainer both loaded successfully at startup.  A load failure leaves
    them ``None`` (server stays up, but predictions will 503).
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_ready(),
    )
