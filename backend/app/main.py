"""
BurnOut Lens — FastAPI application factory.

Run locally:
    cd backend/
    uv run uvicorn app.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import health, predict

app = FastAPI(
    title="BurnOut Lens API",
    description=(
        "Predicts employee burnout risk (0–10) from 9 work and lifestyle "
        "factors using an XGBoost model, with SHAP-based explanations of "
        "which features drove each prediction."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect bare root to Swagger UI."""
    return RedirectResponse(url="/docs")
