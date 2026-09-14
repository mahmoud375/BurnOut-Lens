"""
BurnOut Lens — FastAPI application factory.

Run locally:
    cd backend/
    uv run uvicorn app.main:app --reload --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import health, predict

# CORS origins — defaults to Vite dev server; override via env var for prod
# e.g. CORS_ORIGINS="https://burnoutlens.example.com"
_cors_origins_env = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app = FastAPI(
    title="BurnOut Lens API",
    description=(
        "Predicts employee burnout risk (0-10) from 9 work and lifestyle "
        "factors using an XGBoost model, with SHAP-based explanations of "
        "which features drove each prediction."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect bare root to Swagger UI."""
    return RedirectResponse(url="/docs")
