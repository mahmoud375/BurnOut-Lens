"""
Module-level singleton: loads the XGBoost model and builds the SHAP explainer
exactly once when this module is first imported (i.e. at server startup).

Why a module singleton instead of FastAPI lifespan/dependency injection?
Because this is one model, one explainer, no swapping at runtime. A module
global that's populated at import time is the simplest correct solution —
zero ceremony, zero indirection, trivially testable by checking the booleans.

Timings on the real model (measured empirically):
  model.load_model()    ~110ms
  get_shap_explainer()  ~88ms
Both are one-time costs paid during startup, not per-request.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # must be before any matplotlib import

import xgboost as xgb
from src import explain as _explain

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — populated at import time
# ---------------------------------------------------------------------------

model: xgb.XGBRegressor | None = None
explainer: _explain.shap.TreeExplainer | None = None

try:
    logger.info("Loading model from %s …", settings.model_path)
    _m = xgb.XGBRegressor()
    _m.load_model(settings.model_path)

    logger.info("Building SHAP TreeExplainer …")
    _ex = _explain.get_shap_explainer(_m)

    model = _m
    explainer = _ex
    logger.info("Model and explainer ready.")

except Exception:  # broad catch intentional: any model-load  # pragma: no cover
    logger.exception("Failed to load model")
    # model/explainer remain None — /health will surface this clearly
    # rather than crashing the whole server at import time.


def is_ready() -> bool:
    """Return True if both the model and explainer are loaded successfully."""
    return model is not None and explainer is not None
