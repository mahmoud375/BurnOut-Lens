"""
Core prediction logic: BurnoutRequest -> BurnoutResponse.

Kept as a plain module-level function.  No class, no state, no DI framework.
The model and explainer come from the module singleton in model_loader —
passing them in as arguments would add a parameter to every call site for
zero benefit in a single-model API.
"""
from __future__ import annotations

import pandas as pd

from src import config as ml_config
from src.preprocessing import clip_prediction, preprocess
from src.explain import explain_prediction

from app.core.config import score_to_level
from app.core import model_loader
from app.schemas.burnout import BurnoutRequest, BurnoutResponse, Contribution


def predict_burnout(request: BurnoutRequest) -> BurnoutResponse:
    """Run the full prediction + explanation pipeline for one employee record.

    Steps
    -----
    1. Convert the Pydantic request to a raw DataFrame (matching the CSV schema).
    2. Run ``preprocess()`` — applies clipping and seniority encoding.
    3. Predict with the XGBoost model, clip to [0, 10].
    4. Run SHAP explanation on the single preprocessed row.
    5. Map the score to a severity label.
    6. Return a fully-formed ``BurnoutResponse``.

    Parameters
    ----------
    request:
        A validated ``BurnoutRequest`` from the API layer.

    Returns
    -------
    BurnoutResponse
        Complete prediction result, ready to serialise as JSON.

    Raises
    ------
    RuntimeError
        If the model or explainer failed to load at startup.
    """
    if not model_loader.is_ready():
        raise RuntimeError(
            "Model is not loaded. Check server logs for startup errors."
        )

    # ── 1. Build a raw DataFrame matching the CSV schema ─────────────────────
    # preprocess() expects RAW_FEATURE_COLUMNS — the seniority_level string
    # is passed as-is; encode_seniority() inside preprocess() converts it.
    raw = pd.DataFrame([{
        "seniority_level":          request.seniority_level,
        "work_hours_per_week":      request.work_hours_per_week,
        "meetings_per_day":         request.meetings_per_day,
        "sleep_hours_per_night":    request.sleep_hours_per_night,
        "exercise_days_per_week":   request.exercise_days_per_week,
        "vacation_days_taken":      request.vacation_days_taken,
        "social_support_score":     request.social_support_score,
        "manager_support_score":    request.manager_support_score,
        "deadline_pressure_score":  request.deadline_pressure_score,
    }])

    # ── 2. Preprocess (clip outliers + encode seniority) ─────────────────────
    X = preprocess(raw, include_target=False)   # shape (1, 9), FINAL_FEATURE_COLUMNS

    # ── 3. Predict + clip ─────────────────────────────────────────────────────
    raw_pred = model_loader.model.predict(X)
    score = float(clip_prediction(raw_pred)[0])

    # ── 4. SHAP explanation ───────────────────────────────────────────────────
    shap_result = explain_prediction(model_loader.explainer, X)

    # ── 5. Level label ────────────────────────────────────────────────────────
    level = score_to_level(score)

    # ── 6. Assemble response ──────────────────────────────────────────────────
    contributions = [
        Contribution(
            feature=c["feature"],
            value=c["value"],
            shap_value=c["shap_value"],
            direction=c["direction"],
        )
        for c in shap_result["contributions"]
    ]

    return BurnoutResponse(
        burnout_score=score,
        burnout_level=level,
        base_value=shap_result["base_value"],
        contributions=contributions,
    )
