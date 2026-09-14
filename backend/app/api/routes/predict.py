"""POST /predict — burnout prediction endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.burnout import BurnoutRequest, BurnoutResponse
from app.services.prediction_service import predict_burnout

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=BurnoutResponse,
    summary="Predict burnout score",
    description=(
        "Submit employee feature data and receive a burnout score (0-10), "
        "a severity label (Low / Moderate / High / Severe), and a SHAP-based "
        "explanation of which factors drove the prediction."
    ),
)
def predict(request: BurnoutRequest) -> BurnoutResponse:
    """Run the burnout prediction pipeline.

    Pydantic validates the request body automatically and returns 422 on bad
    input.  Unexpected runtime errors are caught here and returned as 500.
    """
    try:
        return predict_burnout(request)
    except RuntimeError as exc:
        # Model not loaded — surface clearly rather than a generic 500
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
