"""
Backend settings — loaded once at import time.

We use ``pydantic-settings`` for a single reason: the ``MODEL_PATH`` env var.
Without it, you'd need `os.getenv("MODEL_PATH") or str(config.MODEL_PATH)`
in several places.  With it, you get that in two lines plus free validation
and documentation.  Nothing else here warrants a class — log level is
left as a plain constant because nothing reads it dynamically yet.

Environment variables (all optional — defaults work out of the box):
    MODEL_PATH   Override the XGBoost model file location.
                 Default: the path computed by ml/src/config.py (relative
                 to the installed ml/ package, works in Docker too).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Import the ml config to get the default MODEL_PATH computed from the
# installed package location (not hardcoded to this machine).
from src import config as ml_config


# ---------------------------------------------------------------------------
# Burnout level thresholds
# ---------------------------------------------------------------------------
# Based on the training-set distribution (n=80,000):
#   mean=5.40, std=2.67, p25=3.50, p50=5.40, p75=7.40
#
# Chosen bands (inclusive lower, exclusive upper):
#   Low      [0.0, 3.0)  — clearly below average, ~19% of training data
#   Moderate [3.0, 5.5)  — around the lower half median, ~30%
#   High     [5.5, 7.5)  — above mean to p75, ~28%
#   Severe   [7.5, 10.0] — top ~18%, well above p75
#
# The 5.5 cut-off (not 5.0) avoids labelling every score at the exact mean
# as "High"; it gives Moderate a meaningful upper tail.
BURNOUT_LEVEL_LOW      = 3.0
BURNOUT_LEVEL_MODERATE = 5.5
BURNOUT_LEVEL_HIGH     = 7.5


def score_to_level(score: float) -> str:
    """Map a burnout score in [0, 10] to a human-readable severity label."""
    if score < BURNOUT_LEVEL_LOW:
        return "Low"
    if score < BURNOUT_LEVEL_MODERATE:
        return "Moderate"
    if score < BURNOUT_LEVEL_HIGH:
        return "High"
    return "Severe"


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Validated application settings — values overridable via environment vars."""

    model_config = SettingsConfigDict(env_prefix="BURNOUT_")
    # BURNOUT_MODEL_PATH env var overrides model_path

    model_path: Path = ml_config.MODEL_PATH
    """Absolute path to the trained XGBoost model JSON file."""


settings = Settings()
