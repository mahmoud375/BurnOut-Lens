"""
Central configuration for the ML pipeline: paths, feature lists, and hyperparameters.

Single source of truth so preprocessing, training, and inference never drift apart.
Import this module everywhere rather than hard-coding strings or magic numbers.
"""

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent  # points to ml/

RAW_DATA_PATH: Final[Path] = BASE_DIR / "data" / "raw" / "mental_health_burnout_tech_2026.csv"
PROCESSED_DIR: Final[Path] = BASE_DIR / "data" / "processed"
MODELS_DIR: Final[Path] = BASE_DIR / "models"

#: Canonical path for the serialised XGBoost model consumed by the backend.
MODEL_PATH: Final[Path] = MODELS_DIR / "xgb_burnout_model.json"

#: Path where the pre-processed training dataframe is saved (optional artefact).
PROCESSED_FEATURES_PATH: Final[Path] = PROCESSED_DIR / "features_processed.parquet"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL: Final[str] = "burnout_score"
TARGET_MIN: Final[float] = 0.0
TARGET_MAX: Final[float] = 10.0

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Ordinal feature: has a real, meaningful order (Junior -> Principal).
# Values map directly to the integer codes stored in the model.
SENIORITY_ORDER: Final[dict[str, int]] = {
    "Junior": 0,
    "Mid": 1,
    "Senior": 2,
    "Lead": 3,
    "Manager": 4,
    "Principal": 5,
}

# Raw columns pulled from the source dataset before any encoding.
# This is the schema the CSV (and the API request body) must satisfy.
RAW_FEATURE_COLUMNS: Final[list[str]] = [
    "seniority_level",          # ordinal string — encoded to int before training
    "work_hours_per_week",
    "meetings_per_day",
    "sleep_hours_per_night",
    "exercise_days_per_week",
    "vacation_days_taken",
    "social_support_score",
    "manager_support_score",
    "deadline_pressure_score",
]

# Final feature columns the model is trained on (after encoding).
# This is what the API must produce as input to model.predict().
# ORDER MATTERS — must match the column order seen during training.
FINAL_FEATURE_COLUMNS: Final[list[str]] = [
    "work_hours_per_week",
    "meetings_per_day",
    "sleep_hours_per_night",
    "exercise_days_per_week",
    "vacation_days_taken",
    "social_support_score",
    "manager_support_score",
    "deadline_pressure_score",
    "seniority_level_encoded",  # replaces seniority_level after ordinal encoding
]

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------
# Plausible real-world bounds per *numeric* feature (used to clip physically
# impossible values BEFORE they reach the model — both at training and at
# inference time).
# Note: seniority_level_encoded is excluded here because it is validated
# against SENIORITY_ORDER before any clipping can apply.
FEATURE_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "work_hours_per_week":    (20.0, 90.0),
    "meetings_per_day":        (0.0, 15.0),
    "sleep_hours_per_night":   (2.0, 12.0),
    "exercise_days_per_week":  (0.0,  7.0),
    "vacation_days_taken":     (0.0, 30.0),
    "social_support_score":    (1.0, 10.0),
    "manager_support_score":   (1.0, 10.0),
    "deadline_pressure_score": (1.0, 10.0),
}

# ---------------------------------------------------------------------------
# Train / test split & model hyperparameters
# ---------------------------------------------------------------------------
TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42

XGB_PARAMS: Final[dict] = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,           # reduces over-fitting on larger datasets
    "colsample_bytree": 0.8,    # reduces over-fitting on larger datasets
    "random_state": RANDOM_STATE,
    "tree_method": "hist",      # faster training; identical results to "exact" for tabular data
}

# ---------------------------------------------------------------------------
# Public API of this module
# ---------------------------------------------------------------------------
__all__ = [
    "BASE_DIR",
    "FEATURE_BOUNDS",
    "FINAL_FEATURE_COLUMNS",
    "MODELS_DIR",
    "MODEL_PATH",
    "PROCESSED_DIR",
    "PROCESSED_FEATURES_PATH",
    "RANDOM_STATE",
    "RAW_DATA_PATH",
    "RAW_FEATURE_COLUMNS",
    "SENIORITY_ORDER",
    "TARGET_COL",
    "TARGET_MAX",
    "TARGET_MIN",
    "TEST_SIZE",
    "XGB_PARAMS",
]
