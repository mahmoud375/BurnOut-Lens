"""
Central configuration for the ML pipeline: paths, feature lists, and hyperparameters.
Single source of truth so preprocessing, training, and inference never drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # points to ml/
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "mental_health_burnout_tech_2026.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "xgb_burnout_model.json"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET_COL = "burnout_score"
TARGET_MIN = 0.0
TARGET_MAX = 10.0

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Ordinal feature: has a real, meaningful order (Junior -> Principal)
SENIORITY_ORDER = {
    "Junior": 0,
    "Mid": 1,
    "Senior": 2,
    "Lead": 3,
    "Manager": 4,
    "Principal": 5,
}

# Raw columns pulled from the source dataset before any encoding
RAW_FEATURE_COLUMNS = [
    "seniority_level",
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
FINAL_FEATURE_COLUMNS = [
    "work_hours_per_week",
    "meetings_per_day",
    "sleep_hours_per_night",
    "exercise_days_per_week",
    "vacation_days_taken",
    "social_support_score",
    "manager_support_score",
    "deadline_pressure_score",
    "seniority_level_encoded",
]

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------
# Plausible real-world bounds per feature (used to clip physically impossible
# values BEFORE they reach the model — both at training and inference time).
FEATURE_BOUNDS = {
    "work_hours_per_week": (20, 90),
    "meetings_per_day": (0, 15),
    "sleep_hours_per_night": (2, 12),
    "exercise_days_per_week": (0, 7),
    "vacation_days_taken": (0, 30),
    "social_support_score": (1, 10),
    "manager_support_score": (1, 10),
    "deadline_pressure_score": (1, 10),
}

# ---------------------------------------------------------------------------
# Train/test split & model hyperparameters
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
}