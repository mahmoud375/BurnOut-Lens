"""
Pydantic schemas for the /predict endpoint.

BurnoutRequest accepts the *human-facing* form of the data — the API consumer
sends seniority as a string ("Senior"), not the encoded integer (2).  The
service layer handles encoding, keeping that detail out of the API contract.

Field bounds come directly from ml config to ensure training and API
validation are always in sync.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from src import config as ml_config

# Convenience aliases so Field() calls below stay readable
_B = ml_config.FEATURE_BOUNDS


class BurnoutRequest(BaseModel):
    """Input features for a single burnout prediction request."""

    seniority_level: Literal["Junior", "Mid", "Senior", "Lead", "Manager", "Principal"] = Field(
        description="Employee seniority tier."
    )
    work_hours_per_week: float = Field(
        ge=_B["work_hours_per_week"][0],
        le=_B["work_hours_per_week"][1],
        description="Average number of hours worked per week (20–90).",
    )
    meetings_per_day: float = Field(
        ge=_B["meetings_per_day"][0],
        le=_B["meetings_per_day"][1],
        description="Average number of meetings attended per day (0–15).",
    )
    sleep_hours_per_night: float = Field(
        ge=_B["sleep_hours_per_night"][0],
        le=_B["sleep_hours_per_night"][1],
        description="Average hours of sleep per night (2–12).",
    )
    exercise_days_per_week: float = Field(
        ge=_B["exercise_days_per_week"][0],
        le=_B["exercise_days_per_week"][1],
        description="Number of days per week with physical exercise (0–7).",
    )
    vacation_days_taken: float = Field(
        ge=_B["vacation_days_taken"][0],
        le=_B["vacation_days_taken"][1],
        description="Number of vacation days taken in the past year (0–30).",
    )
    social_support_score: float = Field(
        ge=_B["social_support_score"][0],
        le=_B["social_support_score"][1],
        description="Perceived social support from colleagues/friends (1–10).",
    )
    manager_support_score: float = Field(
        ge=_B["manager_support_score"][0],
        le=_B["manager_support_score"][1],
        description="Perceived support from direct manager (1–10).",
    )
    deadline_pressure_score: float = Field(
        ge=_B["deadline_pressure_score"][0],
        le=_B["deadline_pressure_score"][1],
        description="Perceived pressure from deadlines (1–10).",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "seniority_level": "Senior",
            "work_hours_per_week": 48.0,
            "meetings_per_day": 5.0,
            "sleep_hours_per_night": 6.5,
            "exercise_days_per_week": 2.0,
            "vacation_days_taken": 8.0,
            "social_support_score": 6.0,
            "manager_support_score": 7.0,
            "deadline_pressure_score": 7.5,
        }
    }}


class Contribution(BaseModel):
    """SHAP contribution of a single feature to the predicted burnout score."""

    feature: str = Field(description="Feature name (internal model column name).")
    value: float = Field(description="The actual feature value for this prediction.")
    shap_value: float = Field(
        description="SHAP contribution: positive means it pushes the score up."
    )
    direction: Literal["increases", "decreases"] = Field(
        description="Whether this feature increases or decreases the burnout score."
    )


class BurnoutResponse(BaseModel):
    """Full prediction response with score, severity label, and SHAP explanation."""

    burnout_score: float = Field(
        description="Predicted burnout score clipped to [0, 10]."
    )
    burnout_level: Literal["Low", "Moderate", "High", "Severe"] = Field(
        description=(
            "Severity label derived from the score: "
            "Low (<3.0), Moderate (3.0–5.5), High (5.5–7.5), Severe (≥7.5)."
        )
    )
    base_value: float = Field(
        description="SHAP base value — the model's average prediction across training data."
    )
    contributions: list[Contribution] = Field(
        description="Per-feature SHAP contributions, sorted by absolute impact descending."
    )
