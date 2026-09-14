"""
Service-layer tests for predict_burnout().

Tests call the function directly (no HTTP), so failures pinpoint the service
logic rather than routing or serialisation issues.
"""
from __future__ import annotations

import pytest

from src import config as ml_config
from app.schemas.burnout import BurnoutRequest
from app.services.prediction_service import predict_burnout
from app.core.config import (
    BURNOUT_LEVEL_LOW,
    BURNOUT_LEVEL_MODERATE,
    BURNOUT_LEVEL_HIGH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def typical_request() -> BurnoutRequest:
    """A realistic mid-range employee profile."""
    return BurnoutRequest(
        seniority_level="Senior",
        work_hours_per_week=48.0,
        meetings_per_day=5.0,
        sleep_hours_per_night=6.5,
        exercise_days_per_week=2.0,
        vacation_days_taken=8.0,
        social_support_score=6.0,
        manager_support_score=7.0,
        deadline_pressure_score=7.5,
    )


@pytest.fixture
def high_risk_request() -> BurnoutRequest:
    """Profile designed to push toward a high burnout score."""
    return BurnoutRequest(
        seniority_level="Lead",
        work_hours_per_week=85.0,
        meetings_per_day=14.0,
        sleep_hours_per_night=4.0,
        exercise_days_per_week=0.0,
        vacation_days_taken=0.0,
        social_support_score=1.5,
        manager_support_score=1.5,
        deadline_pressure_score=9.5,
    )


@pytest.fixture
def low_risk_request() -> BurnoutRequest:
    """Profile designed to push toward a low burnout score."""
    return BurnoutRequest(
        seniority_level="Mid",
        work_hours_per_week=35.0,
        meetings_per_day=2.0,
        sleep_hours_per_night=8.0,
        exercise_days_per_week=5.0,
        vacation_days_taken=25.0,
        social_support_score=9.0,
        manager_support_score=9.0,
        deadline_pressure_score=2.0,
    )


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

class TestPredictBurnoutShape:
    def test_returns_burnout_response(self, typical_request):
        from app.schemas.burnout import BurnoutResponse
        result = predict_burnout(typical_request)
        assert isinstance(result, BurnoutResponse)

    def test_score_is_float(self, typical_request):
        result = predict_burnout(typical_request)
        assert isinstance(result.burnout_score, float)

    def test_score_in_valid_range(self, typical_request):
        result = predict_burnout(typical_request)
        assert 0.0 <= result.burnout_score <= 10.0

    def test_level_is_valid_label(self, typical_request):
        result = predict_burnout(typical_request)
        assert result.burnout_level in {"Low", "Moderate", "High", "Severe"}

    def test_contributions_non_empty(self, typical_request):
        result = predict_burnout(typical_request)
        assert len(result.contributions) > 0

    def test_contributions_count_equals_feature_count(self, typical_request):
        result = predict_burnout(typical_request)
        assert len(result.contributions) == len(ml_config.FINAL_FEATURE_COLUMNS)

    def test_contribution_features_are_known_names(self, typical_request):
        result = predict_burnout(typical_request)
        known = set(ml_config.FINAL_FEATURE_COLUMNS)
        for c in result.contributions:
            assert c.feature in known, f"Unknown feature: {c.feature}"

    def test_base_value_in_range(self, typical_request):
        result = predict_burnout(typical_request)
        assert 0.0 <= result.base_value <= 10.0


# ---------------------------------------------------------------------------
# Burnout level threshold consistency
# ---------------------------------------------------------------------------

class TestBurnoutLevelMapping:
    def test_score_below_low_threshold_maps_to_low(self):
        """Any score < BURNOUT_LEVEL_LOW must map to 'Low'."""
        from app.core.config import score_to_level
        assert score_to_level(0.0) == "Low"
        assert score_to_level(BURNOUT_LEVEL_LOW - 0.01) == "Low"

    def test_score_at_low_threshold_maps_to_moderate(self):
        from app.core.config import score_to_level
        assert score_to_level(BURNOUT_LEVEL_LOW) == "Moderate"

    def test_score_at_moderate_threshold_maps_to_high(self):
        from app.core.config import score_to_level
        assert score_to_level(BURNOUT_LEVEL_MODERATE) == "High"

    def test_score_at_high_threshold_maps_to_severe(self):
        from app.core.config import score_to_level
        assert score_to_level(BURNOUT_LEVEL_HIGH) == "Severe"

    def test_score_10_maps_to_severe(self):
        from app.core.config import score_to_level
        assert score_to_level(10.0) == "Severe"

    def test_level_consistent_with_score(self, typical_request):
        """level label must be consistent with the returned score."""
        result = predict_burnout(typical_request)
        score = result.burnout_score
        expected_level = (
            "Low" if score < BURNOUT_LEVEL_LOW
            else "Moderate" if score < BURNOUT_LEVEL_MODERATE
            else "High" if score < BURNOUT_LEVEL_HIGH
            else "Severe"
        )
        assert result.burnout_level == expected_level, (
            f"score={score:.3f} → label should be '{expected_level}' "
            f"but got '{result.burnout_level}'"
        )


# ---------------------------------------------------------------------------
# Directional sanity checks
# ---------------------------------------------------------------------------

class TestPredictDirectional:
    def test_high_risk_score_above_low_risk_score(
        self, high_risk_request, low_risk_request
    ):
        """The high-stress profile must predict higher than the low-stress one."""
        high_score = predict_burnout(high_risk_request).burnout_score
        low_score  = predict_burnout(low_risk_request).burnout_score
        assert high_score > low_score, (
            f"Expected high_risk ({high_score:.2f}) > low_risk ({low_score:.2f})"
        )

    def test_high_risk_level_higher_than_low_risk_level(
        self, high_risk_request, low_risk_request
    ):
        level_order = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
        high_level = predict_burnout(high_risk_request).burnout_level
        low_level  = predict_burnout(low_risk_request).burnout_level
        assert level_order[high_level] >= level_order[low_level]

    @pytest.mark.parametrize("seniority", [
        "Junior", "Mid", "Senior", "Lead", "Manager", "Principal"
    ])
    def test_all_seniority_levels_produce_valid_response(self, seniority):
        req = BurnoutRequest(
            seniority_level=seniority,
            work_hours_per_week=45.0,
            meetings_per_day=4.0,
            sleep_hours_per_night=7.0,
            exercise_days_per_week=3.0,
            vacation_days_taken=10.0,
            social_support_score=6.0,
            manager_support_score=6.0,
            deadline_pressure_score=6.0,
        )
        result = predict_burnout(req)
        assert 0.0 <= result.burnout_score <= 10.0
