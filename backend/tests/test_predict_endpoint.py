"""
Endpoint-level tests for POST /predict and GET /health.

Uses FastAPI's TestClient (backed by httpx) — no live server needed.
The model singleton loads at module import time when app.main is imported,
so all tests share the one loaded model (fast, same as production behaviour).
"""
from __future__ import annotations

import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# ---------------------------------------------------------------------------
# A realistic baseline payload — values a real employee might actually submit
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "seniority_level":          "Senior",
    "work_hours_per_week":      48.0,
    "meetings_per_day":         5.0,
    "sleep_hours_per_night":    6.5,
    "exercise_days_per_week":   2.0,
    "vacation_days_taken":      8.0,
    "social_support_score":     6.0,
    "manager_support_score":    7.0,
    "deadline_pressure_score":  7.5,
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "model_loaded" in body

    def test_status_is_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_model_loaded_is_true(self):
        """Model must have loaded successfully at startup."""
        r = client.get("/health")
        assert r.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_valid_request_returns_200(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        body = r.json()
        assert "burnout_score" in body
        assert "burnout_level" in body
        assert "base_value" in body
        assert "contributions" in body

    def test_burnout_score_in_valid_range(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        score = r.json()["burnout_score"]
        assert 0.0 <= score <= 10.0, f"score {score} out of [0, 10]"

    def test_burnout_level_is_valid_label(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        level = r.json()["burnout_level"]
        assert level in {"Low", "Moderate", "High", "Severe"}

    def test_contributions_is_non_empty_list(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        contribs = r.json()["contributions"]
        assert isinstance(contribs, list)
        assert len(contribs) > 0

    def test_each_contribution_has_correct_keys(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        for c in r.json()["contributions"]:
            assert set(c.keys()) == {"feature", "value", "shap_value", "direction"}

    def test_contribution_directions_are_valid(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        for c in r.json()["contributions"]:
            assert c["direction"] in {"increases", "decreases"}

    def test_base_value_is_positive_float(self):
        """Base value is the model mean prediction — should be a non-negative number."""
        r = client.post("/predict", json=VALID_PAYLOAD)
        base = r.json()["base_value"]
        assert isinstance(base, float)
        assert 0.0 <= base <= 10.0

    def test_response_is_json_content_type(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert "application/json" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /predict — different seniority levels all work
# ---------------------------------------------------------------------------

class TestPredictSeniorityLevels:
    @pytest.mark.parametrize("level", ["Junior", "Mid", "Senior", "Lead", "Manager", "Principal"])
    def test_all_seniority_levels_return_200(self, level):
        payload = {**VALID_PAYLOAD, "seniority_level": level}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, f"Failed for seniority_level={level}: {r.json()}"

    @pytest.mark.parametrize("level", ["Junior", "Mid", "Senior", "Lead", "Manager", "Principal"])
    def test_all_seniority_levels_return_valid_score(self, level):
        payload = {**VALID_PAYLOAD, "seniority_level": level}
        score = client.post("/predict", json=payload).json()["burnout_score"]
        assert 0.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# POST /predict — validation errors (422)
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_out_of_bounds_sleep_returns_422(self):
        """sleep_hours_per_night=50 exceeds the max bound of 12."""
        payload = {**VALID_PAYLOAD, "sleep_hours_per_night": 50.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_negative_work_hours_returns_422(self):
        """work_hours_per_week below minimum bound."""
        payload = {**VALID_PAYLOAD, "work_hours_per_week": -5.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_missing_required_field_returns_422(self):
        """Omitting seniority_level must return 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "seniority_level"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_seniority_level_returns_422(self):
        """A seniority_level not in the Literal must return 422."""
        payload = {**VALID_PAYLOAD, "seniority_level": "VP"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_missing_all_fields_returns_422(self):
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_wrong_type_returns_422(self):
        """String where float expected."""
        payload = {**VALID_PAYLOAD, "work_hours_per_week": "lots"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_meetings_per_day_too_high_returns_422(self):
        payload = {**VALID_PAYLOAD, "meetings_per_day": 99.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET / — redirect
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_redirects_to_docs(self):
        """Root should redirect to /docs."""
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (301, 302, 307, 308)
        assert r.headers["location"] == "/docs"
