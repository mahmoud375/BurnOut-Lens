"""
Unit tests for ml/src/explain.py.

A tiny XGBoost model is trained once per test session on synthetic data
that matches ``FINAL_FEATURE_COLUMNS``.  All tests use a ``shap.TreeExplainer``
built from that model — no real dataset is loaded.

Run with:
    cd ml/ && uv run pytest tests/test_explain.py -v
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import shap
import xgboost as xgb

from src import config
from src.explain import (
    explain_batch,
    explain_prediction,
    get_shap_explainer,
    plot_global_importance,
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures  (model trained once, reused across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_X() -> pd.DataFrame:
    """100-row DataFrame matching ``FINAL_FEATURE_COLUMNS`` with valid values."""
    rng = np.random.default_rng(seed=0)
    n = 100
    data = {
        "work_hours_per_week":    rng.uniform(20, 90, n),
        "meetings_per_day":       rng.uniform(0, 15, n),
        "sleep_hours_per_night":  rng.uniform(2, 12, n),
        "exercise_days_per_week": rng.uniform(0, 7, n),
        "vacation_days_taken":    rng.uniform(0, 30, n),
        "social_support_score":   rng.uniform(1, 10, n),
        "manager_support_score":  rng.uniform(1, 10, n),
        "deadline_pressure_score": rng.uniform(1, 10, n),
        "seniority_level_encoded": rng.integers(0, 6, n).astype(float),
    }
    return pd.DataFrame(data, columns=config.FINAL_FEATURE_COLUMNS)


@pytest.fixture(scope="session")
def synthetic_y(synthetic_X: pd.DataFrame) -> np.ndarray:
    """Deterministic synthetic target in [0, 10]."""
    rng = np.random.default_rng(seed=0)
    return rng.uniform(0, 10, len(synthetic_X))


@pytest.fixture(scope="session")
def trained_model(
    synthetic_X: pd.DataFrame, synthetic_y: np.ndarray
) -> xgb.XGBRegressor:
    """Tiny XGBoost model (n_estimators=20) — fast to train, enough for SHAP."""
    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(synthetic_X, synthetic_y)
    return model


@pytest.fixture(scope="session")
def explainer(trained_model: xgb.XGBRegressor) -> shap.TreeExplainer:
    return get_shap_explainer(trained_model)


@pytest.fixture(scope="session")
def single_row(synthetic_X: pd.DataFrame) -> pd.DataFrame:
    """A single-row DataFrame (first row of synthetic_X)."""
    return synthetic_X.iloc[:1].reset_index(drop=True)


@pytest.fixture(scope="session")
def explanation(
    explainer: shap.TreeExplainer,
    single_row: pd.DataFrame,
) -> dict:
    """Pre-computed explanation for the single row — reused across tests."""
    return explain_prediction(explainer, single_row)


# ---------------------------------------------------------------------------
# get_shap_explainer
# ---------------------------------------------------------------------------


class TestGetShapExplainer:
    def test_returns_tree_explainer(self, trained_model):
        ex = get_shap_explainer(trained_model)
        assert isinstance(ex, shap.TreeExplainer)

    def test_expected_value_is_extractable_as_scalar(self, explainer):
        """expected_value must be convertible to a single float.

        SHAP 0.52 returns a 1-d ndarray of shape (1,) for XGBoost regressors;
        older versions returned a plain float32.  We accept either.
        """
        ev = np.asarray(explainer.expected_value)
        assert ev.size == 1, (
            f"expected_value should contain exactly 1 element, got shape {ev.shape}"
        )

    def test_expected_value_in_plausible_range(self, explainer):
        """For data in [0, 10], the mean prediction should also be in [0, 10]."""
        ev = float(np.asarray(explainer.expected_value).flat[0])
        assert config.TARGET_MIN <= ev <= config.TARGET_MAX


# ---------------------------------------------------------------------------
# explain_prediction — dict structure
# ---------------------------------------------------------------------------


class TestExplainPrediction:
    def test_top_level_keys(self, explanation):
        assert set(explanation.keys()) == {"base_value", "prediction", "contributions"}

    def test_base_value_is_python_float(self, explanation):
        assert type(explanation["base_value"]) is float

    def test_prediction_is_python_float(self, explanation):
        assert type(explanation["prediction"]) is float

    def test_contributions_is_list(self, explanation):
        assert isinstance(explanation["contributions"], list)

    def test_contributions_length_matches_feature_count(self, explanation):
        assert len(explanation["contributions"]) == len(config.FINAL_FEATURE_COLUMNS)

    def test_each_contribution_has_correct_keys(self, explanation):
        required = {"feature", "value", "shap_value", "direction"}
        for c in explanation["contributions"]:
            assert set(c.keys()) == required, f"Missing keys in contribution: {c}"

    def test_feature_names_match_final_feature_columns(self, explanation):
        names = [c["feature"] for c in explanation["contributions"]]
        # Same set of names (sorted order may differ since contributions are sorted by |shap|)
        assert set(names) == set(config.FINAL_FEATURE_COLUMNS)

    def test_all_contribution_values_are_python_float(self, explanation):
        for c in explanation["contributions"]:
            assert type(c["value"]) is float, f"'value' in {c['feature']} is not float"
            assert type(c["shap_value"]) is float, f"'shap_value' in {c['feature']} is not float"

    def test_direction_is_valid_string(self, explanation):
        valid = {"increases", "decreases"}
        for c in explanation["contributions"]:
            assert c["direction"] in valid, (
                f"Invalid direction '{c['direction']}' for feature '{c['feature']}'"
            )

    def test_direction_consistent_with_sign(self, explanation):
        """direction must match the sign of shap_value."""
        for c in explanation["contributions"]:
            if c["shap_value"] >= 0.0:
                assert c["direction"] == "increases", (
                    f"{c['feature']}: positive shap_value but direction='{c['direction']}'"
                )
            else:
                assert c["direction"] == "decreases", (
                    f"{c['feature']}: negative shap_value but direction='{c['direction']}'"
                )

    def test_contributions_sorted_by_abs_shap_descending(self, explanation):
        abs_vals = [abs(c["shap_value"]) for c in explanation["contributions"]]
        assert abs_vals == sorted(abs_vals, reverse=True), (
            "Contributions are not sorted by |shap_value| descending"
        )

    # ------------------------------------------------------------------
    # Reconstruction guarantee
    # ------------------------------------------------------------------

    def test_reconstruction_base_plus_shap_equals_prediction(self, explanation):
        """base_value + sum(shap_values) must equal prediction within tolerance."""
        total = explanation["base_value"] + sum(
            c["shap_value"] for c in explanation["contributions"]
        )
        assert math.isclose(total, explanation["prediction"], abs_tol=1e-4), (
            f"Reconstruction failed: {explanation['base_value']} + shap_sum "
            f"≈ {total:.6f} ≠ prediction {explanation['prediction']:.6f}"
        )

    def test_prediction_close_to_model_predict(
        self, trained_model, explainer, single_row
    ):
        """The reconstruction should also be close to model.predict()."""
        model_pred = float(trained_model.predict(single_row)[0])
        explanation = explain_prediction(explainer, single_row)
        assert math.isclose(explanation["prediction"], model_pred, abs_tol=1e-4), (
            f"explanation['prediction']={explanation['prediction']:.6f} "
            f"vs model.predict={model_pred:.6f}"
        )

    # ------------------------------------------------------------------
    # JSON serialisability (explicit end-to-end check)
    # ------------------------------------------------------------------

    def test_output_is_json_serializable(self, explanation):
        """json.dumps must succeed without a custom encoder."""
        try:
            serialised = json.dumps(explanation)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"explain_prediction output is not JSON-serializable: {exc}")
        # Round-trip sanity: deserialise and check top-level keys
        restored = json.loads(serialised)
        assert set(restored.keys()) == {"base_value", "prediction", "contributions"}

    # ------------------------------------------------------------------
    # Error conditions
    # ------------------------------------------------------------------

    def test_raises_on_multi_row_input(self, explainer, synthetic_X):
        with pytest.raises(ValueError, match="1 row"):
            explain_prediction(explainer, synthetic_X.iloc[:3])

    def test_raises_on_wrong_column_names(self, explainer):
        bad_df = pd.DataFrame({"wrong_col": [1.0]})
        with pytest.raises(ValueError):
            explain_prediction(explainer, bad_df)


# ---------------------------------------------------------------------------
# explain_batch
# ---------------------------------------------------------------------------


class TestExplainBatch:
    def test_output_shape(self, explainer, synthetic_X):
        sv = explain_batch(explainer, synthetic_X)
        assert sv.shape == (len(synthetic_X), len(config.FINAL_FEATURE_COLUMNS))

    def test_output_is_ndarray(self, explainer, synthetic_X):
        sv = explain_batch(explainer, synthetic_X)
        assert isinstance(sv, np.ndarray)

    def test_raises_on_wrong_columns(self, explainer):
        bad = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError):
            explain_batch(explainer, bad)

    def test_single_row_consistent_with_explain_prediction(
        self, explainer, single_row, explanation
    ):
        """explain_batch on one row should give the same SHAP values as explain_prediction."""
        batch_sv = explain_batch(explainer, single_row)[0]   # shape (n_features,)
        contrib_sv = np.array(
            [c["shap_value"] for c in sorted(
                explanation["contributions"],
                key=lambda c: config.FINAL_FEATURE_COLUMNS.index(c["feature"])
            )]
        )
        np.testing.assert_allclose(batch_sv, contrib_sv, atol=1e-5)


# ---------------------------------------------------------------------------
# plot_global_importance  (smoke test — just checks it doesn't raise)
# ---------------------------------------------------------------------------


class TestPlotGlobalImportance:
    def test_saves_file_to_disk(self, explainer, synthetic_X, tmp_path):
        out = tmp_path / "global_shap.png"
        plot_global_importance(explainer, synthetic_X, save_path=out)
        assert out.exists(), "Expected plot file was not created"
        assert out.stat().st_size > 0, "Plot file is empty"

    def test_creates_parent_directories(self, explainer, synthetic_X, tmp_path):
        out = tmp_path / "nested" / "dir" / "shap.png"
        plot_global_importance(explainer, synthetic_X, save_path=out)
        assert out.exists()

    def test_raises_on_wrong_columns(self, explainer):
        bad = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError):
            plot_global_importance(explainer, bad, save_path="/tmp/x.png")
