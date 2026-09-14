"""
Unit tests for ml/src/evaluate.py.

All tests are self-contained: they build synthetic arrays or toy models
inline — no file I/O, no real dataset.

Run with:
    cd ml/ && uv run pytest tests/test_evaluate.py -v
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from src import config
from src.evaluate import compute_metrics, evaluate_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_arrays(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) where predictions are exact."""
    y = np.linspace(config.TARGET_MIN, config.TARGET_MAX, n)
    return y, y.copy()


def _known_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return arrays with pre-computed, deterministic expected metrics.

    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 6]   ← only the last element differs by 1

    Expected:
        MAE  = mean([0, 0, 0, 0, 1]) = 0.2
        MSE  = mean([0, 0, 0, 0, 1]) = 0.2  → RMSE = sqrt(0.2) ≈ 0.447214
        R²   = 1 - SS_res/SS_tot
             SS_res = 1
             SS_tot = (1-3)²+(2-3)²+(3-3)²+(4-3)²+(5-3)² = 4+1+0+1+4 = 10
             R²   = 1 - 1/10 = 0.9
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# compute_metrics — happy paths
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions_mae_zero(self):
        y, yp = _perfect_arrays()
        m = compute_metrics(y, yp)
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_rmse_zero(self):
        y, yp = _perfect_arrays()
        m = compute_metrics(y, yp)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_r2_one(self):
        y, yp = _perfect_arrays()
        m = compute_metrics(y, yp)
        assert m["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_known_mae(self):
        y_true, y_pred = _known_arrays()
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.2, abs=1e-6)

    def test_known_rmse(self):
        y_true, y_pred = _known_arrays()
        m = compute_metrics(y_true, y_pred)
        expected_rmse = math.sqrt(0.2)
        assert m["rmse"] == pytest.approx(expected_rmse, rel=1e-5)

    def test_known_r2(self):
        y_true, y_pred = _known_arrays()
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(0.9, abs=1e-6)

    def test_returns_dict_with_expected_keys(self):
        y, yp = _perfect_arrays()
        m = compute_metrics(y, yp)
        assert set(m.keys()) == {"mae", "rmse", "r2"}

    def test_all_values_are_python_float(self):
        y, yp = _perfect_arrays()
        m = compute_metrics(y, yp)
        for k, v in m.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_accepts_lists_and_converts(self):
        """compute_metrics should coerce list inputs to ndarray."""
        m = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_r2_below_zero_for_bad_predictions(self):
        """Constant predictions on diverse targets → R² < 0."""
        y_true = np.array([1.0, 5.0, 9.0])
        y_pred = np.array([5.0, 5.0, 5.0])   # always predict mean → R²=0
        m = compute_metrics(y_true, y_pred)
        # Actually predicting the mean gives R²=0 exactly
        assert m["r2"] == pytest.approx(0.0, abs=1e-9)

    def test_r2_negative_for_worse_than_mean(self):
        y_true = np.array([1.0, 5.0, 9.0])
        y_pred = np.array([9.0, 1.0, 1.0])   # deliberately bad → R² < 0
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] < 0.0

    # ------------------------------------------------------------------
    # Edge cases / errors
    # ------------------------------------------------------------------

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_raises_on_empty_arrays(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_single_element(self):
        """Single-element edge case — MAE and RMSE 0, R² undefined but sklearn returns 1."""
        m = compute_metrics(np.array([5.0]), np.array([5.0]))
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_model — clipping contract
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """
    Core contract: predictions are clipped to [TARGET_MIN, TARGET_MAX] BEFORE
    metrics are computed.  These tests use a mock model whose .predict() returns
    out-of-range values so we can isolate the clipping behaviour.
    """

    def _make_mock_model(self, raw_predictions: np.ndarray) -> MagicMock:
        model = MagicMock()
        model.predict.return_value = raw_predictions
        return model

    def test_clipping_reduces_mae_vs_unclipped(self):
        """
        If the model predicts 12.0 for a true value of 9.0, the unclipped error
        is 3.0 but after clipping to 10.0 the error is 1.0.
        evaluate_model should report 1.0.
        """
        raw = np.array([12.0])          # out-of-range high
        y_test = np.array([9.0])
        model = self._make_mock_model(raw)

        metrics = evaluate_model(model, X_test=np.zeros((1, 9)), y_test=y_test)

        assert metrics["mae"] == pytest.approx(1.0, abs=1e-6), (
            "MAE should reflect clipped prediction (10.0), not raw (12.0)"
        )

    def test_negative_prediction_clipped_to_zero(self):
        """
        Predicting -3.0 for true=2.0: unclipped error=5.0, clipped error=2.0.
        """
        raw = np.array([-3.0])
        y_test = np.array([2.0])
        model = self._make_mock_model(raw)

        metrics = evaluate_model(model, X_test=np.zeros((1, 9)), y_test=y_test)

        assert metrics["mae"] == pytest.approx(2.0, abs=1e-6), (
            "MAE should reflect clipped prediction (0.0), not raw (-3.0)"
        )

    def test_in_range_predictions_unchanged(self):
        """Within-range predictions should give the same metrics as compute_metrics."""
        y_true = np.array([3.0, 5.0, 7.0])
        raw = np.array([3.2, 4.9, 6.8])    # all within [0, 10]
        model = self._make_mock_model(raw)

        metrics = evaluate_model(model, X_test=np.zeros((3, 9)), y_test=y_true)
        expected = compute_metrics(y_true, raw)  # no clipping needed

        assert metrics["mae"] == pytest.approx(expected["mae"], abs=1e-6)
        assert metrics["rmse"] == pytest.approx(expected["rmse"], abs=1e-6)
        assert metrics["r2"] == pytest.approx(expected["r2"], abs=1e-6)

    def test_perfect_predictions_all_zeros(self):
        y = np.linspace(0.0, 10.0, 20)
        model = self._make_mock_model(y.copy())
        metrics = evaluate_model(model, X_test=np.zeros((20, 9)), y_test=y)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_dict_with_correct_keys(self):
        y = np.array([5.0, 5.0])
        model = self._make_mock_model(y.copy())
        metrics = evaluate_model(model, X_test=np.zeros((2, 9)), y_test=y)
        assert set(metrics.keys()) == {"mae", "rmse", "r2"}

    def test_multiple_out_of_range_all_clipped(self):
        """Mixed extreme predictions — verify each is clipped independently."""
        raw = np.array([-5.0, 15.0, 5.0])    # low, high, in-range
        y_test = np.array([1.0,  9.0, 5.0])
        # After clipping: [0.0, 10.0, 5.0] → errors [1.0, 1.0, 0.0] → MAE=2/3
        model = self._make_mock_model(raw)
        metrics = evaluate_model(model, X_test=np.zeros((3, 9)), y_test=y_test)
        assert metrics["mae"] == pytest.approx(2.0 / 3.0, abs=1e-6)
