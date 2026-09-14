"""
Unit tests for ml/src/preprocessing.py.

Each test is self-contained: it builds a minimal DataFrame inline rather than
loading any external file.  This keeps the suite fast, deterministic, and
decoupled from the real dataset.

Run with:
    cd ml/
    uv run pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import config
from src.preprocessing import (
    clip_outliers,
    clip_prediction,
    encode_seniority,
    preprocess,
    select_raw_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(**overrides) -> pd.DataFrame:
    """Return a single-row raw DataFrame with sensible defaults.

    Any column can be overridden via keyword arguments, e.g.
    ``_make_raw_df(work_hours_per_week=120)``.
    """
    defaults: dict = {
        "seniority_level": "Senior",
        "work_hours_per_week": 45.0,
        "meetings_per_day": 4.0,
        "sleep_hours_per_night": 7.0,
        "exercise_days_per_week": 3.0,
        "vacation_days_taken": 10.0,
        "social_support_score": 7.0,
        "manager_support_score": 6.0,
        "deadline_pressure_score": 5.0,
        config.TARGET_COL: 5.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# select_raw_features
# ---------------------------------------------------------------------------

class TestSelectRawFeatures:
    def test_keeps_exactly_raw_and_target_columns(self):
        df = _make_raw_df()
        result = select_raw_features(df, include_target=True)
        expected_cols = config.RAW_FEATURE_COLUMNS + [config.TARGET_COL]
        assert list(result.columns) == expected_cols

    def test_excludes_target_when_requested(self):
        df = _make_raw_df()
        result = select_raw_features(df, include_target=False)
        assert config.TARGET_COL not in result.columns
        assert list(result.columns) == config.RAW_FEATURE_COLUMNS

    def test_drops_extra_columns(self):
        df = _make_raw_df()
        df["irrelevant_column"] = 999
        result = select_raw_features(df, include_target=False)
        assert "irrelevant_column" not in result.columns

    def test_raises_on_missing_feature_column(self):
        df = _make_raw_df()
        df = df.drop(columns=["work_hours_per_week"])
        with pytest.raises(KeyError, match="work_hours_per_week"):
            select_raw_features(df, include_target=False)

    def test_raises_on_missing_target_when_required(self):
        df = _make_raw_df()
        df = df.drop(columns=[config.TARGET_COL])
        with pytest.raises(KeyError, match=config.TARGET_COL):
            select_raw_features(df, include_target=True)

    def test_does_not_raise_on_missing_target_when_not_required(self):
        df = _make_raw_df()
        df = df.drop(columns=[config.TARGET_COL])
        # Should not raise
        result = select_raw_features(df, include_target=False)
        assert config.TARGET_COL not in result.columns

    def test_returns_a_copy(self):
        """Mutations to the result must not affect the original."""
        df = _make_raw_df()
        result = select_raw_features(df, include_target=False)
        result["work_hours_per_week"] = -1
        assert df["work_hours_per_week"].iloc[0] != -1


# ---------------------------------------------------------------------------
# clip_outliers
# ---------------------------------------------------------------------------

class TestClipOutliers:
    def test_values_within_bounds_are_unchanged(self):
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        result = clip_outliers(df_raw)
        for col, (low, high) in config.FEATURE_BOUNDS.items():
            val = result[col].iloc[0]
            assert low <= val <= high, f"{col}: {val} not in [{low}, {high}]"

    def test_values_above_upper_bound_are_clipped(self):
        df = _make_raw_df(work_hours_per_week=999.0)
        df_raw = select_raw_features(df, include_target=False)
        result = clip_outliers(df_raw)
        assert result["work_hours_per_week"].iloc[0] == config.FEATURE_BOUNDS["work_hours_per_week"][1]

    def test_values_below_lower_bound_are_clipped(self):
        df = _make_raw_df(sleep_hours_per_night=-5.0)
        df_raw = select_raw_features(df, include_target=False)
        result = clip_outliers(df_raw)
        assert result["sleep_hours_per_night"].iloc[0] == config.FEATURE_BOUNDS["sleep_hours_per_night"][0]

    def test_all_numeric_bounds_respected_on_extreme_inputs(self):
        extreme_low = _make_raw_df(
            work_hours_per_week=-100,
            meetings_per_day=-50,
            sleep_hours_per_night=-10,
            exercise_days_per_week=-10,
            vacation_days_taken=-10,
            social_support_score=-10,
            manager_support_score=-10,
            deadline_pressure_score=-10,
        )
        df_raw = select_raw_features(extreme_low, include_target=False)
        result = clip_outliers(df_raw)
        for col, (low, _) in config.FEATURE_BOUNDS.items():
            assert result[col].iloc[0] == low, f"{col} lower clip failed"

    def test_seniority_level_column_is_untouched(self):
        """clip_outliers must not touch the string column."""
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        result = clip_outliers(df_raw)
        assert result["seniority_level"].iloc[0] == "Senior"

    def test_returns_a_copy(self):
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        original_val = df_raw["work_hours_per_week"].iloc[0]
        result = clip_outliers(df_raw)
        result["work_hours_per_week"] = -999
        assert df_raw["work_hours_per_week"].iloc[0] == original_val


# ---------------------------------------------------------------------------
# encode_seniority
# ---------------------------------------------------------------------------

class TestEncodeSeniority:
    @pytest.mark.parametrize("level, expected_code", list(config.SENIORITY_ORDER.items()))
    def test_all_valid_levels_map_correctly(self, level: str, expected_code: int):
        df = pd.DataFrame([{"seniority_level": level, "work_hours_per_week": 40}])
        result = encode_seniority(df)
        assert result["seniority_level_encoded"].iloc[0] == expected_code

    def test_encoded_column_is_integer_dtype(self):
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        result = encode_seniority(df_raw)
        assert pd.api.types.is_integer_dtype(result["seniority_level_encoded"])

    def test_original_seniority_level_column_is_dropped(self):
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        result = encode_seniority(df_raw)
        assert "seniority_level" not in result.columns

    def test_raises_on_unknown_seniority_value(self):
        df = pd.DataFrame([{"seniority_level": "Intern", "work_hours_per_week": 40}])
        with pytest.raises(ValueError, match="Intern"):
            encode_seniority(df)

    def test_raises_on_lowercase_seniority_value(self):
        """Encoding is case-sensitive — 'senior' != 'Senior'."""
        df = pd.DataFrame([{"seniority_level": "senior", "work_hours_per_week": 40}])
        with pytest.raises(ValueError, match="senior"):
            encode_seniority(df)

    def test_raises_on_null_seniority_value(self):
        df = pd.DataFrame([{"seniority_level": None, "work_hours_per_week": 40}])
        with pytest.raises(ValueError, match="null"):
            encode_seniority(df)

    def test_raises_on_missing_seniority_column(self):
        df = pd.DataFrame([{"work_hours_per_week": 40}])
        with pytest.raises(KeyError, match="seniority_level"):
            encode_seniority(df)

    def test_multiple_rows_all_encoded_correctly(self):
        levels = list(config.SENIORITY_ORDER.keys())
        df = pd.DataFrame([{"seniority_level": lv, "x": i} for i, lv in enumerate(levels)])
        result = encode_seniority(df)
        for _, row in result.iterrows():
            # We can recover the original level by reversing SENIORITY_ORDER
            reverse = {v: k for k, v in config.SENIORITY_ORDER.items()}
            assert row["seniority_level_encoded"] == config.SENIORITY_ORDER[reverse[row["seniority_level_encoded"]]]

    def test_returns_a_copy(self):
        df = _make_raw_df()
        df_raw = select_raw_features(df, include_target=False)
        _ = encode_seniority(df_raw)
        # Original df_raw should still have the string column
        assert "seniority_level" in df_raw.columns


# ---------------------------------------------------------------------------
# preprocess  (integration — exercises the full pipeline)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_columns_match_final_feature_columns_with_target(self):
        df = _make_raw_df()
        result = preprocess(df, include_target=True)
        expected = config.FINAL_FEATURE_COLUMNS + [config.TARGET_COL]
        assert list(result.columns) == expected

    def test_output_columns_match_final_feature_columns_without_target(self):
        df = _make_raw_df()
        result = preprocess(df, include_target=False)
        assert list(result.columns) == config.FINAL_FEATURE_COLUMNS

    def test_outliers_are_clipped_in_pipeline(self):
        df = _make_raw_df(work_hours_per_week=9999.0)
        result = preprocess(df, include_target=False)
        assert result["work_hours_per_week"].iloc[0] == config.FEATURE_BOUNDS["work_hours_per_week"][1]

    def test_seniority_is_encoded_in_pipeline(self):
        df = _make_raw_df(seniority_level="Lead")
        result = preprocess(df, include_target=False)
        assert result["seniority_level_encoded"].iloc[0] == config.SENIORITY_ORDER["Lead"]

    def test_raises_on_missing_column(self):
        df = _make_raw_df()
        df = df.drop(columns=["meetings_per_day"])
        with pytest.raises(KeyError):
            preprocess(df, include_target=False)

    def test_raises_on_unknown_seniority_in_pipeline(self):
        df = _make_raw_df(seniority_level="Contractor")
        with pytest.raises(ValueError, match="Contractor"):
            preprocess(df, include_target=False)

    def test_all_seniority_levels_produce_valid_output(self):
        for level in config.SENIORITY_ORDER:
            df = _make_raw_df(seniority_level=level)
            result = preprocess(df, include_target=False)
            assert result["seniority_level_encoded"].iloc[0] == config.SENIORITY_ORDER[level]

    def test_no_nans_in_output_for_valid_input(self):
        df = _make_raw_df()
        result = preprocess(df, include_target=True)
        assert not result.isnull().any().any(), "Pipeline produced NaN values for valid input"

    def test_multiple_rows(self):
        rows = [
            _make_raw_df(seniority_level=lv, work_hours_per_week=40 + i)
            for i, lv in enumerate(config.SENIORITY_ORDER)
        ]
        df_multi = pd.concat(rows, ignore_index=True)
        result = preprocess(df_multi, include_target=True)
        assert len(result) == len(config.SENIORITY_ORDER)
        assert list(result.columns) == config.FINAL_FEATURE_COLUMNS + [config.TARGET_COL]


# ---------------------------------------------------------------------------
# clip_prediction
# ---------------------------------------------------------------------------

class TestClipPrediction:
    def test_values_within_range_unchanged(self):
        y = np.array([0.0, 5.0, 10.0])
        np.testing.assert_array_equal(clip_prediction(y), y)

    def test_negative_values_clipped_to_zero(self):
        y = np.array([-3.0, -0.001])
        result = clip_prediction(y)
        assert (result == config.TARGET_MIN).all()

    def test_values_above_ten_clipped_to_ten(self):
        y = np.array([10.001, 999.0])
        result = clip_prediction(y)
        assert (result == config.TARGET_MAX).all()

    def test_scalar_array_works(self):
        y = np.array([-1.0])
        assert clip_prediction(y)[0] == config.TARGET_MIN

    def test_output_is_ndarray(self):
        y = np.array([5.0])
        assert isinstance(clip_prediction(y), np.ndarray)

    def test_mixed_array(self):
        y = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        expected = np.array([0.0, 0.0, 5.0, 10.0, 10.0])
        np.testing.assert_array_equal(clip_prediction(y), expected)
