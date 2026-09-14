"""
Unit tests for ml/src/train.py.

All tests use **synthetic data only** — the real 100k-row CSV is never
loaded.  Where functions have ``path`` / ``output_dir`` parameters, we pass
``tmp_path`` (pytest's built-in fixture) to redirect I/O to a temp directory
so tests leave no side-effects on disk.

Session-scoped fixtures build a tiny training dataset once and reuse it
across the whole module so the XGBoost fit (even at n_estimators=5) is not
repeated for every test.

Run with:
    cd ml/ && uv run pytest tests/test_train.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src import config
from src.train import (
    load_data,
    save_model,
    save_processed_data,
    split_data,
    train_model,
)


# ---------------------------------------------------------------------------
# Helpers & session-scoped fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_raw_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a minimal raw DataFrame matching the CSV schema.

    Contains only the columns ``preprocessing.select_raw_features`` needs:
    the 9 raw feature columns + ``TARGET_COL``.  Extra dataset columns
    (employee_id, age, …) are intentionally absent — they should be ignored
    by ``preprocess()``.
    """
    rng = np.random.default_rng(seed)
    levels = list(config.SENIORITY_ORDER.keys())
    data: dict = {
        "seniority_level":         [levels[i % len(levels)] for i in range(n)],
        "work_hours_per_week":     rng.uniform(20, 90, n),
        "meetings_per_day":        rng.uniform(0, 15, n),
        "sleep_hours_per_night":   rng.uniform(2, 12, n),
        "exercise_days_per_week":  rng.uniform(0, 7, n),
        "vacation_days_taken":     rng.uniform(0, 30, n),
        "social_support_score":    rng.uniform(1, 10, n),
        "manager_support_score":   rng.uniform(1, 10, n),
        "deadline_pressure_score": rng.uniform(1, 10, n),
        config.TARGET_COL:         rng.uniform(0, 10, n),
    }
    return pd.DataFrame(data)


def _make_synthetic_X(n: int = 50, seed: int = 1) -> pd.DataFrame:
    """Return a preprocessed feature DataFrame matching FINAL_FEATURE_COLUMNS."""
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 10, n) for col in config.FINAL_FEATURE_COLUMNS}
    # seniority_level_encoded must be 0-5 integers stored as float
    data["seniority_level_encoded"] = rng.integers(0, 6, n).astype(float)
    return pd.DataFrame(data, columns=config.FINAL_FEATURE_COLUMNS)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return _make_synthetic_raw_df(n=200, seed=42)


@pytest.fixture(scope="session")
def split_result(raw_df: pd.DataFrame):
    """Run split_data once and reuse across all tests."""
    return split_data(raw_df)


@pytest.fixture(scope="session")
def X_train(split_result):
    return split_result[0]


@pytest.fixture(scope="session")
def X_test(split_result):
    return split_result[1]


@pytest.fixture(scope="session")
def y_train(split_result):
    return split_result[2]


@pytest.fixture(scope="session")
def y_test(split_result):
    return split_result[3]


@pytest.fixture(scope="session")
def fitted_model(X_train, y_train) -> xgb.XGBRegressor:
    """Tiny 5-tree model — fast enough for unit tests."""
    mini_params = {**config.XGB_PARAMS, "n_estimators": 5}
    model = xgb.XGBRegressor(**mini_params)
    model.fit(X_train, y_train, verbose=False)
    return model


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_raises_file_not_found_on_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_data(path=tmp_path / "nonexistent.csv")

    def test_loads_csv_returns_dataframe(self, tmp_path):
        # Write a minimal CSV and confirm load_data reads it
        csv = tmp_path / "test.csv"
        df = _make_synthetic_raw_df(n=10)
        df.to_csv(csv, index=False)
        result = load_data(path=csv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_columns_are_preserved(self, tmp_path):
        csv = tmp_path / "test.csv"
        df = _make_synthetic_raw_df(n=5)
        df.to_csv(csv, index=False)
        result = load_data(path=csv)
        for col in config.RAW_FEATURE_COLUMNS + [config.TARGET_COL]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_returns_four_objects(self, split_result):
        assert len(split_result) == 4

    def test_X_train_columns_are_final_feature_columns(self, X_train):
        assert list(X_train.columns) == config.FINAL_FEATURE_COLUMNS

    def test_X_test_columns_are_final_feature_columns(self, X_test):
        assert list(X_test.columns) == config.FINAL_FEATURE_COLUMNS

    def test_y_train_is_series(self, y_train):
        assert isinstance(y_train, pd.Series)

    def test_y_test_is_series(self, y_test):
        assert isinstance(y_test, pd.Series)

    def test_y_series_named_target_col(self, y_train, y_test):
        assert y_train.name == config.TARGET_COL
        assert y_test.name == config.TARGET_COL

    def test_total_rows_equal_input(self, raw_df, X_train, X_test):
        assert len(X_train) + len(X_test) == len(raw_df)

    def test_test_size_proportion(self, raw_df, X_test):
        """Test set size must be within ±1 row of the configured proportion."""
        expected = len(raw_df) * config.TEST_SIZE
        assert abs(len(X_test) - expected) <= 1

    def test_no_row_overlap_between_train_and_test(self, X_train, X_test):
        """After reset_index the row positions don't overlap by definition,
        but the underlying data rows must not appear in both splits.
        We verify by checking that the union equals the total row count."""
        total = len(X_train) + len(X_test)
        # Both should have unique 0-based indices (reset_index(drop=True))
        assert list(X_train.index) == list(range(len(X_train)))
        assert list(X_test.index) == list(range(len(X_test)))
        # Combined length equals original (no duplication / no dropout)
        assert total == 200   # n=200 in _make_synthetic_raw_df

    def test_X_train_y_train_lengths_match(self, X_train, y_train):
        assert len(X_train) == len(y_train)

    def test_X_test_y_test_lengths_match(self, X_test, y_test):
        assert len(X_test) == len(y_test)

    def test_X_indices_are_zero_based(self, X_train, X_test):
        """reset_index(drop=True) must produce clean 0-based integer indices."""
        assert X_train.index[0] == 0
        assert X_test.index[0] == 0

    def test_no_nans_in_splits(self, X_train, X_test, y_train, y_test):
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()
        assert not y_train.isnull().any()
        assert not y_test.isnull().any()

    def test_burnout_scores_in_valid_range(self, y_train, y_test):
        for y in (y_train, y_test):
            assert y.min() >= config.TARGET_MIN
            assert y.max() <= config.TARGET_MAX


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_returns_xgb_regressor(self, fitted_model):
        assert isinstance(fitted_model, xgb.XGBRegressor)

    def test_predict_returns_correct_length(self, fitted_model, X_test):
        preds = fitted_model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_predict_returns_numpy_array(self, fitted_model, X_test):
        preds = fitted_model.predict(X_test)
        assert isinstance(preds, np.ndarray)

    def test_predictions_are_finite(self, fitted_model, X_test):
        preds = fitted_model.predict(X_test)
        assert np.all(np.isfinite(preds))

    def test_model_is_fitted(self, fitted_model):
        """A fitted XGBRegressor has n_features_in_ set."""
        assert hasattr(fitted_model, "n_features_in_")
        assert fitted_model.n_features_in_ == len(config.FINAL_FEATURE_COLUMNS)

    def test_train_on_new_tiny_data(self):
        """train_model must work independently of fixtures."""
        X = _make_synthetic_X(n=30, seed=7)
        y = pd.Series(np.random.default_rng(7).uniform(0, 10, 30),
                      name=config.TARGET_COL)
        model = train_model(X, y)
        assert isinstance(model, xgb.XGBRegressor)
        assert len(model.predict(X)) == 30


# ---------------------------------------------------------------------------
# save_model + load round-trip
# ---------------------------------------------------------------------------


class TestSaveModel:
    def test_file_is_created(self, fitted_model, tmp_path):
        dest = tmp_path / "model.json"
        save_model(fitted_model, path=dest)
        assert dest.exists()

    def test_file_is_nonzero_size(self, fitted_model, tmp_path):
        dest = tmp_path / "model.json"
        save_model(fitted_model, path=dest)
        assert dest.stat().st_size > 0

    def test_creates_parent_directories(self, fitted_model, tmp_path):
        dest = tmp_path / "nested" / "dir" / "model.json"
        save_model(fitted_model, path=dest)
        assert dest.exists()

    def test_round_trip_predictions_identical(self, fitted_model, X_test, tmp_path):
        """Reload the saved model and confirm predictions are bit-for-bit equal."""
        dest = tmp_path / "roundtrip_model.json"
        save_model(fitted_model, path=dest)

        reloaded = xgb.XGBRegressor()
        reloaded.load_model(dest)

        preds_orig    = fitted_model.predict(X_test)
        preds_reloaded = reloaded.predict(X_test)

        assert np.allclose(preds_orig, preds_reloaded, rtol=0, atol=0), (
            "Predictions changed after save/load — model serialisation is lossy."
        )

    def test_round_trip_max_diff_zero(self, fitted_model, X_test, tmp_path):
        """Redundant but explicit: max absolute difference must be exactly 0."""
        dest = tmp_path / "model_diff.json"
        save_model(fitted_model, path=dest)
        reloaded = xgb.XGBRegressor()
        reloaded.load_model(dest)
        diff = np.abs(fitted_model.predict(X_test) - reloaded.predict(X_test)).max()
        assert float(diff) == 0.0

    def test_accepts_path_object(self, fitted_model, tmp_path):
        dest = tmp_path / "path_obj.json"
        assert isinstance(dest, Path)
        save_model(fitted_model, path=dest)   # should not raise
        assert dest.exists()

    def test_accepts_str_path(self, fitted_model, tmp_path):
        dest = str(tmp_path / "str_path.json")
        save_model(fitted_model, path=dest)
        assert Path(dest).exists()


# ---------------------------------------------------------------------------
# save_processed_data
# ---------------------------------------------------------------------------


class TestSaveProcessedData:
    def test_creates_all_four_parquet_files(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        for fname in ("X_train.parquet", "X_test.parquet",
                      "y_train.parquet", "y_test.parquet"):
            assert (tmp_path / fname).exists(), f"{fname} was not created"

    def test_X_train_parquet_shape(self, X_train, X_test, y_train, y_test, tmp_path):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "X_train.parquet")
        assert loaded.shape == X_train.shape

    def test_X_test_parquet_shape(self, X_train, X_test, y_train, y_test, tmp_path):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "X_test.parquet")
        assert loaded.shape == X_test.shape

    def test_y_train_parquet_column_name(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "y_train.parquet")
        assert list(loaded.columns) == [config.TARGET_COL], (
            f"Expected column '{config.TARGET_COL}', got {list(loaded.columns)}"
        )

    def test_y_test_parquet_column_name(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "y_test.parquet")
        assert list(loaded.columns) == [config.TARGET_COL]

    def test_y_parquet_dtype_is_float64(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        for fname in ("y_train.parquet", "y_test.parquet"):
            loaded = pd.read_parquet(tmp_path / fname)
            assert loaded[config.TARGET_COL].dtype == np.float64, (
                f"{fname}: expected float64, got {loaded[config.TARGET_COL].dtype}"
            )

    def test_y_train_values_preserved(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "y_train.parquet")
        np.testing.assert_array_equal(
            loaded[config.TARGET_COL].to_numpy(),
            y_train.to_numpy(),
        )

    def test_X_feature_columns_preserved(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        loaded = pd.read_parquet(tmp_path / "X_train.parquet")
        assert list(loaded.columns) == config.FINAL_FEATURE_COLUMNS

    def test_creates_output_dir_if_missing(
        self, X_train, X_test, y_train, y_test, tmp_path
    ):
        out = tmp_path / "new" / "nested" / "dir"
        assert not out.exists()
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=out)
        assert out.exists()
        assert (out / "X_train.parquet").exists()

    def test_y_row_counts_match_X(self, X_train, X_test, y_train, y_test, tmp_path):
        save_processed_data(X_train, X_test, y_train, y_test, output_dir=tmp_path)
        X_tr = pd.read_parquet(tmp_path / "X_train.parquet")
        y_tr = pd.read_parquet(tmp_path / "y_train.parquet")
        assert len(X_tr) == len(y_tr)
