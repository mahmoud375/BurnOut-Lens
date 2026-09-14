"""
Full training pipeline for the BurnOut Lens XGBoost model.

Entry point
-----------
Run from the ``ml/`` directory::

    uv run python -m src.train

or::

    python -m src.train

Outputs produced by ``main()``
-------------------------------
* ``ml/models/xgb_burnout_model.json``          — serialised XGBoost model
* ``ml/data/processed/X_train.parquet``         — training feature matrix
* ``ml/data/processed/X_test.parquet``          — test feature matrix
* ``ml/data/processed/y_train.parquet``         — training labels
* ``ml/data/processed/y_test.parquet``          — test labels
* ``ml/reports/metrics.json``                   — MAE / RMSE / R²
* ``ml/reports/actual_vs_predicted.png``        — diagnostic scatter plot
* ``ml/reports/residuals.png``                  — residual scatter plot
* ``ml/reports/shap_global_importance.png``     — SHAP feature importance bar

Design notes
------------
* ``os.environ["MPLBACKEND"] = "Agg"`` is set **before every other import**
  so the script is safe in headless environments (Docker, CI). Setting the
  backend via ``matplotlib.use()`` after pyplot has been imported elsewhere
  is silently ignored; the env-var approach is the only reliable method.

* Each pipeline step is a standalone function so individual pieces can be
  imported and tested without running the full pipeline (``main()`` just
  composes them in order).

* ``save_processed_data`` saves all four splits on every run so the
  persisted features are always in sync with the model artefact.  They are
  written to ``PROCESSED_DIR`` (tracked in ``.gitignore``) and are **not**
  committed to version control.
"""

from __future__ import annotations

# ── Set headless backend BEFORE any pyplot / evaluate / explain imports ──────
import os
os.environ.setdefault("MPLBACKEND", "Agg")

# ── Standard library ─────────────────────────────────────────────────────────
import json
import time
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ── Project modules ──────────────────────────────────────────────────────────
from . import config
from .evaluate import evaluate_model, plot_actual_vs_predicted, plot_residuals
from .explain import explain_batch, get_shap_explainer, plot_global_importance
from .preprocessing import clip_prediction, preprocess

__all__ = [
    "load_data",
    "split_data",
    "train_model",
    "save_model",
    "save_processed_data",
    "main",
]

# Default report directory: ml/reports/
_REPORTS_DIR: Path = config.BASE_DIR / "reports"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.  Defaults to ``config.RAW_DATA_PATH``.
        Override in tests to point at a synthetic fixture.

    Returns
    -------
    pd.DataFrame
        Raw data, unmodified.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    csv_path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"load_data: raw CSV not found at {csv_path}.\n"
            "Place the dataset in ml/data/raw/ before running training."
        )
    return pd.read_csv(csv_path)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess ``df`` and split into train / test sets.

    Steps performed:

    1. ``preprocess(df, include_target=True)`` — selects the 9 raw feature
       columns + target, clips outliers, encodes seniority.
    2. Separate ``X`` (``FINAL_FEATURE_COLUMNS``) from ``y`` (``TARGET_COL``).
    3. ``train_test_split`` using ``TEST_SIZE`` and ``RANDOM_STATE`` from
       config — reproducible across runs.

    Parameters
    ----------
    df:
        Raw DataFrame as returned by :func:`load_data`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``X_train, X_test, y_train, y_test``.

        * ``X_train / X_test``: DataFrames with ``FINAL_FEATURE_COLUMNS``
          columns, index reset to 0-based integers.
        * ``y_train / y_test``: Series named ``TARGET_COL``, index reset.
    """
    processed = preprocess(df, include_target=True)

    X = processed[config.FINAL_FEATURE_COLUMNS]
    y = processed[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # Reset indices so parquet files have clean 0-based row numbers
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBRegressor:
    """Fit an ``XGBRegressor`` on the training split.

    Hyperparameters come entirely from ``config.XGB_PARAMS`` — no
    cross-validation or search is performed at this stage.  The notebook
    experiments (``03_modeling.ipynb``) established this baseline achieves
    R² ≈ 0.70 on the held-out test set.

    Parameters
    ----------
    X_train:
        Feature matrix with columns ``FINAL_FEATURE_COLUMNS``.
    y_train:
        Target series with values in ``[TARGET_MIN, TARGET_MAX]``.

    Returns
    -------
    xgb.XGBRegressor
        Fitted model, ready for ``.predict()`` and ``.save_model()``.
    """
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


def save_model(
    model: xgb.XGBRegressor,
    path: Path | str | None = None,
) -> None:
    """Serialise ``model`` to disk in XGBoost's native JSON format.

    Parameters
    ----------
    model:
        A fitted ``XGBRegressor``.
    path:
        Destination path.  Defaults to ``config.MODEL_PATH``.
        ``xgboost.save_model()`` accepts both ``str`` and ``pathlib.Path``.

    Side effects
    ------------
    Creates the parent directory (and any ancestors) if they do not exist.
    """
    dest = Path(path) if path is not None else config.MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(dest)


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path | str | None = None,
) -> None:
    """Persist the four data splits as Parquet files.

    Parquet is preferred over CSV because it preserves dtypes exactly, is
    ~5–10× faster to read/write for columnar data, and is natively supported
    by pandas, PyArrow, and most data-warehouse tools.

    ``y_train`` and ``y_test`` are ``pd.Series`` objects.  Parquet does not
    support bare Series, so each is converted to a single-column DataFrame
    using the series name (``TARGET_COL``) as the column header.  The column
    name and ``float64`` dtype are preserved through the round-trip.

    Parameters
    ----------
    X_train, X_test:
        Feature DataFrames produced by :func:`split_data`.
    y_train, y_test:
        Target Series produced by :func:`split_data`.
    output_dir:
        Directory where the four ``.parquet`` files are written.
        Defaults to ``config.PROCESSED_DIR``.

    Side effects
    ------------
    Creates ``output_dir`` (and parents) if it does not already exist.
    Overwrites existing files silently.
    """
    out = Path(output_dir) if output_dir is not None else config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(out / "X_train.parquet", index=False)
    X_test.to_parquet(out / "X_test.parquet",  index=False)

    # Convert Series → single-column DataFrame to keep target column name
    y_train.to_frame().to_parquet(out / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(out / "y_test.parquet",  index=False)


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete BurnOut Lens training pipeline end-to-end.

    Execution order
    ---------------
    1. Load raw CSV.
    2. Preprocess and split into train / test sets.
    3. Save processed splits to ``PROCESSED_DIR``.
    4. Train XGBoost model.
    5. Evaluate on test set (clipped predictions → MAE / RMSE / R²).
    6. Save model to ``MODEL_PATH``.
    7. Save metrics JSON to ``reports/metrics.json``.
    8. Save diagnostic plots to ``reports/``.
    9. Build SHAP explainer and save global importance plot.
    10. Print a final summary of all artefacts.
    """
    t_start = time.perf_counter()
    sep = "─" * 50

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  BurnOut Lens — Training Pipeline")
    print(sep)
    print("\n[1/8] Loading raw data …")
    df = load_data()
    print(f"      {len(df):,} rows × {len(df.columns)} columns loaded from:\n"
          f"      {config.RAW_DATA_PATH}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n[2/8] Preprocessing & splitting …")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"      train: {len(X_train):,} rows  |  test: {len(X_test):,} rows")

    # ── 3. Save processed data ────────────────────────────────────────────────
    print("\n[3/8] Saving processed splits …")
    save_processed_data(X_train, X_test, y_train, y_test)
    for fname in ("X_train.parquet", "X_test.parquet",
                  "y_train.parquet", "y_test.parquet"):
        p = config.PROCESSED_DIR / fname
        print(f"      {p}  ({p.stat().st_size / 1024:.1f} KB)")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4/8] Training XGBoost model …")
    t_train = time.perf_counter()
    model = train_model(X_train, y_train)
    print(f"      done in {time.perf_counter() - t_train:.1f}s")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/8] Evaluating on test set …")
    metrics = evaluate_model(model, X_test, y_test)

    # ── 6. Save model ─────────────────────────────────────────────────────────
    print("[6/8] Saving model …")
    save_model(model)
    print(f"      {config.MODEL_PATH}  "
          f"({config.MODEL_PATH.stat().st_size / 1024:.1f} KB)")

    # ── 7. Save metrics JSON ──────────────────────────────────────────────────
    print("\n[7/8] Saving reports …")
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = _REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"      metrics  → {metrics_path}")

    # ── 8a. Diagnostic plots ──────────────────────────────────────────────────
    raw_preds = model.predict(X_test)
    clipped_preds = clip_prediction(raw_preds)

    avp_path = _REPORTS_DIR / "actual_vs_predicted.png"
    plot_actual_vs_predicted(y_test.to_numpy(), clipped_preds, save_path=avp_path)
    print(f"      plot     → {avp_path}")

    res_path = _REPORTS_DIR / "residuals.png"
    plot_residuals(y_test.to_numpy(), clipped_preds, save_path=res_path)
    print(f"      plot     → {res_path}")

    # ── 8b. SHAP global importance ────────────────────────────────────────────
    print("\n[8/8] Computing SHAP explanations …")
    explainer = get_shap_explainer(model)
    # Use the test set for global importance (representative, unseen data)
    shap_path = _REPORTS_DIR / "shap_global_importance.png"
    plot_global_importance(explainer, X_test, save_path=shap_path)
    print(f"      shap     → {shap_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n{sep}")
    print("  Training complete")
    print(sep)
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  Time : {elapsed:.1f}s")
    print(sep)
    print(f"\n  Artefacts written to:")
    print(f"    Model   : {config.MODEL_PATH}")
    print(f"    Data    : {config.PROCESSED_DIR}/")
    print(f"    Reports : {_REPORTS_DIR}/")
    print()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
