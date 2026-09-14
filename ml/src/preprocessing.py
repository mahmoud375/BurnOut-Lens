"""
Preprocessing pipeline: raw DataFrame -> model-ready features.

Used by both:
  - ``train.py``  (on the full historical dataset at training time)
  - ``backend/prediction_service.py``  (on a single incoming API request)

Keeping this logic in one shared module guarantees that training and inference
**never diverge** — a classic and costly source of production bugs.

Pipeline order
--------------
1. ``select_raw_features`` — drop irrelevant columns, validate required ones exist.
2. ``clip_outliers``       — clip numeric features to plausible real-world bounds.
3. ``encode_seniority``    — convert the ordinal string to an integer.
4. ``reorder`` (inline)    — put columns in the exact order the model expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "clip_outliers",
    "clip_prediction",
    "encode_seniority",
    "preprocess",
    "select_raw_features",
]


# ---------------------------------------------------------------------------
# Individual transformation steps
# ---------------------------------------------------------------------------


def select_raw_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Return only the columns relevant to this project.

    Parameters
    ----------
    df:
        Source DataFrame (may contain extra columns from the raw CSV or the
        full request payload).
    include_target:
        If ``True`` (default), also keeps ``TARGET_COL``.  Set to ``False``
        when preprocessing a single API request that carries no label.

    Returns
    -------
    pd.DataFrame
        A copy with exactly the required columns, in ``RAW_FEATURE_COLUMNS``
        order (plus ``TARGET_COL`` at the end when requested).

    Raises
    ------
    KeyError
        If any required feature column (or the target, when requested) is
        absent from ``df``.
    """
    required = list(config.RAW_FEATURE_COLUMNS)
    if include_target:
        required.append(config.TARGET_COL)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"select_raw_features: the following required columns are missing "
            f"from the input DataFrame: {missing}"
        )

    return df[required].copy()


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip each numeric feature to a physically plausible range.

    This serves two purposes:
    1. **Training**: reduces the influence of extreme / erroneous values in
       the dataset (we observed noisy tails near the clipped 0 / 10 extremes
       in our EDA).
    2. **Inference**: protects the model from nonsensical API inputs such as
       ``sleep_hours_per_night=-3`` or ``work_hours_per_week=200``.

    Only columns present in both *df* and ``FEATURE_BOUNDS`` are touched.
    ``seniority_level`` (string) is intentionally absent from ``FEATURE_BOUNDS``
    and is left untouched here — it is validated and encoded separately.

    Parameters
    ----------
    df:
        DataFrame after ``select_raw_features``.

    Returns
    -------
    pd.DataFrame
        A copy with outlier values clipped to ``[low, high]`` per feature.
    """
    df = df.copy()
    for col, (low, high) in config.FEATURE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def encode_seniority(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``seniority_level`` (string) -> ``seniority_level_encoded`` (ordinal int).

    The encoding follows ``SENIORITY_ORDER``:
    ``Junior=0, Mid=1, Senior=2, Lead=3, Manager=4, Principal=5``.

    The original ``seniority_level`` column is dropped after encoding so that
    the returned DataFrame only contains numeric features.

    Parameters
    ----------
    df:
        DataFrame that must contain a ``seniority_level`` column.

    Returns
    -------
    pd.DataFrame
        A copy with ``seniority_level`` replaced by ``seniority_level_encoded``
        (dtype ``int64``).

    Raises
    ------
    KeyError
        If ``seniority_level`` column is absent from ``df``.
    ValueError
        If any value in ``seniority_level`` is ``NaN`` or not in
        ``SENIORITY_ORDER`` (e.g. a typo like ``"senior"`` or a new tier not
        yet registered in config).
    """
    if "seniority_level" not in df.columns:
        raise KeyError(
            "encode_seniority: 'seniority_level' column is missing from the DataFrame."
        )

    df = df.copy()

    # Detect NaN inputs *before* mapping so the error message is explicit.
    null_mask = df["seniority_level"].isnull()
    if null_mask.any():
        raise ValueError(
            f"encode_seniority: 'seniority_level' contains {null_mask.sum()} "
            f"null value(s). All rows must have a valid seniority level."
        )

    df["seniority_level_encoded"] = df["seniority_level"].map(config.SENIORITY_ORDER)

    # After mapping, any value that wasn't in SENIORITY_ORDER becomes NaN.
    unknown_mask = df["seniority_level_encoded"].isnull()
    if unknown_mask.any():
        unknown_vals = df.loc[unknown_mask, "seniority_level"].unique().tolist()
        valid_vals = list(config.SENIORITY_ORDER.keys())
        raise ValueError(
            f"encode_seniority: unknown seniority_level value(s) {unknown_vals}. "
            f"Valid values are: {valid_vals}"
        )

    df["seniority_level_encoded"] = df["seniority_level_encoded"].astype(int)
    df = df.drop(columns=["seniority_level"])
    return df


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline, top to bottom.

    Steps (in order):
      1. **select** — keep only relevant columns and validate they exist.
      2. **clip**   — clip numeric features to plausible bounds.
      3. **encode** — convert ``seniority_level`` to an ordinal integer.
      4. **reorder** — enforce column order matching ``FINAL_FEATURE_COLUMNS``
         (plus ``TARGET_COL`` when present).

    This function is the **single entrypoint** that both the training script
    and the backend prediction service call.  Never add ad-hoc transformations
    outside this function.

    Parameters
    ----------
    df:
        Raw input DataFrame.  May contain extra columns (they are dropped by
        ``select_raw_features``).
    include_target:
        Pass ``False`` when calling from the API (no label available).

    Returns
    -------
    pd.DataFrame
        Model-ready DataFrame with dtype-safe numeric columns in the order
        ``FINAL_FEATURE_COLUMNS`` (+ ``TARGET_COL`` appended when requested).

    Raises
    ------
    KeyError
        If required columns are missing.
    ValueError
        If ``seniority_level`` contains unknown or null values.
    """
    df = select_raw_features(df, include_target=include_target)
    df = clip_outliers(df)
    df = encode_seniority(df)

    ordered_cols = list(config.FINAL_FEATURE_COLUMNS)
    if include_target:
        ordered_cols.append(config.TARGET_COL)

    return df[ordered_cols]


# ---------------------------------------------------------------------------
# Post-prediction clipping
# ---------------------------------------------------------------------------


def clip_prediction(y_pred: np.ndarray) -> np.ndarray:
    """Clip model output(s) to the valid target range ``[TARGET_MIN, TARGET_MAX]``.

    XGBoost (like any unconstrained regressor) can predict values outside the
    training range, particularly at the extremes.  Our EDA confirmed negative
    predictions near ``burnout_score = 0``.  This guard ensures the API always
    returns a value in ``[0.0, 10.0]``.

    Parameters
    ----------
    y_pred:
        Raw model predictions as a NumPy array (shape ``(n,)`` or scalar).

    Returns
    -------
    np.ndarray
        Clipped predictions in ``[TARGET_MIN, TARGET_MAX]``.
    """
    return np.clip(y_pred, config.TARGET_MIN, config.TARGET_MAX)
