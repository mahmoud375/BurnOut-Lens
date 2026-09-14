"""
Preprocessing pipeline: raw dataframe -> model-ready features.

Used by both:
  - train.py (on the full historical dataset)
  - backend prediction_service.py (on a single incoming request)

Keeping this logic in one shared module guarantees training and inference
never diverge (a classic source of production bugs).
"""

import numpy as np
import pandas as pd

from . import config


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip each feature to a physically plausible range.

    This does two things:
    1. Protects the model from nonsensical inputs (e.g. someone claims
       40 hours of sleep, or -5 meetings per day) at inference time.
    2. Reduces the influence of extreme/rare values in the training data,
       which we observed were noisy near the dataset's clipped 0/10 tails.
    """
    df = df.copy()
    for col, (low, high) in config.FEATURE_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def encode_seniority(df: pd.DataFrame) -> pd.DataFrame:
    """Map seniority_level (string) -> seniority_level_encoded (ordinal int)."""
    df = df.copy()
    df["seniority_level_encoded"] = df["seniority_level"].map(config.SENIORITY_ORDER)

    if df["seniority_level_encoded"].isnull().any():
        unknown = df.loc[df["seniority_level_encoded"].isnull(), "seniority_level"].unique()
        raise ValueError(f"Unknown seniority_level value(s) encountered: {unknown}")

    df = df.drop(columns=["seniority_level"])
    return df


def select_raw_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Keep only the columns relevant to this project."""
    cols = config.RAW_FEATURE_COLUMNS.copy()
    if include_target:
        cols = cols + [config.TARGET_COL]
    return df[cols].copy()


def preprocess(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline, in order:
      1. select relevant columns
      2. clip outliers to plausible bounds
      3. encode seniority_level
      4. reorder columns to match FINAL_FEATURE_COLUMNS (+ target if present)

    Works identically whether `df` has 1 row (a live API request) or
    100,000 rows (the full training set).
    """
    df = select_raw_features(df, include_target=include_target)
    df = clip_outliers(df)
    df = encode_seniority(df)

    ordered_cols = config.FINAL_FEATURE_COLUMNS.copy()
    if include_target:
        ordered_cols = ordered_cols + [config.TARGET_COL]

    return df[ordered_cols]


def clip_prediction(y_pred: np.ndarray) -> np.ndarray:
    """
    Clip model outputs to the valid target range [0, 10].

    Required because XGBoost (like any regressor) can predict outside the
    training range, especially near the extremes — confirmed in our EDA
    (negative predictions observed near burnout_score = 0).
    """
    return np.clip(y_pred, config.TARGET_MIN, config.TARGET_MAX)