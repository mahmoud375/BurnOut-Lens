"""
SHAP-based explainability for the BurnOut Lens XGBoost model.

Exports
-------
get_shap_explainer
    Build a ``shap.TreeExplainer`` from a fitted XGBoost model.
explain_prediction
    Per-row explanation dict, ready for JSON serialisation in the API.
explain_batch
    Raw SHAP value array for a full DataFrame (global / offline analysis).
plot_global_importance
    SHAP summary (bar) plot saved to disk or shown interactively.

Design notes
------------
``explain_prediction`` is the hot path called by the backend on every request.
Its output contract is:

.. code-block:: python

    {
        "base_value": float,
        "prediction": float,
        "contributions": [
            {"feature": str, "value": float, "shap_value": float,
             "direction": "increases" | "decreases"},
            ...            # sorted by |shap_value| descending
        ]
    }

All numeric values are cast to plain Python ``float`` so the dict is
unconditionally JSON-serialisable without a custom encoder.

Reconstruction guarantee
------------------------
``base_value + sum(c["shap_value"] for c in contributions) ≈ prediction``

XGBoost's ``TreeExplainer`` guarantees this within floating-point precision
(~1e-5 tolerance in practice).  The ``prediction`` field stores the
reconstructed value (not ``model.predict()``), which may differ by up to the
same small epsilon — both reflect the same underlying computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config

matplotlib.use("Agg")  # headless-safe; must be set before any pyplot import

__all__ = [
    "explain_batch",
    "explain_prediction",
    "get_shap_explainer",
    "plot_global_importance",
]


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------


def get_shap_explainer(model: Any) -> shap.TreeExplainer:
    """Build a ``shap.TreeExplainer`` from a fitted XGBoost (or tree) model.

    Parameters
    ----------
    model:
        A fitted tree-based model — in this project always an
        ``xgboost.XGBRegressor``.  Must be already trained.

    Returns
    -------
    shap.TreeExplainer
        The explainer instance.  Building it is cheap (microseconds) so
        callers may construct one per request if needed, but caching it at
        the process level (e.g. in ``model_loader.py``) is recommended to
        amortise any initialisation overhead.

    Notes
    -----
    ``feature_perturbation="tree_path_dependent"`` (the default) is used.
    This is the recommended setting for XGBoost because it uses the model's
    internal structure rather than a background dataset, making it both faster
    and deterministic.
    """
    return shap.TreeExplainer(model)


# ---------------------------------------------------------------------------
# Per-prediction explanation (API hot path)
# ---------------------------------------------------------------------------


def explain_prediction(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
) -> dict[str, Any]:
    """Compute a structured SHAP explanation for a single input row.

    Parameters
    ----------
    explainer:
        A ``shap.TreeExplainer`` built from the fitted model (see
        :func:`get_shap_explainer`).
    X_row:
        A **single-row** DataFrame whose columns are exactly
        ``FINAL_FEATURE_COLUMNS`` in the correct order.  The values should
        already be preprocessed (clipped, encoded).

    Returns
    -------
    dict
        A JSON-serialisable dictionary:

        .. code-block:: python

            {
                "base_value": float,       # model's global mean prediction
                "prediction": float,       # base_value + sum(shap_values)
                "contributions": [
                    {
                        "feature":    str,   # column name
                        "value":      float, # feature value for this row
                        "shap_value": float, # SHAP contribution
                        "direction":  str,   # "increases" or "decreases"
                    },
                    ...  # sorted by |shap_value| descending
                ]
            }

        All numeric fields are native Python ``float`` (not ``numpy.float32``
        or similar) so ``json.dumps()`` works without a custom encoder.

    Raises
    ------
    ValueError
        If ``X_row`` has more or fewer than one row, or if its columns do
        not match ``FINAL_FEATURE_COLUMNS``.
    """
    if len(X_row) != 1:
        raise ValueError(
            f"explain_prediction expects exactly 1 row, got {len(X_row)}."
        )
    _validate_feature_columns(X_row)

    # shap_values returns shape (1, n_features) for a single-row DataFrame
    raw_shap: np.ndarray = explainer.shap_values(X_row)  # shape (1, n_features)
    sv: np.ndarray = raw_shap[0]                         # shape (n_features,)

    # expected_value may be a scalar float32 or a 1-d ndarray of shape (1,)
    # depending on the SHAP version and model type (XGBoost regressor returns
    # shape (1,) in SHAP ≥ 0.46).  Use .flat[0] to safely extract a scalar
    # regardless of shape.
    base_value: float = float(np.asarray(explainer.expected_value).flat[0])
    prediction: float = float(base_value + sv.sum())

    feature_names: list[str] = list(X_row.columns)
    feature_values: list[float] = [float(v) for v in X_row.iloc[0].tolist()]

    contributions = [
        {
            "feature":    fname,
            "value":      fval,
            "shap_value": float(sval),
            "direction":  "increases" if sval >= 0.0 else "decreases",
        }
        for fname, fval, sval in zip(feature_names, feature_values, sv)
    ]

    # Sort by absolute contribution, most impactful first
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return {
        "base_value":    base_value,
        "prediction":    prediction,
        "contributions": contributions,
    }


# ---------------------------------------------------------------------------
# Batch SHAP (training / offline analysis)
# ---------------------------------------------------------------------------


def explain_batch(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute raw SHAP values for every row in ``X``.

    This is intended for offline use (global importance analysis, notebook
    exploration, summary plots) — **not** the API hot path.  Use
    :func:`explain_prediction` for single-row inference.

    Parameters
    ----------
    explainer:
        A ``shap.TreeExplainer`` built from the fitted model.
    X:
        DataFrame with ``FINAL_FEATURE_COLUMNS`` columns.

    Returns
    -------
    np.ndarray
        SHAP value matrix of shape ``(n_samples, n_features)``.  Each entry
        ``[i, j]`` is the SHAP contribution of feature ``j`` for sample ``i``.
    """
    _validate_feature_columns(X)
    return explainer.shap_values(X)


# ---------------------------------------------------------------------------
# Global importance plot
# ---------------------------------------------------------------------------


def plot_global_importance(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """SHAP global feature importance bar chart.

    Plots mean absolute SHAP values per feature (the most interpretable
    single-number global importance measure for regression models).

    Parameters
    ----------
    explainer:
        A ``shap.TreeExplainer`` built from the fitted model.
    X:
        Representative dataset (e.g. the test split) used to compute mean
        SHAP values.  Larger samples give more stable estimates.
    save_path:
        If provided, the figure is saved here and closed (headless-safe).
        If ``None``, ``plt.show()`` is called (interactive / notebook use).
        Parent directories are created automatically.

    Notes
    -----
    Uses a manual bar chart rather than ``shap.summary_plot`` so we can
    apply the ``Agg`` backend consistently and reliably close the figure.
    ``shap.summary_plot`` creates its own figure internally which can
    interfere with the module-level ``Agg`` setting in some environments.
    """
    _validate_feature_columns(X)
    shap_values: np.ndarray = explainer.shap_values(X)      # (n_samples, n_features)
    mean_abs_shap: np.ndarray = np.abs(shap_values).mean(axis=0)  # (n_features,)

    feature_names = list(X.columns)
    order = np.argsort(mean_abs_shap)  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.45)))
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs_shap[order],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_feature_columns(X: pd.DataFrame) -> None:
    """Raise ``ValueError`` if X's columns don't match ``FINAL_FEATURE_COLUMNS``."""
    expected = config.FINAL_FEATURE_COLUMNS
    actual = list(X.columns)
    if actual != expected:
        raise ValueError(
            f"explain: DataFrame columns do not match FINAL_FEATURE_COLUMNS.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {actual}"
        )
