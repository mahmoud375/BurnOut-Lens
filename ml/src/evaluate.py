"""
Model evaluation: metrics computation and diagnostic plots.

Exports
-------
compute_metrics
    Scalar regression metrics (MAE, RMSE, R²) from arrays.
evaluate_model
    End-to-end evaluation on a fitted model + held-out test set.
plot_actual_vs_predicted
    Scatter plot of actuals vs. predictions with identity line.
plot_residuals
    Scatter plot of residuals vs. actual values with zero-line.

Design notes
------------
* All predictions are clipped to ``[TARGET_MIN, TARGET_MAX]`` **before**
  metrics are computed (via ``preprocessing.clip_prediction``).  This mirrors
  the production path — the API always clips before returning a score — so the
  numbers reported here are faithful to what users actually see.
* Plots are always saved to ``save_path`` when provided and the figure is
  closed immediately after saving (``plt.close``).  This is safe for
  non-interactive / headless environments (CI, training scripts running inside
  Docker).  When ``save_path`` is ``None`` the figure is handed to the caller
  via ``plt.show()`` for interactive use in a notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import config
from .preprocessing import clip_prediction

matplotlib.use("Agg")  # non-interactive backend — safe for headless runs

__all__ = [
    "compute_metrics",
    "evaluate_model",
    "plot_actual_vs_predicted",
    "plot_residuals",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics between true and predicted burnout scores.

    Parameters
    ----------
    y_true:
        Ground-truth burnout scores.  Shape ``(n,)``.
    y_pred:
        Predicted burnout scores.  Must already be clipped to
        ``[TARGET_MIN, TARGET_MAX]`` before calling this function.
        Shape ``(n,)``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"`` and their
        corresponding scalar values rounded to 6 decimal places.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"compute_metrics: y_true shape {y_true.shape} != "
            f"y_pred shape {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("compute_metrics: inputs must not be empty.")

    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 6),
        "r2":   round(float(r2_score(y_true, y_pred)), 6),
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Run full evaluation on a fitted model against a held-out test set.

    The prediction pipeline applied here is **identical** to the production
    path:

    1. ``model.predict(X_test)``  — raw model output (may exceed [0, 10]).
    2. ``clip_prediction(raw)``   — clip to ``[TARGET_MIN, TARGET_MAX]``.
    3. ``compute_metrics(...)``   — compute MAE, RMSE, R² on clipped values.

    A formatted summary is printed to stdout so the results are visible when
    running ``train.py`` from the command line.

    Parameters
    ----------
    model:
        A fitted model with a ``.predict(X)`` method (e.g.
        ``xgboost.XGBRegressor``).
    X_test:
        Feature matrix for the test split.  Must have columns in
        ``FINAL_FEATURE_COLUMNS`` order.
    y_test:
        True burnout scores for the test split.

    Returns
    -------
    dict[str, float]
        Same structure as :func:`compute_metrics`: ``{"mae", "rmse", "r2"}``.
    """
    raw_preds: np.ndarray = model.predict(X_test)
    clipped_preds: np.ndarray = clip_prediction(raw_preds)
    metrics = compute_metrics(np.asarray(y_test), clipped_preds)

    print(
        "\n── Evaluation Results ─────────────────────\n"
        f"  MAE  : {metrics['mae']:.4f}\n"
        f"  RMSE : {metrics['rmse']:.4f}\n"
        f"  R²   : {metrics['r2']:.4f}\n"
        "───────────────────────────────────────────\n"
    )
    return metrics


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of actual vs. predicted burnout scores.

    A diagonal identity line (y = x) is drawn as a reference: points on the
    line are perfect predictions, points above the line are over-predictions,
    and points below are under-predictions.

    Parameters
    ----------
    y_true:
        Ground-truth burnout scores.
    y_pred:
        Predicted burnout scores (should be clipped before plotting).
    save_path:
        If provided, the figure is saved to this path and the figure is closed.
        If ``None``, ``plt.show()`` is called (interactive / notebook use).
        Parent directories are created automatically.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="steelblue",
               facecolors="steelblue", linewidths=0.4, s=30, label="Predictions")

    # Identity line spanning the full target range
    lo, hi = config.TARGET_MIN, config.TARGET_MAX
    ax.plot([lo, hi], [lo, hi], color="tomato", linestyle="--",
            linewidth=1.5, label="Perfect prediction (y = x)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Burnout Score", fontsize=11)
    ax.set_ylabel("Predicted Burnout Score", fontsize=11)
    ax.set_title("Actual vs. Predicted Burnout Score", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()

    _save_or_show(fig, save_path)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of residuals (actual - predicted) vs. actual values.

    A horizontal line at residual = 0 is drawn as reference.  Ideally
    residuals should be randomly scattered around zero with no visible pattern.

    Parameters
    ----------
    y_true:
        Ground-truth burnout scores.
    y_pred:
        Predicted burnout scores (should be clipped before plotting).
    save_path:
        If provided, the figure is saved and closed.  If ``None``,
        ``plt.show()`` is called.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(y_true, residuals, alpha=0.5, edgecolors="steelblue",
               facecolors="steelblue", linewidths=0.4, s=30)
    ax.axhline(0, color="tomato", linestyle="--", linewidth=1.5,
               label="Zero residual")

    ax.set_xlabel("Actual Burnout Score", fontsize=11)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_title("Residual Plot", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save ``fig`` to ``save_path`` (creating parents) or call ``plt.show()``."""
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
