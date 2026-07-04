"""Evaluation metric for the Walmart Store Sales Forecasting task.

The competition uses WMAE (Weighted Mean Absolute Error), where weeks flagged
as holidays are weighted 5x and all other weeks are weighted 1x.

IMPORTANT: WMAE needs a per-row weight (IsHoliday) alongside y_true / y_pred.
Standard scikit-learn scorers only receive (y_true, y_pred), so you cannot plug
WMAE straight into `cross_val_score`. Instead compute it manually inside your
own validation loop (see `wmae_from_frame` for convenience).
"""
import numpy as np


def wmae(y_true, y_pred, is_holiday, holiday_weight=5.0):
    """Weighted Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like of true weekly sales
    y_pred : array-like of predicted weekly sales
    is_holiday : array-like of bool (True where the week is a holiday week)
    holiday_weight : weight applied to holiday weeks (default 5)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), holiday_weight, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def wmae_from_frame(
    df,
    y_true_col="Weekly_Sales",
    y_pred_col="y_pred",
    holiday_col="IsHoliday",
    holiday_weight=5.0,
):
    """Convenience wrapper to compute WMAE directly from a DataFrame."""
    return wmae(
        df[y_true_col], df[y_pred_col], df[holiday_col], holiday_weight=holiday_weight
    )