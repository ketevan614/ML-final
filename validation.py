"""Time-series aware validation splits for panel (multi-series) data.

Random K-fold leaks the future into the past. For forecasting we always split
on the time axis: train on everything up to a cutoff date, validate on the
weeks that follow. Because this dataset is a *panel* (many Store x Dept series
share the same dates), we split on the set of unique dates, NOT on row order.
"""
import numpy as np
import pandas as pd


def expanding_window_folds(dates, n_splits=3, horizon=39, step=None):
    """Yield (train_mask, val_mask) boolean arrays over an expanding window.

    Parameters
    ----------
    dates : array-like of datetime (one value per row; many rows share a date)
    n_splits : number of folds
    horizon : validation length in weeks (39 mirrors the real test horizon)
    step : weeks to move the cutoff back between folds (defaults to `horizon`,
           which gives non-overlapping validation windows)

    Folds are yielded earliest-first. Each fold trains on all dates before the
    cutoff and validates on the next `horizon` weeks.
    """
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    unique = np.array(sorted(dates.unique()))
    step = step or horizon

    folds = []
    end = len(unique)
    for _ in range(n_splits):
        val_start = end - horizon
        if val_start <= 0:
            break
        val_dates = set(unique[val_start:end])
        train_dates = set(unique[:val_start])
        folds.append((train_dates, val_dates))
        end -= step
    folds.reverse()

    for train_dates, val_dates in folds:
        train_mask = dates.isin(train_dates).to_numpy()
        val_mask = dates.isin(val_dates).to_numpy()
        yield train_mask, val_mask


def holdout_split(dates, horizon=39):
    """Single train/validation split: the last `horizon` weeks are validation.

    Returns (train_mask, val_mask).
    """
    dates = pd.to_datetime(pd.Series(np.asarray(dates)).reset_index(drop=True))
    unique = np.array(sorted(dates.unique()))
    cutoff = unique[-horizon]
    train_mask = (dates < cutoff).to_numpy()
    val_mask = (dates >= cutoff).to_numpy()
    return train_mask, val_mask