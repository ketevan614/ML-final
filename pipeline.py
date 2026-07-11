"""Raw -> prediction wrapper around the shared features.build_features.

Lets a fitted XGBoost model run directly on the raw Kaggle test rows
(Store, Dept, Date, IsHoliday): it merges the stores/features tables, calls
build_features with the training frame as sales history (so lag/history features
have no future leakage), aligns to the training columns, and predicts. This is
the object saved as the Pipeline and reloaded in model_inference.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline

from dataloader import merge_all
from features import build_features

RAW_COLS = ["Store", "Dept", "Date", "IsHoliday"]


class WalmartRegressor(BaseEstimator, RegressorMixin):
    """Fits an XGBoost model on features built from the raw train rows and keeps
    the training frame so predict() can rebuild history-based features for new
    rows. Holiday rows get weight 5 to match the WMAE metric."""

    def __init__(self, model, features_df, stores_df, clip_zero=True, holiday_weight=5.0):
        self.model = model
        self.features_df = features_df
        self.stores_df = stores_df
        self.clip_zero = clip_zero
        self.holiday_weight = holiday_weight

    def fit(self, X, y):
        raw = X[RAW_COLS].reset_index(drop=True)
        merged = merge_all(raw, self.features_df, self.stores_df)
        merged["Weekly_Sales"] = np.asarray(y, dtype=float)
        self.train_history_ = merged  # kept for lag/history at predict time

        Xtr = build_features(merged, sales_history_df=None,
                             encode_categoricals=True).astype("float32")
        self.feature_names_ = list(Xtr.columns)

        w = np.where(raw["IsHoliday"].astype(bool).to_numpy(), self.holiday_weight, 1.0)
        self.model_ = clone(self.model)
        self.model_.fit(Xtr, np.asarray(y, dtype=float), sample_weight=w)
        return self

    def predict(self, X):
        raw = X[RAW_COLS].reset_index(drop=True)
        merged = merge_all(raw, self.features_df, self.stores_df)
        Xte = build_features(merged, sales_history_df=self.train_history_,
                             encode_categoricals=True)
        Xte = Xte.reindex(columns=self.feature_names_, fill_value=0).astype("float32")
        pred = self.model_.predict(Xte)
        return np.clip(pred, 0, None) if self.clip_zero else pred


def make_pipeline(model, features_df, stores_df, clip_zero=True):
    """Wrap the model as a one-step sklearn Pipeline that runs on raw test rows."""
    return Pipeline([
        ("model", WalmartRegressor(model, features_df, stores_df, clip_zero=clip_zero)),
    ])
