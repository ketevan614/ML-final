"""MLflow pyfunc wrappers for NeuralForecast models.

The project requirement is that logged models accept the raw Walmart test rows,
not a preprocessed feature matrix. This wrapper stores the fitted
NeuralForecast object plus the raw metadata needed to reproduce the Nixtla
long-format frame at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc

from dataloader import merge_all
from features import build_features


class NeuralForecastRawPyFunc(mlflow.pyfunc.PythonModel):
    """Serve a fitted NeuralForecast model from raw Walmart rows."""

    def __init__(
        self,
        neuralforecast_model,
        features_df: pd.DataFrame,
        stores_df: pd.DataFrame,
        train_history_raw: pd.DataFrame,
        model_alias: str,
        uses_exog: bool = True,
        exog_cols: list | None = None,
    ):
        self.neuralforecast_model = neuralforecast_model
        self.features_df = features_df
        self.stores_df = stores_df
        self.train_history_raw = train_history_raw
        self.model_alias = model_alias
        self.uses_exog = uses_exog
        self.exog_cols = list(exog_cols or [])

    def _to_nixtla(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        nf = pd.DataFrame(
            {
                "unique_id": raw_frame["Store"].astype(str) + "_" + raw_frame["Dept"].astype(str),
                "ds": pd.to_datetime(raw_frame["Date"]),
            }
        )
        if not self.exog_cols:  # univariate model (e.g. PatchTST): no covariates to build
            return nf

        merged = merge_all(raw_frame.copy(), self.features_df, self.stores_df)
        X = build_features(merged, sales_history_df=None, encode_categoricals=True)
        X = X.select_dtypes(include=[np.number]).astype("float32")
        # exactly the columns the model was fitted on, in the same order
        X = X.reindex(columns=self.exog_cols, fill_value=0)
        for col in X.columns:
            nf[col] = X[col].to_numpy()
        return nf

    @staticmethod
    def _normalize_predictions(preds: pd.DataFrame) -> pd.DataFrame:
        preds = preds.reset_index()
        unnamed = [col for col in preds.columns if str(col).startswith("level_") or str(col) == "index"]
        if unnamed:
            preds = preds.drop(columns=unnamed)
        return preds

    def predict(self, context, model_input: pd.DataFrame, params=None):
        futr_df = self._to_nixtla(model_input)
        if self.uses_exog:
            preds = self._normalize_predictions(self.neuralforecast_model.predict(futr_df=futr_df))
        else:
            preds = self._normalize_predictions(self.neuralforecast_model.predict())

        aligned = futr_df[["unique_id", "ds"]].merge(
            preds[["unique_id", "ds", self.model_alias]],
            on=["unique_id", "ds"],
            how="left",
        )
        fallback = float(self.train_history_raw["Weekly_Sales"].mean())
        values = aligned[self.model_alias].fillna(fallback).to_numpy(dtype=float)
        return np.clip(values, 0, None)
