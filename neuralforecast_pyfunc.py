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
    ):
        self.neuralforecast_model = neuralforecast_model
        self.features_df = features_df
        self.stores_df = stores_df
        self.train_history_raw = train_history_raw
        self.model_alias = model_alias
        self.uses_exog = uses_exog

    def _to_nixtla(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        raw_frame = raw_frame.copy()
        merged = merge_all(raw_frame, self.features_df, self.stores_df)
        history_merged = merge_all(self.train_history_raw, self.features_df, self.stores_df)
        X = build_features(
            merged,
            sales_history_df=history_merged,
            encode_categoricals=True,
        )

        nf = pd.DataFrame(
            {
                "unique_id": raw_frame["Store"].astype(str) + "_" + raw_frame["Dept"].astype(str),
                "ds": pd.to_datetime(raw_frame["Date"]),
            }
        )
        for col in X.columns:
            values = X[col]
            if str(values.dtype) == "category":
                values = values.astype(str)
            nf[col] = values.to_numpy()
        return nf

    def predict(self, context, model_input: pd.DataFrame, params=None):
        futr_df = self._to_nixtla(model_input)
        if not self.uses_exog:
            futr_df = futr_df[["unique_id", "ds"]]

        preds = self.neuralforecast_model.predict(futr_df=futr_df)
        values = preds[self.model_alias].to_numpy(dtype=float)
        return np.clip(values, 0, None)
