"""MLflow pyfunc wrapper for the per-series Prophet models.

Prophet fits one model per series and does not scale to the ~3,000 (Store, Dept)
pairs in this dataset, so the experiment notebook fits it on a high-volume subset.
This wrapper still meets the project's contract - take the raw Walmart test rows,
return one prediction per row - by serving Prophet where a model exists and falling
back to the series' historical mean everywhere else.

The fallback fraction is reported by `coverage()` and is the honest cost of the
per-series approach; it belongs in the README next to the WMAE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc

RAW_COLS = ["Store", "Dept", "Date"]


class ProphetSubsetPyFunc(mlflow.pyfunc.PythonModel):
    """Serve subset-fitted Prophet models from raw Walmart rows."""

    def __init__(self, models: dict, train_history: pd.DataFrame):
        self.models = models  # {(store, dept): fitted Prophet}
        self.series_mean = (
            train_history.groupby(["Store", "Dept"])["Weekly_Sales"].mean().to_dict()
        )
        self.global_mean = float(train_history["Weekly_Sales"].mean())

    def coverage(self, model_input: pd.DataFrame) -> float:
        """Fraction of rows a fitted Prophet model actually covers."""
        keys = list(zip(model_input["Store"], model_input["Dept"]))
        return float(np.mean([k in self.models for k in keys]))

    def predict(self, context, model_input: pd.DataFrame, params=None):
        raw = model_input[RAW_COLS].copy().reset_index(drop=True)
        raw["Date"] = pd.to_datetime(raw["Date"])
        out = np.empty(len(raw), dtype=float)

        for (store, dept), pos in raw.groupby(["Store", "Dept"]).indices.items():
            model = self.models.get((store, dept))
            if model is None:
                out[pos] = self.series_mean.get((store, dept), self.global_mean)
                continue
            fc = model.predict(pd.DataFrame({"ds": raw["Date"].to_numpy()[pos]}))
            out[pos] = fc["yhat"].to_numpy()

        return np.clip(out, 0, None)
