"""Feature engineering for the Walmart Store Sales Forecasting project.

The central function is `build_features`. It accepts a merged Walmart frame
containing raw train/test rows plus `features.csv` and `stores.csv` columns, and
returns a model matrix.

Target-derived features are deliberately history-only:
- lag features are created by joining sales from earlier calendar weeks;
- historical averages use only rows with dates before the current row.

That matters because the Kaggle test set is a 39-week future horizon. Features
that depend on current/future `Weekly_Sales` would leak the answer and produce
validation scores that cannot be reproduced on the real test set.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


HOLIDAY_WEEK_BY_DATE = {
    "2010-02-12": "super_bowl",
    "2011-02-11": "super_bowl",
    "2012-02-10": "super_bowl",
    "2013-02-08": "super_bowl",
    "2010-09-10": "labor_day",
    "2011-09-09": "labor_day",
    "2012-09-07": "labor_day",
    "2013-09-06": "labor_day",
    "2010-11-26": "thanksgiving",
    "2011-11-25": "thanksgiving",
    "2012-11-23": "thanksgiving",
    "2013-11-29": "thanksgiving",
    "2010-12-31": "christmas",
    "2011-12-30": "christmas",
    "2012-12-28": "christmas",
    "2013-12-27": "christmas",
}

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
BASE_FEATURE_COLS = [
    "Store",
    "Dept",
    "IsHoliday",
    "Type",
    "Size",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    *MARKDOWN_COLS,
]


def _as_datetime(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add date, seasonality, and named-holiday features.

    Walmart rows are weekly and the date is the Friday of that retail week.
    The competition's `IsHoliday` flag marks four events: Super Bowl, Labor Day,
    Thanksgiving, and Christmas. Separate boolean columns help a model learn
    that those holidays have different sales effects.
    """
    out = _as_datetime(df, date_col=date_col)
    date = out[date_col]
    iso = date.dt.isocalendar()

    out["Year"] = date.dt.year.astype("int16")
    out["Month"] = date.dt.month.astype("int8")
    out["Quarter"] = date.dt.quarter.astype("int8")
    out["WeekOfYear"] = iso.week.astype("int16")
    out["DayOfYear"] = date.dt.dayofyear.astype("int16")
    out["DaysFromStart"] = (date - date.min()).dt.days.astype("int16")

    out["WeekSin"] = np.sin(2 * np.pi * out["WeekOfYear"] / 52.0)
    out["WeekCos"] = np.cos(2 * np.pi * out["WeekOfYear"] / 52.0)
    out["MonthSin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["MonthCos"] = np.cos(2 * np.pi * out["Month"] / 12.0)

    holiday_name = date.dt.strftime("%Y-%m-%d").map(HOLIDAY_WEEK_BY_DATE).fillna("none")
    out["HolidayName"] = holiday_name
    for name in ["super_bowl", "labor_day", "thanksgiving", "christmas"]:
        out[f"is_{name}"] = (holiday_name == name).astype("int8")

    christmas = pd.to_datetime(date.dt.year.astype(str) + "-12-25")
    days_to_christmas = (christmas - date).dt.days
    out["DaysToChristmas"] = days_to_christmas.astype("int16")
    out["is_pre_christmas"] = days_to_christmas.between(0, 14).astype("int8")
    out["is_december"] = (out["Month"] == 12).astype("int8")
    return out


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill markdown NaNs with 0 and add missingness/promo intensity features.

    The MarkDown columns are mostly unavailable before November 2011. Filling
    with zero says "no known markdown amount"; the missing flags let the model
    distinguish true zero promotion from unavailable promotion data.
    """
    out = df.copy()
    present_cols = [col for col in MARKDOWN_COLS if col in out.columns]
    for col in present_cols:
        out[f"{col}_missing"] = out[col].isna().astype("int8")
        out[col] = out[col].fillna(0.0)
        out[f"{col}_log1p"] = np.log1p(out[col].clip(lower=0))

    if present_cols:
        out["MarkdownMissingCount"] = out[[f"{c}_missing" for c in present_cols]].sum(axis=1)
        out["AnyMarkdown"] = (out[present_cols].sum(axis=1) > 0).astype("int8")
        out["TotalMarkdown"] = out[present_cols].sum(axis=1)
        out["TotalMarkdown_log1p"] = np.log1p(out["TotalMarkdown"].clip(lower=0))
    return out


def add_external_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute sparse external variables and keep missingness indicators.

    `CPI` and `Unemployment` are missing for some future test weeks. Store-level
    forward/backward fill keeps the values plausible for each store, and the
    `_missing` flags preserve the original data-availability signal.
    """
    out = df.copy()
    for col in ["CPI", "Unemployment"]:
        if col not in out.columns:
            continue
        out[f"{col}_missing"] = out[col].isna().astype("int8")
        if "Store" in out.columns and "Date" in out.columns:
            ordered = out.sort_values(["Store", "Date"])
            filled = ordered.groupby("Store", observed=False)[col].transform(lambda s: s.ffill().bfill())
            out.loc[ordered.index, col] = filled
        out[col] = out[col].fillna(out[col].median())

    for col in ["Temperature", "Fuel_Price", "Size"]:
        if col in out.columns and out[col].isna().any():
            out[f"{col}_missing"] = out[col].isna().astype("int8")
            out[col] = out[col].fillna(out[col].median())
    return out


def add_temperature_fuel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable features derived from temperature and fuel price.

    Raw `Temperature` and `Fuel_Price` are useful, but derived columns can make
    common nonlinear effects easier for a model to learn:
    - extreme weather flags for unusually cold/hot shopping weeks;
    - distance from a comfortable temperature range;
    - store-level deviations, meaning "hotter/colder or more expensive than
      this store's usual value";
    - simple temperature/fuel interactions.
    """
    out = df.copy()

    if "Temperature" in out.columns:
        out["TempCold"] = (out["Temperature"] < 32).astype("int8")
        out["TempHot"] = (out["Temperature"] > 85).astype("int8")
        out["TempMild"] = out["Temperature"].between(55, 75).astype("int8")
        out["TempComfortDistance"] = np.where(
            out["Temperature"] < 55,
            55 - out["Temperature"],
            np.where(out["Temperature"] > 75, out["Temperature"] - 75, 0),
        )
        if "Store" in out.columns:
            store_temp_mean = out.groupby("Store", observed=False)["Temperature"].transform("mean")
            out["TemperatureStoreDeviation"] = out["Temperature"] - store_temp_mean

    if "Fuel_Price" in out.columns:
        if "Store" in out.columns:
            store_fuel_mean = out.groupby("Store", observed=False)["Fuel_Price"].transform("mean")
            out["FuelPriceStoreDeviation"] = out["Fuel_Price"] - store_fuel_mean
        out["FuelPriceHigh"] = (out["Fuel_Price"] > out["Fuel_Price"].quantile(0.75)).astype("int8")

    if {"Temperature", "Fuel_Price"}.issubset(out.columns):
        out["TempFuelInteraction"] = out["Temperature"] * out["Fuel_Price"]
        out["ComfortDistanceFuelInteraction"] = out["TempComfortDistance"] * out["Fuel_Price"]

    return out


def _lag_sales(
    current: pd.DataFrame,
    history: pd.DataFrame,
    lags: Iterable[int],
    group_cols: tuple[str, ...],
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    out = current.copy()
    source_cols = [*group_cols, date_col, target_col]
    source = history[source_cols].dropna(subset=[target_col]).copy()

    for lag in lags:
        hist = source.copy()
        hist[date_col] = hist[date_col] + pd.to_timedelta(7 * lag, unit="D")
        hist = hist.rename(columns={target_col: f"sales_lag_{lag}"})
        out = out.merge(hist, on=[*group_cols, date_col], how="left")
    return out


def _historical_means(
    current: pd.DataFrame,
    history: pd.DataFrame,
    group_cols: tuple[str, ...],
    date_col: str,
    target_col: str,
    history_includes_current: bool = False,
) -> pd.DataFrame:
    """Create expanding historical target means without using the current row."""
    current_marked = current[[*group_cols, date_col]].copy()
    current_marked["_row_id"] = np.arange(len(current_marked))
    current_marked[target_col] = current[target_col] if target_col in current else np.nan

    if history_includes_current:
        combined = current_marked
    else:
        hist = history[[*group_cols, date_col, target_col]].copy()
        hist["_row_id"] = -1
        combined = pd.concat([hist, current_marked], ignore_index=True, sort=False)
    combined = _as_datetime(combined, date_col=date_col)
    combined["WeekOfYear"] = combined[date_col].dt.isocalendar().week.astype("int16")
    combined = combined.sort_values([*group_cols, "WeekOfYear", date_col, "_row_id"])

    def add_expanding_mean(keys: list[str], new_col: str) -> None:
        values = combined[target_col]
        valid = values.notna().astype(int)
        sums = values.fillna(0).groupby([combined[k] for k in keys], sort=False).cumsum()
        counts = valid.groupby([combined[k] for k in keys], sort=False).cumsum()
        prev_sums = sums - values.fillna(0)
        prev_counts = counts - valid
        combined[new_col] = prev_sums / prev_counts.replace(0, np.nan)

    add_expanding_mean([*group_cols, "WeekOfYear"], "same_week_history_mean")
    add_expanding_mean(list(group_cols), "store_dept_history_mean")
    add_expanding_mean(["Dept", "WeekOfYear"], "dept_week_history_mean")
    add_expanding_mean(["Store", "WeekOfYear"], "store_week_history_mean")

    engineered = combined[combined["_row_id"] >= 0].sort_values("_row_id")
    return engineered[
        [
            "same_week_history_mean",
            "store_dept_history_mean",
            "dept_week_history_mean",
            "store_week_history_mean",
        ]
    ].reset_index(drop=True)


def add_sales_history_features(
    df: pd.DataFrame,
    sales_history_df: pd.DataFrame | None = None,
    target_col: str = "Weekly_Sales",
    group_cols: tuple[str, ...] = ("Store", "Dept"),
    date_col: str = "Date",
    lags: tuple[int, ...] = (52,),
) -> pd.DataFrame:
    """Add sales-history features that are available at forecast time.

    `sales_history_df` should contain known historical sales. For training you
    can pass the train frame itself. For test prediction, pass the train frame
    as history and the raw test frame as `df`.
    """
    out = _as_datetime(df, date_col=date_col)
    if sales_history_df is None:
        if target_col not in out.columns:
            return out
        history = out
    else:
        history = _as_datetime(sales_history_df, date_col=date_col)

    if target_col not in history.columns:
        return out

    out = _lag_sales(out, history, lags, group_cols, date_col, target_col)
    means = _historical_means(
        out,
        history,
        group_cols,
        date_col,
        target_col,
        history_includes_current=sales_history_df is None,
    )
    for col in means.columns:
        out[col] = means[col].to_numpy()

    fallback = history[target_col].mean()
    for col in [
        *[f"sales_lag_{lag}" for lag in lags],
        "same_week_history_mean",
        "store_dept_history_mean",
        "dept_week_history_mean",
        "store_week_history_mean",
    ]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
            out[col] = out[col].fillna(fallback)
    return out


def build_features(
    df: pd.DataFrame,
    sales_history_df: pd.DataFrame | None = None,
    target_col: str = "Weekly_Sales",
    encode_categoricals: bool = True,
) -> pd.DataFrame:
    """Build the shared feature matrix.

    Parameters
    ----------
    df:
        Merged train/test rows. Expected columns include Store, Dept, Date,
        IsHoliday, store metadata, and the external features from `features.csv`.
    sales_history_df:
        Known historical sales used for lag and historical-average features.
        Pass the merged training frame when building test features.
    target_col:
        Name of the target column, usually `Weekly_Sales`.
    encode_categoricals:
        If True, one-hot encode store Type and holiday name. If False, keep them
        as pandas categorical columns for libraries such as LightGBM.
    """
    out = add_calendar_features(df)
    out = add_markdown_features(out)
    out = add_external_missing_features(out)
    out = add_temperature_fuel_features(out)
    out = add_sales_history_features(out, sales_history_df=sales_history_df, target_col=target_col)

    if "IsHoliday" in out.columns:
        out["IsHoliday"] = out["IsHoliday"].astype("int8")

    if "Type" in out.columns:
        out["Type"] = pd.Categorical(out["Type"], categories=["A", "B", "C"])
    if "HolidayName" in out.columns:
        out["HolidayName"] = pd.Categorical(
            out["HolidayName"],
            categories=["none", "super_bowl", "labor_day", "thanksgiving", "christmas"],
        )

    drop_cols = ["Date"]
    if target_col in out.columns:
        drop_cols.append(target_col)
    X = out.drop(columns=[col for col in drop_cols if col in out.columns])

    if encode_categoricals:
        X = pd.get_dummies(X, columns=[c for c in ["Type", "HolidayName"] if c in X.columns], dummy_na=False)
    return X
