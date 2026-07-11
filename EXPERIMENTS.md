# Experiment Registry — Walmart Store Sales Forecasting

The full list of experiments we will run, in the exact MLflow structure the brief requires:
**one MLflow experiment per architecture**, and **runs inside it = pipeline stages**
(`_Cleaning → _Feature_Selection → _CrossValidation → _HPO → _Final`). Use nested runs so HPO
trials group under the HPO parent.

> Metric everywhere = **WMAE** (holiday weeks ×5). Never plain MAE/RMSE.
> Validation everywhere = `expanding_window_folds(dates, n_splits=3, horizon=39)`.
> Every `_Final` run logs the saved **Pipeline** (raw test → prediction). Overall best → Registry.

---

## What EVERY run logs (convention — apply to all)

| Kind | What we log |
|---|---|
| **params** | feature set id, encoding, objective, key hyperparams, `holiday_weight`, seed |
| **metrics** | `wmae_fold_0/1/2`, `wmae_cv_mean`, `wmae_cv_std`; for `_Final` also Kaggle public LB WMAE |
| **artifacts** | feature list (txt/json), plots (importance, forecast vs actual), and in `_Final` the Pipeline |
| **tags** | owner (keti/gega), stage, git commit |

---

## 0. `Baseline` experiment  (🟢 both — do FIRST, anchors the leaderboard)

| Run | What it does | Why |
|---|---|---|
| `Baseline_Seasonal_Naive` | predict = same (Store,Dept) week **52 weeks ago**; fallback = that series' mean | the number every real model must beat |
| `Baseline_WeekOfYear_Mean` | predict = mean sales per (Store, Dept, ISO-week) | slightly smarter naive |
| `Baseline_Submit` | build submission from best baseline, upload to Kaggle, record public WMAE | first LB anchor |

---

## 1. `LightGBM_Training`  ⭐ (🟣 Keti — likely overall winner)

| Run | What varies / what we test |
|---|---|
| `LightGBM_Cleaning` | negatives: **keep vs clip-at-0**; markdown NaN: **fill 0 vs missing-flag**; pick best on holdout |
| `LightGBM_Feature_Selection` | fit once → read gain importance → drop dead features → re-score; log kept set |
| `LightGBM_CrossValidation` | 3 expanding folds with default-ish params; objective **`l1` vs `l2` vs `tweedie`**; `sample_weight=5` on holidays **on vs off**; native categoricals (Store/Dept/Type) |
| `LightGBM_HPO` | **Optuna** (nested child run per trial): `num_leaves, learning_rate, min_child_samples, feature_fraction, bagging_fraction, n_estimators` |
| `LightGBM_Final` | best params, fit on full train, wrap in **Pipeline**, `mlflow.sklearn.log_model` |

## 2. `XGBoost_Training`  (🔵 Gega — the other tree model)

| Run | What varies / what we test |
|---|---|
| `XGBoost_Cleaning` | same negatives/markdown decisions as LGBM (reuse the winning choice, but log it) |
| `XGBoost_Feature_Selection` | importance-based pruning; **encoding: ordinal vs one-hot** for Store/Dept/Type |
| `XGBoost_CrossValidation` | objective **`reg:absoluteerror` vs `reg:squarederror` vs `reg:pseudohubererror`**; holiday `sample_weight=5` |
| `XGBoost_HPO` | Optuna: `max_depth, eta, subsample, colsample_bytree, min_child_weight, n_estimators, reg_lambda` |
| `XGBoost_Final` | best params → Pipeline → log_model. **Compare speed & WMAE vs LightGBM** in README |

---

## 3. `DLinear_Training`  (🟣 Keti — via neuralforecast, GPU/Colab)

| Run | What varies / what we test |
|---|---|
| `DLinear_DataPrep` | Nixtla long format (`unique_id=Store_Dept, ds, y` + exogenous); decide handling of short/sparse series |
| `DLinear_CrossValidation` | global model; **with vs without exogenous** (Temp/Fuel/CPI/Unemp/MarkDowns); `input_size` sweep |
| `DLinear_HPO` | `input_size, learning_rate, max_steps` (+ scaler) |
| `DLinear_Final` | best config → **pyfunc** wrapper (raw in → preds), log_model |

## 4. `NBEATS_Training`  (🟣 Keti — via neuralforecast)

| Run | What varies / what we test |
|---|---|
| `NBEATS_DataPrep` | reuse the NF long-format prep |
| `NBEATS_CrossValidation` | **generic vs interpretable** (trend+seasonality) stacks; global over all series |
| `NBEATS_HPO` | `input_size, n_blocks, mlp_units, stack_types, learning_rate, max_steps` |
| `NBEATS_Final` | best config → pyfunc → log_model; **written comparison vs DLinear** (is the complexity worth it?) |

## 5. `PatchTST_Training`  (🔵 Gega — pick **PatchTST _or_ TFT**, one is enough)

| Run | What varies / what we test |
|---|---|
| `PatchTST_DataPrep` | NF long format |
| `PatchTST_CrossValidation` | global transformer; with exogenous |
| `PatchTST_HPO` | `patch_len, stride, hidden_size, n_heads, input_size, learning_rate` |
| `PatchTST_Final` | best config → pyfunc → log_model |

> If we choose **TFT** instead: same run structure, plus log a **variable-importance plot**
> (great for the report — TFT's native interpretability is its selling point).

---

## 6. `SARIMA_Training`  (🟣 Keti — THEORY-focused, tiny subset, don't burn training time)

| Run | What varies / what we test |
|---|---|
| `SARIMA_SubsetSelection` | pick a handful of high-volume (Store,Dept) series |
| `SARIMA_AutoARIMA` | `statsforecast.AutoARIMA`, weekly seasonality **m=52**, on the subset only |
| `SARIMA_Final` | subset forecasts + WMAE; README covers ACF/PACF, differencing, **why it doesn't scale to ~3000 series** |

## 7. `Prophet_Training`  (🔵 Gega — THEORY-focused, subset)

| Run | What varies / what we test |
|---|---|
| `Prophet_SubsetSelection` | same high-volume subset |
| `Prophet_Yearly` | yearly seasonality only (additive decomposition baseline) |
| `Prophet_Holidays` | + the 4 Walmart holidays (SuperBowl/LaborDay/Thanksgiving/Christmas) |
| `Prophet_Final` | subset WMAE; README covers trend+seasonality+holidays decomposition, pros/cons vs trees |

## 8. `TimesFM_Training`  (🔵 Gega — BONUS, optional)

| Run | What varies / what we test |
|---|---|
| `TimesFM_ZeroShot` | Google pretrained, no fine-tune, forecast the horizon |
| `TimesFM_Analysis` | why a general pretrained model may underperform trees tuned on this exact data |

---

## 9. Consolidation (🟢 both — Phase 3, not an "experiment")

1. `Model_Comparison` note — collect **CV WMAE + Kaggle LB WMAE** for every architecture into one table; confirm CV tracks LB.
2. Register overall best → `walmart-best` in the DagsHub **Model Registry**, promote to `@champion`.
3. `model_inference.ipynb` loads `models:/walmart-best/...` → predicts raw test → submission.

---

## Summary — experiment count

| # | MLflow Experiment | Runs | Owner | Required? |
|---|---|---|---|---|
| 0 | `Baseline` | 3 | 🟢 | yes |
| 1 | `LightGBM_Training` | 5 | 🟣 | yes |
| 2 | `XGBoost_Training` | 5 | 🔵 | yes |
| 3 | `DLinear_Training` | 4 | 🟣 | yes |
| 4 | `NBEATS_Training` | 4 | 🟣 | yes |
| 5 | `PatchTST_Training` (or TFT) | 4 | 🔵 | yes (one of the two) |
| 6 | `SARIMA_Training` | 3 | 🟣 | yes (theory) |
| 7 | `Prophet_Training` | 4 | 🔵 | yes (theory) |
| 8 | `TimesFM_Training` | 2 | 🔵 | **bonus / optional** |

**Total: 8 required architecture experiments + baseline (~32 core runs), + optional TimesFM.**
HPO runs spawn many nested child runs each, so the true run count is higher — that's expected.

### Open decisions to lock before starting
- [ ] **PatchTST vs TFT** — pick one (TFT gives a nicer variable-importance story for the report).
- [ ] **Attempting TimesFM bonus?** yes/no.
- [ ] Confirm negatives + markdown-NaN handling once in `LightGBM_Cleaning`, then reuse everywhere.
