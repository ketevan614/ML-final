# Task Board — Keti & Gega

Detailed, dependency-ordered tasks. Each task has: **owner · what to do · deliverable · what to log
to MLflow · depends on**. IDs are stable so you can reference them in commits (`git commit -m "T14: ..."`).

Legend: 🟣 **Keti**  🔵 **Gega**  🟢 **Both**

---

## PHASE 0 — Setup (🟢 both, day 1)

### T01 · 🟢 Fix data extraction so CSVs land where the loader expects
- **Do:** the nested zips didn't fully unpack — `data/.../train.csv` is an empty *folder* next to
  `train.csv.zip`. Fix so the 5 CSVs sit at `data/train.csv`, `data/test.csv`, `data/features.csv`,
  `data/stores.csv`, `data/sampleSubmission.csv` (that's what `dataloader.load_raw("data")` reads).
- **Deliverable:** `python -c "from dataloader import load_raw; print([d.shape for d in load_raw()])"`
  prints 5 shapes with no error.
- **Depends on:** —

### T02 · 🟢 Shared MLflow/DagsHub connection cell
- **Do:** agree on the `dagshub.init(repo_owner='karev23', repo_name='ML-final', mlflow=True)`
  header (Method 1). Put it in a tiny `tracking.py` with a helper `start(experiment)` both import,
  so every notebook connects identically. Each teammate authorizes with their own account.
- **Deliverable:** `tracking.py`; both confirm a dummy run shows on the Experiments page.
- **Depends on:** —

### T03 · 🟢 Add collaborators
- **Do:** make sure **both** people are collaborators on the GitHub repo AND the DagsHub repo, so
  runs + registered models land in one shared place. Add `python-dotenv` is NOT needed (Method 1).
- **Deliverable:** both can push to git and log runs.
- **Depends on:** —

---

## PHASE 1 — Shared foundation (🟢 both, days 2–4)

> Do these together (pair or split the halves below). Everything downstream depends on them, so
> get them RIGHT before splitting into models.


### T05 · 🟢 Shared feature engineering `features.py`
- **Do:** write `build_features(df_merged) -> DataFrame` used by ALL models. Include:
  calendar (Year, Month, ISO WeekOfYear, day-of-year, per-holiday flags, `is_pre_christmas`),
  store meta (Type, Size), exogenous (Temperature, Fuel_Price, CPI, Unemployment, MarkDown1–5 +
  a `markdown_missing` flag), and **lag-52 ("same week last year")** which is the only safe lag for
  a 39-week horizon. Document each feature in a docstring.
- **Deliverable:** `features.py` both teammates trust; a unit sanity check (no leakage: no feature
  uses data from the target week or future).
- **Depends on:** T04

### T06 · 🟢 Agree validation + WMAE harness
- **Do:** confirm everyone uses `validation.expanding_window_folds(dates, n_splits=3, horizon=39)`
  and scores with `metrics.wmae(y, yhat, is_holiday)` — NOT plain MAE. Write one helper
  `cv_score(model_fn) -> (per_fold_wmae, mean_wmae)` in `validation.py` or a `cv.py`, so every model
  notebook computes CV the same way.
- **Deliverable:** shared `cv_score` helper.
- **Depends on:** T05

### T07 · 🟢 Baseline model + first Kaggle submission
- **Do:** seasonal-naive — predict each (Store, Dept, week) as **the same week last year**
  (fallback to that series' mean when no year-ago value). Score locally with WMAE, build a
  submission with `dataloader.make_submission`, upload to Kaggle, record the public WMAE.
- **Deliverable:** `notebooks/01_baseline.ipynb`, one MLflow run `Baseline` in a `Baseline`
  experiment, and a number on the leaderboard that every real model must beat.
- **Depends on:** T06

### T08 · 🟢 `pipeline.py` scaffolding (the raw→prediction contract)
- **Do:** write the custom `WalmartPreprocessor(BaseEstimator, TransformerMixin)` that stores
  `features_df` + `stores_df` and, in `transform`, calls `merge_all` then `build_features`. This is
  the object that lets a saved Pipeline run on the **raw** test set. Both tree models will reuse it.
- **Deliverable:** `pipeline.py` with `WalmartPreprocessor` + a `make_pipeline(model)` helper.
- **Depends on:** T05

---

## PHASE 2 — Model tracks (split, days 5–12)

> For EVERY model notebook, follow the required MLflow structure:
> experiment = `<Arch>_Training`; **nested runs** named `<Arch>_Cleaning`, `<Arch>_Feature_Selection`,
> `<Arch>_CrossValidation`, `<Arch>_HPO`, `<Arch>_Final`. Log params, per-fold WMAE, mean CV WMAE,
> and (in `_Final`) the saved **Pipeline**. Write 2–3 takeaway sentences for the README.

### 🟣 KETI'S TRACK

#### T09 · 🟣 LightGBM — `model_experiment_LightGBM.ipynb`  ⭐ likely overall winner
- **Do:**
  - `_Cleaning`: handle negatives (test clip-at-0 vs keep), markdown NaNs. Log choices.
  - `_Feature_Selection`: fit once, read LightGBM feature importance, drop dead features, re-score.
  - `_CrossValidation`: run `cv_score` over the 3 expanding folds. Log per-fold + mean WMAE.
  - Use **native categoricals** (Store, Dept, Type). Objective `l1` (aligns with WMAE better than
    `l2`). Pass `sample_weight = 5 where IsHoliday` so training matches the metric.
  - `_HPO`: Optuna over `num_leaves, learning_rate, min_child_samples, feature_fraction,
    bagging_fraction, n_estimators`. Log best params.
  - `_Final`: wrap best model with `make_pipeline` (T08), fit on full train, `mlflow.sklearn.log_model`.
- **Deliverable:** tuned LightGBM Pipeline logged + CV/LB WMAE recorded.
- **Depends on:** T06, T08

#### T10 · 🟣 DLinear — `model_experiment_DLinear.ipynb`
- **Do:** via `neuralforecast` (uses the shared NF data-prep from T13). DLinear = simple linear
  trend/seasonal decomposition; fast, strong baseline. Train **globally** across all series with
  exogenous features. Score with WMAE on the val window.
- **Deliverable:** DLinear run(s) + Pipeline-style pyfunc wrapper, CV/LB WMAE.
- **Depends on:** T13 (Gega's NF setup)

#### T11 · 🟣 N-BEATS — `model_experiment_NBEATS.ipynb`
- **Do:** via `neuralforecast`. Pure-DL, interpretable trend/seasonality stacks. Global model over
  all series. Tune horizon/input-size/stacks briefly. Compare to DLinear — is the extra complexity
  worth it here?
- **Deliverable:** N-BEATS runs, CV/LB WMAE, written comparison vs DLinear.
- **Depends on:** T13

#### T12 · 🟣 SARIMA / ARIMA — `model_experiment_SARIMA.ipynb` (theory-focused)
- **Do:** brief says DON'T spend much training time. Fit `AutoARIMA` (from `statsforecast`) with
  weekly seasonality `m=52` on a **small subset** (a handful of high-volume Store×Dept series).
  In the notebook + README explain: what (S)ARIMA is, ACF/PACF, differencing, and **why per-series
  classical models don't scale to ~3,000 series** and got replaced by global ML/DL.
- **Deliverable:** subset runs + a solid theory writeup.
- **Depends on:** T06

### 🔵 GEGA'S TRACK

#### T13 · 🔵 Shared neuralforecast data-prep  (unblocks Keti's T10/T11)
- **Do:** the deep models all need Nixtla's long format: columns `unique_id` (= `Store_Dept`),
  `ds` (Date), `y` (Weekly_Sales) + exogenous columns. Write `nf_prep.py` with
  `to_nixtla(df) -> DataFrame` and the reverse `from_nixtla(preds) -> submission rows`. Decide how to
  handle series too short to train. **Do this early — Keti is blocked on it.**
- **Deliverable:** `nf_prep.py` both DL owners import.
- **Depends on:** T05

#### T14 · 🔵 XGBoost — `model_experiment_XGBoost.ipynb`
- **Do:** same feature matrix as LightGBM but XGBoost needs **encoded categoricals**
  (ordinal or one-hot Store/Dept/Type). Objective `reg:absoluteerror` (or Huber) to match WMAE;
  pass holiday `sample_weight=5`. Same nested-run structure (`_Cleaning → _Feature_Selection →
  _CrossValidation → _HPO → _Final`), Optuna HPO, wrap best in a Pipeline (T08).
- **Deliverable:** tuned XGBoost Pipeline + CV/LB WMAE. Compare speed & score vs Keti's LightGBM.
- **Depends on:** T06, T08

#### T15 · 🔵 PatchTST (or TFT — pick ONE) — `model_experiment_PatchTST.ipynb`
- **Do:** via `neuralforecast`, global model. **PatchTST** = transformer over patched series (strong
  SOTA); **TFT** = attention + native exogenous + variable-importance you can plot (great for the
  report). Only one of the two is required — choose based on which you can get training in time.
- **Deliverable:** runs + CV/LB WMAE; if TFT, include a variable-importance plot for the README.
- **Depends on:** T13

#### T16 · 🔵 Prophet — `model_experiment_Prophet.ipynb` (theory-focused)
- **Do:** Prophet fits one model per series → slow, so demo on a subset. Add US holidays + yearly
  seasonality. Explain additive decomposition (trend + seasonality + holidays) and pros/cons vs
  trees in the writeup.
- **Deliverable:** subset runs + theory writeup.
- **Depends on:** T06

#### T17 · 🔵 TimesFM foundation model (BONUS, optional) — `model_experiment_TimesFM.ipynb`
- **Do:** Google's pretrained model, zero/few-shot. Uncomment `timesfm` in requirements only if you
  attempt it. Interesting to report even if it loses to tuned trees — comment on why a general
  pretrained model may underperform a model tuned on this exact data.
- **Deliverable:** optional runs + a paragraph of analysis.
- **Depends on:** T13

---

## PHASE 3 — Consolidation (🟢 both, days 13–14)

### T18 · 🟢 Model comparison table
- **Do:** collect **mean CV WMAE + Kaggle public LB WMAE** for every architecture into one table
  (pull numbers from MLflow / the Compare view). Confirm CV tracks LB.
- **Deliverable:** comparison table (goes straight into README).
- **Depends on:** all Phase 2 tasks

### T19 · 🟣 Register the overall best in the Model Registry
- **Do:** pick the winner (expect LightGBM/XGBoost). `mlflow.register_model` it as `walmart-best`,
  promote to `Production`/`@champion` in the DagsHub Models tab.
- **Deliverable:** one registered model version marked champion.
- **Depends on:** T18

### T20 · 🟣 `model_inference.ipynb`
- **Do:** `mlflow.pyfunc.load_model("models:/walmart-best/Production")` → load **raw** test/features/
  stores → `preds = model.predict(raw_test)` (pipeline does merge+features internally) →
  `make_submission` → upload to Kaggle → record final public WMAE.
- **Deliverable:** working inference notebook loading from Registry (no local pickle), final submission.
- **Depends on:** T19

### T21 · 🟢 Georgian `README.md` (30% of grade)
- **Do:** write in Georgian: problem → data/EDA → feature engineering → validation (WMAE, expanding
  window) → per-architecture (what tried, params, result, why) → comparison table (CV vs LB) →
  best model + Registry → conclusions (why trees won, where DL/classical struggled). Emphasize the
  **comparison and the "why"** — that's what's graded.
- **Split idea:** each person writes the sections for the models they built; both write EDA/conclusions.
- **Depends on:** T18, T20

### T22 · 🟢 Presentation (10-min, 10% of grade)
- **Do:** summarize README's main points. Slides optional but recommended.
- **Split idea:** each presents their own model track.
- **Depends on:** T21

---

## Workload balance check

| | Keti 🟣 | Gega 🔵 |
|---|---|---|
| Tree | LightGBM (T09) | XGBoost (T14) |
| Deep | DLinear (T10), N-BEATS (T11) | PatchTST/TFT (T15) |
| Classical | SARIMA (T12) | Prophet (T16) |
| Bonus | — | TimesFM (T17) |
| Infra | pipeline.py (T08), Registry (T19), inference (T20) | NF data-prep (T13), W&B if used |

Keti carries the winning-pipeline + registry/inference plumbing; Gega carries the extra deep model +
the shared neuralforecast setup + the bonus. Roughly even. Adjust freely — just keep **one tree
model each** and don't both block on the same shared file.

## Critical path (what unblocks what)
`T01 → T04 → T05 → {T06, T08, T13}` → all models → `T18 → T19 → T20 → T21 → T22`.
The three files everyone waits on: **`features.py` (T05)**, **`pipeline.py` (T08)**,
**`nf_prep.py` (T13)**. Finish those first.
