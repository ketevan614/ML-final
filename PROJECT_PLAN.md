# Walmart Store Sales Forecasting — Implementation Plan

> Working plan for a 2-person team. Not graded (the graded README is separate and in Georgian).
> Keep this in English/whatever is convenient for you two; convert the findings into the Georgian `README.md` at the end.

---

## 0. The problem in one paragraph

Predict `Weekly_Sales` for **45 stores × ~99 departments** over a **39-week future horizon**
(test = 2012-11-02 → 2013-07-26). It is a **panel / multi-series** forecasting problem: ~3,000
individual (Store, Dept) time series that all share the same weekly calendar. The metric is
**WMAE** — Weighted Mean Absolute Error where **holiday weeks count 5×**. Lower is better.

Data files (already downloaded in `data/`):
- `train.csv` — Store, Dept, Date, Weekly_Sales, IsHoliday (2010-02-05 → 2012-10-26)
- `test.csv` — Store, Dept, Date, IsHoliday (the 39 weeks to predict)
- `features.csv` — Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday
- `stores.csv` — Store, Type (A/B/C), Size
- `sampleSubmission.csv` — Id (`Store_Dept_Date`), Weekly_Sales

You already have `dataloader.py` (load + merge + submission builder), `metrics.py` (WMAE),
`validation.py` (expanding-window / holdout splits). Reuse these everywhere.

---

## 1. Infrastructure: DagsHub + MLflow (do this FIRST, together)

**DagsHub** = a GitHub-like host that gives you a **free hosted MLflow tracking server + Model
Registry** out of the box. This is exactly what you want so *both teammates push runs and
registered models to the same place*.

### One-time setup (30 min, both people present)
1. Both create accounts at https://dagshub.com.
2. **Person A** creates the repo on DagsHub — easiest path: "Create → Connect a repository" and
   point it at your GitHub repo (DagsHub mirrors it). Or create the repo on DagsHub and add GitHub
   as a remote. Either way you get **one** repo both can push to.
3. Add **Person B** as a collaborator (Settings → Collaborators).
4. On the DagsHub repo page click **"Remote → Experiments"** → it shows you:
   - `MLFLOW_TRACKING_URI` (looks like `https://dagshub.com/<user>/<repo>.mlflow`)
   - `MLFLOW_TRACKING_USERNAME` = your DagsHub username
   - `MLFLOW_TRACKING_PASSWORD` = a token (each teammate uses their own token)

### Put credentials in a local `.env` (never commit it — add `.env` to `.gitignore`)
```
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

### Standard header for EVERY experiment notebook
```python
import os, mlflow
from dotenv import load_dotenv
load_dotenv()                      # reads .env
# MLFLOW_TRACKING_URI / USERNAME / PASSWORD are picked up from env automatically
mlflow.set_experiment("LightGBM_Training")   # one experiment per architecture
```
> Tip: `pip install python-dotenv`. Add it to requirements.

**Neural nets:** the assignment lets you use W&B instead of MLflow for DL. Pick ONE and be
consistent. Simplest: use **MLflow for everything** (registry lives there anyway). If you want
nicer training curves for the deep models, add W&B *in addition* just for those.

---

## 2. Required repository structure (what the graders check)

```
ML-final/
├── data/                              # gitignored (Kaggle data)
├── dataloader.py                      # ✅ have it
├── metrics.py                         # ✅ WMAE
├── validation.py                      # ✅ time-series splits
├── features.py                        # ➕ ADD: shared feature engineering (see §4)
├── pipeline.py                        # ➕ ADD: sklearn-compatible raw→prediction pipeline (see §7)
├── requirements.txt                   # ✅ (add python-dotenv, kaggle already there)
├── notebooks/
│   ├── 00_EDA.ipynb                   # ➕ shared EDA
│   ├── model_experiment_LightGBM.ipynb        # ✅ stub exists
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_DLinear.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   ├── model_experiment_PatchTST.ipynb        # (PatchTST OR TFT — one is enough)
│   ├── model_experiment_SARIMA.ipynb          # classical: theory-focused, small subset
│   ├── model_experiment_Prophet.ipynb
│   ├── model_experiment_TimesFM.ipynb         # bonus, optional
│   └── model_inference.ipynb          # ✅ loads BEST model from Registry → predicts raw test → submission
└── README.md                          # 🇬🇪 Georgian, detailed report (30% of grade)
```

**Hard requirements to keep in mind while building:**
- **One MLflow experiment per architecture** (`XGBoost_Training`, `LightGBM_Training`, …).
- **Runs inside it** correspond to stages: `_Cleaning`, `_Feature_Selection`, `_CrossValidation`,
  `_HPO`, `_Final`. Use nested runs (`mlflow.start_run(nested=True)`) so they group nicely.
- **Best model of each architecture = a saved sklearn `Pipeline`** that accepts the **raw,
  un-preprocessed** test set and outputs predictions. (§7 — this is the trickiest requirement.)
- **Best model overall → registered in the Model Registry.** `model_inference.ipynb` must
  `mlflow.pyfunc.load_model("models:/<name>/<stage>")` and call `.predict()` — no local pickle.

---

## 3. Phase timeline & who does what

Trees almost always win this competition, so **both** of you should understand the tree pipeline.
Deep-learning + classical are more about *breadth and the report*.

| Phase | Task | Owner | Output |
|------|------|-------|--------|
| **P0** | DagsHub + MLflow wired, `.env`, verify a dummy run appears | **Both** | working tracking server |
| **P1** | EDA (`00_EDA.ipynb`) — seasonality, holidays, markdowns, negatives | **Both** | plots + written findings |
| **P2** | `features.py` shared feature engineering + `validation.py` sanity check | **Both** | one feature function both trust |
| **P3** | **Baseline** (seasonal-naive: last-year same week) → first Kaggle submission | **Both** | leaderboard anchor + WMAE target |
| **P4a** | **Tree models**: LightGBM + XGBoost, HPO, feature selection | **Person A** | 2 architectures, tuned |
| **P4b** | **Deep models**: DLinear + N-BEATS + (PatchTST *or* TFT) via `neuralforecast` | **Person B** | 3 architectures |
| **P4c** | **Classical**: SARIMA (theory + small subset) + Prophet | **Person A** | theory writeup + runs |
| **P4d** | **TimesFM** foundation model (bonus) | **Person B** | optional runs |
| **P5** | Wrap each architecture's best as a **Pipeline**, log to MLflow | **each owner** | pipelines |
| **P6** | Compare all → register overall best → `model_inference.ipynb` → final Kaggle submit | **Both** | registered model + submission |
| **P7** | Write Georgian `README.md` + 10-min presentation | **Both** | 30% + 10% of grade |

Rule from the brief: **EDA + feature engineering together**, model training independently, share
often. Sync every couple of days on WMAE numbers so you catch validation mistakes early.

---

## 4. EDA (P1) — what to actually look for

Put findings in `00_EDA.ipynb` AND jot them for the README. Look for:
- **Yearly seasonality** — strong ~52-week cycle; huge spikes at **Thanksgiving** and the
  **week before Christmas**. Plot total weekly sales.
- **The 4 holidays** the metric cares about: Super Bowl (Feb), Labor Day (Sep), Thanksgiving
  (Nov), Christmas (late Dec). Only these weeks are flagged `IsHoliday` and weighted 5×.
- **MarkDown1–5 are mostly NaN before Nov 2011** — anti-markdown promo data only exists later.
  Decide: fill with 0, or add "markdown_missing" flag.
- **Negative sales exist** (returns) — clip at 0 for predictions? Test empirically.
- **Store Type (A/B/C) & Size** strongly separate sales levels.
- **Series count** — ~3,000 Store×Dept combos; some Depts don't exist in every store; some series
  are short/sparse. Note which (Store, Dept) pairs are in test but sparse in train.
- **Known leaderboard trick (mention in README):** because the ISO calendar shifts, the
  Christmas peak lands in different test weeks than train weeks; a small **post-hoc "shift ~2.5
  days" of the holiday-week predictions** historically improved WMAE. Good depth point even if you
  don't implement it.

---

## 5. Validation strategy (agree once, use everywhere)

Your `validation.py` already does the right thing: **split on unique dates, not rows** (panel
data), expanding window, `horizon=39` to mirror the real test length.

- Use **`expanding_window_folds(dates, n_splits=3, horizon=39)`** for model comparison.
- **Always evaluate with WMAE** using the per-row `IsHoliday`, never plain MAE/RMSE — otherwise
  your local score won't track the leaderboard.
- Never let a random KFold near this data. No shuffling. No leakage of future dates.
- Report **local CV WMAE next to Kaggle public LB WMAE** for each model in the README — showing
  they track is a maturity signal graders love.

---

## 6. Feature engineering (`features.py`, P2) — build ONE function both use

Signature idea: `build_features(df_merged) -> X` so trees and (optionally) DL share it.

Feature groups:
- **Calendar:** Year, Month, WeekOfYear (ISO), day-of-year, `is_pre_christmas`, per-holiday flags
  (SuperBowl/LaborDay/Thanksgiving/Christmas) derived from Date.
- **Store meta:** Type (A/B/C → categorical), Size.
- **Exogenous:** Temperature, Fuel_Price, CPI, Unemployment; MarkDown1–5 with a missing-flag.
- **Lags / seasonality — CAREFUL with the 39-week gap.** Test starts 1 week after train ends but
  you must predict 39 weeks out, so a lag-1 feature isn't available for week 39 at inference.
  Two clean options:
  1. **Only use lags ≥ 39** (esp. **lag-52 = "same week last year"**, which is always available
     and very predictive here). Simplest, no leakage, no recursion. **Recommended default.**
  2. **Recursive multi-step**: predict week t, feed it back as the lag for t+1. More code, risk of
     error accumulation. Do this only if you have time.
- **Encode categoricals** for trees: LightGBM handles native categoricals; XGBoost needs
  one-hot/ordinal.

Log the chosen feature set as a param/artifact in the `_Feature_Selection` run.

---

## 7. The Pipeline requirement (§ most people get wrong)

Requirement: *the best model per architecture is saved as a **Pipeline** that runs directly on the
**raw, un-preprocessed** test set.* So preprocessing must live **inside** the model object.

Build a custom sklearn transformer in `pipeline.py` that bakes in `features.csv` + `stores.csv`
so the pipeline is self-contained:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class WalmartPreprocessor(BaseEstimator, TransformerMixin):
    """Takes RAW test rows (Store, Dept, Date, IsHoliday) and produces the model matrix.
    features/stores tables are stored on the object so predict() needs no external files."""
    def __init__(self, features_df, stores_df):
        self.features_df = features_df
        self.stores_df = stores_df
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        merged = merge_all(X, self.features_df, self.stores_df)  # from dataloader.py
        return build_features(merged)                            # from features.py

pipe = Pipeline([
    ("prep", WalmartPreprocessor(features, stores)),
    ("model", lgbm_regressor),
])
pipe.fit(raw_train, y_train)
mlflow.sklearn.log_model(pipe, artifact_path="model",
                         registered_model_name=None)  # register only the overall best
```

Now `pipe.predict(raw_test_df)` works end-to-end. That is exactly what `model_inference.ipynb`
loads from the registry and calls. For **neural / statsforecast** models that aren't sklearn
estimators, wrap them as an `mlflow.pyfunc.PythonModel` with the same "raw in → predictions out"
contract.

---

## 8. Per-architecture notes (what to try, what to log)

For **every** architecture: create experiment `<Arch>_Training`; nested runs for
`_Cleaning → _Feature_Selection → _CrossValidation → _HPO → _Final`; log params, per-fold WMAE,
mean CV WMAE, and the final Pipeline. Write 2–3 sentences of takeaways for the README.

- **LightGBM (Person A):** strongest candidate. Native categorical Store/Dept/Type. Objective
  `l1` (MAE) aligns with WMAE better than `l2`. Try `sample_weight = 5 where holiday`. HPO with
  Optuna over num_leaves, learning_rate, min_child_samples, feature/bagging fraction. Feature
  importance → drop useless markdowns if they don't help.
- **XGBoost (Person A):** same features, ordinal/one-hot encode categoricals, `reg:absoluteerror`
  or Huber. Compare speed/score vs LightGBM.
- **SARIMA (Person A):** classical, **theory-focused** — the brief says don't burn time training.
  Fit `AutoARIMA` (via `statsforecast`) on a **small subset** (a few high-volume Store×Dept series)
  with weekly seasonality m=52. In README: explain (S)ARIMA, ACF/PACF, why it scales badly to
  3,000 series and why global ML models replaced it.
- **Prophet (Person A):** one model per series is slow; demo on a subset. Add US holidays +
  yearly seasonality. Discuss additive decomposition, pros/cons vs trees.
- **DLinear (Person B):** simple linear decomposition baseline; fast; surprisingly strong.
  Via `neuralforecast`. Good first DL model.
- **N-BEATS (Person B):** pure DL, interpretable trend/seasonality stacks. Global model over all
  series with `neuralforecast` (handles exogenous + many series under one API).
- **PatchTST *or* TFT (Person B):** pick ONE. TFT = attention + native exogenous + interpretable
  variable importance (great for the report). PatchTST = transformer on patched series, strong SOTA.
- **TimesFM (Person B, bonus):** Google pretrained foundation model — zero/few-shot. Interesting
  to report even if it underperforms tuned trees. Uncomment in requirements only if attempting.

`neuralforecast` (Nixtla) gives N-BEATS, NHITS, DLinear, PatchTST, TFT under one API with global
cross-series training + exogenous vars — so Person B implements once and swaps model classes.

---

## 9. Model comparison, registry, inference (P6)

1. Collect mean CV WMAE + Kaggle public LB WMAE for every architecture into one table.
2. Pick the winner (expect LightGBM or XGBoost).
3. Register it: `mlflow.register_model(run_uri, "walmart-best")`, promote to `Production`/`@champion`.
4. `model_inference.ipynb`:
   ```python
   model = mlflow.pyfunc.load_model("models:/walmart-best/Production")
   raw_test, features, stores = ...          # raw, un-preprocessed
   preds = model.predict(raw_test)           # pipeline does merge+features internally
   make_submission(raw_test, preds, "submission.csv")   # from dataloader.py
   ```
5. Upload `submission.csv` to Kaggle (competition still accepts late submissions for scoring).
   Record the public WMAE.

---

## 10. Deliverables checklist

- [ ] GitHub repo link submitted on Classroom (shared, both committing)
- [ ] MLflow on DagsHub: one experiment per architecture, staged nested runs
- [ ] `model_experiment_*.ipynb` for each architecture
- [ ] Each architecture's best saved as a **Pipeline** (raw test → prediction)
- [ ] Overall best in **Model Registry**
- [ ] `model_inference.ipynb` loads from Registry → predicts → submission
- [ ] `submission.csv` scored on Kaggle
- [ ] **README.md in Georgian** — every approach tried + its result (30%)
- [ ] 10-min presentation summarizing the README (10%)

---

## 11. Georgian README structure (fill at the end — 30% of grade)

Sections to cover (write in Georgian): პრობლემის აღწერა → მონაცემები/EDA → Feature Engineering →
ვალიდაციის სტრატეგია (WMAE, expanding window) → თითო არქიტექტურა (რა სცადეთ, პარამეტრები, შედეგი,
რატომ) → შედარების ცხრილი (CV WMAE vs Kaggle LB) → საუკეთესო მოდელი და Registry → დასკვნები
(რატომ მოიგო tree-based, სად ვერ გაართვა თავი DL/classical). Emphasize the *comparison and the
"why"* — that is what this project is graded on.
```
