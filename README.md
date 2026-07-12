# Walmart Store Sales Forecasting

Final machine-learning project for the Kaggle competition **Walmart Recruiting - Store Sales Forecasting**.

The goal is to predict weekly sales for each `Store-Dept-Date` row using historical sales, store metadata, holiday information, markdowns, and external economic variables. The final output is a Kaggle submission file with one prediction per row in `test.csv`.

## Current Status

This README is written as the final report structure. Some result values are marked `TODO` because LightGBM, XGBoost, and N-BEATS still need to be trained or verified in DagsHub before the final winner can be selected.

| Item | Status | Evidence |
|---|---|---|
| SARIMA experiment | Done | `SARIMA_CrossValidation` logged 3 folds and `mean_subset_cv_wmae = 21215.70` |
| SARIMA final run | Done | `SARIMA_Final` finished and logged `sarima_subset_forecast.csv`; not a production candidate |
| DLinear final run | Done | `DLinear_Final` finished with `input_size = 26`, `max_steps = 500` |
| DLinear bug fix | Done | final HPO params are converted to integers before building the PyTorch model |
| Inference notebook | Prepared | `model_inference.ipynb` loads `models:/walmart-best/Production` and writes `models/best_model_submission.csv` |
| Git ignore rules | Prepared | data, credentials, local MLflow folders, generated submissions, and model artifacts are ignored |

Still required before final submission:

| Item | Status |
|---|---|
| LightGBM training and final metrics | TODO |
| XGBoost training and final metrics | TODO |
| N-BEATS training and final metrics | TODO |
| DLinear CV/HPO metric copied from DagsHub | TODO |
| Best model selected from full-data CV metrics | TODO |
| Best model registered as `walmart-best` | TODO |
| Final Kaggle score added | TODO |

## Dataset

The Kaggle data contains:

| File | Purpose |
|---|---|
| `train.csv` | historical weekly sales with target `Weekly_Sales` |
| `test.csv` | future rows where sales must be predicted |
| `features.csv` | markdowns, fuel price, temperature, CPI, unemployment, holidays |
| `stores.csv` | store type and store size |
| `sampleSubmission.csv` | required Kaggle submission format |

The prediction grain is:

```text
Store + Dept + Date
```

## Metric

The competition metric is **Weighted Mean Absolute Error**:

```text
WMAE = sum(w_i * |y_i - yhat_i|) / sum(w_i)

w_i = 5 if IsHoliday = True
w_i = 1 otherwise
```

Holiday weeks matter more because Walmart sales can spike around events such as Thanksgiving, Christmas, Labor Day, and the Super Bowl. Tree-based models can use holiday sample weights directly during training. All models are evaluated with the same WMAE function.

## Validation Strategy

Random splitting is not valid for this task because it would leak future information into training. The project uses expanding-window validation:

```text
Fold 1: train on earlier weeks -> validate on next 39 weeks
Fold 2: train on a larger historical window -> validate on next 39 weeks
Fold 3: train on an even larger window -> validate on next 39 weeks
```

The validation horizon is 39 weeks, matching the future forecasting period.

## Feature Engineering

Shared feature engineering is implemented mainly in `features.py`.

| Feature group | Examples | Why used |
|---|---|---|
| Calendar | year, month, ISO week, quarter, day-of-year | captures seasonality |
| Cyclical time | `WeekSin`, `WeekCos`, `MonthSin`, `MonthCos` | represents repeated cycles smoothly |
| Holidays | holiday flag and named holiday indicators | important because WMAE weights holidays 5x |
| Store metadata | `Type`, `Size` | captures store format and scale |
| Economic variables | `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` | external demand signals |
| Markdowns | `MarkDown1-5`, total markdown, missing flags | promotions affect weekly sales |
| Historical sales | lag/year-over-year features and historical means | gives tabular models memory of previous demand |

Leakage control:

- Future target values are not used for test rows.
- Lag features use known historical values only.
- Expanding-window CV keeps validation dates strictly after training dates.

## Models

### LightGBM

LightGBM is a gradient boosted decision tree model. It is included because Walmart forecasting is partly a tabular problem with strong nonlinear interactions between store, department, holiday, markdown, and economic features.

Strengths:

- strong with engineered tabular features;
- handles nonlinear effects well;
- fast enough for repeated CV and HPO;
- supports holiday sample weights for WMAE alignment;
- feature importance is useful for explanation.

Weaknesses:

- does not naturally understand time-series sequence dynamics;
- depends on good lag/date features;
- can miss patterns that are not represented in features.

Current status:

```text
TODO: train/verify LightGBM_CrossValidation, LightGBM_HPO, and LightGBM_Final.
TODO: copy mean/best CV WMAE here.
```

### XGBoost

XGBoost is another boosted tree model. It is included as a direct comparison to LightGBM.

Strengths:

- strong nonlinear tabular model;
- robust and widely used;
- useful check that results are not specific to one boosting implementation;
- can also use holiday sample weights.

Weaknesses:

- often slower than LightGBM on larger tabular data;
- needs careful categorical encoding;
- still depends on engineered time features.

Current status:

```text
TODO: train/verify XGBoost_CrossValidation, XGBoost_HPO, and XGBoost_Final.
TODO: copy mean/best CV WMAE here.
```

### DLinear

DLinear is a neural forecasting baseline from `neuralforecast`. It decomposes a series into trend and seasonal/residual parts and learns mostly linear mappings from an input window to the forecast horizon.

Strengths:

- fast and simple compared with larger neural models;
- useful baseline for sequence learning;
- global model across many Store-Dept series;
- less complex than N-BEATS.

Weaknesses:

- mostly linear, so it can miss sharp nonlinear holiday/markdown effects;
- weaker for categorical interactions than boosted trees;
- sensitive to input window and training step choices.

Current status:

```text
DLinear_Final: done.
Best final params shown in DagsHub:
- input_size = 26
- max_steps = 500

TODO: copy DLinear_CrossValidation mean_cv_wmae and DLinear_HPO best_mean_cv_wmae from DagsHub.
```

Implementation note:

The final run originally failed because HPO values from a DataFrame could become floats, for example `26.0`. PyTorch layer sizes must be integers. The final cell now converts them:

```python
final_params = {
    "input_size": int(final_params["input_size"]),
    "max_steps": int(final_params["max_steps"]),
}
```

### N-BEATS

N-BEATS is a neural forecasting architecture based on fully connected blocks that produce backcast and forecast components. It is more expressive than DLinear and can model nonlinear temporal patterns.

Strengths:

- learns nonlinear temporal structure;
- global model across many series;
- trend/seasonality explanation is useful for the report.

Weaknesses:

- slower than DLinear;
- more tuning-sensitive;
- may still underperform tree models if markdown, holiday, and categorical effects dominate.

Current status:

```text
TODO: train/verify NBEATS_CrossValidation, NBEATS_HPO, and NBEATS_Final.
TODO: copy mean/best CV WMAE here.
```

### SARIMA / AutoARIMA

SARIMA is a classical statistical time-series model:

```text
SARIMA(p,d,q)(P,D,Q,m)
```

For weekly Walmart data, yearly seasonality is represented with:

```text
m = 52
```

SARIMA is included as a theory-focused baseline, not as a production model.

Strengths:

- interpretable;
- good for explaining autoregression, differencing, moving-average errors, and seasonality;
- useful on a small number of high-volume series.

Weaknesses:

- does not scale well to thousands of Store-Dept series;
- fits separate models instead of sharing information globally;
- does not naturally use store metadata, markdowns, CPI, unemployment, or categorical interactions;
- not suitable as the final production model for this project.

Verified SARIMA result:

| Metric | Value |
|---|---:|
| `fold_1_subset_wmae` | 29620.60 |
| `fold_2_subset_wmae` | 19762.93 |
| `fold_3_subset_wmae` | 14263.56 |
| `mean_subset_cv_wmae` | 21215.70 |

Conclusion: SARIMA is complete as a **theory-focused subset baseline** and is intentionally marked `production_candidate = False`.

## Experiment Tracking

Experiments are tracked in DagsHub MLflow:

```python
import dagshub

dagshub.init(repo_owner="karev23", repo_name="ML-final", mlflow=True)
```

Expected MLflow experiments:

| Experiment | Purpose |
|---|---|
| `LightGBM_Training` | boosted tree candidate |
| `XGBoost_Training` | boosted tree comparison |
| `DLinear_Training` | simple neural forecasting baseline |
| `NBEATS_Training` | more expressive neural forecasting model |
| `SARIMA_Training` | classical subset baseline |

Each model notebook follows this stage structure:

```text
*_Cleaning
*_Feature_Selection
*_CrossValidation
*_HPO
*_Final
```

SARIMA is slightly different because AutoARIMA performs its own internal order search and the final artifact is a subset forecast CSV rather than a production pyfunc model.

## Results Table

This table should be completed after all remaining models finish training.

| Model | Validation scope | Main metric | Value | Production candidate? | Notes |
|---|---|---|---:|---|---|
| LightGBM | full data | `best_mean_cv_wmae` or `mean_cv_wmae` | TODO | yes | likely strongest tabular model |
| XGBoost | full data | `wmae_cv_best` or `wmae_cv_mean` | TODO | yes | tree comparison against LightGBM |
| DLinear | full data | `best_mean_cv_wmae` / `mean_cv_wmae` | TODO | maybe | final run completed with `input_size=26`, `max_steps=500` |
| N-BEATS | full data | `best_mean_cv_wmae` / `mean_cv_wmae` | TODO | maybe | pending training |
| SARIMA | subset only | `mean_subset_cv_wmae` | 21215.70 | no | not directly comparable to full-data models |

Selection rule:

```text
Choose the lowest full-data CV WMAE among LightGBM, XGBoost, DLinear, and N-BEATS.
Do not choose SARIMA as the final model because it is subset-only.
```

## Final Model Registry

After the best model is selected, register it in MLflow Model Registry as:

```text
walmart-best
```

Then promote or alias the selected version as the production/champion model.

Current status:

```text
TODO: wait until LightGBM, XGBoost, and N-BEATS results are available.
TODO: register the winning full-data model as walmart-best.
```

## Inference

The final inference notebook is:

```text
model_inference.ipynb
```

It loads the registered best model:

```python
MODEL_URI = "models:/walmart-best/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)
```

It writes:

```text
models/best_model_submission.csv
```

This step should be run only after `walmart-best` is registered.

## How To Run In Colab

Clone the repo:

```python
%cd /content

import os

if not os.path.exists("/content/ML-final"):
    !git clone https://github.com/ketevan614/ML-final.git

%cd /content/ML-final
```

Install dependencies:

```python
!pip -q install -r requirements.txt
```

For neural models:

```python
!pip -q install -r requirements-dl.txt
```

Connect DagsHub:

```python
import dagshub
import mlflow

dagshub.init(
    repo_owner="karev23",
    repo_name="ML-final",
    mlflow=True,
)

print("MLflow tracking URI:", mlflow.get_tracking_uri())
```

Set Kaggle credentials with Colab Secrets:

```python
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
```

Download data:

```python
!mkdir -p data
!kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data
!unzip -o "data/*.zip" -d data
!unzip -o "data/*.csv.zip" -d data
!ls -lh data
```

Expected files:

```text
train.csv
test.csv
features.csv
stores.csv
sampleSubmission.csv
```

## Repository Structure

```text
.
├── dataloader.py
├── features.py
├── metrics.py
├── validation.py
├── neuralforecast_pyfunc.py
├── model_inference.ipynb
├── models/
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_DLinear.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   └── model_experiment_SARIMA.ipynb
├── requirements.txt
├── requirements-dl.txt
└── README.md
```

## Final Checklist

- [x] SARIMA subset baseline logged.
- [x] DLinear final run completed.
- [x] DLinear final parameter type bug fixed.
- [x] DagsHub setup documented.
- [x] Kaggle setup documented.
- [x] Inference notebook prepared.
- [ ] LightGBM final run completed.
- [ ] XGBoost final run completed.
- [ ] N-BEATS final run completed.
- [ ] Full results table filled.
- [ ] Best model registered as `walmart-best`.
- [ ] Final inference submission generated.
- [ ] Kaggle score added.
