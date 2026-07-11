# Walmart Recruiting - Store Sales Forecasting

ეს რეპოზიტორია არის ფინალური პროექტი Kaggle competition-ისთვის: **Walmart Recruiting - Store Sales Forecasting**. ამოცანა არის time-series forecasting: ისტორიული გაყიდვების, მაღაზიის მახასიათებლების, დღესასწაულების და ეკონომიკური/markdown ცვლადების მიხედვით უნდა ვიწინასწარმეტყველოთ მომავალი კვირების გაყიდვები `Store-Dept-Date` დონეზე.

მთავარი შეფასების მეტრიკაა **WMAE** - Weighted Mean Absolute Error. ჩვეულებრივი MAE-სგან განსხვავებით, holiday კვირებზე შეცდომა 5-ჯერ უფრო მძიმედ ითვლება:

```text
WMAE = sum(w_i * |y_i - yhat_i|) / sum(w_i)

w_i = 5, თუ IsHoliday=True
w_i = 1, სხვა შემთხვევაში
```

ამიტომ მოდელების ტრენინგშიც მნიშვნელოვანია holiday rows უფრო დიდი წონით შევიყვანოთ, განსაკუთრებით tree-based მოდელებში.

## ფაილების სტრუქტურა

სავალდებულო notebook-ები root დირექტორიაშია:

- `model_experiment_LightGBM.ipynb`
- `model_experiment_DLinear.ipynb`
- `model_experiment_NBEATS.ipynb`
- `model_experiment_SARIMA.ipynb`
- `model_inference.ipynb`

დამხმარე ფაილები:

- `dataloader.py` - raw Kaggle CSV-ების ჩატვირთვა, merge და submission-ის შექმნა.
- `features.py` - feature engineering: calendar, holiday, markdown, external, lag-52 და historical mean features.
- `metrics.py` - WMAE.
- `validation.py` - expanding-window time-series validation.
- `neuralforecast_pyfunc.py` - DLinear/N-BEATS-ის MLflow pyfunc wrapper, რომ logged model პირდაპირ raw test set-ზე გაეშვას.

## Validation Strategy

Random K-Fold time-series ამოცანაში არ შეიძლება, რადგან მომავლის ინფორმაცია წარსულში გაჟონავს. ამიტომ ვიყენებთ expanding-window validation-ს:

```text
Fold 1: train წარსულზე -> validate შემდეგ 39 კვირაზე
Fold 2: train უფრო დიდ წარსულზე -> validate შემდეგ 39 კვირაზე
Fold 3: train კიდევ უფრო დიდ წარსულზე -> validate შემდეგ 39 კვირაზე
```

ეს ჰგავს რეალურ Kaggle სცენარს, სადაც train set გვაქვს წარსულში და test set არის მომავალი horizon.

## Feature Engineering

ყველა მოდელისთვის საერთო იდეებია:

- calendar features: წელი, თვე, კვირის ნომერი, quarter, day-of-year;
- cyclic features: `WeekSin`, `WeekCos`, `MonthSin`, `MonthCos`;
- holiday flags: Super Bowl, Labor Day, Thanksgiving, Christmas;
- store metadata: `Type`, `Size`;
- external variables: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`;
- markdown features: `MarkDown1-5`, missing flags, total markdown;
- safe history features: `sales_lag_52`, historical means.

ყველაზე მნიშვნელოვანი leakage control არის ის, რომ test პერიოდისთვის არ ვიყენებთ target-ის მომავალ მნიშვნელობებს. `lag_52` უსაფრთხოა, რადგან 39-week horizon-ზე ერთი წლის წინანდელი გაყიდვები უკვე ცნობილია.

## მოდელები

### LightGBM

LightGBM არის gradient boosted decision tree მოდელი. ის აშენებს ბევრ პატარა decision tree-ს მიმდევრობით. ყოველი ახალი ხე ცდილობს წინა ხეების შეცდომის გასწორებას.

მათემატიკურად საბოლოო პროგნოზი არის ხეების ჯამი:

```text
F(x) = f1(x) + f2(x) + ... + fM(x)
```

თუ objective არის L1, მოდელი ამცირებს აბსოლუტურ შეცდომას:

```text
Loss = sum |y_i - F(x_i)|
```

ჩვენი Kaggle metric არის WMAE, ამიტომ LightGBM-ში ვიყენებთ holiday sample weights-ს:

```text
sample_weight = 5, თუ IsHoliday=True
sample_weight = 1, სხვა შემთხვევაში
```

რატომ არის კარგი ამ ამოცანაზე:

- Walmart data არის tabular time-series;
- ძლიერი categorical interactions არსებობს: Store, Dept, Type, Holiday;
- markdown და holiday effects ხშირად nonlinear არის;
- LightGBM native categorical support-ს იყენებს;
- სწრაფად ტრენინგდება და feature importance-ს გვაძლევს.

სუსტი მხარე:

- time-series დინამიკას პირდაპირ არ სწავლობს;
- დროის ცოდნა მხოლოდ feature engineering-ით აქვს;
- თუ მნიშვნელოვანი pattern feature-ებში არ ჩავდეთ, თვითონ ვერ დაინახავს.

ამ პროექტში LightGBM არის მთავარი production candidate.

### DLinear

DLinear არის მარტივი deep-learning forecasting მოდელი. მისი იდეაა, რომ წარსული window-დან მომავალი horizon შეიძლება დავაპროგნოზოთ linear mapping-ით. იგი ხშირად ყოფს series-ს trend და seasonal/residual ნაწილებად:

```text
y = trend + seasonal
forecast = Linear(trend) + Linear(seasonal)
```

ანუ მოდელი სწავლობს წარსული მნიშვნელობების წონიან კომბინაციას:

```text
yhat[t+1:t+H] = W * y[t-L:t]
```

რატომ არის სასარგებლო:

- ძალიან სწრაფია neural model-ებს შორის;
- კარგი baseline არის deep forecasting-ისთვის;
- global model-ად შეიძლება ყველა Store-Dept series-ზე ერთად ტრენინგი;
- კარგად მუშაობს, როცა trend/seasonality სტაბილურია.

სუსტი მხარე:

- mostly linear მოდელია;
- holiday spikes და markdown shocks შეიძლება ზედმეტად გაასწოროს;
- რთულ categorical interactions-ს LightGBM-ზე სუსტად იჭერს.

DLinear საჭიროა იმის შესამოწმებლად, არის თუ არა მარტივი neural sequence model საკმარისი.

### N-BEATS

N-BEATS არის neural forecasting architecture, რომელიც fully-connected blocks-ს იყენებს. თითოეული block ცდილობს წარსულის ნაწილის ახსნას და მომავლის ნაწილის პროგნოზს.

ძირითადი იდეა:

```text
input window -> block -> backcast + forecast
residual = input - backcast
final forecast = forecast_1 + forecast_2 + ...
```

Interpretable N-BEATS-ში block-ები შეიძლება trend და seasonality კომპონენტებად დაიყოს. Trend შეიძლება polynomial basis-ით აიხსნას:

```text
trend(t) = a0 + a1*t + a2*t^2 + ...
```

Seasonality ხშირად Fourier-like basis-ით აიხსნება:

```text
seasonality(t) = sum a_k sin(2*pi*k*t) + b_k cos(2*pi*k*t)
```

რატომ არის კარგი:

- nonlinear temporal patterns-ს სწავლობს;
- global model-ად ბევრ series-ზე ერთად სწავლობს;
- DLinear-ზე უფრო expressive არის;
- trend/seasonality decomposition-ის კარგი თეორიული ახსნა აქვს.

სუსტი მხარე:

- უფრო ნელია;
- მეტი tuning სჭირდება;
- plain N-BEATS exogenous variables-ს ყოველთვის კარგად არ იყენებს;
- Walmart-ის sharp holiday/markdown effects შეიძლება tree models-მა უკეთ დაიჭიროს.

N-BEATS-ის მთავარი კითხვა ამ პროექტშია: ღირს თუ არა მეტი neural complexity DLinear-თან და LightGBM-თან შედარებით?

### SARIMA / AutoARIMA

SARIMA არის classical statistical time-series მოდელი. ARIMA შედგება სამი ნაწილისგან:

```text
ARIMA(p, d, q)
```

- `p` - autoregressive lags;
- `d` - differencing order;
- `q` - moving average error terms.

Autoregression ნიშნავს, რომ მიმდინარე მნიშვნელობა წინა მნიშვნელობებზეა დამოკიდებული:

```text
y_t = c + phi_1*y_{t-1} + phi_2*y_{t-2} + ... + error_t
```

Differencing გამოიყენება trend/non-stationarity-ის მოსაშორებლად:

```text
y'_t = y_t - y_{t-1}
```

SARIMA ამატებს seasonal ნაწილს:

```text
SARIMA(p,d,q)(P,D,Q,m)
```

weekly Walmart data-სთვის yearly seasonality არის:

```text
m = 52
```

რატომ არის მნიშვნელოვანი:

- თეორიულად ძალიან კარგი ასახსნელია;
- ACF/PACF, differencing და seasonality-ის განხილვის საშუალებას იძლევა;
- high-volume subset-ზე შეიძლება სწრაფად ვაჩვენოთ.

რატომ არ არის production winner:

- თითო Store-Dept series-ზე ცალკე მოდელი სჭირდება;
- Walmart-ში ათასობით series არის;
- არ აზიარებს ინფორმაციას sparse departments-ს შორის;
- store type, markdowns, CPI, unemployment და categorical interactions ბუნებრივად არ იყენებს.

ამიტომ SARIMA ამ პროექტში theory-focused მოდელია და არა მთავარი Kaggle მოდელი.

## მოდელების შედარება

მოსალოდნელი practical ranking ასეთია:

1. **LightGBM** - ყველაზე ძლიერი production candidate tabular/categorical/time features-ის გამო.
2. **N-BEATS** - ძლიერი neural comparison, თუ საკმარისი compute და tuning გვაქვს.
3. **DLinear** - სწრაფი neural baseline.
4. **SARIMA** - თეორიული/კლასიკური baseline subset-ზე.

რატომ შეიძლება LightGBM-მ მოიგოს:

- competition data-ში ბევრი categorical interaction არის;
- holiday/markdown effects sharp და nonlinear არის;
- feature engineering-ით year-over-year pattern უკვე მიწოდებულია;
- WMAE-სთან sample weights პირდაპირ შეგვიძლია შევათავსოთ.

## როგორ გავუშვათ პროექტი

### Local Windows Terminal

თუ `.venv` უკვე გააქტიურებულია:

```cmd
python -m pip install -r requirements.txt
```

თუ `!pip` დაწერე terminal-ში, არ იმუშავებს. `!pip` მხოლოდ Colab/Jupyter cell-ში გამოიყენება. Windows terminal-ში გამოიყენე:

```cmd
python -m pip install ...
```

### Colab - Core Models

LightGBM და SARIMA გაუშვი CPU runtime-ზე:

```python
!git clone <YOUR_REPO_URL>
%cd ML-final
!pip install -r requirements.txt
```

შემდეგ გაუშვი:

```text
model_experiment_LightGBM.ipynb
model_experiment_SARIMA.ipynb
```

### Colab - Deep Learning Models

DLinear და N-BEATS გაუშვი GPU runtime-ზე:

```text
Runtime -> Change runtime type -> T4 GPU
```

შემდეგ:

```python
!git clone <YOUR_REPO_URL>
%cd ML-final
!pip install -r requirements-dl.txt
```

გაუშვი:

```text
model_experiment_DLinear.ipynb
model_experiment_NBEATS.ipynb
```

### Kaggle Data

Colab-ში ატვირთე `kaggle.json`:

```python
from google.colab import files
files.upload()
```

შემდეგ:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c walmart-recruiting-store-sales-forecasting -p data
!unzip -q data/walmart-recruiting-store-sales-forecasting.zip -d data
!unzip -q data/train.csv.zip -d data
!unzip -q data/test.csv.zip -d data
!unzip -q data/features.csv.zip -d data
!unzip -q data/stores.csv.zip -d data
!unzip -q data/sampleSubmission.csv.zip -d data
```

### MLflow / DagsHub

ყველა notebook-ში MLflow experiment ცალკეა:

- `LightGBM_Training`
- `DLinear_Training`
- `NBEATS_Training`
- `SARIMA_Training`

DagsHub-ისთვის გამოიყენე:

```python
import dagshub
dagshub.init(repo_owner="karev23", repo_name="ML-final", mlflow=True)
```

ყოველ architecture-ში run-ები უნდა იყოს:

```text
*_Cleaning
*_Feature_Selection / *_DataPrep
*_CrossValidation
*_HPO
*_Final
```

### Final Inference

საუკეთესო მოდელი Model Registry-ში უნდა დარეგისტრირდეს, მაგალითად:

```text
walmart-best
```

შემდეგ `model_inference.ipynb` ტვირთავს:

```python
MODEL_URI = "models:/walmart-best/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)
preds = model.predict(test_raw)
```

და ქმნის Kaggle submission-ს:

```text
models/best_model_submission.csv
```

## Current Implementation Status

| Model | Code | Training | Final logging |
|---|---:|---:|---:|
| LightGBM | yes | yes | sklearn Pipeline |
| DLinear | yes | yes | MLflow pyfunc |
| N-BEATS | yes | yes | MLflow pyfunc |
| SARIMA | yes | subset only | artifact only, not production |

SARIMA intentionally is not registered as production model because it is theory-focused and subset-only.
