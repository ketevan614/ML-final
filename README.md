# Walmart Store Sales Forecasting

Kaggle-ის შეჯიბრი [Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).
მიზანი: 45 მაღაზიის დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზი 39 კვირით წინ.

პროექტში შედარებულია ოთხი ოჯახის მოდელი (Tree-Based, Deep Learning, Classical, Foundation)
ერთსა და იმავე მონაცემებზე, ერთი მეტრიკით და ერთი ვალიდაციის სქემით.

MLflow tracking: https://dagshub.com/karev23/ML-final.mlflow

გუნდი: ქეთევან არევაძე (`karev23`), გიორგი აბაშიძე (`gabas22`).

## 1. მონაცემები

| | |
|---|---|
| სატრენინგო სტრიქონები | 421,570 |
| სატესტო სტრიქონები | 115,064 |
| სერიების რაოდენობა `(Store, Dept)` | 3,331 |
| ისტორიის სიგრძე | 143 კვირა (2010-02-05 დან 2012-10-26 მდე) |
| საპროგნოზო ჰორიზონტი | 39 კვირა |
| უარყოფითი გაყიდვები | 1,285 სტრიქონი |

ეს panel (multi-series) ამოცანაა: 3,331 დროითი მწკრივი, რომლებიც ერთსა და იმავე თარიღებს
იზიარებენ. სერიების სიგრძე ძალიან არათანაბარია: მედიანა 143 კვირაა, მაგრამ მინიმუმი მხოლოდ
1 კვირა. ეს ფაქტი პროექტის ბევრ გადაწყვეტილებას განსაზღვრავს.

დამატებითი ცხრილები: `features.csv` (Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5)
და `stores.csv` (Type, Size).

## 2. მეტრიკა: WMAE

```
WMAE = Σ wᵢ · |yᵢ − ŷᵢ| / Σ wᵢ        wᵢ = 5,  თუ კვირა სადღესასწაულოა
                                       wᵢ = 1,  სხვა შემთხვევაში
```

სადღესასწაულო კვირები 5-ჯერ იწონება. ეს ერთი დეტალი პროექტის თითქმის ყველა გადაწყვეტილებას
განსაზღვრავს:

- ხის მოდელებში სატრენინგო `sample_weight = 5` სადღესასწაულო სტრიქონებზე;
- XGBoost-ში `objective="reg:absoluteerror"`, რადგან L1 loss პირდაპირ WMAE-ს ესადაგება, MSE კი არა;
- მოდელები, რომლებმაც არ იციან, რომ კვირა Thanksgiving-ია, ზუსტად იქ ცდებიან, სადაც ჯარიმა
  ყველაზე მძიმეა (იხ. PatchTST და TimesFM ქვემოთ).

იმპლემენტაცია: `metrics.py`.

## 3. Feature Engineering

დეტალური აღწერა: `FEATURES.md`. კოდი: `features.py` (73 სვეტი).

| ჯგუფი | Feature-ები |
|---|---|
| კალენდარი | `Year`, `Month`, `WeekOfYear`, `WeekSin/Cos`, `MonthSin/Cos` |
| დღესასწაულები | `is_super_bowl`, `is_labor_day`, `is_thanksgiving`, `is_christmas`, `DaysToChristmas`, `is_pre_christmas` |
| Markdown | `MarkDown1-5` და `*_missing` flag-ები, `TotalMarkdown`, `TotalMarkdown_log1p` |
| გარემო | `Temperature`, `Fuel_Price`, `TempCold/Hot/Mild`, `TempComfortDistance`, `TemperatureStoreDeviation` |
| ეკონომიკა | `CPI`, `Unemployment` და `*_missing` |
| გაყიდვების ისტორია | `sales_lag_52`, `same_week_history_mean`, `store_dept_history_mean`, `dept_week_history_mean`, `store_week_history_mean` და `*_missing` |

### 3.1. Leakage-ის მთავარი წესი

სატესტო ჰორიზონტი 39 კვირაა. ბოლო სატესტო კვირის პროგნოზისას ხელთ გვაქვს მხოლოდ 39 კვირით
ძველი ინფორმაცია. აქედან გამომდინარე გამოსადეგია მხოლოდ `lag >= 39`. `lag_1`, `lag_4`, `lag_12`
დაუშვებელია, რადგან ისინი სატესტო პერიოდში უბრალოდ არ არსებობს.

ამიტომ `sales_lag_52` ("იგივე კვირა შარშან") ერთადერთი ბუნებრივი და ამავე დროს უსაფრთხო lag-ია.
ის ამ პროექტის ცენტრალური feature-ია.

ვალიდაციისას history-ზე დაფუძნებული feature-ები აიგება მხოლოდ სატრენინგო ნაწილიდან:

```python
build_features(val.drop(columns=["Weekly_Sales"]), sales_history_df=train_fold)
```

ასე სავალიდაციო გაყიდვები საკუთარ feature-ებში ვერ მოხვდება, ანუ leakage გამორიცხულია.

### 3.2. Feature Importance (XGBoost)

| Feature | Importance |
|---|---|
| `store_dept_history_mean` | 0.274 |
| `store_dept_history_mean_missing` | 0.090 |
| `sales_lag_52_missing` | 0.072 |
| `same_week_history_mean_missing` | 0.035 |
| `dept_week_history_mean` | 0.034 |
| `sales_lag_52` | 0.033 |
| `Type_B` | 0.022 |

საინტერესო დაკვირვება: `*_missing` flag-ები თითქმის ისეთივე მნიშვნელოვანია, როგორც თავად
მნიშვნელობები. მოდელი სწავლობს, რომ "ამ სერიას გამოსადეგი ისტორია არ აქვს" თავისთავად ძლიერი
სიგნალია, და ასეთ სტრიქონებზე fallback-საშუალოს იყენებს.

## 4. ვალიდაცია

Random K-Fold აქ დაუშვებელია, რადგან ის მომავლის ინფორმაციას წარსულში ატანს, ანუ პირდაპირ
leakage-ს იწვევს. ვიყენებთ expanding window-ს უნიკალურ თარიღებზე (და არა სტრიქონებზე, რადგან
panel-ია):

| fold | სატრენინგო ისტორია | ვალიდაცია |
|---|---|---|
| 1 | 26 კვირა | შემდეგი 39 კვირა |
| 2 | 65 კვირა | შემდეგი 39 კვირა |
| 3 | 104 კვირა | ბოლო 39 კვირა |

იმპლემენტაცია: `validation.py` (`expanding_window_folds`, `holdout_split`).

### PatchTST-ის გამონაკლისი

PatchTST-ს ერთი სატრენინგო ფანჯრისთვის სჭირდება `input_size + h` კვირა თითო სერიაზე.
`h = 39`-ისა და სულ 143 კვირის პირობებში expanding window-ის არცერთ fold-ს არ ჰყოფნის ისტორია
(26, 65, 104 კვირა). ამიტომ PatchTST იყენებს ერთ 39-კვირიან holdout-ს (104 სატრენინგო კვირა,
`input_size = 52`).

ორივე შემთხვევაში საქმე ეხება ერთსა და იმავე ბოლო 39 კვირას, ამიტომ expanding window-ის მე-3
fold და PatchTST-ის holdout სრულად შედარებადი გაზომვებია. სწორედ ამას იყენებს ქვემოთ მოცემული
მთავარი ცხრილი.

## 5. შედეგები

### 5.1. მთავარი ცხრილი: ბოლო 39 კვირა (სრული panel, სრულად შედარებადი)

| მოდელი | ოჯახი | WMAE |
|---|---|---|
| LightGBM | Tree-Based | 1,608.5 |
| XGBoost | Tree-Based | 1,780.1 |
| N-BEATS | Deep Learning | 1,845.6 |
| DLinear | Deep Learning | 2,482.7 |
| PatchTST | Deep Learning | 2,581.9 |

### 5.2. სრული Cross-Validation (3 fold-ის საშუალო)

| მოდელი | fold 1 (26კვ) | fold 2 (65კვ) | fold 3 (104კვ) | საშუალო |
|---|---|---|---|---|
| LightGBM | 2,836.7 | 1,944.5 | 1,608.5 | 2,129.9 |
| XGBoost | 4,312.5 | 2,292.1 | 1,780.1 | 2,755.4 |
| N-BEATS | 4,027.7 | 3,448.3 | 1,845.6 | 3,107.2 |
| DLinear | 11,936.1 | 5,833.4 | 2,482.7 | 6,750.7 |

ყველა მოდელი მონოტონურად უმჯობესდება fold-იდან fold-ში. ეს შემთხვევითობა არ არის: რაც მეტი
ისტორიაა, მით უფრო სავსეა `sales_lag_52` და history-საშუალოები. fold 1-ს მხოლოდ 26 კვირა აქვს,
ანუ lag-52 პრაქტიკულად ცარიელია, და შედეგიც შესაბამისია.

### 5.3. Subset-ის შედეგები (5 ყველაზე მსხვილი სერია)

Prophet და TimesFM ერთსა და იმავე 5 სერიაზეა გაზომილი, ამიტომ მხოლოდ ერთმანეთთან არიან
შედარებადი. აბსოლუტური WMAE აქ ბევრად დიდია, რადგან სერიები დიდია. ეს მოდელის ხარისხზე
არაფერს ამბობს.

| მოდელი | WMAE | რა იცის |
|---|---|---|
| TimesFM 2.5 (zero-shot) | 11,243.0 | ამ მონაცემების შესახებ არაფერი |
| Prophet (per-series fit) | 12,328.4 | თითო სერიის 104 კვირა |
| seasonal-naive (lag-52) | 16,730.8 | ერთი დაკვირვება შარშანდელიდან |
| SARIMA (8 სერია) | 21,215.7 | თითო სერიის ისტორია |

### 5.4. HPO

| მოდელი | HPO-მდე | HPO-ს შემდეგ |
|---|---|---|
| XGBoost | 2,794.9 | 2,755.4 (15 Optuna trial) |
| PatchTST | 2,819.3 | 2,581.9 (5 Optuna trial) |
| N-BEATS | 3,107.2 | 3,107.2 |
| DLinear | 6,750.7 | 6,321.0 |

## 6. მთავარი დასკვნები

### 6.1. Feature Engineering აჯობა არქიტექტურას

ხის მოდელებმა ყველა deep learning მოდელი დაამარცხეს: LightGBM (1,608.5) საუკეთესო DL მოდელის
N-BEATS-ის (1,845.6) წინააღმდეგ. PatchTST კი XGBoost-ს 45%-ით ჩამორჩება იდენტურ ფანჯარაზე
(2,581.9 vs 1,780.1).

მიზეზი პირდაპირია: PatchTST univariate არქიტექტურაა. `neuralforecast`-ის იმპლემენტაციაში
`EXOGENOUS_FUTR = False`, და `futr_exog_list`-ის გადაცემა პირდაპირ იძლევა შეცდომას
`PatchTST does not support future exogenous variables`. ანუ PatchTST ვერ ხედავს markdown-ებს,
დღესასწაულებს, CPI-ს, ტემპერატურას და მაღაზიის ტიპს. ყველაფერი მხოლოდ გაყიდვების მწკრივიდან
უნდა ამოიცნოს.

XGBoost-ს კი იგივე ინფორმაცია მზა სვეტად გადაეცემა: `sales_lag_52` და `store_dept_history_mean`.
ხეს year-over-year სეზონურობის სწავლა არ სჭირდება, ის მას უბრალოდ კითხულობს.

WMAE ამ სხვაობას კიდევ უფრო ამძიმებს: სადღესასწაულო კვირები 5-ჯერ იწონება, ხოლო PatchTST-ს
საერთოდ არ შეუძლია იცოდეს, რომ კვირა Thanksgiving-ია.

დასკვნა: ამ მონაცემებზე feature engineering უფრო მნიშვნელოვანია, ვიდრე არქიტექტურა. სწორედ
ამიტომ არიან ამ Kaggle-შეჯიბრის გამარჯვებულები gradient boosting, და არა ტრანსფორმერები.

### 6.2. რატომ სჯობს LightGBM XGBoost-ს

განსხვავება ყველაზე დიდია fold 1-ზე (2,836.7 vs 4,312.5), ანუ იქ, სადაც ისტორია ყველაზე მწირია.
LightGBM კატეგორიულ სვეტებს native-ად ამუშავებს (pandas `Categorical`), XGBoost კი one-hot
encoding-ს იყენებს. მწირი ისტორიის პირობებში ეს LightGBM-ს უპირატესობას აძლევს. LightGBM ასევე
58 შერჩეულ feature-ს იყენებს 73-დან.

### 6.3. Foundation მოდელმა Prophet დაამარცხა

TimesFM-მა (11,243.0), რომელსაც ამ მონაცემების არც ერთი სტრიქონი უნახავს, დაამარცხა Prophet
(12,328.4), რომელიც ინდივიდუალურად იყო დატრენინგებული თითოეულ ამ 5 სერიაზე. ის ასევე 32.8%-ით
ჯობნის seasonal-naive baseline-ს.

მიზეზი: Prophet-ის მოდელი ფიქსირებული ფუნქციური ფორმაა (piecewise-linear trend, Fourier
სეზონურობა, holiday dummy-ები). დაახლოებით 2-წლიანი ისტორიით მან წლიური სეზონური მრუდი
თითოეული კალენდარული კვირის მხოლოდ ორი დაკვირვებიდან უნდა შეაფასოს. TimesFM კი მზა სეზონურ
prior-ს იღებს უზარმაზარი კორპუსიდან და მას მხოლოდ ამოცნობა სჭირდება, და არა შეფასება.
მოკლე ისტორიაზე ნასესხები სტრუქტურა სჯობს შეფასებულ სტრუქტურას.

### 6.4. Prophet-ს დღესასწაულებმა შედეგი გაუარესა

| Prophet კონფიგურაცია | subset WMAE |
|---|---|
| მხოლოდ წლიური სეზონურობა | 12,328.4 |
| და 4 Walmart დღესასწაული | 13,376.1 |

შედეგად `USE_HOLIDAYS = False`.

ეს კონტრინტუიციურია, რადგან WMAE ხომ სწორედ სადღესასწაულო კვირებს იწონის 5-ჯერ. მიზეზი ისევ
მონაცემების სიმცირეა: 143 კვირაში თითოეული დღესასწაული მხოლოდ 2-ჯერ გვხვდება, ამიტომ holiday
dummy ორ დაკვირვებაზე ფიტდება და overfit-ს იწვევს.

ეს ზუსტად აჩვენებს განსხვავებას global და per-series მიდგომებს შორის: XGBoost-ის
`is_thanksgiving` flag მუშაობს, რადგან ის დღესასწაულის ეფექტს 3,331-ვე სერიაზე ერთდროულად
სწავლობს. Prophet კი თითო სერიას იზოლირებულად უყურებს, და საკმარისი მონაცემი არ რჩება.

### 6.5. Prophet-ის მასშტაბირების ფასი

Prophet თითო სერიაზე ცალკე მოდელს ფიტავს. რეგისტრირებული `Prophet_Walmart`-ის
`subset_coverage = 0.17%`, ანუ მას რეალური Prophet-პროგნოზი მხოლოდ სატესტო სტრიქონების
0.17%-ისთვის აქვს. დანარჩენ 99.83%-ზე ის ისტორიულ საშუალოს აბრუნებს.

ამიტომ `submission_prophet.csv`-ის Kaggle-ქულა არ არის Prophet-ის შედეგი. ის პრაქტიკულად
საშუალოს baseline-ს ზომავს. Prophet-ის ერთადერთი სანდო რიცხვები subset-ის შედეგებია.

ეს ხარვეზი კი არა, არამედ განცხადების "Prophet არ მასშტაბირდება 3,331-სერიან panel-ზე"
რიცხვად გადაქცევაა.

## 7. MLflow და Model Registry

თითო არქიტექტურაზე ერთი experiment, შიგნით ეტაპობრივი run-ები:

| Experiment | Run-ები |
|---|---|
| `LightGBM_Training` | `Cleaning`, `Feature_Selection`, `CrossValidation` |
| `XGBoost_Training` | `Cleaning`, `Feature_Selection`, `CrossValidation`, `HPO` (15 nested trial), `Final` |
| `NBEATS_Training` | `Cleaning`, `Feature_Selection`, `CrossValidation`, `HPO` |
| `DLinear_Training` | `Cleaning`, `Feature_Selection`, `CrossValidation`, `HPO`, `Final` |
| `PatchTST_Training` | `DataPrep`, `CrossValidation`, `HPO` (5 nested trial), `Final` |
| `SARIMA_Training` | `Cleaning`, `CrossValidation`, `Final` |
| `Prophet_Training` | `SubsetSelection`, `Yearly`, `Holidays`, `Final` |
| `TimesFM_Training` | `ZeroShot`, `Analysis` |

### Raw-test კონტრაქტი

რეგისტრირებული მოდელი პირდაპირ raw სატესტო სტრიქონებზე მუშაობს: `Store`, `Dept`, `Date`,
`IsHoliday`. მომხმარებელს არანაირი preprocessing არ სჭირდება, რადგან მოდელი თავად აკეთებს
merge-ს, feature-ების აგებას და სვეტების გასწორებას.

| რეგისტრირებული მოდელი | Wrapper | ფაილი |
|---|---|---|
| `XGBoost_Walmart` | sklearn `Pipeline`, `WalmartRegressor` | `pipeline.py` |
| `PatchTST_Walmart` | `NeuralForecastRawPyFunc` | `neuralforecast_pyfunc.py` |
| `Prophet_Walmart` | `ProphetSubsetPyFunc` | `prophet_pyfunc.py` |
| `walmart-best` | საუკეთესო მოდელი (champion) | |

ჩატვირთვა: `model_inference.ipynb`.

## 8. რეპოზიტორიის სტრუქტურა

```text
├── dataloader.py              # load_raw, merge_all, make_submission
├── features.py                # build_features, 73 სვეტი
├── metrics.py                 # wmae
├── validation.py              # expanding_window_folds, holdout_split
├── pipeline.py                # XGBoost-ის raw-to-prediction Pipeline
├── neuralforecast_pyfunc.py   # PatchTST / N-BEATS / DLinear pyfunc
├── prophet_pyfunc.py          # Prophet subset pyfunc
├── model_inference.ipynb      # Model Registry-დან საუკეთესო მოდელის ჩატვირთვა
├── models/
│   ├── model_experiment_LightGBM.ipynb
│   ├── model_experiment_XGBoost.ipynb
│   ├── model_experiment_NBEATS.ipynb
│   ├── model_experiment_DLinear.ipynb
│   ├── model_experiment_PatchTST.ipynb
│   ├── model_experiment_SARIMA.ipynb
│   ├── model_experiment_Prophet.ipynb
│   └── model_experiment_TimesFM.ipynb
├── FEATURES.md                # feature-ების დეტალური აღწერა
├── EXPERIMENTS.md             # ექსპერიმენტების რეესტრი
├── requirements.txt           # ხის და კლასიკური მოდელები
└── requirements-dl.txt        # deep learning მოდელები
```

## 9. გაშვება

ექსპერიმენტები Google Colab-ზეა გაშვებული. თითოეული notebook თვითკმარია.

1. გახსენით notebook Colab-ში, GitHub-იდან.
2. Secrets პანელში ჩართეთ Notebook access ორივესთვის:
   - `KAGGLE_JSON`, Kaggle API key (`kaggle.json`-ის შიგთავსი)
   - `DAGSHUB_TOKEN`, DagsHub access token
3. Runtime, Run all.

პირველი უჯრა თავად აკლონებს რეპოს, დააინსტალირებს დამოკიდებულებებს, ჩამოტვირთავს Kaggle-ის
მონაცემებს და დააკავშირებს MLflow-ს DagsHub-თან.

ლოკალურად:

```bash
pip install -r requirements.txt        # ხის და კლასიკური მოდელები
pip install -r requirements-dl.txt     # deep learning მოდელები
```
