# Ames Housing Price Predictor

A production-style regression project trained on the [Kaggle Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). It pairs a carefully engineered preprocessing pipeline with XGBoost and LightGBM, then surfaces predictions through a Streamlit web app that treats the model like a real product rather than a notebook demo.

---

## What this project does

Given a set of property characteristics — lot size, neighborhood, build quality, basement finish, garage type, and roughly 75 other features — the model estimates a home's sale price in dollars. The full workflow covers:

- Domain-aware missing value handling (not generic mean/mode imputation)
- Rare category grouping to reduce encoding noise
- Feature engineering for total area, property age, and quality composites
- 5-fold cross-validated grid search across XGBoost and LightGBM
- `log1p` target transformation with `expm1` inversion at inference time
- A Streamlit UI that seeds all form inputs from the training distribution and shows a CV-based confidence band alongside the point estimate

---

## Project structure

```
ames-housing-price-predictor/
├── app.py                    # Streamlit prediction interface
├── requirements.txt
├── data/
│   └── raw/                  # Place train.csv, test.csv, sample_submission.csv here
├── models/                   # Serialized model bundle written here after training
├── outputs/                  # Kaggle submission CSV written here
├── reports/
│   ├── figures/
│   └── metrics/              # metrics.json and cv_results.csv written here
└── src/
    ├── config.py             # Centralized paths and training hyperparameters
    ├── data_preprocessing.py # Full sklearn-compatible pipeline
    ├── model_utils.py        # Model candidates, bundle serialization, inference
    ├── train.py              # Training entrypoint (CLI)
    ├── predict.py            # Inference helpers used by the app
    └── logging_utils.py
```

---

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/Louaighoul/ames-housing-price-predictor.git
cd ames-housing-price-predictor
pip install -r requirements.txt
```

**2. Add the data**

Download the competition files from Kaggle and place them under `data/raw/`:

```
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

**3. Train**

```bash
python -m src.train
```

This runs a 5-fold grid search over both XGBoost and LightGBM, picks the best model by validation RMSE, writes a serialized bundle to `models/best_model.pkl`, and saves a Kaggle-ready submission to `outputs/submission.csv`.

**4. Launch the app**

```bash
streamlit run app.py
```

The UI loads the trained bundle and seeds every input field from the training distribution. Fill in the property details and click **Predict Sale Price**.

---

## How the pipeline works

### Missing value handling

Most real estate datasets treat a missing basement or garage field as "no basement / no garage." The preprocessing module handles these cases explicitly using Ames-specific rules rather than blanket imputation, which avoids introducing signal where there is none.

### Rare category grouping

Several categorical features in the dataset have sparse levels that appear in fewer than 1% of rows. `RareCategoryGrouper` collapses these into a single `__RARE__` bucket before encoding, reducing dimensionality and keeping the model from fitting to one-off labels that won't generalize.

### Feature engineering

The pipeline constructs a handful of composite features — total square footage, overall quality-condition interaction, years since build and remodel — before the data reaches the encoder. These tend to carry more predictive weight than the raw columns they're derived from.

### Model selection

Both XGBoost (`reg:squarederror`, `tree_method=hist`) and LightGBM (`regression` objective) are evaluated over a small grid. The search uses `neg_root_mean_squared_error` on the log-scaled target as the refit criterion. Whichever model wins the cross-validation comparison is the one serialized and served.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data processing | pandas, NumPy, scikit-learn |
| Models | XGBoost, LightGBM |
| Serialization | joblib |
| UI | Streamlit |
| Python | 3.9+ |

---

## Configuration

All paths and training knobs live in `src/config.py`. The relevant defaults:

| Setting | Default | Notes |
|---|---|---|
| `n_splits` | 5 | KFold cross-validation |
| `random_state` | 42 | Seeds both CV and model estimators |
| `n_jobs` | -1 | Uses all available CPU cores |
| `min_category_frequency` | 0.01 | Threshold for rare label grouping |

To point the pipeline at a different data location or model output path, update the `ProjectPaths` dataclass rather than touching individual scripts.

---

## Output files

After `python -m src.train` completes:

- `models/best_model.pkl` — the full fitted pipeline bundle, including the model name, CV RMSE, and feature list
- `reports/metrics/metrics.json` — best CV RMSE, MAE, R², and the winning hyperparameters
- `reports/metrics/cv_results.csv` — full grid search results for both candidates
- `outputs/submission.csv` — Kaggle submission formatted with `Id` and `SalePrice`

---

## Dataset

The Ames Housing dataset was compiled by Dean De Cock and covers residential property sales in Ames, Iowa from 2006–2010. It contains 79 explanatory variables and is widely used as a more nuanced alternative to the Boston Housing dataset for regression benchmarking.

Source: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

---

## License

This project is released for educational and portfolio purposes. The Ames Housing dataset is subject to Kaggle's terms of use.
