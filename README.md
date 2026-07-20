# Ames Housing Price Prediction

Production-style end-to-end machine learning project for the Kaggle Ames Housing dataset.

This repository is intentionally structured like a real-world ML product instead of a single notebook. It includes domain-aware preprocessing, leak-safe model training, CPU-friendly cross-validation, model serialization, and a lightweight Streamlit demo for inference.

## STAR Summary

### Situation
Real estate pricing is a classic tabular ML problem with messy missing values, mixed feature types, and correlated variables. The Ames dataset is especially useful for interviews because it contains both obvious features and subtle domain-specific missingness.

### Task
Build a professional, modular Python repository that:

- handles Ames-specific missing data correctly
- reduces multicollinearity through feature engineering
- trains strong CPU-friendly models
- serializes a reusable model artifact
- exposes predictions through a simple web interface

### Action
I designed a clean pipeline with:

- domain-aware imputations
- rare-category consolidation
- feature engineering for total area, age, and quality signals
- log-target training for sale price stability
- cross-validated grid search over XGBoost and LightGBM
- Streamlit-based inference UI

### Result
The repository is now organized for portfolio use, technical interviews, and iterative model improvement. It demonstrates end-to-end ML engineering rather than only model fitting.

## Repository Structure

```text
src/
├─ __init__.py
├─ config.py
├─ data_preprocessing.py
├─ logging_utils.py
├─ model_utils.py
├─ predict.py
└─ train.py
app.py
requirements.txt
README.md
data/raw/
├─ train.csv
├─ test.csv
└─ sample_submission.csv
models/
└─ best_model.pkl
reports/metrics/
└─ metrics.json
```

## Data Pain Points and Techniques Used

### 1. Heavy Missing Data
The Ames dataset has many missing values, but they are not all the same kind of missingness.

I handled them with distinct strategies:

- `None` imputation for features where missing means the feature does not exist
  - examples: `GarageType`, `BsmtQual`, `FireplaceQu`, `PoolQC`, `Fence`
- median imputation for numeric features with genuine gaps
  - examples: `LotFrontage`, `GarageYrBlt`, remaining numeric columns
- mode imputation for true data-quality missingness in categorical columns
  - examples: `Electrical`, `MSZoning`, `Exterior1st`, `KitchenQual`
- zero fill for area/count columns when absence is meaningful
  - examples: `GarageArea`, `TotalBsmtSF`, `BsmtFullBath`, `PoolArea`

I also added binary missingness flags such as:

- `HasGarage`
- `HasBasement`
- `HasFireplace`
- `HasPool`

These flags let the model learn from the absence itself.

### 2. High Dimensionality and Multicollinearity
The raw dataset has many correlated variables.

To reduce redundancy, I engineered aggregate features:

- `HouseAge`
- `RemodelAge`
- `YearsSinceRemodel`
- `TotalSF`
- `TotalBath`
- `TotalPorchSF`
- `QualityConditionScore`
- `QualityGap`
- `GarageAreaPerCar`
- `LivingAreaToLotRatio`

I also cast `MSSubClass` to string because it is an encoded category, not a true numeric quantity.

### 3. Advanced Categorical Encoding
I used a rare-category strategy to collapse sparse labels into a single bucket before one-hot encoding.

Why this helps:

- reduces dimensionality
- improves stability on sparse categories
- prevents the model from overfitting to one-off labels

The encoding strategy is leak-safe because it is fit on training folds only inside the model pipeline.

## Modeling Approach

The training pipeline focuses on strong CPU-efficient tree-based models:

- XGBoost
- LightGBM

The training objective uses `log1p(SalePrice)` so the target distribution is more stable and closer to what the Kaggle competition evaluates.

Cross-validation details:

- `KFold` with 5 splits
- shuffle enabled
- fixed random seed for reproducibility
- `GridSearchCV` over compact parameter grids
- scoring tracked on:
  - RMSE
  - MAE
  - R2

The best model is serialized as a `.pkl` artifact with `joblib`.

## CPU Optimization Decisions

This project is optimized for local laptop development:

- tree models use histogram-based algorithms where available
- preprocessing is handled with scikit-learn pipelines
- one-hot encoding is sparse and memory-conscious
- rare category grouping reduces feature explosion
- the parameter grids are intentionally compact so CV remains practical on CPU

## Core Modules

### `src/data_preprocessing.py`
Domain-aware preprocessing module with:

- train/test loading
- target splitting
- domain-specific imputations
- feature engineering
- rare category handling
- preprocessing artifacts for inference

### `src/train.py`
CLI training entrypoint that:

- loads raw train/test CSV files
- runs cross-validated grid search
- compares XGBoost and LightGBM
- saves metrics to JSON and CSV
- serializes the best model bundle
- optionally creates a Kaggle submission file

### `src/predict.py`
Simple inference helpers for loading the saved model bundle and predicting prices on raw feature rows.

### `app.py`
Streamlit interface that:

- loads the trained `.pkl` model
- presents a form for house features
- returns a live price estimate

## How to Run

### 0. Download the Kaggle data

Place `train.csv`, `test.csv`, and `sample_submission.csv` from the Kaggle archive into:

```text
data/raw/
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train --train-csv data/raw/train.csv --test-csv data/raw/test.csv
```

This will create:

- `models/best_model.pkl`
- `reports/metrics/metrics.json`
- `reports/metrics/cv_results.csv`
- `outputs/submission.csv`

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

## Artifacts Produced

- `models/best_model.pkl`
  - serialized pipeline + model bundle
- `reports/metrics/metrics.json`
  - summary of CV and diagnostic metrics
- `reports/metrics/cv_results.csv`
  - detailed grid search output
- `outputs/submission.csv`
  - Kaggle-ready prediction file for the test set

## Notes for Interview Discussion

If asked why this project is strong, emphasize:

- domain-specific missingness handling instead of generic imputation
- feature engineering that reflects housing economics
- leak-safe pipeline design
- CPU-aware model selection
- production-style project structure
- Streamlit inference experience

## Future Improvements

- add SHAP-based explainability
- add model comparison with CatBoost
- add experiment tracking with MLflow
- add unit tests for preprocessing edge cases
- containerize the app with Docker
