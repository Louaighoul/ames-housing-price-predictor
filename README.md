Ames Housing Price Prediction
A modular, production-style machine learning pipeline for the Ames Housing dataset. Built to move beyond exploratory notebooks into clean, structured Python code featuring domain-aware preprocessing, leak-safe cross-validation, CPU-optimized gradient boosting, and real-time inference via Streamlit.

├── data/
│   └── raw/               # train.csv, test.csv, sample_submission.csv
├── models/
│   └── best_model.pkl     # Serialized pipeline + model artifact
├── reports/
│   └── metrics/           # metrics.json, cv_results.csv
├── src/
│   ├── config.py          # Paths, features, and grid search hyperparams
│   ├── data_preprocessing.py # Pipeline transformers & feature engineering
│   ├── logging_utils.py   # Standard logging setup
│   ├── model_utils.py     # Training loops, CV, & metrics calculation
│   ├── predict.py         # Batch and single-row inference helpers
│   └── train.py           # CLI entrypoint for training & serialization
├── app.py                 # Streamlit web UI for live predictions
└── requirements.txt
Technical Highlights & Design Choices
1. Domain-Aware Missing Value Handling
Standard SimpleImputer(strategy='mean') distorts real estate datasets because missing values in structural attributes indicate absence, not missing data.

Explicit Negation (None / 0): Categorical features like GarageType, BsmtQual, FireplaceQu, and PoolQC use 'None' because missing values mean the property lacks that amenity. Corresponding area/count features use 0.

Stratified Medians: LotFrontage uses median imputation based on neighborhood groupings.

True Missingness: Mode imputation is reserved strictly for low-count data quality gaps (Electrical, MSZoning).

Absence Indicators: Added binary flags (HasGarage, HasBasement, HasPool, HasFireplace) so tree-based models can split directly on feature presence.

2. Feature Engineering & Multicollinearity
Raw housing attributes exhibit heavy collinearity (GrLivArea vs TotRmsAbvGrd, floor square footage). The following aggregations streamline these signals:

Aggregated Totals: Built TotalSF (Basement + 1st/2nd floor), TotalBath (Full + 0.5*Half across main level and basement), and TotalPorchSF.

Temporal Dynamics: Derived HouseAge, RemodelAge, and YearsSinceRemodel relative to YrSold.

Spatial Ratios: Added LivingAreaToLotRatio and GarageAreaPerCar.

Type Normalization: Cast MSSubClass to string categorical (it represents dwelling type codes, not continuous values).

3. Leak-Safe Pipeline & Categorical Encoding
Rare Category Grouping: Sparse categorical levels (e.g., rare Exterior1st materials) are grouped into an 'Other' bucket to prevent high-dimensional expansion and overfitting.

Strict Cross-Validation Bounds: All encoders, scalers, and custom transformers are encapsulated in Scikit-learn Pipelines and fitted strictly on training folds inside cross-validation.

4. Training & Target Optimization
Log-Transformed Target: Models target log(1+SalePrice) to mitigate right-skewness and match RMSLE evaluation metrics.

Model Selection: Compares XGBoost and LightGBM using 5-fold cross-validation (KFold).

CPU Optimization: Hyperparameter search grids use histogram-based tree algorithms (tree_method='hist' / boosting_type='gbdt') to maintain CV runtimes under 2 minutes on standard quad-core CPUs.

Quickstart
1. Environment Setup
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Place Raw Data
Download the data from Kaggle and place train.csv and test.csv inside data/raw/.

3. Run the Training Pipeline
Execute the CLI to preprocess data, perform grid search cross-validation, save performance reports, serialize the best model bundle, and generate predictions:

Bash
python -m src.train --train-csv data/raw/train.csv --test-csv data/raw/test.csv
Outputs generated:

models/best_model.pkl — Complete pipeline artifact (transformers + estimator).

reports/metrics/metrics.json — Out-of-fold RMSE, MAE, and R 
2
  performance metrics.

outputs/submission.csv — Predictions formatted for Kaggle submission.

4. Launch Local Inference App
Run the interactive Streamlit dashboard to test predictions against the serialized model artifact:

Bash
streamlit run app.py
