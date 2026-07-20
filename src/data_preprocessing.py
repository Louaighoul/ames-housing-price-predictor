"""Ames Housing preprocessing pipeline.

This module is intentionally designed for production-style reuse:
- explicit type hints
- logging and guarded I/O
- domain-aware missing-value handling
- feature engineering that reduces noise and improves signal
- leak-safe categorical encoding for train/test workflows


"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LOGGER = logging.getLogger(__name__)


def _build_one_hot_encoder() -> OneHotEncoder:
    """Create a version-tolerant OneHotEncoder.

    Why this exists:
    scikit-learn changed the `sparse` argument to `sparse_output` in newer
    releases. This helper keeps the repository usable across local dev setups.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,  # type: ignore[arg-type]
            dtype=np.float32,
        )


@dataclass(frozen=True)
class AmesPreprocessingConfig:
    """Configuration knobs for the preprocessing pipeline."""

    target_col: str = "SalePrice"
    id_col: str = "Id"
    rare_label: str = "__RARE__"
    min_category_frequency: float = 0.01
    drop_original_total_sf_sources: bool = False


@dataclass(frozen=True)
class FittedPreprocessingArtifacts:
    """Container for fitted preprocessing components.

    Keeping the fitted steps together makes the training script and the Streamlit
    app easy to wire up without leaking implementation details across files.
    """

    domain_imputer: "AmesDomainImputer"
    feature_engineer: "AmesFeatureEngineer"
    preprocessor: ColumnTransformer
    feature_names: List[str]


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse infrequent categorical labels into a single rare bucket.

    Why this matters:
    Ames has several categorical variables with small, sparse levels. Grouping
    rare labels reduces dimensionality, lowers variance, and keeps the encoder
    from learning unstable one-off signals that do not generalize.
    """

    def __init__(self, min_frequency: float = 0.01, rare_label: str = "__RARE__"):
        self.min_frequency = min_frequency
        self.rare_label = rare_label
        self.frequent_levels_: Dict[str, set[str]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RareCategoryGrouper":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_ = list(X.columns)
        n_rows = max(len(X), 1)
        self.frequent_levels_ = {}

        for column in self.columns_:
            value_counts = X[column].astype("string").fillna("None").value_counts(dropna=False)
            frequent = value_counts[value_counts / n_rows >= self.min_frequency].index.astype(str)
            self.frequent_levels_[column] = set(frequent.tolist())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)

        transformed = X.copy()
        for column in self.columns_:
            allowed = self.frequent_levels_.get(column, set())
            transformed[column] = (
                transformed[column]
                .astype("string")
                .fillna("None")
                .where(transformed[column].astype("string").fillna("None").isin(allowed), self.rare_label)
            )
        return transformed


class AmesDomainImputer(BaseEstimator, TransformerMixin):
    """Apply domain-aware imputations before feature engineering.

    Missingness in Ames is not random. Some nulls mean "feature absent" rather
    than truly missing data. Handling those cases explicitly preserves signal
    and prevents the model from treating absence as an arbitrary numeric zero.
    """

    categorical_none_cols = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MasVnrType",
    ]

    categorical_mode_cols = [
        "Electrical",
        "MSZoning",
        "Exterior1st",
        "Exterior2nd",
        "KitchenQual",
        "Functional",
        "SaleType",
        "Utilities",
        "BldgType",
        "HouseStyle",
        "LandSlope",
        "LotShape",
        "PavedDrive",
        "Street",
    ]

    zero_when_absent_numeric_cols = [
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
    ]

    group_median_cols = ["LotFrontage"]

    def __init__(self) -> None:
        self.mode_map_: Dict[str, object] = {}
        self.numeric_median_map_: Dict[str, float] = {}
        self.neighborhood_lotfrontage_median_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AmesDomainImputer":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.mode_map_ = {}
        self.numeric_median_map_ = {}
        self.neighborhood_lotfrontage_median_ = {}

        for column in self.categorical_mode_cols:
            if column in X.columns:
                mode_series = X[column].dropna()
                self.mode_map_[column] = mode_series.mode().iloc[0] if not mode_series.empty else "None"

        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                median_value = X[column].median()
                self.numeric_median_map_[column] = float(median_value) if pd.notna(median_value) else 0.0

        if "Neighborhood" in X.columns and "LotFrontage" in X.columns:
            grouped = X[["Neighborhood", "LotFrontage"]].dropna(subset=["Neighborhood"])
            medians = grouped.groupby("Neighborhood")["LotFrontage"].median()
            self.neighborhood_lotfrontage_median_ = {
                str(index): float(value) for index, value in medians.items() if pd.notna(value)
            }

        return self

    def _add_missingness_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add binary indicators so the model can learn from absence itself."""

        flagged = X.copy()
        flagged["HasGarage"] = np.where(flagged["GarageType"].isna(), 0, 1)
        flagged["HasBasement"] = np.where(flagged["BsmtQual"].isna(), 0, 1)
        flagged["HasFireplace"] = np.where(flagged["FireplaceQu"].isna(), 0, 1)
        flagged["HasPool"] = np.where(flagged["PoolQC"].isna(), 0, 1)
        flagged["HasFence"] = np.where(flagged["Fence"].isna(), 0, 1)
        flagged["HasAlley"] = np.where(flagged["Alley"].isna(), 0, 1)
        return flagged

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        transformed = self._add_missingness_flags(X)

        # Missing categories that semantically mean "feature not present".
        for column in self.categorical_none_cols:
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna("None")

        # These columns genuinely need a modal fill because the missingness is
        # a data quality issue, not an absence signal.
        for column, value in self.mode_map_.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(value)

        # LotFrontage is sparse and neighborhood-aware. Using the neighborhood
        # median preserves local lot-size structure better than a global mean.
        if "LotFrontage" in transformed.columns:
            if "Neighborhood" in transformed.columns and self.neighborhood_lotfrontage_median_:
                transformed["LotFrontage"] = transformed["LotFrontage"].fillna(
                    transformed["Neighborhood"].map(self.neighborhood_lotfrontage_median_)
                )
            transformed["LotFrontage"] = transformed["LotFrontage"].fillna(
                self.numeric_median_map_.get("LotFrontage", 0.0)
            )

        # Numeric features tied to the absence of a structure should default to
        # zero. This is especially important for tree models because zero is a
        # meaningful "no square feet / no area" state.
        for column in self.zero_when_absent_numeric_cols:
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(0)

        # Garage year is a special case: if there is no garage, a zero value is
        # more informative than a fabricated median construction year.
        if "GarageYrBlt" in transformed.columns:
            if "GarageType" in transformed.columns:
                transformed.loc[transformed["GarageType"].eq("None"), "GarageYrBlt"] = 0
            transformed["GarageYrBlt"] = transformed["GarageYrBlt"].fillna(
                self.numeric_median_map_.get("GarageYrBlt", 0.0)
            )

        if "MasVnrArea" in transformed.columns and "MasVnrType" in transformed.columns:
            transformed.loc[transformed["MasVnrType"].eq("None"), "MasVnrArea"] = 0

        # Final safety net for any remaining numeric gaps.
        numeric_columns = transformed.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            transformed[column] = transformed[column].fillna(self.numeric_median_map_.get(column, 0.0))

        # Any remaining object columns that are still null are imputed to None.
        object_columns = transformed.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            transformed[column] = transformed[column].fillna("None")

        return transformed


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create aggregate features that reduce redundancy and improve signal.

    These features are deliberately chosen to:
    - summarize correlated raw columns into a single stronger signal
    - reduce effective multicollinearity
    - preserve interpretability for hiring-manager style feature walkthroughs
    """

    def __init__(self, drop_original_total_sf_sources: bool = False) -> None:
        self.drop_original_total_sf_sources = drop_original_total_sf_sources

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AmesFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        engineered = X.copy()

        year_built = engineered.get("YearBuilt")
        year_remod = engineered.get("YearRemodAdd")
        year_sold = engineered.get("YrSold")

        if year_built is not None and year_sold is not None:
            engineered["HouseAge"] = year_sold - year_built
        if year_remod is not None and year_sold is not None:
            engineered["RemodelAge"] = year_sold - year_remod
        if year_built is not None and year_remod is not None:
            engineered["YearsSinceRemodel"] = year_remod - year_built

        required_total_sf_cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
        if all(col in engineered.columns for col in required_total_sf_cols):
            engineered["TotalSF"] = (
                engineered["TotalBsmtSF"] + engineered["1stFlrSF"] + engineered["2ndFlrSF"]
            )

        bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        if all(col in engineered.columns for col in bath_cols):
            engineered["TotalBath"] = (
                engineered["FullBath"]
                + 0.5 * engineered["HalfBath"]
                + engineered["BsmtFullBath"]
                + 0.5 * engineered["BsmtHalfBath"]
            )

        porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        if all(col in engineered.columns for col in porch_cols):
            engineered["TotalPorchSF"] = sum(engineered[col] for col in porch_cols)

        quality_cols = ["OverallQual", "OverallCond"]
        if all(col in engineered.columns for col in quality_cols):
            engineered["QualityConditionScore"] = engineered["OverallQual"] * engineered["OverallCond"]
            engineered["QualityGap"] = engineered["OverallQual"] - engineered["OverallCond"]

        if "GarageArea" in engineered.columns and "GarageCars" in engineered.columns:
            engineered["GarageAreaPerCar"] = engineered["GarageArea"] / (engineered["GarageCars"] + 1.0)

        if "GrLivArea" in engineered.columns and "LotArea" in engineered.columns:
            engineered["LivingAreaToLotRatio"] = engineered["GrLivArea"] / (engineered["LotArea"] + 1.0)

        # MSSubClass is stored as an integer code, but semantically it behaves
        # like a nominal category. Casting to string prevents the model from
        # pretending that the class codes have a meaningful numeric distance.
        if "MSSubClass" in engineered.columns:
            engineered["MSSubClass"] = engineered["MSSubClass"].astype("string")

        if self.drop_original_total_sf_sources:
            for column in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]:
                if column in engineered.columns:
                    engineered.drop(columns=column, inplace=True)

        return engineered


def load_raw_data(train_path: str | Path, test_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle-style train/test CSV files with guardrails."""

    try:
        train_df = pd.read_csv(Path(train_path))
        test_df = pd.read_csv(Path(test_path))
        LOGGER.info("Loaded train shape=%s, test shape=%s", train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to load raw Ames data")
        raise RuntimeError("Unable to load raw Ames housing CSV files.") from exc


def split_target(
    train_df: pd.DataFrame,
    target_col: str = "SalePrice",
    id_col: str = "Id",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target from features and drop identifier columns."""

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the training data.")

    y = train_df[target_col].copy()
    X = train_df.drop(columns=[target_col], errors="ignore")
    X = X.drop(columns=[id_col], errors="ignore")
    return X, y


def drop_identifier_columns(df: pd.DataFrame, id_col: str = "Id") -> pd.DataFrame:
    """Drop the identifier column because it carries no predictive signal."""

    return df.drop(columns=[id_col], errors="ignore")


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split the dataframe into numeric and categorical feature names."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    config: AmesPreprocessingConfig,
) -> ColumnTransformer:
    """Create the final sklearn ColumnTransformer.

    Why a ColumnTransformer:
    - keeps preprocessing leak-safe and fit on train only
    - preserves a single end-to-end artifact for later serialization
    - keeps the training script simple and reproducible
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("rare_grouper", RareCategoryGrouper(
                min_frequency=config.min_category_frequency,
                rare_label=config.rare_label,
            )),
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", _build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, selector(dtype_include=["object", "string", "category"])),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def preprocess_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[AmesPreprocessingConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """Run the full preprocessing workflow on train/test data.

    Returns:
        X_train_processed, y_train, X_test_processed, fitted_preprocessor, feature_names
    """

    config = config or AmesPreprocessingConfig()

    try:
        X_train, y_train = split_target(train_df, target_col=config.target_col, id_col=config.id_col)
        X_test = drop_identifier_columns(test_df, id_col=config.id_col)

        artifacts = fit_preprocessing_artifacts(X_train, config=config)

        X_train_processed = artifacts.feature_engineer.transform(
            artifacts.domain_imputer.transform(X_train)
        )
        X_test_processed = artifacts.feature_engineer.transform(
            artifacts.domain_imputer.transform(X_test)
        )

        X_train_matrix = artifacts.preprocessor.transform(X_train_processed)
        X_test_matrix = artifacts.preprocessor.transform(X_test_processed)
        feature_names = artifacts.feature_names

        LOGGER.info(
            "Preprocessing complete: X_train=%s, X_test=%s, features=%d",
            getattr(X_train_matrix, "shape", None),
            getattr(X_test_matrix, "shape", None),
            len(feature_names),
        )

        return X_train_matrix, y_train.to_numpy(), X_test_matrix, artifacts.preprocessor, feature_names
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Ames preprocessing failed")
        raise RuntimeError("Ames preprocessing pipeline failed.") from exc


def fit_preprocessing_artifacts(
    X_train: pd.DataFrame,
    config: Optional[AmesPreprocessingConfig] = None,
) -> FittedPreprocessingArtifacts:
    """Fit preprocessing components on train data only.

    The returned artifacts can be serialized alongside the final model or used
    inside the Streamlit app for inference-time transformations.
    """

    config = config or AmesPreprocessingConfig()

    domain_imputer = AmesDomainImputer()
    feature_engineer = AmesFeatureEngineer(
        drop_original_total_sf_sources=config.drop_original_total_sf_sources
    )

    # Fit the domain-specific steps first, then discover the transformed schema.
    X_train_processed = feature_engineer.fit_transform(domain_imputer.fit_transform(X_train))
    preprocessor = build_preprocessor(config=config)
    preprocessor.fit(X_train_processed)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(preprocessor.transform(X_train_processed).shape[1])]

    return FittedPreprocessingArtifacts(
        domain_imputer=domain_imputer,
        feature_engineer=feature_engineer,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


def build_full_model_pipeline(
    estimator: BaseEstimator,
    config: Optional[AmesPreprocessingConfig] = None,
) -> Pipeline:
    """Build a raw-data-to-prediction pipeline for training and CV."""

    config = config or AmesPreprocessingConfig()

    return Pipeline(
        steps=[
            ("domain_imputer", AmesDomainImputer()),
            (
                "feature_engineer",
                AmesFeatureEngineer(drop_original_total_sf_sources=config.drop_original_total_sf_sources),
            ),
            ("preprocessor", build_preprocessor(config=config)),
            ("model", estimator),
        ]
    )


def preprocess_for_inference(
    raw_df: pd.DataFrame,
    fitted_imputer: AmesDomainImputer,
    fitted_feature_engineer: AmesFeatureEngineer,
    fitted_preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Transform a single inference dataframe using fitted preprocessing objects."""

    try:
        transformed = fitted_imputer.transform(raw_df)
        transformed = fitted_feature_engineer.transform(transformed)
        return fitted_preprocessor.transform(transformed)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Inference preprocessing failed")
        raise RuntimeError("Failed to preprocess inference data.") from exc


def validate_expected_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Fail fast when a required raw column is missing from the dataset."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
