"""Model selection, serialization, and prediction helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Persisted training artifact used by the Streamlit app and inference."""

    pipeline: Any
    best_model_name: str
    cv_rmse_log: float
    metrics: Dict[str, float]
    feature_names: List[str]
    target_is_log1p: bool = True


def try_import_xgboost() -> Any | None:
    """Import XGBoost lazily so the repo still works when the package is absent."""

    try:
        from xgboost import XGBRegressor

        return XGBRegressor
    except Exception:
        LOGGER.warning("xgboost is not installed; skipping XGBoost candidate.")
        return None


def try_import_lightgbm() -> Any | None:
    """Import LightGBM lazily so the repo still works when the package is absent."""

    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor
    except Exception:
        LOGGER.warning("lightgbm is not installed; skipping LightGBM candidate.")
        return None


def build_model_candidates(random_state: int, n_jobs: int) -> List[Tuple[str, BaseEstimator, Dict[str, List[Any]]]]:
    """Return small CPU-friendly grid-search spaces for the two target models."""

    candidates: List[Tuple[str, BaseEstimator, Dict[str, List[Any]]]] = []

    XGBRegressor = try_import_xgboost()
    if XGBRegressor is not None:
        candidates.append(
            (
                "xgboost",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_estimators=600,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    eval_metric="rmse",
                ),
                {
                    "model__max_depth": [3, 4],
                    "model__learning_rate": [0.03, 0.05],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                    "model__min_child_weight": [1, 5],
                    "model__reg_alpha": [0.0],
                    "model__reg_lambda": [1.0],
                },
            )
        )

    LGBMRegressor = try_import_lightgbm()
    if LGBMRegressor is not None:
        candidates.append(
            (
                "lightgbm",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=800,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=-1,
                ),
                {
                    "model__num_leaves": [31, 63],
                    "model__learning_rate": [0.03, 0.05],
                    "model__subsample": [0.8],
                    "model__colsample_bytree": [0.8],
                    "model__min_child_samples": [20, 30],
                    "model__reg_alpha": [0.0],
                    "model__reg_lambda": [0.0, 1.0],
                },
            )
        )

    return candidates


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a small JSON artifact with robust folder creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_bundle(path: Path, bundle: ModelBundle) -> None:
    """Serialize the trained bundle with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    """Load the persisted model bundle."""

    return joblib.load(path)


def predict_prices(bundle: ModelBundle, raw_features: pd.DataFrame) -> np.ndarray:
    """Return house price predictions on the original dollar scale."""

    cleaned_features = raw_features.drop(columns=["Id", "SalePrice"], errors="ignore")
    raw_predictions = bundle.pipeline.predict(cleaned_features)
    if bundle.target_is_log1p:
        return np.expm1(raw_predictions)
    return np.asarray(raw_predictions, dtype=float)
