"""Training entrypoint for the Ames Housing project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from .config import ProjectPaths, TrainConfig
from .data_preprocessing import AmesPreprocessingConfig, build_full_model_pipeline, load_raw_data
from .logging_utils import configure_logging
from .model_utils import ModelBundle, build_model_candidates, save_bundle, save_json


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train XGBoost/LightGBM models on Ames Housing.")
    parser.add_argument("--train-csv", type=Path, default=ProjectPaths.train_csv)
    parser.add_argument("--test-csv", type=Path, default=ProjectPaths.test_csv)
    parser.add_argument("--model-path", type=Path, default=ProjectPaths.final_model_path)
    parser.add_argument("--metrics-path", type=Path, default=ProjectPaths.metrics_json_path)
    parser.add_argument("--cv-results-path", type=Path, default=ProjectPaths.cv_results_csv_path)
    parser.add_argument("--submission-path", type=Path, default=ProjectPaths.root / "outputs" / "submission.csv")
    return parser.parse_args()


def evaluate_holdout_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics on the transformed target scale."""

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
    }


def choose_best_search(searches: List[GridSearchCV]) -> GridSearchCV:
    """Select the search object with the best RMSE on the validation folds."""

    if not searches:
        raise RuntimeError("No model candidates were available for grid search.")

    return max(searches, key=lambda search: search.best_score_)


def build_submission(test_df: pd.DataFrame, predictions: np.ndarray, output_path: Path) -> None:
    """Save a Kaggle submission file using the test set IDs."""

    submission = pd.DataFrame({"Id": test_df["Id"].astype(int), "SalePrice": predictions})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    LOGGER.info("Saved submission file to %s", output_path)


def main() -> None:
    """Train the best model and persist the final bundle."""

    args = parse_args()
    configure_logging()

    train_config = TrainConfig()
    preprocess_config = AmesPreprocessingConfig(
        min_category_frequency=train_config.min_category_frequency,
        drop_original_total_sf_sources=False,
    )

    train_df, test_df = load_raw_data(args.train_csv, args.test_csv)
    if "SalePrice" not in train_df.columns:
        raise ValueError("The training file must contain a SalePrice target column.")

    X_train = train_df.drop(columns=["SalePrice", "Id"], errors="ignore")
    y_train = np.log1p(train_df["SalePrice"].astype(float).to_numpy())

    cv = KFold(n_splits=train_config.n_splits, shuffle=True, random_state=train_config.random_state)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    searches: List[GridSearchCV] = []
    model_candidates = build_model_candidates(
        random_state=train_config.random_state,
        n_jobs=train_config.n_jobs,
    )

    if not model_candidates:
        raise RuntimeError(
            "Neither xgboost nor lightgbm is installed. Install at least one to run training."
        )

    for model_name, estimator, param_grid in model_candidates:
        LOGGER.info("Starting grid search for %s", model_name)
        pipeline = build_full_model_pipeline(estimator, config=preprocess_config)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="rmse",
            cv=cv,
            n_jobs=train_config.n_jobs,
            verbose=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        LOGGER.info(
            "%s best CV RMSE(log): %.5f with params=%s",
            model_name,
            -search.best_score_,
            search.best_params_,
        )
        searches.append(search)

    best_search = choose_best_search(searches)
    best_model_name = best_search.best_estimator_.named_steps["model"].__class__.__name__

    cv_results_df = pd.DataFrame(best_search.cv_results_)
    cv_results_df["target_scale"] = "log1p"
    cv_results_df["best_model"] = best_model_name
    args.cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results_df.to_csv(args.cv_results_path, index=False)

    best_params_summary = {
        key: value for key, value in best_search.best_params_.items()
    }
    fold_metrics = {
        "best_cv_rmse_log": float(-best_search.best_score_),
        "best_cv_mae_log": float(-best_search.cv_results_["mean_test_mae"][best_search.best_index_]),
        "best_cv_r2_log": float(best_search.cv_results_["mean_test_r2"][best_search.best_index_]),
        "best_model_name": best_model_name,
    }

    # Train-set diagnostic predictions are useful for quick sanity checks and
    # to populate a compact metrics file. The real performance estimate remains
    # the cross-validation score above.
    train_pred_log = best_search.best_estimator_.predict(X_train)
    diagnostics = evaluate_holdout_metrics(y_train, train_pred_log)
    metrics = {**fold_metrics, **diagnostics, "best_params": best_params_summary}

    save_json(args.metrics_path, metrics)

    feature_names: List[str]
    try:
        feature_names = best_search.best_estimator_.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    bundle = ModelBundle(
        pipeline=best_search.best_estimator_,
        best_model_name=best_model_name,
        cv_rmse_log=float(-best_search.best_score_),
        metrics={k: float(v) for k, v in diagnostics.items()},
        feature_names=feature_names,
        target_is_log1p=True,
    )
    save_bundle(args.model_path, bundle)
    LOGGER.info("Saved model bundle to %s", args.model_path)

    if args.test_csv.exists():
        raw_test = pd.read_csv(args.test_csv)
        test_features = raw_test.drop(columns=["Id"], errors="ignore")
        test_predictions = np.expm1(best_search.best_estimator_.predict(test_features))
        build_submission(raw_test, test_predictions, args.submission_path)

    LOGGER.info("Training complete. Best model: %s", best_model_name)


if __name__ == "__main__":
    main()
