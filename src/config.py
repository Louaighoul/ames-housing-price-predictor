"""Project configuration for the Ames Housing workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized file paths used across the project."""

    root: Path = get_project_root()
    raw_data_dir: Path = root / "data" / "raw"
    processed_data_dir: Path = root / "data" / "processed"
    models_dir: Path = root / "models"
    metrics_dir: Path = root / "reports" / "metrics"
    figures_dir: Path = root / "reports" / "figures"
    artifacts_dir: Path = root / "artifacts"
    train_csv: Path = raw_data_dir / "train.csv"
    test_csv: Path = raw_data_dir / "test.csv"
    sample_submission_csv: Path = raw_data_dir / "sample_submission.csv"
    final_model_path: Path = models_dir / "best_model.pkl"
    metrics_json_path: Path = metrics_dir / "metrics.json"
    cv_results_csv_path: Path = metrics_dir / "cv_results.csv"
    feature_names_json_path: Path = artifacts_dir / "feature_names.json"


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters and CV settings."""

    random_state: int = 42
    n_splits: int = 5
    n_jobs: int = -1
    min_category_frequency: float = 0.01

