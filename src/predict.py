"""Inference helpers for the Ames Housing project."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .model_utils import ModelBundle, load_bundle, predict_prices


LOGGER = logging.getLogger(__name__)


def load_model(path: Path = ProjectPaths.final_model_path) -> ModelBundle:
    """Load the persisted model bundle from disk."""

    try:
        return load_bundle(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to load model bundle from %s", path)
        raise RuntimeError(f"Could not load model bundle from {path}.") from exc


def predict_dataframe(bundle: ModelBundle, raw_features: pd.DataFrame) -> np.ndarray:
    """Predict sale prices for a dataframe of raw Ames features."""

    try:
        return predict_prices(bundle, raw_features)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Prediction failed")
        raise RuntimeError("Prediction failed.") from exc

