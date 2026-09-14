"""Unified model interface.

All models consume NumPy feature matrices ``(n_samples, n_features)`` and a
1-D target vector, and return predictions as a 1-D NumPy array. The polars
<-> NumPy plumbing lives in :mod:`quantlab.models.trainer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Base class for regression models used for cross-sectional stock ranking."""

    def __init__(self) -> None:
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> Model:
        """Fit on training data, optionally early-stopping on validation data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for ``X``."""

    def feature_importances(self, feature_names: list[str]) -> dict[str, float] | None:
        """Optional per-feature importance (LightGBM); ``None`` if unsupported."""
        return None
