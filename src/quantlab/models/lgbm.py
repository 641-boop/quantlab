"""LightGBM regression wrapper (the primary, robust baseline)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.models.base import Model


class LGBMModel(Model):
    """Gradient-boosted regression with early stopping on a validation set."""

    def __init__(self, params: dict | None = None, num_boost_round: int = 1000, early_stopping: int = 30) -> None:
        super().__init__()
        defaults: dict = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if params:
            defaults.update(params)
        self.params = defaults
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.model: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> LGBMModel:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(n_estimators=self.num_boost_round, **self.params)
        callbacks: list[Any] = [lgb.log_evaluation(0)]
        if X_valid is not None and y_valid is not None:
            model.fit(
                X,
                y,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False), *callbacks],
            )
        else:
            model.fit(X, y, callbacks=callbacks)
        self.model = model
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "model must be fitted before predict"
        return self.model.predict(X)

    def feature_importances(self, feature_names: list[str]) -> dict[str, float] | None:
        if self.model is None:
            return None
        imp = self.model.feature_importances_
        return dict(zip(feature_names, imp, strict=False))
