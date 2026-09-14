"""Training plumbing: date-based splitting and polars <-> NumPy conversion."""

from __future__ import annotations

import numpy as np
import polars as pl

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.models.base import Model


def split_train_valid_test(
    df: pl.DataFrame, train_end: str, valid_end: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a long panel chronologically into train / valid / test."""
    train_end_d = pl.lit(train_end).str.to_date()
    valid_end_d = pl.lit(valid_end).str.to_date()
    train = df.filter(pl.col(DATE) <= train_end_d)
    valid = df.filter((pl.col(DATE) > train_end_d) & (pl.col(DATE) <= valid_end_d))
    test = df.filter(pl.col(DATE) > valid_end_d)
    return train, valid, test


def _to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float32)


def fit_predict(
    model: Model,
    df: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    train_end: str,
    valid_end: str,
) -> tuple[Model, pl.DataFrame]:
    """Fit ``model`` on train, early-stop on valid, and return out-of-sample test predictions.

    ``df`` is a long panel with ``[date, instrument, *feature_cols, label_col]``.
    Missing features are zero-filled; rows with a missing label are dropped.
    """
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in feature_cols])
    df = df.drop_nulls(subset=[label_col])
    train, valid, test = split_train_valid_test(df, train_end, valid_end)
    if len(test) == 0:
        raise ValueError("empty test set; check train_end / valid_end boundaries")

    Xtr, ytr = _to_numpy(train, feature_cols), train.select(label_col).to_numpy().ravel()
    Xv, yv = _to_numpy(valid, feature_cols), valid.select(label_col).to_numpy().ravel()
    model.fit(Xtr, ytr, Xv, yv)

    pred = model.predict(_to_numpy(test, feature_cols))
    pred_df = test.select([pl.col(DATE), pl.col(INSTRUMENT)]).with_columns(pl.Series("predict", pred))
    return model, pred_df
