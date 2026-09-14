"""Portfolio construction strategies that turn predictions into target weights.

The TopK strategy reuses the equal-weight ``to_signal`` idea from the original
``QlibTopKStrategy``: rank a day's predictions, keep the top ``top_k`` fraction,
and allocate equal weight across the survivors.
"""

from __future__ import annotations

import polars as pl

from quantlab.data.base import DATE, INSTRUMENT


class TopKStrategy:
    """Long-only equal-weight portfolio over the top ``top_k`` fraction of names."""

    def __init__(self, top_k: float = 0.2) -> None:
        self.top_k = top_k

    def target_weights(self, day: pl.DataFrame, predict_col: str = "predict") -> dict[str, float]:
        """Return ``{instrument: weight}`` for a single trading day.

        ``day`` is one day's slice of the prediction panel with columns
        ``[date, instrument, predict]``.
        """
        day = day.drop_nulls(subset=[predict_col])
        n = len(day)
        if n == 0:
            return {}
        k = max(1, int(n * self.top_k + 0.5))
        top = day.sort(predict_col, descending=True).head(k)
        w = 1.0 / k
        return dict.fromkeys(top[INSTRUMENT].to_list(), w)


def raw_prediction_to_signal(pred: pl.DataFrame, top_k: float = 0.2) -> pl.DataFrame:
    """Cross-sectionally rank predictions into a binary long signal (1 = top-k).

    Kept as a standalone helper mirroring the original ``raw_prediction_to_signal``.
    """
    return pred.with_columns(
        (pl.col("predict").rank(method="average", descending=True).over(DATE) <= pl.len().over(DATE) * top_k)
        .cast(pl.Int8)
        .alias("signal")
    )
