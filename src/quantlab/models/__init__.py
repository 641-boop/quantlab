"""Model layer: a unified interface over LightGBM and PyTorch regressors.

PyTorch models are imported lazily (see :func:`__getattr__`) so LightGBM-only
runs don't pay torch's heavy startup cost — or require a working torch runtime
on machines that only need the LGBM baseline.
"""

from __future__ import annotations

from quantlab.models.base import Model
from quantlab.models.lgbm import LGBMModel
from quantlab.models.trainer import fit_predict, split_train_valid_test

__all__ = [
    "Model",
    "LGBMModel",
    "MLPModel",
    "TransformerModel",
    "split_train_valid_test",
    "fit_predict",
]


def __getattr__(name: str):
    if name in {"MLPModel", "TransformerModel"}:
        from quantlab.models.torch_model import MLPModel, TransformerModel

        return {"MLPModel": MLPModel, "TransformerModel": TransformerModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
