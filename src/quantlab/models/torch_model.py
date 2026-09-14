"""PyTorch regression models: a plain MLP and a simplified FT-Transformer.

Both share a mini-batch training loop with early stopping on validation loss and
per-epoch validation IC reporting (reusing the ``val_ic`` idea from the original
``models.py``).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from quantlab.models.base import Model


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class _TorchRegressor(Model):
    def __init__(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        patience: int = 5,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.device = device
        self.net: nn.Module | None = None

    def _build_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> _TorchRegressor:
        net = self._build_net(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device).view(-1, 1)
        has_valid = X_valid is not None and y_valid is not None
        if has_valid:
            Xv = torch.tensor(X_valid, dtype=torch.float32, device=self.device)
            yv = torch.tensor(y_valid, dtype=torch.float32, device=self.device).view(-1, 1)

        best_loss = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        patience_left = self.patience
        n = len(Xt)

        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                out = net(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()

            if has_valid:
                net.eval()
                with torch.no_grad():
                    pv = net(Xv).view(-1)
                    val_loss = float(loss_fn(pv, yv.view(-1)))
                    val_ic = _pearson(pv.cpu().numpy(), yv.view(-1).cpu().numpy())
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(net.state_dict())
                    patience_left = self.patience
                else:
                    patience_left -= 1
                if epoch == 1 or epoch % 5 == 0:
                    print(f"  epoch {epoch:>3}: val_loss {val_loss:.5f}  val_ic {val_ic:+.4f}")
                if patience_left <= 0:
                    print("  early stopping")
                    break

        if has_valid:
            net.load_state_dict(best_state)
        self.net = net
        self.fitted = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "model must be fitted before predict"
        self.net.eval()
        out = self.net(torch.tensor(X, dtype=torch.float32, device=self.device)).view(-1)
        return out.cpu().numpy()


class MLPModel(_TorchRegressor):
    """A plain multi-layer perceptron regressor."""

    def __init__(self, hidden_sizes: tuple[int, ...] = (64, 32), **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes

    def _build_net(self, n_features: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, n_features + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True, dim_feedforward=4 * d_model
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f = x.shape
        x = self.embed(x.unsqueeze(-1))  # (B, F, d)
        cls = self.cls.expand(b, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)  # (B, F+1, d)
        x = x + self.pos[:, : f + 1, :]
        x = self.encoder(x)
        return self.head(x[:, 0, :])  # CLS token


class TransformerModel(_TorchRegressor):
    """A simplified FT-Transformer over the feature vector."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def _build_net(self, n_features: int) -> nn.Module:
        return _TransformerNet(n_features, self.d_model, self.n_heads, self.n_layers, self.dropout)
