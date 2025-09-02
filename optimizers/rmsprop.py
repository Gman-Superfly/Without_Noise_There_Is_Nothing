from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch.optim import RMSprop as _RMSprop
except Exception:  # pragma: no cover - optional torch environment
    torch = None
    _RMSprop = None

from mlviz.registry import OptimizerRegistry


def _ensure_torch() -> None:
    assert torch is not None and _RMSprop is not None, "PyTorch RMSprop not available"


def _factory(parameters: Iterable[object], lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False):
    _ensure_torch()
    return _RMSprop(parameters, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)


OptimizerRegistry.register("rmsprop", _factory)


