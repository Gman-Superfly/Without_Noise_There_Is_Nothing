from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch.optim import SGD as _SGD
except Exception:  # pragma: no cover - optional torch environment
    torch = None
    _SGD = None

from mlviz.registry import OptimizerRegistry


def _ensure_torch() -> None:
    assert torch is not None and _SGD is not None, "PyTorch SGD not available"


def _factory(parameters: Iterable[object], lr: float = 1e-2, momentum: float = 0.0, nesterov: bool = False):
    _ensure_torch()
    return _SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov)


OptimizerRegistry.register("sgd", _factory)


