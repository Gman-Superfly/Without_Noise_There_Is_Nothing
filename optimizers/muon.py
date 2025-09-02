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
    assert torch is not None and _SGD is not None, "PyTorch not available"


class MuonLikeOptimizer:
    def __init__(self, parameters: Iterable[object], lr: float = 1e-3):
        _ensure_torch()
        self._opt = _SGD(parameters, lr=lr)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._opt.step()


def _factory(parameters: Iterable[object], lr: float = 1e-3):
    return MuonLikeOptimizer(parameters, lr=lr)


OptimizerRegistry.register("muon", _factory)


