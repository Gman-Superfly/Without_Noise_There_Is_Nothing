from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch.optim import AdamW as _AdamW
except Exception:  # pragma: no cover - optional torch environment
    torch = None
    _AdamW = None

from mlviz.registry import OptimizerRegistry


def _ensure_torch() -> None:
    assert torch is not None and _AdamW is not None, "PyTorch AdamW not available"


def _factory(parameters: Iterable[object], lr: float = 1e-3, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.999)):
    _ensure_torch()
    return _AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)


OptimizerRegistry.register("adamw", _factory)


