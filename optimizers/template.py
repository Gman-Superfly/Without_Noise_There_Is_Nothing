from __future__ import annotations

"""
Optimizer template

Instructions:
- Define an optimizer class or use an existing torch.optim class.
- Provide a factory function `_factory(parameters, **kwargs)` returning the optimizer.
- Register with `OptimizerRegistry.register("your_name", _factory)` at import time.

Design notes:
- Validate inputs aggressively (asserts) to fail fast.
- Keep one purpose per class/function; no side effects beyond optimizer state.
- Provide clear, typed signatures.
"""

from typing import Iterable

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from mlviz.registry import OptimizerRegistry


def _ensure_torch() -> None:
    assert torch is not None, "PyTorch not available"


class YourOptimizer:
    def __init__(self, parameters: Iterable[object], lr: float = 1e-3):
        _ensure_torch()
        assert lr > 0.0, "lr must be positive"
        self._params = [p for p in parameters]
        assert len(self._params) > 0, "no parameters provided"
        self.lr = float(lr)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self._params:
            if getattr(p, "grad", None) is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        # Example: SGD-like step (replace with your logic)
        for p in self._params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            p.data.add_(g, alpha=-self.lr)


def _factory(parameters: Iterable[object], lr: float = 1e-3):
    return YourOptimizer(parameters, lr=lr)


# Example registration (rename "your_optimizer" accordingly)
OptimizerRegistry.register("your_optimizer", _factory)


