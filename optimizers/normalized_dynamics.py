from __future__ import annotations

from typing import Iterable

try:
    import torch
except Exception:  # pragma: no cover - optional torch environment
    torch = None

from mlviz.registry import OptimizerRegistry


def _ensure_torch() -> None:
    assert torch is not None, "PyTorch not available"


class NormalizedDynamicsOptimizer:
    def __init__(self, parameters: Iterable[object], lr: float = 1e-3, eps: float = 1e-8, weight_decay: float = 0.0):
        _ensure_torch()
        assert lr > 0, "lr must be positive"
        assert eps > 0, "eps must be positive"
        self._params = [p for p in parameters]
        assert len(self._params) > 0, "no parameters provided"
        self.lr = float(lr)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

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
        for p in self._params:
            grad = getattr(p, "grad", None)
            if grad is None:
                continue
            g = grad
            if self.weight_decay != 0.0:
                g = g.add(self.weight_decay, p.data)
            norm = torch.linalg.vector_norm(g).clamp_min(self.eps)
            p.data.add_(g, alpha=-self.lr / norm)


def _factory(parameters: Iterable[object], lr: float = 1e-3, eps: float = 1e-8, weight_decay: float = 0.0):
    return NormalizedDynamicsOptimizer(parameters, lr=lr, eps=eps, weight_decay=weight_decay)


OptimizerRegistry.register("normalized_dynamics", _factory)
OptimizerRegistry.register("normalizeddynamics", _factory)


