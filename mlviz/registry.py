from __future__ import annotations

from typing import Callable, Dict, Optional


class OptimizerRegistry:
    """Simple name → factory registry for optimizers.

    Factories should be callables that accept a `parameters` iterable and
    keyword arguments, returning an optimizer-like object.
    """

    _registry: Dict[str, Callable[..., object]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., object]) -> None:
        assert isinstance(name, str) and name, "Optimizer name must be non-empty string"
        assert callable(factory), "Factory must be callable"
        cls._registry[name.lower()] = factory

    @classmethod
    def get(cls, name: str) -> Callable[..., object]:
        key = name.lower()
        assert key in cls._registry, f"Optimizer '{name}' not found"
        return cls._registry[key]

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> object:
        factory = cls.get(name)
        optimizer = factory(*args, **kwargs)
        assert optimizer is not None, "Optimizer factory returned None"
        return optimizer

    @classmethod
    def maybe_get(cls, name: str) -> Optional[Callable[..., object]]:
        return cls._registry.get(name.lower())

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry.keys())


