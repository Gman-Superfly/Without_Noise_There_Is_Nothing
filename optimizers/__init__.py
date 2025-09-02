"""Optimizer plugins package.

Drop-in files that register optimizers with `mlviz.registry.OptimizerRegistry` on import.
Each module should call `OptimizerRegistry.register(name, factory)` at import time.
"""


