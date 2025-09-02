"""Machine learning noise visualization core.

Provides typed entities, event schemas, optimizer registry, and a
minimal training loop that streams metrics over WebSockets.
"""

from .entities import (
    TrainingRunEntity,
    NoiseScheduleEntity,
    StepMetricsEvent,
)

from .registry import OptimizerRegistry

__all__ = [
    "TrainingRunEntity",
    "NoiseScheduleEntity",
    "StepMetricsEvent",
    "OptimizerRegistry",
]


