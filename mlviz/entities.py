from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class TrainingRunEntity(BaseModel):
    """Represents a single training run configuration and identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID = Field(default_factory=uuid4, description="Unique run identifier")
    description: str = Field(default="", description="Human-readable run description")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NoiseScheduleEntity(BaseModel):
    """Noise schedule configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal[
        "none",
        "gradient_gaussian",
        "sgld",
        "parameter_jitter",
        "input_jitter",
        "label_corrupt",
    ] = Field(default="none")
    magnitude: float = Field(ge=0.0, default=0.0)
    schedule: Literal["constant", "cosine", "cyclical", "inverse_anneal"] = Field(
        default="constant"
    )
    period: Optional[int] = Field(default=None, ge=1, description="For cyclical schedules")


class StepMetricsEvent(BaseModel):
    """Event carrying step-level metrics for visualization."""

    model_config = ConfigDict(extra="forbid")

    event: Literal["step_metrics"] = Field(default="step_metrics")
    run_id: UUID
    step: int = Field(ge=0)
    train_loss: float
    val_loss: Optional[float] = None
    perplexity: Optional[float] = None
    temperature: Optional[float] = None
    grad_norm: Optional[float] = None
    hessian_trace_est: Optional[float] = None
    sharpness_proxy: Optional[float] = None
    noise_mode: Optional[str] = None
    schedule_phase: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))



