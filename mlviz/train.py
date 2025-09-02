from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Iterable, Tuple, TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover - optional torch environment
    torch = None

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor  # type: ignore
else:
    class TorchTensor:  # minimal runtime placeholder
        pass


@runtime_checkable
class DatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int): ...

from .entities import TrainingRunEntity, NoiseScheduleEntity, StepMetricsEvent
from .registry import OptimizerRegistry
from .optimizer_loader import load_plugins_from


def _assert_torch():
    assert torch is not None, "PyTorch is required to run training"


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: 'TorchTensor') -> 'TorchTensor':
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def _make_synthetic_dataset(n: int = 4096, d: int = 32) -> Tuple[DatasetLike, DatasetLike]:
    _assert_torch()
    rng = np.random.default_rng(42)
    w_true = rng.normal(size=(d,))
    X = rng.normal(size=(n, d)).astype(np.float32)
    y_score = X @ w_true + 0.25 * rng.normal(size=(n,))
    y = (y_score > 0).astype(np.int64)
    split = int(0.8 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    assert len(tr) > 0 and len(va) > 0
    return tr, va


def _apply_input_noise(batch_x: 'TorchTensor', schedule: NoiseScheduleEntity, step: int) -> 'TorchTensor':
    if schedule.mode != "input_jitter" or schedule.magnitude <= 0:
        return batch_x
    sigma = _scheduled_value(schedule, step)
    return batch_x + sigma * torch.randn_like(batch_x)


def _scheduled_value(schedule: NoiseScheduleEntity, step: int) -> float:
    mag = schedule.magnitude
    if schedule.schedule == "constant":
        return mag
    if schedule.schedule == "cosine":
        return 0.5 * mag * (1 + math.cos(math.pi * step / max(1, schedule.period or 1000)))
    if schedule.schedule == "cyclical":
        T = float(schedule.period or 1000)
        phase = (step % int(T)) / T
        return mag * (0.5 * (1 - math.cos(2 * math.pi * phase)))
    if schedule.schedule == "inverse_anneal":
        return mag / (1.0 + 0.001 * step)
    return mag


async def train_and_stream(
    run: TrainingRunEntity,
    noise: NoiseScheduleEntity,
    optimizer_name: str = "adamw",
    steps: int = 2000,
    batch_size: int = 128,
) -> AsyncIterator[StepMetricsEvent]:
    _assert_torch()
    load_plugins_from("optimizers")

    model = TinyMLP()
    tr, va = _make_synthetic_dataset()
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=512)

    opt = OptimizerRegistry.create(optimizer_name, model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    step = 0
    while step < steps:
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            xb_noisy = _apply_input_noise(xb, noise, step)

            logits = model(xb_noisy)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.sqrt(sum((p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None))).item()
            opt.step()

            if step % 20 == 0:
                val_loss = _evaluate(model, val_loader, device)
                event = StepMetricsEvent(
                    run_id=run.run_id,
                    step=step,
                    train_loss=float(loss.item()),
                    val_loss=float(val_loss),
                    temperature=_scheduled_value(noise, step) if noise.mode != "none" else 0.0,
                    grad_norm=float(grad_norm),
                    noise_mode=noise.mode,
                    schedule_phase=noise.schedule,
                )
                yield event
                await asyncio.sleep(0)  # cooperative

            step += 1
            if step >= steps:
                break


@torch.no_grad() if torch is not None else (lambda f: f)
def _evaluate(model, loader, device) -> float:
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.item())
        total += int(yb.shape[0])
    assert total > 0, "Empty validation set"
    return total_loss / total



