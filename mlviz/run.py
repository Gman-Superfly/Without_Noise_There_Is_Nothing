from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from .entities import TrainingRunEntity, NoiseScheduleEntity
from .stream import EventStreamer
from .train import train_and_stream


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WNIN training + streaming")
    p.add_argument("--ws", type=str, default="ws://127.0.0.1:8765", help="WebSocket URI")
    p.add_argument("--optimizer", type=str, default="adamw", help="Optimizer name")
    p.add_argument("--steps", type=int, default=1000, help="Training steps")
    p.add_argument("--noise-mode", type=str, default="none", help="Noise mode")
    p.add_argument("--noise-mag", type=float, default=0.0, help="Noise magnitude")
    p.add_argument("--noise-schedule", type=str, default="constant", help="Schedule")
    p.add_argument("--noise-period", type=int, default=0, help="Schedule period if applicable")
    return p.parse_args(argv)


async def _main_async(ns: argparse.Namespace) -> int:
    run = TrainingRunEntity(description="WNIN demo run")
    noise = NoiseScheduleEntity(
        mode=ns.noise_mode,
        magnitude=max(0.0, ns.noise_mag),
        schedule=ns.noise_schedule,
        period=ns.noise_period or None,
    )

    streamer = EventStreamer(ns.ws)

    async def events():
        async for ev in train_and_stream(run, noise, optimizer_name=ns.optimizer, steps=ns.steps):
            yield ev

    await streamer.send_events(events())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ns = parse_args(argv)
    try:
        return asyncio.run(_main_async(ns))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


