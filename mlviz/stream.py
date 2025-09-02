from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from pydantic import BaseModel
import websockets


class EventStreamer:
    """Streams pydantic models as JSON over WebSockets."""

    def __init__(self, uri: str):
        assert isinstance(uri, str) and uri.startswith("ws"), "Invalid WebSocket URI"
        self._uri = uri

    async def send_events(self, events: AsyncIterator[BaseModel]) -> None:
        async with websockets.connect(self._uri, max_size=None) as ws:
            async for event in events:
                payload = event.model_dump(mode="json")  # Pydantic v2 JSON dict
                await ws.send(json.dumps(payload))



