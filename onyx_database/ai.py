"""Helpers for Onyx AI endpoints (chat completions, models, approvals)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, IO, Iterator, Optional

from .http import parse_json_allow_nan


def iter_sse(stream: IO[bytes]) -> Iterator[Any]:
    """Yield parsed JSON objects from an SSE text/event-stream response."""
    try:
        for raw_line in iter(lambda: stream.readline(), b""):
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield parse_json_allow_nan(data)
            except Exception:
                continue
    finally:
        try:
            stream.close()
        except Exception:
            pass


async def iter_sse_async(stream: IO[bytes], loop: Optional[asyncio.AbstractEventLoop] = None) -> AsyncIterator[Any]:
    """Async variant of iter_sse using a background executor to read lines."""
    loop = loop or asyncio.get_running_loop()

    async def _readline() -> bytes:
        return await loop.run_in_executor(None, stream.readline)

    try:
        while True:
            raw_line = await _readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                yield parse_json_allow_nan(data)
            except Exception:
                continue
    finally:
        try:
            stream.close()
        except Exception:
            pass
