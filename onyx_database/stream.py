"""Streaming helpers for query changefeeds (std lib only)."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, IO

from .http import parse_json_allow_nan


StreamHandler = Callable[[Any], None]


def _dispatch(action: Optional[str], entity: Any, handlers: Dict[str, StreamHandler]) -> None:
    if not action:
        return
    upper = action.upper()
    if upper in {"CREATE", "CREATED", "ADD", "ADDED", "INSERT", "INSERTED"}:
        if handlers.get("on_item_added"):
            handlers["on_item_added"](entity)
        if handlers.get("on_item"):
            handlers["on_item"](entity, "CREATE")
    elif upper in {"UPDATE", "UPDATED"}:
        if handlers.get("on_item_updated"):
            handlers["on_item_updated"](entity)
        if handlers.get("on_item"):
            handlers["on_item"](entity, "UPDATE")
    elif upper in {"DELETE", "DELETED", "REMOVE", "REMOVED"}:
        if handlers.get("on_item_deleted"):
            handlers["on_item_deleted"](entity)
        if handlers.get("on_item"):
            handlers["on_item"](entity, "DELETE")


def open_json_lines_stream(
    opener: Callable[[], IO[bytes]],
    *,
    handlers: Optional[Dict[str, StreamHandler]] = None,
    max_retries: int = 4,
) -> Dict[str, Callable[[], None]]:
    """Open a streaming connection and dispatch JSON-lines events."""
    cancel_event = threading.Event()
    handlers = handlers or {}
    current_stream: Dict[str, IO[bytes]] = {}

    def process_line(line: str) -> None:
        txt = line.strip()
        if not txt or txt.startswith(":"):
            return
        if txt.startswith("data:"):
            txt = txt[5:].strip()
        try:
            obj = parse_json_allow_nan(txt)
        except Exception:
            return
        if isinstance(obj, str):
            return
        action = None
        if isinstance(obj, dict):
            action = (
                obj.get("action")
                or obj.get("event")
                or obj.get("type")
                or obj.get("eventType")
                or obj.get("changeType")
            )
            entity = obj.get("entity")
        else:
            entity = None
        _dispatch(action, entity, handlers)  # type: ignore[arg-type]

    def worker() -> None:
        retries = 0
        while not cancel_event.is_set() and retries <= max_retries:
            try:
                stream = opener()
                current_stream["stream"] = stream
                retries = 0
                while not cancel_event.is_set():
                    line_bytes = stream.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace")
                    process_line(line)
                if cancel_event.is_set():
                    break
            except Exception:
                retries += 1
                if retries > max_retries or cancel_event.is_set():
                    break
                time.sleep(min(1 * (2 ** (retries - 1)), 30))
                continue
        # close on exit
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def cancel() -> None:
        cancel_event.set()
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:
            pass

    return {"cancel": cancel}
