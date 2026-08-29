"""Streaming helpers for query changefeeds (std lib only)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import IO, Any

from .entity_wire import MESSAGE_PACK_MEDIA_TYPE, iter_unpack_entities
from .http import parse_json_allow_nan

StreamHandler = Callable[[Any], None]


def _dispatch(action: str | None, entity: Any, handlers: dict[str, StreamHandler]) -> None:
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


def _dispatch_value(obj: Any, handlers: dict[str, StreamHandler]) -> None:
    if obj is None or isinstance(obj, str):
        return
    action = None
    entity = None
    if isinstance(obj, dict):
        action = (
            obj.get("action")
            or obj.get("event")
            or obj.get("type")
            or obj.get("eventType")
            or obj.get("changeType")
        )
        entity = obj.get("entity")
    _dispatch(action, entity, handlers)


def open_json_lines_stream(
    opener: Callable[[], IO[bytes]],
    *,
    handlers: dict[str, StreamHandler] | None = None,
    max_retries: int = 4,
) -> dict[str, Callable[[], None]]:
    """Open a streaming connection and dispatch JSON-lines events."""
    cancel_event = threading.Event()
    handlers = handlers or {}
    current_stream: dict[str, IO[bytes]] = {}

    def process_line(line: str) -> None:
        txt = line.strip()
        if not txt or txt.startswith(":"):
            return
        if txt.startswith("data:"):
            txt = txt[5:].strip()
        try:
            obj = parse_json_allow_nan(txt)
        except Exception:  # noqa: BLE001 - malformed events must not stop the stream
            return
        _dispatch_value(obj, handlers)

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
            except Exception:  # noqa: BLE001 - reconnect after transport/handler failures
                retries += 1
                if retries > max_retries or cancel_event.is_set():
                    break
                time.sleep(min(1 * (2 ** (retries - 1)), 30))
                continue
        # close on exit
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:  # noqa: BLE001, S110 - best-effort close during worker exit
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def cancel() -> None:
        cancel_event.set()
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:  # noqa: BLE001, S110 - cancellation close is best-effort
            pass

    return {"cancel": cancel}


def open_entity_stream(
    opener: Callable[[], IO[bytes]],
    *,
    handlers: dict[str, StreamHandler] | None = None,
    max_retries: int = 4,
) -> dict[str, Callable[[], None]]:
    """Dispatch MessagePack stream values, with JSON-lines response fallback."""

    cancel_event = threading.Event()
    handlers = handlers or {}
    current_stream: dict[str, IO[bytes]] = {}

    def process_json_line(line_bytes: bytes) -> None:
        text = line_bytes.decode("utf-8", errors="replace").strip()
        if not text or text.startswith(":"):
            return
        if text.startswith("data:"):
            text = text[5:].strip()
        try:
            _dispatch_value(parse_json_allow_nan(text), handlers)
        except Exception:  # noqa: BLE001 - malformed events must not stop the stream
            return

    def worker() -> None:
        retries = 0
        while not cancel_event.is_set() and retries <= max_retries:
            try:
                stream = opener()
                current_stream["stream"] = stream
                retries = 0
                response_headers = getattr(stream, "headers", {})
                content_type = ""
                if response_headers is not None:
                    try:
                        content_type = response_headers.get("Content-Type", "")
                    except Exception:  # noqa: BLE001 - unusual header adapters fall back to JSON
                        content_type = ""
                is_message_pack = (
                    content_type.lower().split(";", 1)[0].strip() == MESSAGE_PACK_MEDIA_TYPE
                )
                if is_message_pack:
                    for value in iter_unpack_entities(stream):
                        if cancel_event.is_set():
                            break
                        # A nil value is an optional flush/connection sentinel.
                        if value is not None:
                            _dispatch_value(value, handlers)
                else:
                    while not cancel_event.is_set():
                        line_bytes = stream.readline()
                        if not line_bytes:
                            break
                        process_json_line(line_bytes)
                if cancel_event.is_set():
                    break
            except Exception:  # noqa: BLE001 - reconnect after transport/handler failures
                retries += 1
                if retries > max_retries or cancel_event.is_set():
                    break
                time.sleep(min(1 * (2 ** (retries - 1)), 30))
                continue
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:  # noqa: BLE001, S110 - best-effort close during worker exit
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def cancel() -> None:
        cancel_event.set()
        try:
            if current_stream.get("stream"):
                current_stream["stream"].close()
        except Exception:  # noqa: BLE001, S110 - cancellation close is best-effort
            pass

    return {"cancel": cancel}
