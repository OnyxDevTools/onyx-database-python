"""MessagePack v1 wire profile for Onyx entity routes."""

from __future__ import annotations

import datetime
import math
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Any, BinaryIO

import msgpack

MESSAGE_PACK_MEDIA_TYPE = "application/vnd.msgpack"
MESSAGE_PACK_ACCEPT = f"{MESSAGE_PACK_MEDIA_TYPE}, application/json;q=0.9"

MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_DEPTH = 128
MAX_CONTAINER_ITEMS = 1_000_000
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_NODES = 2_000_000
MAX_FENCED_SAVE_ENTITIES = 500
MIN_SIGNED_INT64 = -(2**63)
MAX_SIGNED_INT64 = 2**63 - 1

_CYCLE_SENTINEL = {"cyclicReference": "detected"}


class EntityWireError(ValueError):
    """Raised when a value falls outside the portable entity wire profile."""


class _TraversalState:
    def __init__(self) -> None:
        self.nodes = 0
        self.active: set[int] = set()

    def visit(self, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise EntityWireError(f"entity value exceeds maximum depth of {MAX_DEPTH}")
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise EntityWireError(f"entity value exceeds maximum node count of {MAX_NODES}")


def _validate_string(value: str) -> str:
    # Four bytes is the maximum UTF-8 width of a Unicode scalar. Most entity
    # strings therefore need no temporary encoded copy just to enforce size.
    if len(value) <= MAX_STRING_BYTES // 4:
        return value
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise EntityWireError("entity strings must contain valid UTF-8") from exc
    if size > MAX_STRING_BYTES:
        raise EntityWireError(f"entity string exceeds maximum size of {MAX_STRING_BYTES} bytes")
    return value


def _datetime_to_wire(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    value = value.astimezone(datetime.UTC)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _normalize(value: Any, state: _TraversalState, depth: int) -> Any:
    if isinstance(value, Enum):
        return _normalize(value.value, state, depth)
    state.visit(depth)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < MIN_SIGNED_INT64 or value > MAX_SIGNED_INT64:
            raise EntityWireError("entity integers must fit in a signed 64-bit value")
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EntityWireError("entity floats must be finite")
        return float(value)
    if isinstance(value, str):
        return str(_validate_string(value))
    if isinstance(value, datetime.datetime):
        return _validate_string(_datetime_to_wire(value))
    if isinstance(value, datetime.date):
        return _validate_string(value.isoformat())
    if isinstance(value, (bytes, bytearray, memoryview, msgpack.ExtType, msgpack.Timestamp)):
        raise EntityWireError("binary and extension values are not supported by the entity wire profile")

    object_id = id(value)
    if object_id in state.active:
        return dict(_CYCLE_SENTINEL)

    if isinstance(value, Mapping):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity map exceeds maximum size of {MAX_CONTAINER_ITEMS} entries")
        state.active.add(object_id)
        try:
            normalized = {}
            for key, child in value.items():
                # Map keys are wire values too. Count them at the same child
                # depth as their associated values, matching the server and
                # the other SDK codecs.
                state.visit(depth + 1)
                if not isinstance(key, str):
                    raise EntityWireError("entity map keys must be strings")
                normalized[str(_validate_string(key))] = _normalize(child, state, depth + 1)
            return normalized
        finally:
            state.active.remove(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity array exceeds maximum size of {MAX_CONTAINER_ITEMS} items")
        state.active.add(object_id)
        try:
            return [_normalize(child, state, depth + 1) for child in value]
        finally:
            state.active.remove(object_id)

    if hasattr(value, "__dict__"):
        state.active.add(object_id)
        try:
            attributes = vars(value)
            if len(attributes) > MAX_CONTAINER_ITEMS:
                raise EntityWireError(f"entity object exceeds maximum size of {MAX_CONTAINER_ITEMS} fields")
            normalized = {}
            for key, child in attributes.items():
                state.visit(depth + 1)
                if not isinstance(key, str):
                    raise EntityWireError("entity object field names must be strings")
                normalized[str(_validate_string(key))] = _normalize(child, state, depth + 1)
            return normalized
        finally:
            state.active.remove(object_id)

    raise EntityWireError(f"unsupported entity value type: {type(value).__name__}")


def normalize_entity_value(value: Any) -> Any:
    """Convert an arbitrary entity graph to the portable recursive wire profile."""

    return _normalize(value, _TraversalState(), 0)


def require_single_entity(value: Any) -> Any:
    """Reject batch/scalar inputs before issuing an atomic create request."""

    if isinstance(value, Mapping) or hasattr(value, "__dict__"):
        return value
    raise EntityWireError("create requires exactly one entity object")


def require_fenced_entities(value: Any) -> list[Any]:
    """Normalize one entity or a bounded explicit batch for fenced persistence."""

    if isinstance(value, Mapping) or hasattr(value, "__dict__"):
        entities = [value]
    elif isinstance(value, (list, tuple)):
        entities = list(value)
    else:
        raise EntityWireError("fenced_save requires one entity object or an explicit list/tuple")
    if not 1 <= len(entities) <= MAX_FENCED_SAVE_ENTITIES:
        raise EntityWireError(
            f"fenced_save requires between 1 and {MAX_FENCED_SAVE_ENTITIES} entities"
        )
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping) and not hasattr(entity, "__dict__"):
            raise EntityWireError(f"fenced_save entity {index} must be an object")
    return entities


def require_mutation_guard(value: Any) -> Any:
    """Validate the stable fenced guard envelope without performing a server read."""

    if not isinstance(value, Mapping):
        raise EntityWireError("guard must be an object")
    if not isinstance(value.get("table"), str) or not value["table"].strip():
        raise EntityWireError("guard.table must be a non-empty string")
    if "id" not in value or value["id"] is None:
        raise EntityWireError("guard.id is required")
    expected = value.get("expected")
    if not isinstance(expected, Mapping) or not expected:
        raise EntityWireError("guard.expected must be a non-empty object")
    unsupported = set(value) - {"table", "id", "partition", "expected"}
    if unsupported:
        raise EntityWireError("guard contains unsupported fields")
    return value


def require_query_condition(value: Any) -> Any:
    """Require the serialized QueryCondition shape used by fenced mutation filters."""

    if not isinstance(value, Mapping):
        raise EntityWireError("filters must be a QueryCondition object")
    condition_type = value.get("conditionType")
    if condition_type not in {"SingleCondition", "CompoundCondition"}:
        raise EntityWireError(
            "filters must be a serialized QueryCondition with conditionType "
            "SingleCondition or CompoundCondition"
        )
    return value


def require_fenced_updates(value: Any) -> Any:
    """Require a non-empty update map with concrete field names."""

    if not isinstance(value, Mapping) or not value:
        raise EntityWireError("updates must be a non-empty object")
    if any(not isinstance(field, str) or not field.strip() for field in value):
        raise EntityWireError("updates field names must be non-empty strings")
    return value


def require_concrete_partition(value: Any) -> str:
    """Reject missing, blank, and cross-partition fenced mutation targets."""

    if not isinstance(value, str) or not value.strip() or value.strip().upper() == "ALL":
        raise EntityWireError("partition must name one concrete partition")
    return value


def _is_native_entity_graph(value: Any, state: list[Any] | None = None, depth: int = 0) -> bool:
    """Validate an already-normalized dict/list graph without copying it."""

    if state is None:
        state = [0, set()]
    if depth > MAX_DEPTH:
        raise EntityWireError(f"entity value exceeds maximum depth of {MAX_DEPTH}")
    state[0] += 1
    if state[0] > MAX_NODES:
        raise EntityWireError(f"entity value exceeds maximum node count of {MAX_NODES}")
    value_type = type(value)

    if value is None or value_type is bool:
        return True
    if value_type is int:
        if value < MIN_SIGNED_INT64 or value > MAX_SIGNED_INT64:
            raise EntityWireError("entity integers must fit in a signed 64-bit value")
        return True
    if value_type is float:
        if not math.isfinite(value):
            raise EntityWireError("entity floats must be finite")
        return True
    if value_type is str:
        if len(value) > MAX_STRING_BYTES // 4:
            _validate_string(value)
        return True
    if value_type in {bytes, bytearray, memoryview}:
        raise EntityWireError("binary values are not supported by the entity wire profile")

    if value_type is list:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity array exceeds maximum size of {MAX_CONTAINER_ITEMS} items")
        object_id = id(value)
        active = state[1]
        if object_id in active:
            return False
        active.add(object_id)
        try:
            for child in value:
                if not _is_native_entity_graph(child, state, depth + 1):
                    return False
            return True
        finally:
            active.remove(object_id)

    if value_type is dict:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity map exceeds maximum size of {MAX_CONTAINER_ITEMS} entries")
        object_id = id(value)
        active = state[1]
        if object_id in active:
            return False
        active.add(object_id)
        try:
            for key, child in value.items():
                if depth + 1 > MAX_DEPTH:
                    raise EntityWireError(f"entity value exceeds maximum depth of {MAX_DEPTH}")
                state[0] += 1
                if state[0] > MAX_NODES:
                    raise EntityWireError(f"entity value exceeds maximum node count of {MAX_NODES}")
                if type(key) is not str:
                    raise EntityWireError("entity map keys must be strings")
                if len(key) > MAX_STRING_BYTES // 4:
                    _validate_string(key)
                if not _is_native_entity_graph(child, state, depth + 1):
                    return False
            return True
        finally:
            active.remove(object_id)

    # Dates, enums, tuples/sets, mapping subclasses, and model objects need
    # the generic normalizer. It also produces the JSON cycle sentinel.
    return False


def pack_entity(value: Any) -> bytes:
    """Encode one entity value as canonical MessagePack without extensions."""

    normalized = value if _is_native_entity_graph(value) else normalize_entity_value(value)
    try:
        packed = msgpack.packb(normalized, use_bin_type=True, strict_types=True)
    except Exception as exc:
        raise EntityWireError(f"invalid entity value: {exc}") from exc
    if len(packed) > MAX_BODY_BYTES:
        raise EntityWireError(f"entity body exceeds maximum size of {MAX_BODY_BYTES} bytes")
    return packed


def _reject_extension(code: int, data: bytes) -> Any:
    raise EntityWireError(f"MessagePack extension type {code} is not supported")


def _strict_object_pairs(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    """Build a map without allowing duplicate wire keys to disappear."""

    # Let CPython build the dict in C, then compare cardinality before the map
    # can reach recursive validation. This keeps strict decoding substantially
    # cheaper than a second Python-level insertion loop.
    result = dict(pairs)
    if len(result) != len(pairs):
        raise EntityWireError("entity map contains duplicate keys")
    return result


def _unpack_options() -> dict[str, Any]:
    return {
        "raw": False,
        "strict_map_key": True,
        "object_pairs_hook": _strict_object_pairs,
        "ext_hook": _reject_extension,
        "max_str_len": MAX_STRING_BYTES,
        "max_bin_len": 0,
        "max_array_len": MAX_CONTAINER_ITEMS,
        "max_map_len": MAX_CONTAINER_ITEMS,
        "max_ext_len": 0,
    }


def _validate_decoded(
    value: Any,
    state: list[int] | None = None,
    depth: int = 0,
) -> Any:
    # Validate in place. Rebuilding a graph already allocated by the C decoder
    # would add substantial latency and memory without changing its values.
    # msgpack's max_* options bound individual allocations, but it has no
    # total-node, depth, signed-int64, or finite-float decoder options.
    if state is None:
        state = [0]
    if depth > MAX_DEPTH:
        raise EntityWireError(f"entity value exceeds maximum depth of {MAX_DEPTH}")
    state[0] += 1
    if state[0] > MAX_NODES:
        raise EntityWireError(f"entity value exceeds maximum node count of {MAX_NODES}")

    value_type = type(value)
    if value is None or value_type is bool or value_type is str:
        return value
    if value_type is int:
        if value < MIN_SIGNED_INT64 or value > MAX_SIGNED_INT64:
            raise EntityWireError("entity integers must fit in a signed 64-bit value")
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise EntityWireError("entity floats must be finite")
        return value
    if value_type is list:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity array exceeds maximum size of {MAX_CONTAINER_ITEMS} items")
        for child in value:
            _validate_decoded(child, state, depth + 1)
        return value
    if value_type is dict:
        if len(value) > MAX_CONTAINER_ITEMS:
            raise EntityWireError(f"entity map exceeds maximum size of {MAX_CONTAINER_ITEMS} entries")
        for key, child in value.items():
            if depth + 1 > MAX_DEPTH:
                raise EntityWireError(f"entity value exceeds maximum depth of {MAX_DEPTH}")
            state[0] += 1
            if state[0] > MAX_NODES:
                raise EntityWireError(f"entity value exceeds maximum node count of {MAX_NODES}")
            if type(key) is not str:
                raise EntityWireError("entity map keys must be strings")
            _validate_decoded(child, state, depth + 1)
        return value
    raise EntityWireError("binary and extension values are not supported by the entity wire profile")


def unpack_entity(payload: bytes | bytearray | memoryview) -> Any:
    """Decode exactly one MessagePack entity value and enforce profile limits."""

    raw_payload = bytes(payload)
    if len(raw_payload) > MAX_BODY_BYTES:
        raise EntityWireError(f"entity body exceeds maximum size of {MAX_BODY_BYTES} bytes")
    try:
        value = msgpack.unpackb(raw_payload, **_unpack_options())
    except EntityWireError:
        raise
    except Exception as exc:
        raise EntityWireError(f"invalid entity MessagePack payload: {exc}") from exc
    return _validate_decoded(value)


def iter_unpack_entities(stream: BinaryIO) -> Iterator[Any]:
    """Decode concatenated, self-delimiting MessagePack values from a stream."""

    options = _unpack_options()
    try:
        unpacker = msgpack.Unpacker(stream, max_buffer_size=MAX_BODY_BYTES, **options)
        for value in unpacker:
            yield _validate_decoded(value)
    except EntityWireError:
        raise
    except Exception as exc:
        raise EntityWireError(f"invalid entity MessagePack stream: {exc}") from exc
