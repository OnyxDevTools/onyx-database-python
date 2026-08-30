"""Validated wire helpers for native bounded and semantic search."""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Final, cast

from ..types import (
    ApproximateIndexCandidateQuery,
    HnswSearchQuery,
    Int64WireInput,
    SearchMatch,
    SearchMode,
    SemanticVectorSignature,
    VectorSearchQuery,
)

HNSW_QUERY_FORMAT_VERSION: Final = 1
DEFAULT_HNSW_CANDIDATES = 1_000
DEFAULT_HNSW_EF_SEARCH = 1_000
MAX_HNSW_CANDIDATES = 5_000
MAX_HNSW_EF_SEARCH = 20_000
MAX_HNSW_VECTOR_DIMENSION = 16_384

DEFAULT_APPROXIMATE_INDEX_CANDIDATES = 1_000
MAX_APPROXIMATE_INDEX_CANDIDATES = 5_000
MAX_APPROXIMATE_INDEX_ROUTE_VALUES = 5_000
MAX_VECTOR_SEARCH_CANDIDATES = 5_000

_SIGNED_INT64_MIN = -(1 << 63)
_SIGNED_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MAX = (1 << 31) - 1
_SEMANTIC_BAND_COUNT = 4
_MISSING = object()
_DECIMAL_RE = re.compile(r"^-?\d+$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_SEARCH_MODES = {"lexical", "semantic", "hybrid"}
_SEARCH_MATCH_POLICIES = {"all", "any"}
_SEARCH_OPTION_FIELDS = {
    "mode",
    "match",
    "minScore",
    "min_score",
    "maxCandidates",
    "max_candidates",
}


def _mapping(input_value: Mapping[str, Any] | None, field: str) -> Mapping[str, Any]:
    if input_value is None:
        return {}
    if not isinstance(input_value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in input_value):
        raise TypeError(f"{field} field names must be strings")
    return input_value


def _value(
    values: Mapping[str, Any],
    *names: str,
    default: Any = _MISSING,
) -> Any:
    for name in names:
        if name in values:
            return values[name]
    if default is _MISSING:
        raise TypeError(f"{names[0]} is required")
    return default


def _aliased_search_option(
    values: Mapping[str, Any],
    wire_name: str,
    snake_name: str,
    *,
    default: Any = _MISSING,
) -> Any:
    present = [name for name in (wire_name, snake_name) if name in values]
    if len(present) > 1:
        raise TypeError(f"provide only one of {wire_name} and {snake_name}")
    if present:
        return values[present[0]]
    if default is _MISSING:
        raise TypeError(f"{snake_name} is required")
    return default


def high_level_search_query(
    text: Any,
    options: Mapping[str, Any] | None = None,
    *,
    mode: Any = _MISSING,
    match: Any = _MISSING,
    min_score: Any = _MISSING,
    max_candidates: Any = _MISSING,
) -> dict[str, Any]:
    """Validate and canonicalize a text-first lexical, semantic, or hybrid query."""

    if not isinstance(text, str):
        raise TypeError("search text must be a string")
    if not text.strip():
        raise ValueError("search text must not be blank")

    has_options_mapping = options is not None
    values = _mapping(options, "SearchOptions")
    unknown = sorted(set(values) - _SEARCH_OPTION_FIELDS)
    if unknown:
        raise TypeError(f"unsupported search option: {unknown[0]}")

    keyword_values = (mode, match, min_score, max_candidates)
    if has_options_mapping and any(
        value is not _MISSING for value in keyword_values
    ):
        raise TypeError("do not combine a search options mapping with search keywords")

    if has_options_mapping:
        mode = _value(values, "mode", default="hybrid")
        if mode is None:
            raise TypeError("search mode must be a string")
        match = _value(values, "match", default="any")
        min_score = _aliased_search_option(
            values, "minScore", "min_score", default=None
        )
        max_candidates = _aliased_search_option(
            values, "maxCandidates", "max_candidates", default=1_000
        )
    else:
        if mode is _MISSING or mode is None:
            mode = "hybrid"
        if match is _MISSING:
            match = "any"
        if min_score is _MISSING:
            min_score = None
        if max_candidates is _MISSING or max_candidates is None:
            max_candidates = 1_000

    if mode is _MISSING or mode is None:
        mode = "hybrid"
    if not isinstance(mode, str):
        raise TypeError("search mode must be a string")
    if mode not in _SEARCH_MODES:
        raise ValueError("search mode must be lexical, semantic, or hybrid")
    if not isinstance(match, str):
        raise TypeError("search match must be a string")
    if match not in _SEARCH_MATCH_POLICIES:
        raise ValueError("search match must be all or any")

    canonical_min_score = None
    if min_score is not None:
        canonical_min_score, _ = _finite_float32(min_score, "minScore")
        if canonical_min_score < 0 or canonical_min_score > 1:
            raise ValueError("minScore must be between 0 and 1")
    candidate_count = _require_integer(max_candidates, "maxCandidates")
    if candidate_count < 1 or candidate_count > MAX_VECTOR_SEARCH_CANDIDATES:
        raise ValueError(
            f"maxCandidates must be between 1 and {MAX_VECTOR_SEARCH_CANDIDATES}"
        )
    if mode == "hybrid" and candidate_count < 2:
        raise ValueError("maxCandidates must be at least 2 for hybrid search")

    return {
        "text": text,
        "mode": cast(SearchMode, mode),
        "match": cast(SearchMatch, match),
        "minScore": canonical_min_score,
        "maxCandidates": candidate_count,
    }


def _require_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _require_int32(value: Any, field: str) -> int:
    integer = _require_integer(value, field)
    if integer < -(1 << 31) or integer > _INT32_MAX:
        raise ValueError(f"{field} is outside the signed 32-bit range")
    return integer


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_float32(value: Any, field: str) -> tuple[float, float]:
    """Return the wire number and the finite value Kotlin sees after ``toFloat``."""

    wire_number = _finite_number(value, field)
    try:
        narrowed = struct.unpack("!f", struct.pack("!f", wire_number))[0]
    except OverflowError as error:
        raise ValueError(f"{field} must be finite after Float32 conversion") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{field} must be finite after Float32 conversion")
    return wire_number, narrowed


def _signed_int64(value: Int64WireInput, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a signed decimal 64-bit value")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not _DECIMAL_RE.fullmatch(text):
            raise TypeError(f"{field} must be a signed decimal 64-bit value")
        parsed = int(text, 10)
    else:
        raise TypeError(f"{field} must be a signed decimal 64-bit value")
    if parsed < _SIGNED_INT64_MIN or parsed > _SIGNED_INT64_MAX:
        raise ValueError(f"{field} exceeds the signed 64-bit range")
    return str(parsed)


def _semantic_word(value: Int64WireInput, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be signed decimal or unsigned hexadecimal")
    if isinstance(value, int):
        parsed = value
        if parsed < _SIGNED_INT64_MIN or parsed > _SIGNED_INT64_MAX:
            raise ValueError(f"{field} exceeds the signed 64-bit range")
        return parsed & _UINT64_MAX
    if not isinstance(value, str):
        raise TypeError(f"{field} must be signed decimal or unsigned hexadecimal")

    text = value.strip()
    if text.lower().startswith("0x"):
        digits = text[2:]
        if not digits or not _HEX_RE.fullmatch(digits):
            raise TypeError(f"{field} must be signed decimal or unsigned hexadecimal")
        parsed = int(digits, 16)
        if parsed > _UINT64_MAX:
            raise ValueError(f"{field} exceeds 64 bits")
        return parsed
    if any(character in "abcdefABCDEF" for character in text):
        if not text or not _HEX_RE.fullmatch(text):
            raise TypeError(f"{field} must be signed decimal or unsigned hexadecimal")
        parsed = int(text, 16)
        if parsed > _UINT64_MAX:
            raise ValueError(f"{field} exceeds 64 bits")
        return parsed
    if not _DECIMAL_RE.fullmatch(text):
        raise TypeError(f"{field} must be signed decimal or unsigned hexadecimal")
    parsed = int(text, 10)
    if parsed < _SIGNED_INT64_MIN or parsed > _SIGNED_INT64_MAX:
        raise ValueError(f"{field} exceeds the signed 64-bit range")
    return parsed & _UINT64_MAX


def _wire_word(value: int) -> str:
    return f"0x{value & _UINT64_MAX:016x}"


def _semantic_bands(words: Sequence[int]) -> list[int]:
    bit_count = len(words) * 64
    band_bits = bit_count // _SEMANTIC_BAND_COUNT
    mask = (1 << band_bits) - 1
    packed = 0
    for index, word in enumerate(words):
        packed |= word << (index * 64)
    return [
        (packed >> (index * band_bits)) & mask for index in range(_SEMANTIC_BAND_COUNT)
    ]


def semantic_vector_signature(
    signature: Mapping[str, Any] | None = None,
    *,
    calibration_id: Any = _MISSING,
    bucket_id: Any = _MISSING,
    cells: Any = _MISSING,
    cell_counts: Any = _MISSING,
    fingerprint: Any = _MISSING,
    bands: Any = _MISSING,
    boundary_confidence: Any = _MISSING,
) -> SemanticVectorSignature:
    """Validate and canonicalize one lossless semantic routing signature.

    A mapping may use Cloud's camelCase fields or Python snake_case aliases. Keyword
    arguments use snake_case. Fingerprint bands are computed when omitted.
    """

    values = _mapping(signature, "SemanticVectorSignature")
    calibration_id = (
        _value(values, "calibrationId", "calibration_id")
        if calibration_id is _MISSING
        else calibration_id
    )
    bucket_id = (
        _value(values, "bucketId", "bucket_id") if bucket_id is _MISSING else bucket_id
    )
    cells = _value(values, "cells") if cells is _MISSING else cells
    cell_counts = (
        _value(values, "cellCounts", "cell_counts")
        if cell_counts is _MISSING
        else cell_counts
    )
    fingerprint = (
        _value(values, "fingerprint", "fingerprintWords", "fingerprint_words")
        if fingerprint is _MISSING
        else fingerprint
    )
    bands = (
        _value(values, "bands", "semanticBands", "semantic_bands", default=None)
        if bands is _MISSING
        else bands
    )
    boundary_confidence = (
        _value(
            values,
            "boundaryConfidence",
            "boundary_confidence",
            default=0.0,
        )
        if boundary_confidence is _MISSING
        else boundary_confidence
    )

    calibration = _signed_int64(calibration_id, "calibrationId")
    if calibration == "0":
        raise ValueError("calibrationId must be non-zero")
    bucket = _require_int32(bucket_id, "bucketId")
    if bucket < 0:
        raise ValueError("bucketId must be non-negative")

    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes, bytearray)):
        raise TypeError("cells must be a sequence")
    if not isinstance(cell_counts, Sequence) or isinstance(
        cell_counts, (str, bytes, bytearray)
    ):
        raise TypeError("cellCounts must be a sequence")
    canonical_cells = [
        _require_int32(cell, f"cells[{index}]") for index, cell in enumerate(cells)
    ]
    canonical_counts = [
        _require_int32(count, f"cellCounts[{index}]")
        for index, count in enumerate(cell_counts)
    ]
    if not canonical_cells:
        raise ValueError("at least one product cell is required")
    if len(canonical_cells) != len(canonical_counts):
        raise ValueError("cellCounts must contain one cardinality per product cell")

    packed_bucket = 0
    bucket_space = 1
    for axis, (cell, count) in enumerate(zip(canonical_cells, canonical_counts)):
        if count < 2:
            raise ValueError(f"cellCounts[{axis}] must be at least 2")
        if cell < 0 or cell >= count:
            raise ValueError(f"cells[{axis}] is outside its cell count")
        bucket_space *= count
        if bucket_space > _INT32_MAX:
            raise ValueError(
                "product-cell space exceeds the supported Int bucket domain"
            )
        packed_bucket = packed_bucket * count + cell
    if bucket != packed_bucket:
        raise ValueError("bucketId does not match the mixed-radix product cells")

    if not isinstance(fingerprint, Sequence) or isinstance(
        fingerprint, (str, bytes, bytearray)
    ):
        raise TypeError("fingerprint must be a sequence")
    if len(fingerprint) < 1 or len(fingerprint) > 4:
        raise ValueError("fingerprint must contain between 1 and 4 64-bit words")
    fingerprint_words = [
        _semantic_word(word, f"fingerprint[{index}]")
        for index, word in enumerate(fingerprint)
    ]
    expected_bands = _semantic_bands(fingerprint_words)
    if bands is not None:
        if not isinstance(bands, Sequence) or isinstance(
            bands, (str, bytes, bytearray)
        ):
            raise TypeError("bands must be a sequence")
        if len(bands) != _SEMANTIC_BAND_COUNT:
            raise ValueError("bands must contain exactly four values")
        supplied_bands = [
            _semantic_word(band, f"bands[{index}]") for index, band in enumerate(bands)
        ]
        if supplied_bands != expected_bands:
            raise ValueError(
                "bands do not represent four equal portions of the fingerprint"
            )

    confidence, narrowed_confidence = _finite_float32(
        boundary_confidence, "boundaryConfidence"
    )
    if narrowed_confidence < 0 or narrowed_confidence > 1:
        raise ValueError("boundaryConfidence must be between 0 and 1")
    return {
        "calibrationId": calibration,
        "bucketId": bucket,
        "cells": canonical_cells,
        "cellCounts": canonical_counts,
        "fingerprint": [_wire_word(word) for word in fingerprint_words],
        "bands": [_wire_word(band) for band in expected_bands],
        "boundaryConfidence": confidence,
    }


def _looks_like_semantic_signature(values: Mapping[str, Any]) -> bool:
    return (
        ("calibrationId" in values or "calibration_id" in values)
        and ("bucketId" in values or "bucket_id" in values)
        and (
            "fingerprint" in values
            or "fingerprintWords" in values
            or "fingerprint_words" in values
        )
    )


def vector_search_query(
    query: Mapping[str, Any] | None = None,
    *,
    text: Any = _MISSING,
    semantic: Any = _MISSING,
    min_score: Any = _MISSING,
    nearby_bucket_radius: Any = _MISSING,
    max_candidates: Any = _MISSING,
    require_all_terms: Any = _MISSING,
) -> VectorSearchQuery:
    """Validate and canonicalize a native lexical, semantic, or hybrid query."""

    values = _mapping(query, "VectorSearchQuery")
    if _looks_like_semantic_signature(values) and "semantic" not in values:
        values = {"semantic": values}
    text = (
        _value(values, "text", "queryText", "query_text", default=None)
        if text is _MISSING
        else text
    )
    semantic = (
        _value(
            values, "semantic", "semanticSignature", "semantic_signature", default=None
        )
        if semantic is _MISSING
        else semantic
    )
    min_score = (
        _value(values, "minScore", "min_score", default=None)
        if min_score is _MISSING
        else min_score
    )
    nearby_bucket_radius = (
        _value(values, "nearbyBucketRadius", "nearby_bucket_radius", default=1)
        if nearby_bucket_radius is _MISSING
        else nearby_bucket_radius
    )
    max_candidates = (
        _value(values, "maxCandidates", "max_candidates", default=1_000)
        if max_candidates is _MISSING
        else max_candidates
    )
    require_all_terms = (
        _value(values, "requireAllTerms", "require_all_terms", default=True)
        if require_all_terms is _MISSING
        else require_all_terms
    )

    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise TypeError("text must be non-blank when supplied")
    canonical_semantic = None
    if semantic is not None:
        canonical_semantic = semantic_vector_signature(
            cast(Mapping[str, Any], semantic)
        )
    if text is None and canonical_semantic is None:
        raise ValueError(
            "VectorSearchQuery must contain text and/or a semantic signature"
        )

    canonical_min_score = None
    if min_score is not None:
        canonical_min_score, _ = _finite_float32(min_score, "minScore")
    radius = _require_int32(nearby_bucket_radius, "nearbyBucketRadius")
    if radius < 0:
        raise ValueError("nearbyBucketRadius must be non-negative")
    candidate_count = _require_integer(max_candidates, "maxCandidates")
    if candidate_count < 1 or candidate_count > MAX_VECTOR_SEARCH_CANDIDATES:
        raise ValueError(
            f"maxCandidates must be between 1 and {MAX_VECTOR_SEARCH_CANDIDATES}"
        )
    if not isinstance(require_all_terms, bool):
        raise TypeError("requireAllTerms must be boolean")
    return {
        "text": text,
        "semantic": canonical_semantic,
        "minScore": canonical_min_score,
        "nearbyBucketRadius": radius,
        "maxCandidates": candidate_count,
        "requireAllTerms": require_all_terms,
    }


def hnsw_search_query(
    query: Mapping[str, Any] | None = None,
    *,
    calibration_id: Any = _MISSING,
    vector: Any = _MISSING,
    max_candidates: Any = _MISSING,
    ef_search: Any = _MISSING,
    min_score: Any = _MISSING,
    format_version: Any = _MISSING,
) -> HnswSearchQuery:
    """Validate and canonicalize a bounded native-HNSW candidate request."""

    values = _mapping(query, "HnswSearchQuery")
    calibration_id = (
        _value(values, "calibrationId", "calibration_id")
        if calibration_id is _MISSING
        else calibration_id
    )
    vector = _value(values, "vector") if vector is _MISSING else vector
    max_candidates = (
        _value(
            values, "maxCandidates", "max_candidates", default=DEFAULT_HNSW_CANDIDATES
        )
        if max_candidates is _MISSING
        else max_candidates
    )
    format_version = (
        _value(
            values, "formatVersion", "format_version", default=HNSW_QUERY_FORMAT_VERSION
        )
        if format_version is _MISSING
        else format_version
    )
    min_score = (
        _value(values, "minScore", "min_score", default=None)
        if min_score is _MISSING
        else min_score
    )

    version = _require_integer(format_version, "HNSW formatVersion")
    if version != HNSW_QUERY_FORMAT_VERSION:
        raise ValueError(
            f"unsupported HNSW query formatVersion {version}; expected {HNSW_QUERY_FORMAT_VERSION}"
        )
    calibration = _signed_int64(calibration_id, "HNSW calibrationId")
    if calibration == "0":
        raise ValueError("HNSW calibrationId must be non-zero")
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise TypeError("HNSW vector must be a sequence")
    if len(vector) < 1 or len(vector) > MAX_HNSW_VECTOR_DIMENSION:
        raise ValueError(
            f"HNSW vector dimensions must be between 1 and {MAX_HNSW_VECTOR_DIMENSION}"
        )
    canonical_vector = []
    narrowed_vector = []
    for index, value in enumerate(vector):
        wire_value, narrowed_value = _finite_float32(value, f"HNSW vector[{index}]")
        canonical_vector.append(wire_value)
        narrowed_vector.append(narrowed_value)
    squared_magnitude = 0.0
    for value in narrowed_vector:
        squared_magnitude += value * value
        if not math.isfinite(squared_magnitude):
            break
    if not math.isfinite(squared_magnitude) or squared_magnitude <= 0:
        raise ValueError("HNSW vector must have a non-zero finite norm")

    candidate_count = _require_integer(max_candidates, "HNSW maxCandidates")
    if candidate_count < 1 or candidate_count > MAX_HNSW_CANDIDATES:
        raise ValueError(
            f"HNSW maxCandidates must be between 1 and {MAX_HNSW_CANDIDATES}"
        )
    if ef_search is _MISSING:
        ef_search = _value(
            values,
            "efSearch",
            "ef_search",
            default=max(DEFAULT_HNSW_EF_SEARCH, candidate_count),
        )
    ef_search_value = _require_integer(ef_search, "HNSW efSearch")
    if ef_search_value < candidate_count or ef_search_value > MAX_HNSW_EF_SEARCH:
        raise ValueError(
            f"HNSW efSearch must be between maxCandidates and {MAX_HNSW_EF_SEARCH}"
        )
    canonical_min_score = None
    if min_score is not None:
        canonical_min_score, narrowed_min_score = _finite_float32(
            min_score, "HNSW minScore"
        )
        if narrowed_min_score < -1 or narrowed_min_score > 1:
            raise ValueError("HNSW minScore must be between -1 and 1")
    return {
        "calibrationId": calibration,
        "vector": canonical_vector,
        "maxCandidates": candidate_count,
        "efSearch": ef_search_value,
        "minScore": canonical_min_score,
        "formatVersion": HNSW_QUERY_FORMAT_VERSION,
    }


def approximate_index_candidate_query(
    value_or_values: Any,
    max_candidates: int = DEFAULT_APPROXIMATE_INDEX_CANDIDATES,
) -> ApproximateIndexCandidateQuery:
    """Validate and canonicalize one bounded ordinary-index candidate route."""

    candidate_count = _require_integer(max_candidates, "maxCandidates")
    if candidate_count < 1 or candidate_count > MAX_APPROXIMATE_INDEX_CANDIDATES:
        raise ValueError(
            f"maxCandidates must be between 1 and {MAX_APPROXIMATE_INDEX_CANDIDATES}"
        )
    if isinstance(value_or_values, Sequence) and not isinstance(
        value_or_values, (str, bytes, bytearray)
    ):
        values = list(value_or_values)
    else:
        values = [value_or_values]
    if not values:
        raise ValueError(
            "approximate index candidates require at least one route value"
        )
    if len(values) > MAX_APPROXIMATE_INDEX_ROUTE_VALUES:
        raise ValueError(
            "approximate index candidate routes cannot exceed "
            f"{MAX_APPROXIMATE_INDEX_ROUTE_VALUES} values"
        )
    if any(value is None for value in values):
        raise TypeError("approximate index candidate route values cannot be null")
    return {"values": values, "maxCandidates": candidate_count}
