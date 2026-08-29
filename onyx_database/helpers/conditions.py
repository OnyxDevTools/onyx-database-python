"""Condition helper functions mirroring the TypeScript SDK operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence, Union

from ..types import Condition, QueryBuilderLike
from .candidate_search import (
    DEFAULT_APPROXIMATE_INDEX_CANDIDATES,
    approximate_index_candidate_query,
    hnsw_search_query,
    vector_search_query,
)


def _condition(field: str, operator: str, value: Any = None) -> Condition:
    return {"field": field, "operator": operator, "value": value}


def eq(field: str, value: Any) -> Condition:
    return _condition(field, "EQUAL", value)


def neq(field: str, value: Any) -> Condition:
    return _condition(field, "NOT_EQUAL", value)


def _normalize_values(values: Union[str, Sequence[Any], QueryBuilderLike]) -> Any:
    if isinstance(values, str):
        return [v.strip() for v in values.split(",") if v.strip()]
    return values


def in_op(field: str, values: Union[str, Sequence[Any], QueryBuilderLike]) -> Condition:
    return _condition(field, "IN", _normalize_values(values))


def within(field: str, values: Union[str, Sequence[Any], QueryBuilderLike]) -> Condition:
    return in_op(field, values)


def not_in(field: str, values: Union[str, Sequence[Any], QueryBuilderLike]) -> Condition:
    return _condition(field, "NOT_IN", _normalize_values(values))


def not_within(field: str, values: Union[str, Sequence[Any], QueryBuilderLike]) -> Condition:
    return not_in(field, values)


def between(field: str, lower: Any, upper: Any) -> Condition:
    return _condition(field, "BETWEEN", [lower, upper])


def not_between(field: str, lower: Any, upper: Any) -> Condition:
    return _condition(field, "NOT_BETWEEN", [lower, upper])


def gt(field: str, value: Any) -> Condition:
    return _condition(field, "GREATER_THAN", value)


def gte(field: str, value: Any) -> Condition:
    return _condition(field, "GREATER_THAN_EQUAL", value)


def lt(field: str, value: Any) -> Condition:
    return _condition(field, "LESS_THAN", value)


def lte(field: str, value: Any) -> Condition:
    return _condition(field, "LESS_THAN_EQUAL", value)


def matches(field: str, regex: str) -> Condition:
    return _condition(field, "MATCHES", regex)


def not_matches(field: str, regex: str) -> Condition:
    return _condition(field, "NOT_MATCHES", regex)


def like(field: str, pattern: str) -> Condition:
    return _condition(field, "LIKE", pattern)


def not_like(field: str, pattern: str) -> Condition:
    return _condition(field, "NOT_LIKE", pattern)


def contains(field: str, value: Any) -> Condition:
    return _condition(field, "CONTAINS", value)


def not_contains(field: str, value: Any) -> Condition:
    return _condition(field, "NOT_CONTAINS", value)


def starts_with(field: str, prefix: str) -> Condition:
    return _condition(field, "STARTS_WITH", prefix)


def not_starts_with(field: str, prefix: str) -> Condition:
    return _condition(field, "NOT_STARTS_WITH", prefix)


def is_null(field: str) -> Condition:
    return {"field": field, "operator": "IS_NULL"}


def not_null(field: str) -> Condition:
    return {"field": field, "operator": "NOT_NULL"}


# Convenience aliases mirroring TS containsIgnoreCase/notContainsIgnoreCase
def contains_ignore_case(field: str, value: Any) -> Condition:
    return _condition(field, "CONTAINS_IGNORE_CASE", value)


def not_contains_ignore_case(field: str, value: Any) -> Condition:
    return _condition(field, "NOT_CONTAINS_IGNORE_CASE", value)


def search(
    query_text_or_search: Any,
    min_score: Optional[float] = None,
    *,
    semantic: Any = None,
    nearby_bucket_radius: Optional[int] = None,
    max_candidates: Optional[int] = None,
    require_all_terms: Optional[bool] = None,
) -> Condition:
    """Create an exact native full-text, semantic, or hybrid search condition.

    Strings retain the established ``queryText`` wire shape. A mapping is
    validated and emitted as a canonical ``VectorSearchQuery``.
    """

    has_vector_options = (
        semantic is not None
        or nearby_bucket_radius is not None
        or max_candidates is not None
        or require_all_terms is not None
    )
    if isinstance(query_text_or_search, str) and not has_vector_options:
        value: Any = {"queryText": query_text_or_search, "minScore": min_score}
    else:
        options: dict[str, Any] = {}
        if min_score is not None:
            options["min_score"] = min_score
        if semantic is not None:
            options["semantic"] = semantic
        if nearby_bucket_radius is not None:
            options["nearby_bucket_radius"] = nearby_bucket_radius
        if max_candidates is not None:
            options["max_candidates"] = max_candidates
        if require_all_terms is not None:
            options["require_all_terms"] = require_all_terms
        value = (
            vector_search_query(text=query_text_or_search, **options)
            if isinstance(query_text_or_search, str)
            else vector_search_query(query_text_or_search, **options)
        )
    return _condition("__full_text__", "MATCHES", value)


def approximate_search(
    query_text_or_search: Any,
    min_score: Any = None,
    *,
    max_candidates: int = 1_000,
    require_all_terms: bool = True,
) -> Condition:
    """Create a sole-root ``SEARCH_CANDIDATES`` lexical admission condition."""

    if isinstance(query_text_or_search, str) and isinstance(min_score, Mapping):
        if max_candidates != 1_000 or require_all_terms is not True:
            raise TypeError(
                "do not combine an options mapping with approximate search keywords"
            )
        query = vector_search_query(min_score, text=query_text_or_search)
    elif isinstance(query_text_or_search, str):
        query = vector_search_query(
            text=query_text_or_search,
            min_score=min_score,
            max_candidates=max_candidates,
            require_all_terms=require_all_terms,
        )
    else:
        if (
            min_score is not None
            or max_candidates != 1_000
            or require_all_terms is not True
        ):
            raise TypeError(
                "pass approximate search options inside the VectorSearchQuery mapping"
            )
        query = vector_search_query(query_text_or_search)
    if query["text"] is None or query["semantic"] is not None:
        raise ValueError("SEARCH_CANDIDATES supports text-only VectorSearchQuery values")
    return _condition("__full_text__", "SEARCH_CANDIDATES", query)


def hnsw_candidates(search_query: Any) -> Condition:
    """Create a sole-root ``HNSW_CANDIDATES`` nearest-neighbor condition."""

    return _condition(
        "__full_text__", "HNSW_CANDIDATES", hnsw_search_query(search_query)
    )


def approximate_candidates(
    attribute: str,
    value_or_values: Any,
    max_candidates: int = DEFAULT_APPROXIMATE_INDEX_CANDIDATES,
) -> Condition:
    """Create a sole-root bounded ordinary-index candidate condition."""

    if not isinstance(attribute, str) or not attribute.strip():
        raise TypeError("candidate attribute must not be blank")
    return _condition(
        attribute,
        "CANDIDATES",
        approximate_index_candidate_query(value_or_values, max_candidates),
    )
