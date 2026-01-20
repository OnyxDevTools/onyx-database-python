"""Condition helper functions mirroring the TypeScript SDK operators."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..types import Condition, QueryBuilderLike


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
    return _condition(field, "IS_NULL")


def not_null(field: str) -> Condition:
    return _condition(field, "NOT_NULL")


# Convenience aliases mirroring TS containsIgnoreCase/notContainsIgnoreCase
def contains_ignore_case(field: str, value: Any) -> Condition:
    return _condition(field, "CONTAINS_IGNORE_CASE", value)


def not_contains_ignore_case(field: str, value: Any) -> Condition:
    return _condition(field, "NOT_CONTAINS_IGNORE_CASE", value)


def search(query_text: str, min_score: Optional[float] = None) -> Condition:
    """Full-text search using the __full_text__ pseudo-field."""
    return _condition("__full_text__", "MATCHES", {"queryText": query_text, "minScore": min_score})
