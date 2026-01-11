"""Sort helper functions."""

from __future__ import annotations

from ..types import Sort


def asc(field: str) -> Sort:
    return {"field": field, "direction": "ASC"}


def desc(field: str) -> Sort:
    return {"field": field, "direction": "DESC"}
