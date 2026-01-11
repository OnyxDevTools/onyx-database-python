"""Lightweight shared type aliases used across the SDK."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict


class QueryBuilderLike(Protocol):
    def to_query_object(self) -> Dict[str, Any]:
        ...


Condition = Dict[str, Any]
Sort = Dict[str, str]


class QueryPage(TypedDict, total=False):
    items: List[Any]
    next_page: Optional[str]
    total_count: Optional[int]


class StreamHandlers(TypedDict, total=False):
    on_item_added: Optional[callable]
    on_item_updated: Optional[callable]
    on_item_deleted: Optional[callable]
    on_item: Optional[callable]


class SchemaDiff(TypedDict, total=False):
    added_tables: List[str]
    removed_tables: List[str]
    changed_tables: List[str]
