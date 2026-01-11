"""Helper container for paged query results."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


class QueryResults(list):
    def __init__(self, items: Iterable[Any], next_page: Optional[str] = None, fetcher=None):
        super().__init__(items)
        self.next_page = next_page
        self._fetcher: Optional[Callable[[str], "QueryResults"]] = fetcher

    def values(self, field: str) -> list:
        vals = []
        for item in self:
            if isinstance(item, dict):
                vals.append(item.get(field))
            else:
                vals.append(getattr(item, field, None))
        return vals

    def first_or_none(self) -> Any:
        return self[0] if self else None

    def size(self) -> int:
        return len(self)

    def page(self, next_page: Optional[str] = None) -> Optional["QueryResults"]:
        if next_page is None:
            next_page = self.next_page
        if not next_page or not self._fetcher:
            return None
        return self._fetcher(next_page)
