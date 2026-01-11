"""Async helper container for paged query results."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable, Optional


class AsyncQueryResults(list):
    def __init__(self, items: Iterable[Any], next_page: Optional[str] = None, fetcher: Optional[Callable[[str], Awaitable["AsyncQueryResults"]]] = None):
        super().__init__(items)
        self.next_page = next_page
        self._fetcher = fetcher

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

    async def page(self, next_page: Optional[str] = None) -> Optional["AsyncQueryResults"]:
        token = next_page or self.next_page
        if not token or not self._fetcher:
            return None
        return await self._fetcher(token)

    def __aiter__(self):
        async def _gen():
            for item in self:
                yield item
        return _gen()
