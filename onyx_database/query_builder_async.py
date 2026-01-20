"""Async query builder mirroring the synchronous QueryBuilder."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .query_builder import _flatten_strings, _normalize_condition
from .query_results_async import AsyncQueryResults
from .types import Sort
from .helpers.conditions import search as search_condition


class AsyncQueryBuilder:
    def __init__(self, executor, table: Optional[str] = None, partition: Optional[str] = None):
        self._exec = executor
        self._table = table
        self._fields: Optional[List[str]] = None
        self._resolvers: Optional[List[str]] = None
        self._conditions: Optional[Dict[str, Any]] = None
        self._sort: Optional[List[Sort]] = None
        self._limit: Optional[int] = None
        self._distinct = False
        self._group_by: Optional[List[str]] = None
        self._resolver_types: Dict[str, Any] = {}
        self._partition = partition
        self._page_size: Optional[int] = None
        self._next_page: Optional[str] = None
        self._mode: str = "select"
        self._updates: Optional[Dict[str, Any]] = None

    def ensure_table(self) -> str:
        if not self._table:
            raise ValueError("Table is not defined. Call from_table(<table>) first.")
        return self._table

    def to_select_query(self) -> Dict[str, Any]:
        return {
            "type": "SelectQuery",
            "fields": self._fields,
            "conditions": _normalize_condition(self._conditions),
            "sort": self._sort,
            "limit": self._limit,
            "distinct": self._distinct,
            "groupBy": self._group_by,
            "partition": self._partition,
            "resolvers": self._resolvers,
        }

    def to_update_query(self) -> Dict[str, Any]:
        return {
            "type": "UpdateQuery",
            "conditions": _normalize_condition(self._conditions),
            "updates": self._updates or {},
            "sort": self._sort,
            "limit": self._limit,
            "partition": self._partition,
        }

    def to_query_object(self) -> Dict[str, Any]:
        payload = self.to_update_query() if self._mode == "update" else self.to_select_query()
        return {**payload, "table": self.ensure_table()}

    # Fluent modifiers
    def from_table(self, table: str):
        self._table = table
        return self

    def select(self, *fields):
        flat = _flatten_strings(fields)
        self._fields = flat or None
        return self

    def resolve(self, *values):
        resolver_names: List[str] = []
        for v in values:
            if isinstance(v, tuple) and len(v) == 2:
                name, model = v
                resolver_names.append(name)
                if name and model:
                    self._resolver_types[str(name)] = model
            else:
                resolver_names.append(v)
        flat = _flatten_strings(resolver_names)
        if flat:
            existing = list(self._resolvers) if self._resolvers else []
            existing.extend(flat)
            self._resolvers = existing
        return self

    def where(self, condition):
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if not self._conditions:
            self._conditions = cond
        else:
            self._conditions = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [self._conditions, cond],
            }
        return self

    def search(self, query_text: str, min_score: Optional[float] = None):
        cond = _normalize_condition(search_condition(query_text, min_score))
        if not cond:
            return self
        if self._conditions and self._conditions.get("conditionType") == "CompoundCondition" and self._conditions.get("operator") == "AND":
            self._conditions["conditions"].append(cond)
        elif self._conditions:
            self._conditions = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [self._conditions, cond],
            }
        else:
            self._conditions = cond
        return self

    def and_(self, condition):
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if self._conditions and self._conditions.get("conditionType") == "CompoundCondition" and self._conditions.get("operator") == "AND":
            self._conditions["conditions"].append(cond)
        elif self._conditions:
            self._conditions = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [self._conditions, cond],
            }
        else:
            self._conditions = cond
        return self

    def and_where(self, condition):
        return self.and_(condition)

    def or_(self, condition):
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if self._conditions and self._conditions.get("conditionType") == "CompoundCondition" and self._conditions.get("operator") == "OR":
            self._conditions["conditions"].append(cond)
        elif self._conditions:
            self._conditions = {
                "conditionType": "CompoundCondition",
                "operator": "OR",
                "conditions": [self._conditions, cond],
            }
        else:
            self._conditions = cond
        return self

    def order_by(self, *sorts: Sort):
        self._sort = list(sorts) if sorts else None
        return self

    def group_by(self, *fields: str):
        self._group_by = list(fields) if fields else None
        return self

    def distinct(self):
        self._distinct = True
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    def in_partition(self, partition: str):
        self._partition = partition
        return self

    def page_size(self, n: int):
        self._page_size = n
        return self

    def next_page(self, token: str):
        self._next_page = token
        return self

    def set_updates(self, updates: Dict[str, Any]):
        self._mode = "update"
        self._updates = updates
        return self

    def _default_model(self):
        getter = getattr(self._exec, "get_model_for_table", None)
        if callable(getter):
            try:
                return getter(self.ensure_table())
            except Exception:
                return None
        return None

    def _coerce_resolver_value(self, value, model):
        if model is None:
            return value
        if value is None:
            return None
        if isinstance(value, list):
            return [self._coerce_resolver_value(v, model) for v in value]
        if isinstance(value, model):
            return value
        if isinstance(value, dict):
            return model(**value)
        return model(value)

    def _apply_resolver_types(self, item: Any, resolver_types: Dict[str, Any]):
        if not resolver_types or not isinstance(item, dict):
            return item
        new_item = dict(item)
        for name, model in resolver_types.items():
            if name in new_item:
                new_item[name] = self._coerce_resolver_value(new_item[name], model)
        return new_item

    def _apply_model(self, items, model):
        if model is None and not self._resolver_types:
            return items
        out = []
        for item in items:
            working = self._apply_resolver_types(item, self._resolver_types) if isinstance(item, dict) else item
            if model is None:
                out.append(working)
            elif isinstance(working, model):
                out.append(working)
            elif isinstance(working, dict):
                out.append(model(**working))
            else:
                out.append(model(working))
        return out

    async def count(self) -> int:
        if self._mode != "select":
            raise ValueError("Cannot call count() in update mode.")
        return await self._exec.count(self.ensure_table(), self.to_select_query(), self._partition)

    async def page(self, page_size: Optional[int] = None, next_page: Optional[str] = None, model=None):
        if self._mode != "select":
            raise ValueError("Cannot call page() in update mode.")
        size = page_size or self._page_size
        token = next_page or self._next_page
        chosen_model = None if (self._fields and model is None) else model or self._default_model()
        res = await self._exec.query_page(self.ensure_table(), self.to_select_query(), {"pageSize": size, "nextPage": token, "partition": self._partition})
        records = res.get("records", [])
        mapped = self._apply_model(records, chosen_model)
        return {"records": mapped, "nextPage": res.get("nextPage") or res.get("next_page")}

    async def list(self, page_size: Optional[int] = None, next_page: Optional[str] = None, model=None) -> AsyncQueryResults:
        chosen_model = None if (self._fields and model is None) else model or self._default_model()
        pg = await self.page(page_size=page_size, next_page=next_page, model=chosen_model)
        async def fetcher(token):
            self.next_page(token)
            return await self.list(page_size=page_size, model=chosen_model)
        return AsyncQueryResults(pg.get("records", []), pg.get("nextPage") or pg.get("next_page"), fetcher)

    async def first_or_none(self, model=None):
        if self._mode != "select":
            raise ValueError("Cannot call first_or_none() in update mode.")
        if not self._conditions:
            raise ValueError("first_or_none() requires a where() clause.")
        self._limit = 1
        chosen_model = None if (self._fields and model is None) else model or self._default_model()
        pg = await self.page(model=chosen_model)
        records = pg.get("records") or []
        return records[0] if records else None

    async def delete(self):
        if self._mode != "select":
            raise ValueError("delete() is only applicable in select mode.")
        return await self._exec.delete_by_query(self.ensure_table(), self.to_select_query(), self._partition)

    async def update(self):
        if self._mode != "update":
            raise ValueError("Call set_updates(...) before update().")
        return await self._exec.update(self.ensure_table(), self.to_update_query(), self._partition)
