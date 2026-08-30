"""Query builder mirroring the TypeScript SDK shape."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, overload

from .query_results import QueryResults
from .types import SearchMatch, SearchMode, SearchOptions, Sort
from .helpers.conditions import (
    approximate_candidates as approximate_candidates_condition,
    approximate_search as approximate_search_condition,
    hnsw_candidates as hnsw_candidates_condition,
    search as search_condition,
)

_CANDIDATE_OPERATORS = {"CANDIDATES", "SEARCH_CANDIDATES", "HNSW_CANDIDATES"}
_READ_ONLY_SEARCH_OPERATOR = "SEARCH"


def _flatten_strings(values) -> List[str]:
    flat: List[str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            flat.extend(_flatten_strings(v))
        else:
            flat.append(str(v))
    return flat


def _normalize_condition(condition: Any) -> Optional[Dict[str, Any]]:
    if condition is None:
        return None
    if isinstance(condition, dict) and condition.get("conditionType"):
        conds = condition.get("conditions")
        if isinstance(conds, list):
            condition["conditions"] = [c for c in (_normalize_condition(c) for c in conds) if c]
        return condition
    if isinstance(condition, dict) and "field" in condition and "operator" in condition:
        value = condition.get("value")
        if hasattr(value, "to_query_object"):
            value = value.to_query_object()
        criteria = {**condition, "value": value}
        if condition["operator"] in {"IS_NULL", "NOT_NULL"}:
            criteria.pop("value", None)
        return {"conditionType": "SingleCondition", "criteria": criteria}
    return None


def _iter_criteria(
    condition: Optional[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Yield criteria from every level of a normalized condition tree."""

    if not condition:
        return
    if condition.get("conditionType") == "SingleCondition":
        criteria = condition.get("criteria")
        if isinstance(criteria, dict):
            yield criteria
        return
    conditions = condition.get("conditions")
    if isinstance(conditions, list):
        for child in conditions:
            if isinstance(child, dict):
                yield from _iter_criteria(child)


def _candidate_operator(condition: Optional[Dict[str, Any]]) -> Optional[str]:
    return next(
        (
            operator
            for criteria in _iter_criteria(condition)
            if (operator := criteria.get("operator")) in _CANDIDATE_OPERATORS
        ),
        None,
    )


def _read_only_search_operator(
    condition: Optional[Dict[str, Any]],
) -> Optional[str]:
    return next(
        (
            operator
            for criteria in _iter_criteria(condition)
            if (operator := criteria.get("operator"))
            in {*_CANDIDATE_OPERATORS, _READ_ONLY_SEARCH_OPERATOR}
        ),
        None,
    )


def _validate_search_composition(
    existing: Optional[Dict[str, Any]], incoming: Dict[str, Any]
) -> None:
    criteria = [*_iter_criteria(existing), *_iter_criteria(incoming)]
    if any(
        item.get("operator") == _READ_ONLY_SEARCH_OPERATOR
        and item.get("field") != "__full_text__"
        for item in criteria
    ):
        raise ValueError("SEARCH must target __full_text__")
    search_count = sum(
        item.get("operator") == _READ_ONLY_SEARCH_OPERATOR for item in criteria
    )
    if search_count > 1:
        raise ValueError("SEARCH may appear at most once in a query")
    full_text_count = sum(item.get("field") == "__full_text__" for item in criteria)
    if search_count == 1 and full_text_count > 1:
        raise ValueError(
            "SEARCH cannot be combined with another __full_text__ criterion"
        )


class QueryBuilder:
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
        self._on_item_added = None
        self._on_item_updated = None
        self._on_item_deleted = None
        self._on_item = None
        self._candidate_root_operator: Optional[str] = None

    def _require_composable_root(self):
        if self._candidate_root_operator is not None:
            raise ValueError(f"{self._candidate_root_operator} must be the sole root criterion")

    def _require_mutable_root(self, operation: str):
        read_only_operator = _read_only_search_operator(self._conditions)
        if read_only_operator is not None:
            raise ValueError(
                f"{read_only_operator} is read-only and cannot execute {operation}"
            )

    def _require_streamable_root(self):
        read_only_operator = _read_only_search_operator(self._conditions)
        if read_only_operator is not None:
            raise ValueError(
                f"{read_only_operator} cannot be used with live query streams"
            )

    def _adopt_candidate_root(self, condition: Dict[str, Any]) -> bool:
        operator = _candidate_operator(condition)
        if operator is None:
            return False
        if self._conditions is not None or condition.get("conditionType") != "SingleCondition":
            raise ValueError(f"{operator} must be the sole root criterion")
        self._conditions = condition
        self._candidate_root_operator = operator
        return True

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
        self._require_composable_root()
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if self._adopt_candidate_root(cond):
            return self
        _validate_search_composition(self._conditions, cond)
        if not self._conditions:
            self._conditions = cond
        else:
            self._conditions = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [self._conditions, cond],
            }
        return self

    @overload
    def search(
        self,
        query_text_or_search: str,
        min_score: SearchOptions,
        *,
        mode: None = None,
        match: None = None,
        semantic: None = None,
        nearby_bucket_radius: None = None,
        max_candidates: None = None,
        require_all_terms: None = None,
    ) -> QueryBuilder:
        ...

    @overload
    def search(
        self,
        query_text_or_search: Any,
        min_score: Optional[float] = None,
        *,
        mode: Optional[SearchMode] = None,
        match: Optional[SearchMatch] = None,
        semantic: Any = None,
        nearby_bucket_radius: Optional[int] = None,
        max_candidates: Optional[int] = None,
        require_all_terms: Optional[bool] = None,
    ) -> QueryBuilder:
        ...

    def search(
        self,
        query_text_or_search: Any,
        min_score: Optional[float] | SearchOptions = None,
        *,
        mode: Optional[SearchMode] = None,
        match: Optional[SearchMatch] = None,
        semantic: Any = None,
        nearby_bucket_radius: Optional[int] = None,
        max_candidates: Optional[int] = None,
        require_all_terms: Optional[bool] = None,
    ) -> QueryBuilder:
        """Add a legacy predicate or high-level lexical, semantic, or hybrid search."""

        self._require_composable_root()
        cond = _normalize_condition(
            search_condition(
                query_text_or_search,
                min_score,
                mode=mode,
                match=match,
                semantic=semantic,
                nearby_bucket_radius=nearby_bucket_radius,
                max_candidates=max_candidates,
                require_all_terms=require_all_terms,
            )
        )
        if not cond:
            return self
        _validate_search_composition(self._conditions, cond)
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

    def approximate_search(
        self,
        query_text_or_search: Any,
        min_score: Any = None,
        *,
        max_candidates: int = 1_000,
        require_all_terms: bool = True,
    ):
        """Seed a bounded lexical candidate request as the sole root criterion."""

        if self._conditions is not None:
            raise ValueError("SEARCH_CANDIDATES must be the sole root criterion")
        self._conditions = _normalize_condition(
            approximate_search_condition(
                query_text_or_search,
                min_score,
                max_candidates=max_candidates,
                require_all_terms=require_all_terms,
            )
        )
        self._candidate_root_operator = "SEARCH_CANDIDATES"
        return self

    def hnsw_candidates(self, search_query: Any):
        """Seed a bounded native-HNSW request as the sole root criterion."""

        if self._conditions is not None:
            raise ValueError("HNSW_CANDIDATES must be the sole root criterion")
        self._conditions = _normalize_condition(hnsw_candidates_condition(search_query))
        self._candidate_root_operator = "HNSW_CANDIDATES"
        return self

    def approximate_candidates(
        self,
        attribute: str,
        value_or_values: Any,
        max_candidates: int = 1_000,
    ):
        """Seed bounded ordinary-index admission as the sole root criterion."""

        if self._conditions is not None:
            raise ValueError("CANDIDATES must be the sole root criterion")
        self._conditions = _normalize_condition(
            approximate_candidates_condition(attribute, value_or_values, max_candidates)
        )
        self._candidate_root_operator = "CANDIDATES"
        return self

    def and_(self, condition):
        self._require_composable_root()
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if self._adopt_candidate_root(cond):
            return self
        _validate_search_composition(self._conditions, cond)
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
        """Alias for and_ to avoid reserved keyword concerns."""
        return self.and_(condition)

    def or_(self, condition):
        self._require_composable_root()
        cond = _normalize_condition(condition)
        if not cond:
            return self
        if self._adopt_candidate_root(cond):
            return self
        _validate_search_composition(self._conditions, cond)
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

    # Execution
    def count(self) -> int:
        if self._mode != "select":
            raise ValueError("Cannot call count() in update mode.")
        return self._exec.count(self.ensure_table(), self.to_select_query(), self._partition)

    def page(self, page_size: Optional[int] = None, next_page: Optional[str] = None, model=None):
        if self._mode != "select":
            raise ValueError("Cannot call page() in update mode.")
        size = page_size or self._page_size
        token = next_page or self._next_page
        # When specific fields are selected, default to dicts unless a model is explicitly provided.
        if self._fields and model is None:
            model = None
        else:
            model = model or self._default_model()
        res = self._exec.query_page(self.ensure_table(), self.to_select_query(), {"pageSize": size, "nextPage": token, "partition": self._partition})
        records = res.get("records", [])
        mapped = self._apply_model(records, model)
        return {"records": mapped, "nextPage": res.get("nextPage") or res.get("next_page")}

    def list(self, page_size: Optional[int] = None, next_page: Optional[str] = None, model=None) -> QueryResults:
        # Default to dicts when select() is used unless a model is explicitly passed.
        chosen_model = None if (self._fields and model is None) else model or self._default_model()
        pg = self.page(page_size=page_size, next_page=next_page, model=chosen_model)
        fetcher = lambda token: self.next_page(token).list(page_size=page_size, model=chosen_model)  # noqa: E731
        return QueryResults(pg.get("records", []), pg.get("nextPage") or pg.get("next_page"), fetcher)

    def first_or_none(self, model=None):
        if self._mode != "select":
            raise ValueError("Cannot call first_or_none() in update mode.")
        if not self._conditions:
            raise ValueError("first_or_none() requires a where() clause.")
        self._limit = 1
        chosen_model = None if (self._fields and model is None) else model or self._default_model()
        pg = self.page(model=chosen_model)
        records = pg.get("records") or []
        return records[0] if records else None

    def delete(self):
        self._require_mutable_root("delete")
        if self._mode != "select":
            raise ValueError("delete() is only applicable in select mode.")
        return self._exec.delete_by_query(self.ensure_table(), self.to_select_query(), self._partition)

    def update(self):
        self._require_mutable_root("update")
        if self._mode != "update":
            raise ValueError("Call set_updates(...) before update().")
        return self._exec.update(self.ensure_table(), self.to_update_query(), self._partition)

    def on_item_added(self, fn):
        self._on_item_added = fn
        return self

    def on_item_updated(self, fn):
        self._on_item_updated = fn
        return self

    def on_item_deleted(self, fn):
        self._on_item_deleted = fn
        return self

    def on_item(self, fn):
        self._on_item = fn
        return self

    def stream(self, include_query_results: bool = True, keep_alive: bool = False):
        if self._mode != "select":
            raise ValueError("Streaming is only applicable in select mode.")
        self._require_streamable_root()
        handlers = {
            "on_item_added": self._on_item_added,
            "on_item_updated": self._on_item_updated,
            "on_item_deleted": self._on_item_deleted,
            "on_item": self._on_item,
        }
        return self._exec.stream(
            self.ensure_table(),
            self.to_select_query(),
            include_query_results,
            keep_alive,
            handlers,
        )
