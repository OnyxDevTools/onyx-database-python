# Changelog

## Unreleased

## 2.4.0 - 2026-08-29

- Added one high-level `search` API for lexical, semantic, and hybrid retrieval
  across sync and async table builders and database-wide facades. The new form
  accepts an options mapping or Python keyword arguments, emits the fail-closed
  `SEARCH` wire operator, defaults to hybrid/any retrieval, validates scores in
  `0..1`, and remains composable with structured filters while rejecting
  mutations.
- Preserved the exact legacy `MATCHES` wire shape for `search(text)` and
  `search(text, min_score)` calls.
- Added typed `SearchOptions` overloads and a PEP 561 marker while retaining the
  legacy numeric-score overloads on sync and async builders and database-wide
  facades.
- Added recursive guards for duplicate or mis-targeted `SEARCH`, mixed
  `__full_text__` predicates, and unsupported live query streams. Database-wide
  search no longer inherits a configured default partition.

## 2.3.0 - 2026-08-29

- Added typed, hard-bounded native `CANDIDATES`, `SEARCH_CANDIDATES`, and
  `HNSW_CANDIDATES` query helpers to sync and async builders.
- Added lossless semantic-signature and HNSW wire validation, including signed
  64-bit calibration identifiers, mixed-radix bucket checks, vector/work bounds,
  and sole-root admission enforcement.
- Added the missing public `NOT_BETWEEN` operator/helper so the Python SDK now
  covers every Cloud query operator.

## 2.2.0 - 2026-08-29

- Added sync and async `fenced_save`, `fenced_delete_where`, and `fenced_update_where` helpers for the
  server's atomic guard-check-plus-mutation endpoint. Fenced writes never retry,
  perform a client-side guard read, or fall back on older servers.
- Bounded fenced saves to 500 rows and documented the serialized
  `QueryCondition` shape required for fenced delete filters.
- Fenced delete issues one request capped at 500 affected rows; callers can
  repeat while `affected == 500` so each batch receives a fresh guard check.
- Fenced update issues one request, strictly requires a concrete partition,
  serialized `QueryCondition`, and non-empty update map, and affects at most one row.
- `IS_NULL` and `NOT_NULL` query helpers now omit the wire `value` field, so
  nullable compare-and-set expectations serialize to Cloud's canonical shape.

## 2.1.0 - 2026-08-29

- Added an opt-in MessagePack v1 transport for sync and async entity CRUD, query,
  and query-stream routes while retaining JSON as the default and for all
  non-entity APIs.
- Added sync and async `create(table, entity)` helpers for the server's atomic
  create-if-absent endpoint. They reject batch input and never fall back to an
  upsert when an older server lacks the endpoint.
- Added bounded recursive wire validation, actual-response content negotiation,
  cross-client golden fixtures, transport regression tests, and a reproducible
  codec/packet-size benchmark.
