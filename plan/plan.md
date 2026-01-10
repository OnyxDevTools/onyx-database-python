# Build Plan: Onyx Database Python SDK

- Catalog parity targets: read the TypeScript SDK surface (README, `src/types/public.ts`, builders, HTTP/stream) to derive a feature matrix covering config, CRUD/queries, streaming, schema/secrets/documents APIs, helpers, codegen, and CLI flags.
- Architecture & scaffolding: define Python package layout (`onyx_database/` core, builders, helpers, codegen, cli), choose minimal runtime deps (prefer stdlib; optional HTTP adapter), and set up dev tooling (ruff/mypy/pytest + build backend).
- Config resolution: implement the same chain (explicit kwargs ➜ env vars ➜ `ONYX_CONFIG_PATH` ➜ project config ➜ home profile), TTL caching, logging toggles (`request_logging_enabled`, `response_logging_enabled`, `ONYX_DEBUG`), and partition defaults.
- HTTP layer: build a small client with retries (GET/query paths), header signing (`x-onyx-key`/`x-onyx-secret`), JSON parsing that tolerates NaN, optional custom fetch adapter, and request/response logging with secrets redacted.
- Public client (`onyx.init`): expose sync-first client (optional async flavor) wiring save/batch_save/delete/find_by_id, query builder entry `from_table`, cascade builder, utility methods (`clear_cache_config`, TTL override).
- Query builder: support select/where/and_/or_/limit/offset/order_by, page/list/first/first_or_none/values/size, nested queries for `within`/`not_within`, update/delete via query endpoints, and resolve paths for graph fetching; match TypeScript request shapes.
- Condition & sort helpers: ship eq/neq/gt/gte/lt/lte/between/within/not_within/in_op/not_in/like/not_like/contains/not_contains/starts_with/not_starts_with/matches/not_matches/is_null/not_null plus asc/desc helpers, serializing to the TS contract.
- Save & cascade builders: implement save/batch_save semantics (partition handling, batches), cascade graph syntax (`field:Type(target, source)` and builder variant), and cascade delete options.
- Streaming: implement JSON-lines/SSE reader with callbacks (on_item_added/updated/deleted/on_item), keep-alive handling, and auto-reconnect with backoff; gate debug via `ONYX_STREAM_DEBUG`.
- Schema/Secrets/Documents APIs: add get_schema/history/validate/update/publish/diff, secrets CRUD, and document CRUD with base64 support; keep payload/response shapes aligned to the TS SDK.
- Codegen CLI: implement `onyx-py gen` (source api|file, multi `--out`, `--timestamps` modes, optional JSON emit) emitting Pydantic models, `tables` helper, `SCHEMA` map, and `__init__.py` re-exports; allow extra fields.
- Schema CLI: implement `onyx-py schema` (get/print/publish/validate/diff, table filtering, stdout-only for subsets) reusing the same config chain.
- Docs & examples: update README and example scripts to use `from_table`, helpers, config paths, streaming, codegen usage; include sample generated package layout.
- Testing & quality: add unit tests for config chain, HTTP retry/logging, builders, helpers, streaming (mocked), codegen outputs, and CLI commands; mypy/ruff/format + CI workflow; package metadata for PyPI.
