# Acceptance Criteria: Onyx Database Python SDK

- Public API matches the TypeScript feature set: `onyx.init` with config cache/TTL, `from_table` query builder with list/page/first/values/update/delete, save/batch_save, cascade builder, streaming callbacks, schema/secrets/documents methods.
- Config resolution honors precedence (explicit ➜ env ➜ `ONYX_CONFIG_PATH` ➜ project ➜ home) with cache TTL and logging toggles; partition defaults applied where appropriate.
- HTTP client sets required headers, retries GET/query paths, serializes datetimes, redacts secrets in logs, and surfaces structured `OnyxHTTPError`/`OnyxConfigError`.
- Helpers (conditions/sort) serialize identically to the TypeScript expectations; nested queries allowed via builders in `within`/`not_within`.
- Codegen outputs Pydantic models, `tables`, and `SCHEMA`, allowing extra fields; `onyx-py gen` supports API/file sources, multi-output, timestamp modes, and optional JSON emit.
- `onyx-py schema` supports get/print/publish/validate/diff with table filters; uses the same config chain.
- Streaming reads JSON-lines/SSE, dispatches canonical actions, supports keep-alives, and retries with backoff.
- Docs updated with new API shape (`from_table`), config locations (`config/onyx-database.json`/`onyx.schema.json`), and examples for queries/saves/cascades/streaming/codegen/CLI.
- Tests cover config chain, HTTP retry/logging, builders (query/save/cascade), helpers, streaming, codegen, and CLI entrypoints; lint/typecheck/build pipelines pass.
- Packaging ready for PyPI (console scripts `onyx-py gen`/`onyx-py schema`, Python 3.11+, minimal runtime deps).
