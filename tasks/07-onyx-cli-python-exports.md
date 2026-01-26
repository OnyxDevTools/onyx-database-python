# Onyx CLI: Python generator should export schema + model map

**Gap**: The `onyx gen --python` output currently writes `onyx/__init__.py` with only a comment, so consumers cannot `from onyx import SCHEMA, MODEL_MAP, tables, User, ...`. Examples in this repo expect those exports for `onyx.init(schema=SCHEMA)` and typed results.

**Plan**
- Update the Onyx CLI Python generator to emit `__init__.py` that:
  - re-exports `tables`, `SCHEMA_JSON`, all generated models, and a `MODEL_MAP` dict `{table: ModelClass}`.
  - defines `SCHEMA = MODEL_MAP` (back-compat with existing SDK examples).
  - populates `__all__` accordingly.
- Ensure the generator keeps idempotent overwrites and preserves formatting used in other emitted files.
- Add/adjust golden test fixtures in onyx-cli for the Python language target to cover `__init__.py` contents.
- Verify downstream: running `onyx gen --python --source api --out ./onyx` lets `examples/seed.py` import `SCHEMA` and run without modification.

**Reference**
- Behavior mirrored from prior SDK-embedded generator (`onyx_database/codegen.py`), now removed in favor of onyx-cli.
