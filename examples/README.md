# Onyx Database Python SDK Examples

Run these examples from the repo root after configuring credentials (env vars or `config/onyx-database.json`). Each script uses `onyx_database` with `from_table` and the same query/save patterns shown in the TypeScript SDK.

Examples are grouped by feature area:

- `query/` – basic filtering, paging, nested queries, updates
- `save/` – save/batch save/cascade saves
- `delete/` – delete by id or by query
- `stream/` – streaming change events
- `document/` – document save/get/delete
- `schema/` – schema get/validate/publish/diff
- `secrets/` – secrets CRUD

Invoke an example:

```bash
python3 examples/query/basic.py
```
