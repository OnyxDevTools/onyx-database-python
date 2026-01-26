# Onyx Database Python SDK Examples

Run these examples from the repo root after configuring credentials (env vars or `config/onyx-database.json`). Each script uses `onyx_database` with `from_table` and the same query/save patterns shown in the TypeScript SDK.

Before running, install the external **Onyx CLI** and generate the local stubs the examples import:

```bash
brew tap OnyxDevTools/onyx-cli
brew install onyx-cli
onyx schema get onyx.schema.json
onyx gen --python --source api --out ./onyx
```

Examples are grouped by feature area:

- `ai/` – Onyx AI chat completions and models
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

```bash
python3 examples/ai/chat.py
python3 examples/ai/streaming.py
python3 examples/ai/models.py
python3 examples/ai/script_approval.py
```
