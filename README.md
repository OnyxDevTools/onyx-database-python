
# onyx-database-python (Python)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![codecov](https://codecov.io/gh/OnyxDevTools/onyx-database-python/branch/main/graph/badge.svg)](https://codecov.io/gh/OnyxDevTools/onyx-database-python)
[![PyPI version](https://img.shields.io/pypi/v/onyx-database.svg)](https://pypi.org/project/onyx-database/)

Python client SDK for **Onyx Cloud Database** — a small, typed, builder-pattern API for querying and persisting data in Onyx. Includes:

- A runtime SDK (sync-first; async optional if enabled)
- Credential resolution (explicit config ➜ env ➜ config file chain ➜ home profile)
- Optional generated tables/models via the external **Onyx CLI** (`onyx gen --python`) that produces table-safe Pydantic models and a `tables` helper

- **Website:** https://onyx.dev/
- **Cloud Console:** https://cloud.onyx.dev
- **Docs hub:** https://onyx.dev/documentation/
- **Cloud API docs:** https://onyx.dev/documentation/api-documentation/

---

## Getting started (Cloud ➜ keys ➜ connect)

1. **Sign up & create resources** at **https://cloud.onyx.dev**  
   Create an **Organization**, then a **Database**, define your **Schema** (e.g., `User`, `Role`, `Permission`), and create **API Keys**.

2. **Note your connection parameters**:
   You will need to setup an apiKey to connect to your database in the onyx console at <https://cloud.onyx.dev>.  After creating the apiKey, you can download the `onyx-database.json`. Save it to the `config` folder
   The SDK and the external Onyx CLI share the same config resolution chain. A recommended layout is:

    ```
    your-project/
    ├── config/
    │   └── onyx-database.json
    ├── onyx.schema.json          # fetched via `onyx schema get`
    └── onyx/                     # generated via `onyx gen --python`
    ```

3. **Install the SDK** in your project:

   ```bash
   pip install onyx-database
   ```

4. **Initialize the client** using config files, env vars, or explicit config.

> Supports Python **3.11+**.

---

## Install

```bash
pip install onyx-database
```

The package installs the importable module: `onyx_database`.

### CLI tooling (external, recommended globally)

Install the standalone **Onyx CLI** for schema commands and code generation:

```bash
# macOS (Homebrew tap)
brew tap OnyxDevTools/onyx-cli
brew install onyx-cli
onyx version

# or portable install script (uses latest release)
curl -fsSL https://raw.githubusercontent.com/OnyxDevTools/onyx-cli/main/scripts/install.sh | bash
```

### Install from a repo checkout (local dev)

```bash
# from repo root
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

---

## Initialize the client

This SDK resolves credentials automatically using the chain:

**explicit config ➜ environment variables ➜ `ONYX_CONFIG_PATH` file ➜ project config file ➜ home profile**

Call `onyx.init(database_id="database-id")` to target a specific database, or omit the
`database_id` to use the default from config resolution. You can also pass credentials
directly via config.

### Option A) Config files (recommended)

Create a project config file:

- `./config/onyx-database.json` (recommended layout), or
- `./onyx-database.json` (repo root)

Example: `config/onyx-database.json`

```json
{
  "databaseId": "YOUR_DATABASE_ID",
  "baseUrl": "https://api.onyx.dev",
  "aiBaseUrl": "https://ai.onyx.dev",
  "defaultModel": "onyx",
  "apiKey": "YOUR_DATABASE_KEY",
  "apiSecret": "YOUR_API_SECRET"
}
```

Then initialize:

```py
from onyx_database import onyx

db = onyx.init()  # resolves config via the standard chain
```

#### AWS Secrets Manager credentials (optional)

If your config uses AWS Secrets Manager, the SDK can resolve credentials at runtime
(install with `pip install "onyx-database[aws]"`):

```json
{
  "baseUrl": "https://api.onyx.dev",
  "databaseId": "YOUR_DATABASE_ID",
  "auth": {
    "type": "aws_secrets_manager",
    "secretId": "onyx/demo/database/credentials",
    "apiKeyField": "apiKey",
    "apiSecretField": "apiSecret"
  }
}
```

### Option B) Environment variables

Set the following:

- `ONYX_DATABASE_ID`
- `ONYX_DATABASE_BASE_URL`
- `ONYX_AI_BASE_URL` (defaults to `https://ai.onyx.dev`)
- `ONYX_DEFAULT_MODEL` (defaults to `onyx`)
- `ONYX_DATABASE_API_KEY`
- `ONYX_DATABASE_API_SECRET`

```py
from onyx_database import onyx

db = onyx.init(database_id="YOUR_DATABASE_ID")
```

### Option C) Explicit config (direct)

```py
from onyx_database import onyx

db = onyx.init(
    base_url="https://api.onyx.dev",
    database_id="YOUR_DATABASE_ID",
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
    partition="tenantA",
    request_logging_enabled=True,
    response_logging_enabled=True,
)
```

#### Default partition + logging

- `partition` sets a default partition for queries, `find_by_id`, and deletes by primary key.
- Save operations use the partition field on the entity itself (if present).
- `request_logging_enabled` logs HTTP requests and JSON bodies.
- `response_logging_enabled` logs HTTP responses and JSON bodies.
- Setting `ONYX_DEBUG=true` enables both request/response logging and also logs which credential source was used.

---

## Use Onyx AI (ChatGPT-compatible)

Onyx AI shares the same key/secret as the database client. Use `db.ai` for chat, models, and script approvals; `db.chat()`/`db.chat('...')` remain supported. Shorthand chat defaults to `defaultModel` (or `ONYX_DEFAULT_MODEL`), which falls back to `onyx`. The AI base URL defaults to `https://ai.onyx.dev` and can be overridden via `aiBaseUrl`/`ai_base_url` in config or `ONYX_AI_BASE_URL` in the environment.

```py
from onyx_database import onyx

db = onyx.init()

# Shorthand chat completion (returns first message content)
quick = db.chat("Summarize last week's signups.")
print(quick)

# Shorthand defaults:
# - model: config defaultModel / ONYX_DEFAULT_MODEL (fallback "onyx")
# - role: "user"
# - stream: False
# - return value: first message content (set raw=True for full response)

# Full request via db.ai
completion = db.ai.chat(
    {
        "model": "onyx-chat",
        "messages": [{"role": "user", "content": "Summarize last week's signups."}],
    }
)
print(completion["choices"][0]["message"]["content"])

# Override shorthand defaults (raw=True returns full response)
custom = db.chat(
    "List three colors.",
    model="onyx-chat",
    role="user",
    temperature=0.2,
    raw=True,
)

# Models metadata
models = db.ai.get_models()
print([m["id"] for m in models["data"]])
```

- Tool calls and other OpenAI fields can be passed through when using a full request.
- `db.chat()` returns a chat client for full requests: `db.chat().create({...})`.
- Pass `database_id="..."` to scope grounding/billing for chat; defaults to the database ID from `onyx.init()`.
- Shorthand streaming returns the SSE iterator (set `stream=True`); non-streaming shorthand returns the first message content unless `raw=True`.
- Script mutation approvals:

```py
approval = db.ai.request_script_approval("db.save({ 'table': 'User', 'id': '123' })")
if approval["requiresApproval"]:
    print("Approval needed:", approval["findings"])
```

- Streaming chat:

```py
stream = db.ai.chat(
    {
        "model": "onyx-chat",
        "stream": True,
        "messages": [{"role": "user", "content": "Draft an onboarding email."}],
    }
)
for chunk in stream:
    delta = chunk["choices"][0].get("delta", {})
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
```

- Override the AI base URL (self-hosted/testing):

```py
db = onyx.init(
    base_url="https://api.onyx.dev",
    ai_base_url="http://localhost:8787",
    api_key="YOUR_KEY",
    api_secret="YOUR_SECRET",
)
```

Example scripts:

- Chat completion: `examples/ai/chat.py`
- Streaming chat (full request): `examples/ai/chat_stream.py`
- Shorthand chat: `examples/ai/chat_shorthand.py`
- Shorthand streaming: `examples/ai/chat_shorthand_stream.py`
- List/retrieve models: `examples/ai/models.py`

### Connection handling

Calling `onyx.init()` returns a lightweight client. Configuration is resolved once and cached
for a short TTL (configurable) to avoid repeated credential lookups. Each database instance
keeps a single internal HTTP client (connection pooling is handled by the HTTP library). Reuse
the returned `db` for multiple operations.

---

## Optional: generate Python models from your schema (via Onyx CLI)

Use the standalone **Onyx CLI** to emit Python stubs (models, `tables` helper, and `SCHEMA` mapping). The CLI mirrors this SDK’s credential/config resolution.

### Generate directly from the API (preferred)

```bash
onyx gen --python --source api --out ./onyx
```

### Generate from a local schema file

```bash
onyx schema get onyx.schema.json                # fetch if you don't have one yet
onyx gen --python --schema ./onyx.schema.json --out ./onyx
```

Notes:

- Defaults: source `file`, schema `./onyx.schema.json`, output `./onyx`, overwrite on.
- Use `--tables User,Role` to print only selected entities to stdout instead of writing files.
- If you omit `--python`, the CLI falls back to `codegenLanguage` in config or `ONYX_CODEGEN_LANGUAGE`.

---

## Manage schemas from the CLI (Onyx CLI)

Use the external Onyx CLI for schema download/publish/validate/diff:

```bash
# Download to onyx.schema.json (default path)
onyx schema get onyx.schema.json

# Publish local schema (validates first)
onyx schema publish onyx.schema.json

# Validate without publishing
onyx schema validate onyx.schema.json

# Diff local schema vs API
onyx schema diff onyx.schema.json

# Print only selected tables to stdout
onyx schema get --tables=User,Profile
```

The CLI uses the same credential/config resolution chain as the SDK (explicit config ➜ env vars ➜ `ONYX_CONFIG_PATH` ➜ project config ➜ home profile).

Programmatic diffing is also available:

```py
from onyx_database import onyx

db = onyx.init()
diff = db.diff_schema(local_schema)  # SchemaUpsertRequest-like dict
print(diff.changed_tables)
```

---

## Use in code (with generated stubs)

```py
from onyx_database import onyx, eq, asc
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

active_users = (
    db.from_table(tables.User)
      .where(eq("status", "active"))
      .order_by(asc("createdAt"))
      .limit(20)
      .list()  # returns generated User instances when schema/model map is provided
)

for u in active_users:
    print(u.id, u.email)
```

---

## Modeling users, roles, and permissions

`User` and `Role` form a many-to-many relationship through a `UserRole` join table.
`Role` and `Permission` are connected the same way via `RolePermission`.

- **`userRoles` / `rolePermissions` resolvers** return join-table rows. Use these when cascading saves or deletes to add or remove associations.
- **`roles` / `permissions` resolvers** traverse those joins and return `Role` or `Permission` records for display.

Define these resolvers in your `onyx.schema.json`:

```json
"resolvers": [
  {
    "name": "roles",
    "resolver": "db.from(\"Role\")\n  .where(\n    inOp(\"id\", \n        db.from(\"UserRole\")\n            .where(eq(\"userId\", this.id))\n            .list()\n            .values('roleId')\n    )\n)\n .list()"
  },
  {
    "name": "profile",
    "resolver": "db.from(\"UserProfile\")\n .where(eq(\"userId\", this.id))\n .firstOrNull()"
  },
  {
    "name": "userRoles",
    "resolver": "db.from(\"UserRole\")\n  .where(eq(\"userId\", this.id))\n  .list()"
  }
]
```

Save a user and attach roles in one operation:

```py
db.cascade("userRoles:UserRole(userId, id)").save("User", {
    "id": "user_126",
    "email": "dana@example.com",
    "userRoles": [
        {"roleId": "role_admin"},
        {"roleId": "role_editor"},
    ],
})
```

Fetch a user with roles and each role's permissions:

```py
detailed = (
    db.from_table("User")
      .resolve("roles.permissions", "profile")
      .first_or_none()
)

# detailed["roles"] -> list[Role]
# detailed["roles"][0]["permissions"] -> list[Permission]
```

Remove a role and its permission links:

```py
db.cascade("rolePermissions").delete("Role", "role_temp")
```

---

## Query helpers at a glance

Importable helpers for conditions and sort:

```py
from onyx_database import (
    eq, neq, within, not_within,
    in_op, not_in,
    between,
    gt, gte, lt, lte,
    like, not_like, contains, not_contains,
    starts_with, not_starts_with, matches, not_matches,
    is_null, not_null,
    asc, desc,
)
```

- Prefer `within` / `not_within` for inclusion checks (supports arrays, comma-separated strings, or inner queries).
- `in_op` / `not_in` remain available for backward compatibility and are exact aliases.

Aggregate / string helpers for `select()` expressions:

```py
from onyx_database import avg, sum, count, min, max, std, variance, median, upper, lower, substring, replace, format, percentile

db.select(avg("age")).from_table(tables.UserProfile).list()   # -> [{"avg(age)": 42}]
db.from_table(tables.User).select("isActive", count("id")).group_by("isActive").list()
db.from_table(tables.User).select("id", format("createdAt", "%tF")).list()
db.from_table(tables.UserProfile).select("id", format("age", "%.1f")).list()
```

When `select()` is used (including aggregates), `list()` returns dictionaries by default to avoid dropping custom field names; pass `model=User` to map records to a model explicitly.

`format(field, formatter)` uses Java `String.format`-style patterns (for example, `%tF` for dates or `%.2f` for numbers) and works with any type supported by the formatter.

### Inner queries (IN/NOT IN with sub-selects)

You can pass another query builder to `within` or `not_within` to create nested filters. The SDK
serializes the inner query (including its table) before sending the request.

```py
from onyx_database import onyx, within, not_within, eq
from myservice.db.generated.tables import tables

db = onyx.init()

# Users that HAVE the admin role
users_with_admin = (
    db.from_table(tables.User)
      .where(
          within(
              "id",
              db.select("userId").from_table(tables.UserRole).where(eq("roleId", "role-admin")),
          )
      )
      .list()
)

# Roles that DO NOT include a specific permission
roles_missing_permission = (
    db.from_table(tables.Role)
      .where(
          not_within(
              "id",
              db.from_table(tables.RolePermission).where(eq("permissionId", "perm-manage-users")),
          )
      )
      .list()
)
```

### Full-text (Lucene) search

Use `.search(...)` on a builder or the `search` predicate helper to add a `MATCHES` condition against the `__full_text__` pseudo-field. `db.search(...)` sets `table = "ALL"` and seeds a query builder with that condition (extras like `partition`, `pageSize`, `nextPage` remain querystring params when provided).

```py
from onyx_database import onyx, search, eq
from myservice.db.generated.tables import tables

db = onyx.init()

# Table-specific
db.from_table(tables.User).search("Text", 4.4).list()
db.from_table(tables.User).search("Text").list()              # sends "minScore": null

# Across all tables
db.search("Text", 4.4).list()
db.search("Text").list()                                      # sends "minScore": null

# Combine with structured filters
db.from_table(tables.User).where(search("text", 4.4)).and_(eq("status", "active")).list()
```

Practical Lucene examples:

```py
# Lucene phrase + boolean within one table
lucene_query = '"product engineer" AND remote'
db.from_table(tables.User).search(lucene_query, 0).list()

# All-table search with a phrase branch OR an email wildcard
lucene_all_query = '("product manager" AND remote) OR email:*.ux*'
db.search(lucene_all_query).list()
```

Example request bodies emitted by the SDK:

- Table search with `minScore`

```json
{
  "type": "SelectQuery",
  "conditions": {
    "criteria": {
      "field": "__full_text__",
      "operator": "MATCHES",
      "value": { "queryText": "Text", "minScore": 4.4 }
    },
    "conditionType": "SingleCondition"
  },
  "distinct": false,
  "table": "Table"
}
```

- Table search with `minScore: null`

```json
{
  "type": "SelectQuery",
  "conditions": {
    "criteria": {
      "field": "__full_text__",
      "operator": "MATCHES",
      "value": { "queryText": "Text", "minScore": null }
    },
    "conditionType": "SingleCondition"
  },
  "distinct": false,
  "table": "Table"
}
```

- All tables via `db.search("Text", 4.4)`

```json
{
  "type": "SelectQuery",
  "conditions": {
    "criteria": {
      "field": "__full_text__",
      "operator": "MATCHES",
      "value": { "queryText": "Text", "minScore": 4.4 }
    },
    "conditionType": "SingleCondition"
  },
  "distinct": false,
  "table": "ALL"
}
```

- All tables with `minScore: null`

```json
{
  "type": "SelectQuery",
  "conditions": {
    "criteria": {
      "field": "__full_text__",
      "operator": "MATCHES",
      "value": { "queryText": "Text", "minScore": null }
    },
    "conditionType": "SingleCondition"
  },
  "distinct": false,
  "table": "ALL"
}
```

- Combined search predicate with another filter

```json
{
  "type": "SelectQuery",
  "conditions": {
    "operator": "AND",
    "conditions": [
      {
        "criteria": {
          "field": "__full_text__",
          "operator": "MATCHES",
          "value": { "queryText": "text", "minScore": 4.4 }
        },
        "conditionType": "SingleCondition"
      },
      {
        "criteria": {
          "field": "status",
          "operator": "EQUAL",
          "value": "active"
        },
        "conditionType": "SingleCondition"
      }
    ],
    "conditionType": "CompoundCondition"
  },
  "distinct": false,
  "table": "Table"
}
```

---

## Usage examples with `User`, `Role`, `Permission`

> The examples assume your schema has tables named `User`, `Role`, and `Permission`.
> If you generated stubs, prefer `tables.User`, `tables.Role`, etc.

### 1) List (query & paging)

```py
from onyx_database import onyx, eq, contains, asc
from myservice.db.generated.tables import tables

db = onyx.init()

# Fetch first 25 active Users whose email contains "@example.com"
page1 = (
    db.from_table(tables.User)
      .where(eq("status", "active"))
      .and_(contains("email", "@example.com"))
      .order_by(asc("createdAt"))
      .limit(25)
      .page()
)

items = list(page1.items)
while page1.next_page:
    page1 = (
        db.from_table(tables.User)
          .where(eq("status", "active"))
          .and_(contains("email", "@example.com"))
          .order_by(asc("createdAt"))
          .limit(25)
          .page(next_page=page1.next_page)
    )
    items.extend(page1.items)
```

### 1b) First or none

```py
maybe_user = (
    db.from_table(tables.User)
      .where(eq("email", "alice@example.com"))
      .first_or_none()
)
```

### 2) Save (create/update)

```py
# Upsert a single user
db.save("User", {
    "id": "user_123",
    "email": "alice@example.com",
    "status": "active",
})

# Batch upsert Users
db.save("User", [
    {"id": "user_124", "email": "bob@example.com", "status": "active"},
    {"id": "user_125", "email": "carol@example.com", "status": "invited"},
])

# Save many users in batches of 500
db.batch_save("User", large_user_array, batch_size=500)
```

### 3) Delete (by primary key)

```py
db.delete("User", "user_125")

# Delete cascading relationships
db.cascade("rolePermissions").delete("Role", "role_temp")
```

### 4) Delete using query

```py
deleted_count = (
    db.from_table(tables.User)
      .where(eq("status", "inactive"))
      .delete()
)
```

### 5) Schema API

```py
schema = db.get_schema(tables=["User", "Profile"])
history = db.get_schema_history()

db.validate_schema({
    "revisionDescription": "Add profile triggers",
    "entities": [
        {
            "name": "Profile",
            "identifier": {"name": "id", "generator": "UUID"},
            "attributes": [
                {"name": "id", "type": "String", "isNullable": False},
                {"name": "userId", "type": "String", "isNullable": False},
            ],
        }
    ],
})

db.update_schema(
    {
        "revisionDescription": "Publish profile changes",
        "entities": [
            {
                "name": "Profile",
                "identifier": {"name": "id", "generator": "UUID"},
                "attributes": [
                    {"name": "id", "type": "String", "isNullable": False},
                    {"name": "userId", "type": "String", "isNullable": False},
                ],
            }
        ],
    },
    publish=True,
)
```

### 6) Secrets API

```py
secrets = db.list_secrets()
secret = db.get_secret("api-key")

db.put_secret("api-key", {
    "value": "super-secret",
    "purpose": "Access to external API",
})

db.delete_secret("api-key")
```

### 7) Documents API (binary assets)

```py
# Save / upload a document (Base64 content)
doc = {
    "documentId": "logo.png",
    "path": "/brand/logo.png",
    "mimeType": "image/png",
    "content": "iVBORw0KGgoAAA...",  # base64
}
db.save_document(doc)

image = db.get_document("logo.png", width=128, height=128)
db.delete_document("logo.png")
```

### 8) Streaming (live changes)

```py
from onyx_database import onyx, eq

db = onyx.init()

handle = (
    db.from_table("User")
      .where(eq("status", "active"))
      .on_item_added(lambda u: print("USER ADDED", u))
      .on_item_updated(lambda u: print("USER UPDATED", u))
      .on_item_deleted(lambda u: print("USER DELETED", u))
      .on_item(lambda entity, action: print("STREAM EVENT", action, entity))
      .stream(include_query_results=True)
)

# Later, cancel:
handle.cancel()
```

> Debugging: set `ONYX_STREAM_DEBUG=1` to log stream connection details.

---

## Error handling

- **OnyxConfigError** – thrown by `init()` if required connection parameters are missing.
- **OnyxHTTPError** – thrown for non-2xx API responses, with status and message from the server.

Use standard `try/except` patterns:

```py
from onyx_database import onyx
from onyx_database.errors import OnyxConfigError, OnyxHTTPError

try:
    db = onyx.init()
    # ...perform queries...
except (OnyxConfigError, OnyxHTTPError) as err:
    print("Onyx error:", err)
```


## Release workflow

A typical release flow for this repository:

1. Update the version in `onyx_database/_version.py` (or use your preferred versioning tool).
2. Build: `python -m build`
3. Publish: `twine upload dist/*`

---

## Related links

- Onyx website: https://onyx.dev/
- Cloud console: https://cloud.onyx.dev
- Docs hub: https://onyx.dev/documentation/
- Cloud API docs: https://onyx.dev/documentation/api-documentation/

---

## Security

See [SECURITY.md](./SECURITY.md) for our security policy and vulnerability reporting process.

---

## License

MIT © Onyx Dev Tools. See [LICENSE](./LICENSE).

---

> **Keywords:** Onyx Database Python SDK, Onyx Cloud Database, Onyx NoSQL Graph Database client, Python query builder, tables helper, typed database client, Pydantic models, streaming, schema API
