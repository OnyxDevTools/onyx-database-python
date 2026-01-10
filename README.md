
# onyx-database-python (Python)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![codecov](https://codecov.io/gh/OnyxDevTools/onyx-database-python/branch/main/graph/badge.svg)](https://codecov.io/gh/OnyxDevTools/onyx-database-python)
[![PyPI version](https://img.shields.io/pypi/v/onyx-database.svg)](https://pypi.org/project/onyx-database/)

Python client SDK for **Onyx Cloud Database** — a small, typed, builder-pattern API for querying and persisting data in Onyx. Includes:

- A runtime SDK (sync-first; async optional if enabled)
- Credential resolution (explicit config ➜ env ➜ config file chain ➜ home profile)
- Optional **schema code generator** that produces table-safe Pydantic models and a `tables` helper

- **Website:** https://onyx.dev/
- **Cloud Console:** https://cloud.onyx.dev
- **Docs hub:** https://onyx.dev/documentation/
- **Cloud API docs:** https://onyx.dev/documentation/api-documentation/

---

## Getting started (Cloud ➜ keys ➜ connect)

1. **Sign up & create resources** at **https://cloud.onyx.dev**  
   Create an **Organization**, then a **Database**, define your **Schema** (e.g., `User`, `Role`, `Permission`), and create **API Keys**.

2. **Note your connection parameters**:
   You will need to setup an apiKey to connect to your database in the onyx console at <https://cloud.onyx.dev>.  After creting the apiKey, you can download the `onyx-database.json`. Save it to the `config` folder
   The default place where the sdk and the generator and schema cli tools look for the config and schema file is here: 

    ```
    your-project/
    ├── config/
    │   └── onyx-database.json
    └── schema/
        └── onyx.schema.json
    ```

3. **Install the SDK** in your project:

   ```bash
   pip install onyx-database-python
   ```

4. **Initialize the client** using config files, env vars, or explicit config.

> Supports Python **3.11+**.

---

## Install

```bash
pip install onyx-database-python
```

The package installs:

- the importable module: `onyx_database`
- the CLIs: `onyx-py gen` and `onyx-py schema`

### CLI-only global install (recommended)

If you want `onyx-py gen` / `onyx-py schema` available globally without polluting your project venv:

```bash
pipx install onyx-database-python
```

### Install from a repo checkout (local dev)

```bash
# from repo root
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

onyx-py gen --help
onyx-py schema --help
onyx-py info
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
  "baseUrl": "https://api.onyx.dev",
  "databaseId": "YOUR_DATABASE_ID",
  "auth": {
    "type": "inline",
    "apiKey": "YOUR_KEY",
    "apiSecret": "YOUR_SECRET"
  },
  "defaults": {
    "partition": "",
    "requestTimeoutSeconds": 30
  }
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

### Connection handling

Calling `onyx.init()` returns a lightweight client. Configuration is resolved once and cached
for a short TTL (configurable) to avoid repeated credential lookups. Each database instance
keeps a single internal HTTP client (connection pooling is handled by the HTTP library). Reuse
the returned `db` for multiple operations.

---

## Optional: generate Python models from your schema

The package ships a small codegen CLI that emits:

- a `tables` helper (table-name constants)
- a `SCHEMA` metadata mapping
- one Pydantic model per table
- an `__init__.py` that re-exports the generated symbols

Generated models allow extra properties so resolver-attached fields or embedded objects remain safe:
they use Pydantic config `extra="allow"`.

### Generate directly from the API

Generate by downloading the schema from Onyx (using the same credential resolver as `init()`):

```bash
onyx-py gen --source api --out ./onyx --package onyx
```

With `--source api`, `onyx-py gen` calls the Schema API (same as `onyx-py schema get`) using the
standard config chain (env, project file, home profile).

### Generate from a local schema file

Export `onyx.schema.json` from the console and generate locally:

```bash
onyx-py gen --source file --schema ./schema/onyx.schema.json --out ./onyx --package onyx
```

Run it with no flags to use defaults:

- reads `./schema/onyx.schema.json` (or `./onyx.schema.json` if present)
- writes to `./onyx` by default
- if `--package` is omitted, the package name defaults to the final folder name from `--out` (e.g., `--out ./examples/onyx` => package `onyx`)
- use `--models pydantic` to emit Pydantic models (default is plain classes; extra fields allowed)
- CLI flags for HTTP behavior: `--timeout <seconds>`, `--max-retries <n>`, `--retry-backoff <seconds>`
- Subset to stdout: `onyx-py gen --tables User Role` (prints only those entities instead of writing files)

### Emit to multiple output paths

Comma-separated or repeated `--out`:

```bash
onyx-py gen --out ./onyx,./apps/admin/onyx
# or
onyx-py gen --out ./onyx --out ./apps/admin/onyx
```

### Timestamp handling

Timestamp attributes are emitted as `datetime.datetime` by default. When saving, `datetime`
values are automatically serialized to ISO-8601 timestamp strings. Pass:

- `--timestamps string` to keep timestamps as ISO strings in generated models.

---

## Manage schemas from the CLI

Publish or download schema JSON directly via API using the `onyx-py schema` helper:

```bash
# Publish ./schema/onyx.schema.json (publish=true by default)
onyx-py schema publish

# Overwrite ./schema/onyx.schema.json with the remote schema
onyx-py schema get

# Print the remote schema without writing a file
onyx-py schema get --print

# Fetch only selected tables (prints to stdout; does not overwrite files)
onyx-py schema get --tables=User,Profile

# Validate a schema file without publishing
onyx-py schema validate ./schema/onyx.schema.json

# Diff local schema vs API
onyx-py schema diff ./schema/onyx.schema.json
```

When `--tables` is provided, the subset is printed to stdout instead of writing a file.
Otherwise, the CLI writes to `./schema/onyx.schema.json` by default.

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
from onyx_database import avg, sum, count, min, max, std, variance, median, upper, lower, substring, replace, percentile

db.select(avg("age")).from_table(tables.UserProfile).list()   # -> [{"avg(age)": 42}]
db.from_table(tables.User).select("isActive", count("id")).group_by("isActive").list()
```

When `select()` is used (including aggregates), `list()` returns dictionaries by default to avoid dropping custom field names; pass `model=User` to map records to a model explicitly.

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

---

## CLI (codegen + schema)
- From checkout: `python3 -m pip install -e .` exposes `onyx-py`
- Or globally via pipx: `pipx install .`
- Check help: `onyx-py --help`, `onyx-py gen --help`, `onyx-py schema --help`

```
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| Command             | Flags                                       | Defaults / notes                                             |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py info        | --json                                      | Shows DB ID, base URL, masked keys, config path, status.     |
|                     |                                             | Uses standard config resolution.                             |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py schema get  | --out <path>                                | Writes ./schema/onyx.schema.json if --out not set.           |
|                     | --tables a,b                                | Creates schema/ if missing. --print skips writing.           |
|                     | --print                                     |                                                              |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py schema publish | --schema <path>                          | Reads ./schema/onyx.schema.json (or ./onyx.schema.json)      |
|                        |                                           | by default; publishes to API.                               |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py schema validate | --schema <path>                         | Validates local schema file (same default path as publish).  |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py schema diff | --schema <path>                             | Diffs local schema (same default path as publish) vs remote. |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
| onyx-py gen         | --source api|file (default file)            | Generates models/tables/schema helpers.                      |
|                     | --schema <path> (default ./schema/onyx.schema.json | Multiple --out allowed (repeat or space-separated).   |
|                     |    or ./onyx.schema.json)                   | Default package inferred from final folder name of each      |
|                     | --out <paths...> (default ./onyx)           | --out when --package is omitted.                             |
|                     | --package <name>                            |                                                              |
|                     | --timestamps datetime|string|number         | Default timestamps: datetime.                                |
+---------------------+---------------------------------------------+--------------------------------------------------------------+
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

> **Keywords:** Onyx Database Python SDK, Onyx Cloud Database, Onyx NoSQL Graph Database client, Python query builder, tables helper, schema code generation, typed database client, Pydantic models, streaming, schema CLI
