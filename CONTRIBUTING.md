# Contributing to onyx-database-python

Thanks for helping build the Onyx Database Python SDK. This guide walks you from checkout to shipping changes with the same flows used in the TypeScript SDK.

# Quick start on macOS (fresh machine)
```bash

# first time setup
xcode-select --install                                    # command line tools (once)
brew install python@3.11 pipx                             # Python + pipx from Homebrew
python3 -m venv .venv && source .venv/bin/activate        # isolate from PEP 668 protections
python -m pip install --upgrade pip                       # upgrade pip inside the venv
git clone https://github.com/OnyxDevTools/onyx-database-python.git
cd onyx-database-python

# building and installing locally
python -m pip install -e .                                # install SDK + CLI locally
python -m py_compile $(find onyx_database -name '*.py')   # quick sanity check
# Download onyx-database.json from https://cloud.onyx.dev and save to ./config/onyx-database.json
onyx-py info                                              # user this to verify your installation and connection
onyx-py schema get                                        # fetch schema (writes ./schema/onyx.schema.json)
onyx-py gen --out ./onyx                                  # generate the onyx/ package used by examples
python examples/seed.py                                   # creates sample data that the examples depend on
python examples/query/basic.py                            # run a sample
 # Optional: set ONYX_DEBUG=true to log requests/responses
```

## If you make a code change, you just need to: 
pip install -e .
python -m py_compile $(find onyx_database -name '*.py')
## or this is done for you if you run an example using the vscode/launch.json config `debug example`


##  Writing code with the SDK
- Initialize using your config: 
  ```py
  from onyx_database import onyx
  db = onyx.init()  # uses onyx-database.json via the resolver chain
  ```
- Simple save and query (drop into `examples/` or your app):
  ```py
  from onyx_database import eq, asc

  db.save("User", {"id": "user_1", "email": "a@example.com", "status": "active"})
  users = (
      db.from_table("User")
        .where(eq("status", "active"))
        .order_by(asc("createdAt"))
        .limit(10)
        .list()
  )
  ```

## Configuring credentials
- Preferred locations (checked in order): `./config/onyx-database.json`, then `./onyx-database.json`
- Obtain the file from https://cloud.onyx.dev (API Keys -> download `onyx-database.json`) and place it under `./config/`
- Fallbacks: set `ONYX_CONFIG_PATH` to an absolute/relative JSON file; home profiles are also read if present.
- Schema defaults: `./schema/onyx.schema.json` or `./onyx.schema.json`

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

## 10) Release / publish the library
- Bump version in `pyproject.toml` (and any `_version.py` once added)
- Build: `python -m build`
- Publish: `twine upload dist/*`

## 11) Install the public package
- `pip install onyx-database`
- For CLI-only without touching a venv: `pipx install onyx-database`

## CLI tools (schema + codegen)
Install the CLI (`onyx-py`) locally or globally, then use `schema` / `gen`:

Install:
- In repo venv: `python -m pip install -e .`
- Global (isolated) from this checkout: `pipx install .` (PyPI name not published yet). If pipx warns about PATH, run `pipx ensurepath`
- Verify: `onyx-py --help`

Schema commands:
- Download remote schema: `onyx-py schema get` (writes `./schema/onyx.schema.json` by default)
- Print only: `onyx-py schema get --print`
- Publish local schema: `onyx-py schema publish --schema ./schema/onyx.schema.json`
- Validate/diff: `onyx-py schema validate <path>`, `onyx-py schema diff <path>`

Codegen commands:
- From API: `onyx-py gen --source api --out ./onyx --package onyx`
- From file: `onyx-py gen --source file --schema ./schema/onyx.schema.json --out ./onyx --package onyx`
- Multiple outputs: `onyx-py gen --out ./onyx --out ./apps/admin/onyx`
- Timestamp mode: `--timestamps string` to keep ISO strings in generated models.
- Validation models: `--models pydantic` to emit Pydantic models (default is plain classes; extra fields allowed).
- HTTP flags: `--timeout <seconds>`, `--max-retries <n>`, `--retry-backoff <seconds>`
- Subset to stdout: `onyx-py gen --tables User Role` (prints those entities instead of writing files)

Config/env knobs:
- `ONYX_CONFIG_PATH` to point to a specific `onyx-database.json` (otherwise uses `./config/onyx-database.json`, then `./onyx-database.json`, then home profile).
- `ONYX_SCHEMA_PATH` to point to a schema file for `schema` and `gen --source file`.
- `ONYX_DEBUG=true` to log request/response bodies and credential resolution.
