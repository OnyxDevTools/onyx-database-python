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

# build and install locally (inside venv)
python -m pip install -e .                                # install SDK locally
python -m py_compile $(find onyx_database -name '*.py')   # quick sanity check
# Download onyx-database.json from https://cloud.onyx.dev and save to ./config/onyx-database.json 
# or use onyx cli to get the schema (see below)

# install the standalone Onyx CLI (global)
brew tap OnyxDevTools/onyx-cli
brew install onyx-cli

# configure + generate stubs used by examples
onyx info                                                 # verify your installation and connection
onyx schema get onyx.schema.json                          # fetch schema (writes ./onyx.schema.json by default)
onyx gen --python --source api --out ./onyx               # generate the onyx/ package used by examples

# run sample data + example script
python examples/seed.py
python examples/query/basic.py
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
- Schema default: `./onyx.schema.json` (used by the Onyx CLI; pass `--schema` if you store it elsewhere)

## Onyx CLI (schema + codegen)
- Install (macOS): `brew tap OnyxDevTools/onyx-cli && brew install onyx-cli`
- Verify connection: `onyx info`
- Fetch schema: `onyx schema get onyx.schema.json`
- Publish/validate/diff: `onyx schema publish|validate|diff onyx.schema.json`
- Generate Python stubs: `onyx gen --python --source api --out ./onyx` (or `--schema ./onyx.schema.json`)
- Print a subset: `onyx schema get --tables=User,Role --print`

The CLI uses the same credential/config resolution chain as the SDK.

## 10) Release / publish the library
- Bump version in `pyproject.toml` (and any `_version.py` once added)
- Build: `python -m build`
- Publish: `twine upload dist/*`

## 11) Install the public package
- `pip install onyx-database`
