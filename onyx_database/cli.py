"""CLI entrypoint for `onyx-py` (schema + codegen)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from . import onyx
from .config import resolve_config, _candidate_paths, OnyxConfigError
from .codegen import generate_models


def _load_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _default_schema_path() -> Path:
    candidates = [Path("schema/onyx.schema.json"), Path("onyx.schema.json")]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _detect_config_path() -> Path | None:
    env_path = os.environ.get("ONYX_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    for c in _candidate_paths(None):
        if c.exists():
            return c
    return None


def handle_info(args: argparse.Namespace) -> int:
    config_path = _detect_config_path()
    schema_env = os.environ.get("ONYX_SCHEMA_PATH")
    schema_path = Path(schema_env) if schema_env else _default_schema_path()

    env_present = any(
        os.environ.get(k)
        for k in ["ONYX_DATABASE_ID", "ONYX_DATABASE_API_KEY", "ONYX_DATABASE_API_SECRET", "ONYX_DATABASE_BASE_URL"]
    )
    source_guess = "file" if (config_path and Path(config_path).exists()) else ("env" if env_present else "default")

    def _mask(value: str | None) -> str | None:
        if not value:
            return None
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}...{value[-2:]}"

    info: Dict[str, Any] = {
        "configPath": str(config_path) if config_path else None,
        "configPathExists": bool(config_path and Path(config_path).exists()),
        "schemaPath": str(schema_path),
        "schemaPathExists": schema_path.exists(),
        "source": source_guess,
    }

    try:
        resolved = resolve_config()
        info.update(
            {
                "baseUrl": resolved.base_url,
                "databaseId": resolved.database_id,
                "partition": resolved.partition,
                "requestLoggingEnabled": resolved.request_logging_enabled,
                "responseLoggingEnabled": resolved.response_logging_enabled,
                "apiKeyMasked": _mask(resolved.api_key),
                "apiSecretMasked": _mask(resolved.api_secret),
                "connection": "ok",
            }
        )
    except OnyxConfigError as exc:
        info["configError"] = str(exc)
        info["connection"] = f"error: {exc}"

    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
    else:
        if "configError" in info:
            print(f"Config error: {info['configError']}")
        else:
            print(f"Database ID: {info.get('databaseId')} (source: {info.get('source')})")
            print(f"Base URL   : {info.get('baseUrl')} (source: {info.get('source')})")
            print(f"API Key    : {info.get('apiKeyMasked')} (source: {info.get('source')})")
            print(f"API Secret : {info.get('apiSecretMasked')} (source: {info.get('source')})")
            print(f"Config file: {info.get('configPath')}")
            print(f"Connection : {info.get('connection')}")
    return 0


def handle_schema(args: argparse.Namespace) -> int:
    action: str = args.action
    try:
        db = onyx.init(
            request_timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff,
        )
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1

    if action == "get":
        tables: list[str] = []
        if args.tables:
            tables = [t.strip() for t in args.tables.split(",") if t.strip()]
        schema = db.get_schema(tables=tables if tables else None)
        if args.print_only or args.tables:
            print(json.dumps(schema, indent=2))
            return 0
        out_path = Path(args.out or _default_schema_path())
        _write_json(out_path, schema)
        print(f"schema written to {out_path}")
        return 0

    schema_path = Path(args.schema or _default_schema_path())
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    local_schema = _load_json(schema_path)

    if action == "publish":
        res = db.update_schema(local_schema, publish=True)
        print(json.dumps(res, indent=2))
        return 0

    if action == "validate":
        try:
            res = db.validate_schema(local_schema)
        except Exception as exc:  # validation API should respond 200 + errors list; if not, treat as invalid
            print(f"Schema at {schema_path} is INVALID:")
            print(f"  validation call failed: {exc}")
            return 1

        errors = []
        valid_flag = None
        if isinstance(res, dict):
            errors = res.get("errors") or res.get("validationErrors") or []
            valid_flag = res.get("valid")

        if errors or valid_flag is False:
            print(f"Schema at {schema_path} is INVALID:")
            if errors:
                print(json.dumps(errors, indent=2))
            elif valid_flag is False:
                print("  validation returned valid=false")
            return 1

        print(f"Schema at {schema_path} is valid.")
        return 0

    if action == "diff":
        res = db.diff_schema(local_schema)
        # Pretty print in a human-friendly, non-JSON style
        print("newTables:", res.get("added_tables", []))
        print("removedTables:", res.get("removed_tables", []))
        print("changedTables:")
        for table in res.get("changed_tables", []):
            print(f"  - name: \"{table.get('name')}\"")
            attrs = table.get("attributes") or {}
            print("    attributes:")
            print(f"      added: {attrs.get('added', [])}")
            print(f"      removed: {attrs.get('removed', [])}")
            changed_attrs = attrs.get("changed", [])
            if changed_attrs:
                print("      changed:")
                for ch in changed_attrs:
                    print(f"        - name: \"{ch.get('name')}\"")
                    from_attr = ch.get("from") or {}
                    to_attr = ch.get("to") or {}
                    diffs = []
                    for key in sorted(set(from_attr.keys()) | set(to_attr.keys())):
                        if from_attr.get(key) != to_attr.get(key):
                            diffs.append(f"{key} ({from_attr.get(key)} -> {to_attr.get(key)})")
                    if diffs:
                        print(f"          change: {', '.join(diffs)}")
                    else:
                        print("          change: none")
            else:
                print("      changed: []")
        return 0

    raise ValueError(f"Unknown schema action: {action}")


def handle_gen(args: argparse.Namespace) -> int:
    source = args.source
    schema_path = Path(args.schema) if args.schema else _default_schema_path()
    out_paths = args.out or ["./onyx"]
    timestamp_mode = args.timestamps

    try:
        db = onyx.init(
            request_timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff,
        )
    except Exception as exc:
        print(f"Config error: {exc}")
        return 1
    if source == "api":
        schema = db.get_schema()
    else:
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        schema = _load_json(schema_path)

    if args.tables:
        # subset to stdout only
        subset = {**schema, "entities": [e for e in schema.get("entities", []) if e.get("name") in args.tables]}
        print(json.dumps(subset, indent=2))
        return 0

    for out in out_paths:
        out_path = Path(out)
        pkg = args.package
        if not pkg:
            pkg = out_path.stem if out_path.suffix else out_path.name
        generate_models(schema, out_path, package=pkg, timestamp_mode=timestamp_mode, models_mode=args.models)
        print(f"generated models at {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onyx-py", description="Onyx Database Python SDK CLI")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("gen", help="Generate Python helpers/models from schema")
    gen.add_argument("--source", choices=["api", "file"], default="file")
    gen.add_argument("--schema", help="Path to schema JSON (when --source=file)")
    gen.add_argument("--out", nargs="+", help="Output path(s) (dir or .py file). Default: ./generated")
    gen.add_argument("--package", help="Package name when writing a package (optional)")
    gen.add_argument("--timestamps", choices=["datetime", "string", "number"], default="datetime", help="Timestamp annotation mode")
    gen.add_argument("--models", choices=["plain", "pydantic"], default="plain", help="Model generation style (plain classes or Pydantic BaseModel)")
    gen.add_argument("--timeout", type=float, help="Request timeout (seconds)")
    gen.add_argument("--max-retries", type=int, help="Max retries for GET/query requests")
    gen.add_argument("--retry-backoff", type=float, help="Initial retry backoff in seconds")
    gen.add_argument("--tables", nargs="+", help="When provided, print a subset of entities to stdout instead of writing files")

    schema = sub.add_parser("schema", help="Schema management")
    schema.add_argument("action", choices=["get", "publish", "validate", "diff"])
    schema.add_argument("--schema", help="Local schema path (default: ./schema/onyx.schema.json or ./onyx.schema.json)")
    schema.add_argument("--out", help="Output path for `schema get` (default: ./schema/onyx.schema.json)")
    schema.add_argument("--tables", help="Comma-separated table names (for get)")
    schema.add_argument("--timeout", type=float, help="Request timeout (seconds)")
    schema.add_argument("--max-retries", type=int, help="Max retries for GET/query requests")
    schema.add_argument("--retry-backoff", type=float, help="Initial retry backoff in seconds")
    schema.add_argument("--print", dest="print_only", action="store_true", help="Print only for get")

    info = sub.add_parser("info", help="Show resolved configuration and paths")
    info.add_argument("--json", action="store_true", help="Print JSON output")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "schema":
        return handle_schema(args)
    if args.command == "gen":
        return handle_gen(args)
    if args.command == "info":
        return handle_info(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
