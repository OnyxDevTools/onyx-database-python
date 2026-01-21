"""Configuration resolution chain for the Onyx Database Python SDK."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time

from .errors import OnyxConfigError

DEFAULT_BASE_URL = "https://api.onyx.dev"
DEFAULT_AI_BASE_URL = "https://ai.onyx.dev"
DEFAULT_AI_MODEL = "onyx"
DEFAULT_CACHE_TTL_SECONDS = 5 * 60
DEFAULT_TIMEOUT_SECONDS = None  # keep default behavior (blocking) unless set
DEFAULT_MAX_RETRIES = None  # fall back to HttpClient logic (GET/query -> 3)
DEFAULT_RETRY_BACKOFF_SECONDS = 0.1


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _sanitize_base_url(url: str) -> str:
    return url.rstrip("/")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
        # remove accidental newlines that may break JSON parsing
        cleaned = content.replace("\r", "").replace("\n", "")
        return json.loads(cleaned)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - edge cases
        raise OnyxConfigError(f"Failed to read {path}: {exc}") from exc


def _extract_auth(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    api_key = config.get("apiKey") or config.get("api_key")
    api_secret = config.get("apiSecret") or config.get("api_secret")
    auth = config.get("auth") or {}
    auth_type = auth.get("type") or auth.get("authType")
    if auth:
        api_key = auth.get("apiKey") or auth.get("api_key") or api_key
        api_secret = auth.get("apiSecret") or auth.get("api_secret") or api_secret
        if auth_type and auth_type != "inline":
            raise OnyxConfigError(
                f"Unsupported auth type '{auth_type}'. Inline API key/secret is required in this build."
            )
    return api_key, api_secret


def _normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    base_url = raw.get("baseUrl") or raw.get("base_url")
    ai_base_url = raw.get("aiBaseUrl") or raw.get("ai_base_url")
    default_model = raw.get("defaultModel") or raw.get("default_model")
    database_id = raw.get("databaseId") or raw.get("database_id")
    api_key, api_secret = _extract_auth(raw)
    partition = (
        raw.get("partition")
        or (raw.get("defaults") or {}).get("partition")
        or (raw.get("defaultPartition"))
    )
    request_logging_enabled = raw.get("requestLoggingEnabled") or raw.get("request_logging_enabled")
    response_logging_enabled = raw.get("responseLoggingEnabled") or raw.get("response_logging_enabled")
    ttl = raw.get("ttl") or (raw.get("defaults") or {}).get("ttl")
    request_timeout_seconds = raw.get("requestTimeoutSeconds") or raw.get("request_timeout_seconds")
    max_retries = raw.get("maxRetries") or raw.get("max_retries")
    retry_backoff_seconds = raw.get("retryBackoffSeconds") or raw.get("retry_backoff_seconds")
    return _drop_none(
        {
            "base_url": base_url,
            "ai_base_url": ai_base_url,
            "default_model": default_model,
            "database_id": database_id,
            "api_key": api_key,
            "api_secret": api_secret,
            "partition": partition,
            "request_logging_enabled": request_logging_enabled,
            "response_logging_enabled": response_logging_enabled,
            "ttl": ttl,
            "request_timeout_seconds": request_timeout_seconds,
            "max_retries": max_retries,
            "retry_backoff_seconds": retry_backoff_seconds,
        }
    )


def _read_env(target_id: Optional[str]) -> Dict[str, Any]:
    env = os.environ
    env_id = env.get("ONYX_DATABASE_ID")
    if target_id and env_id and env_id != target_id:
        return {}
    data = _drop_none(
        {
            "base_url": env.get("ONYX_DATABASE_BASE_URL"),
            "ai_base_url": env.get("ONYX_AI_BASE_URL"),
            "default_model": env.get("ONYX_DEFAULT_MODEL"),
            "database_id": env_id,
            "api_key": env.get("ONYX_DATABASE_API_KEY"),
            "api_secret": env.get("ONYX_DATABASE_API_SECRET"),
            "request_timeout_seconds": env.get("ONYX_REQUEST_TIMEOUT_SECONDS"),
            "max_retries": env.get("ONYX_MAX_RETRIES"),
            "retry_backoff_seconds": env.get("ONYX_RETRY_BACKOFF_SECONDS"),
        }
    )
    return data


def _candidate_paths(database_id: Optional[str]) -> Tuple[Path, ...]:
    cwd = Path.cwd()
    candidates = []
    if database_id:
        candidates.append(cwd / f"onyx-database-{database_id}.json")
    candidates.append(cwd / "config" / "onyx-database.json")
    candidates.append(cwd / "onyx-database.json")

    home = Path.home()
    onyx_dir = home / ".onyx"
    if database_id:
        candidates.append(onyx_dir / f"onyx-database-{database_id}.json")
    candidates.append(onyx_dir / "onyx-database.json")
    candidates.append(home / "onyx-database.json")
    return tuple(candidates)


_config_cache: Dict[str, Any] = {}


@dataclass
class ResolvedConfig:
    base_url: str
    ai_base_url: str
    default_model: str
    database_id: str
    api_key: str
    api_secret: str
    partition: Optional[str]
    request_logging_enabled: bool
    response_logging_enabled: bool
    ttl_seconds: int
    request_timeout_seconds: Optional[float]
    max_retries: Optional[int]
    retry_backoff_seconds: Optional[float]


def clear_config_cache() -> None:
    """Clear cached configuration resolution."""
    _config_cache.clear()


def resolve_config(explicit: Optional[Dict[str, Any]] = None) -> ResolvedConfig:
    """
    Resolve configuration using precedence:
      explicit config > env vars > ONYX_CONFIG_PATH > project config > home profile.
    """
    explicit = _normalize_config(explicit or {})
    ttl_seconds = int(explicit.get("ttl") or DEFAULT_CACHE_TTL_SECONDS)

    cache_key = json.dumps(explicit, sort_keys=True)
    cached = _config_cache.get(cache_key)
    if cached and cached["expires_at"] > time.time():
        return cached["value"]

    config_path_env = os.environ.get("ONYX_CONFIG_PATH")
    cfg_from_path = {}
    if config_path_env:
        p = Path(config_path_env)
        if not p.is_absolute():
            p = Path.cwd() / p
        cfg_from_path = _normalize_config(_read_json(p))

    env_cfg = _read_env(explicit.get("database_id") or cfg_from_path.get("database_id"))

    target_id = explicit.get("database_id") or env_cfg.get("database_id") or cfg_from_path.get("database_id")

    file_cfg: Dict[str, Any] = {}
    for candidate in _candidate_paths(target_id):
        if candidate.exists():
            try:
                file_cfg = _normalize_config(_read_json(candidate))
                break
            except FileNotFoundError:
                continue

    merged: Dict[str, Any] = {}
    for src in (file_cfg, cfg_from_path, env_cfg, explicit):
        merged.update({k: v for k, v in src.items() if v is not None})

    base_url = merged.get("base_url") or DEFAULT_BASE_URL
    ai_base_url = merged.get("ai_base_url") or DEFAULT_AI_BASE_URL
    default_model_raw = merged.get("default_model")
    if isinstance(default_model_raw, str) and default_model_raw.strip():
        default_model = default_model_raw.strip()
    else:
        default_model = DEFAULT_AI_MODEL
    database_id = merged.get("database_id")
    api_key = merged.get("api_key")
    api_secret = merged.get("api_secret")
    if not database_id:
        raise OnyxConfigError("database_id is required (env ONYX_DATABASE_ID or config file)")
    if not api_key or not api_secret:
        raise OnyxConfigError(
            "api_key and api_secret are required (env ONYX_DATABASE_API_KEY / ONYX_DATABASE_API_SECRET or config file)"
        )

    env_debug = os.environ.get("ONYX_DEBUG") == "true"
    request_logging_enabled = bool(merged.get("request_logging_enabled") or env_debug)
    response_logging_enabled = bool(merged.get("response_logging_enabled") or env_debug)
    timeout_val = merged.get("request_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout_seconds = float(timeout_val) if timeout_val is not None else DEFAULT_TIMEOUT_SECONDS
    except Exception:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    max_retries_val = merged.get("max_retries", DEFAULT_MAX_RETRIES)
    try:
        max_retries_int = int(max_retries_val) if max_retries_val is not None else DEFAULT_MAX_RETRIES
    except Exception:
        max_retries_int = DEFAULT_MAX_RETRIES
    backoff_val = merged.get("retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS)
    try:
        backoff_seconds = float(backoff_val) if backoff_val is not None else DEFAULT_RETRY_BACKOFF_SECONDS
    except Exception:
        backoff_seconds = DEFAULT_RETRY_BACKOFF_SECONDS

    resolved = ResolvedConfig(
        base_url=_sanitize_base_url(base_url),
        ai_base_url=_sanitize_base_url(ai_base_url),
        default_model=default_model,
        database_id=database_id,
        api_key=api_key,
        api_secret=api_secret,
        partition=merged.get("partition"),
        request_logging_enabled=request_logging_enabled,
        response_logging_enabled=response_logging_enabled,
        ttl_seconds=ttl_seconds,
        request_timeout_seconds=timeout_seconds,
        max_retries=max_retries_int,
        retry_backoff_seconds=backoff_seconds,
    )

    _config_cache[cache_key] = {"value": resolved, "expires_at": time.time() + ttl_seconds}
    return resolved
