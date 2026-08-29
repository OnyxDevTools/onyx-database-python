"""HTTP client wrapper with retry, logging, and entity wire negotiation."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .config import WireFormat
from .entity_wire import (
    MESSAGE_PACK_ACCEPT,
    MESSAGE_PACK_MEDIA_TYPE,
    pack_entity,
    unpack_entity,
)
from .errors import (
    OnyxClientError,
    OnyxConfigError,
    OnyxHTTPError,
    OnyxNotFoundError,
    OnyxRateLimitedError,
    OnyxServerError,
    OnyxTimeoutError,
    OnyxUnauthorizedError,
)


def parse_json_allow_nan(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r"(:\s*)(NaN|Infinity|-Infinity)(\s*[,}])", r"\1null\3", text)
        return json.loads(fixed)


def serialize_dates(value: Any) -> Any:
    # Normalize datetime objects to RFC3339 with millisecond precision.
    if isinstance(value, datetime.datetime):
        try:
            dt = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
            dt = dt.astimezone(datetime.timezone.utc)
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except Exception:
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass
    # Support plain Python objects (e.g., generated models) by using their __dict__.
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return {k: serialize_dates(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [serialize_dates(v) for v in value]
    if isinstance(value, tuple):
        return [serialize_dates(v) for v in value]
    if isinstance(value, set):
        return [serialize_dates(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_dates(v) for k, v in value.items()}
    return value


class HttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        *,
        default_headers: Optional[Dict[str, str]] = None,
        request_logging_enabled: bool = False,
        response_logging_enabled: bool = False,
        request_timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_backoff_seconds: Optional[float] = None,
    ) -> None:
        if not base_url:
            raise OnyxConfigError("baseUrl is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.defaults = dict(default_headers or {})
        env_debug = os.environ.get("ONYX_DEBUG") == "true"
        self.request_logging_enabled = request_logging_enabled or env_debug
        self.response_logging_enabled = response_logging_enabled or env_debug
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds if retry_backoff_seconds is not None else 0.1

    def headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        merged = {
            "x-onyx-key": self.api_key,
            "x-onyx-secret": self.api_secret,
            "Accept": "application/json",
            "Content-Type": "application/json",
            # Explicit UA helps avoid overly aggressive WAF/browser-signature blocks.
            "User-Agent": "onyx-database-python",
            **self.defaults,
        }
        if extra:
            merged.update(extra)
        merged.pop("x-onyx-key", None)
        merged.pop("x-onyx-secret", None)
        merged.update(
            {
                "x-onyx-key": self.api_key,
                "x-onyx-secret": self.api_secret,
            }
        )
        return merged

    def _log_request(self, method: str, url: str, headers: Dict[str, str], body: Any) -> None:
        if not self.request_logging_enabled:
            return
        print(f"{method} {url}")
        redacted = {**headers, "x-onyx-secret": "[REDACTED]"}
        print("Headers:", redacted)
        if body is not None:
            if isinstance(body, str):
                print(body)
            elif isinstance(body, bytes):
                print(f"<{len(body)} bytes>")
            else:
                try:
                    print(json.dumps(serialize_dates(body)))
                except Exception:
                    print(repr(body))

    def _log_response(
        self,
        status: int,
        reason: str,
        raw: str | bytes,
        content_type: str = "",
    ) -> None:
        if not self.response_logging_enabled:
            return
        print(f"{status} {reason}".strip())
        if isinstance(raw, bytes):
            if raw:
                if content_type.lower().split(";", 1)[0].strip() == MESSAGE_PACK_MEDIA_TYPE:
                    print(f"<{len(raw)} MessagePack bytes>")
                else:
                    print(raw.decode("utf-8", errors="replace"))
        elif raw.strip():
            print(raw)

    def _do_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: Optional[bytes],
    ) -> Tuple[int, str, Dict[str, str], bytes]:
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            timeout = self.request_timeout_seconds
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
                status = resp.getcode() or 0
                reason = resp.reason or ""
                hdrs = {k: v for k, v in resp.headers.items()}
                return status, reason, hdrs, raw_bytes
        except urllib.error.HTTPError as err:
            raw_bytes = err.read()
            hdrs = {k: v for k, v in (err.headers or {}).items()}
            return err.code or 0, err.reason or "", hdrs, raw_bytes
        except urllib.error.URLError as err:
            # Detect timeouts explicitly
            message = str(err.reason or err)
            if "timed out" in message.lower():
                raise OnyxTimeoutError(message, None, None, None, None)
            raise OnyxHTTPError(message, None, None, None, None)

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        extra_headers: Optional[Dict[str, str]] = None,
        *,
        wire_format: WireFormat | str = WireFormat.JSON,
    ) -> Any:
        if not path.startswith("/"):
            raise OnyxConfigError("path must start with /")
        url = f"{self.base_url}{path}"
        headers = self.headers(extra_headers)
        selected_wire_format = WireFormat.parse(wire_format)
        use_message_pack = selected_wire_format is WireFormat.MESSAGE_PACK

        payload: Optional[bytes] = None
        if body is not None:
            if use_message_pack:
                headers["Accept"] = MESSAGE_PACK_ACCEPT
                headers["Content-Type"] = MESSAGE_PACK_MEDIA_TYPE
                payload = pack_entity(body)
            elif isinstance(body, (str, bytes)):
                payload = body if isinstance(body, bytes) else body.encode("utf-8")
            else:
                payload = json.dumps(serialize_dates(body)).encode("utf-8")
        else:
            if use_message_pack:
                headers["Accept"] = MESSAGE_PACK_ACCEPT
                headers.pop("Content-Type", None)
            elif "Content-Type" in headers and extra_headers is None:
                headers.pop("Content-Type", None)

        self._log_request(method, url, headers, body if body is not None else "")

        is_query = "/query/" in path and not re.search(r"/query/(update|delete)/", path)
        can_retry = method.upper() == "GET" or is_query
        max_attempts = (self.max_retries or 3) if can_retry else 1
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            status = 0
            reason = ""
            raw: str | bytes = b""
            try:
                status, reason, resp_headers, raw = self._do_request(method, url, headers, payload)
                content_type = next(
                    (value for key, value in resp_headers.items() if key.lower() == "content-type"),
                    "",
                )
                self._log_response(status, reason, raw, content_type)
                raw_bytes = raw if isinstance(raw, bytes) else raw.encode("utf-8")
                is_message_pack = content_type.lower().split(";", 1)[0].strip() == MESSAGE_PACK_MEDIA_TYPE
                if not raw_bytes:
                    data = ""
                    raw_text = ""
                elif is_message_pack:
                    data = unpack_entity(raw_bytes)
                    raw_text = repr(raw_bytes)
                else:
                    encoding = "utf-8"
                    match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
                    if match:
                        encoding = match.group(1).strip('"')
                    try:
                        raw_text = raw_bytes.decode(encoding, errors="replace")
                    except LookupError:
                        raw_text = raw_bytes.decode("utf-8", errors="replace")
                    is_json = bool(raw_text.strip()) and (
                        "application/json" in content_type.lower()
                        or raw_text.strip().startswith(("{", "["))
                    )
                    data = parse_json_allow_nan(raw_text) if is_json else raw_text
                if status < 200 or status >= 300:
                    msg = (
                        data.get("error", {}).get("message")
                        if isinstance(data, dict)
                        else f"{status} {reason}"
                    )
                    if can_retry and status >= 500 and attempt + 1 < max_attempts:
                        time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                        continue
                    err_cls = OnyxHTTPError
                    if status == 401 or status == 403:
                        err_cls = OnyxUnauthorizedError
                    elif status == 404:
                        err_cls = OnyxNotFoundError
                    elif status == 429:
                        err_cls = OnyxRateLimitedError
                    elif status >= 500:
                        err_cls = OnyxServerError
                    elif status >= 400:
                        err_cls = OnyxClientError
                    raise err_cls(str(msg), status, reason, data, raw_text)
                return data
            except Exception as err:
                last_error = err
                retryable = can_retry and (
                    not isinstance(err, OnyxHTTPError) or (getattr(err, "status", None) is not None and getattr(err, "status") >= 500)
                )
                if attempt + 1 < max_attempts and retryable:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                break
        if last_error:
            raise last_error
        raise OnyxHTTPError("Request failed")

    def open_stream(
        self,
        path: str,
        *,
        method: str = "PUT",
        body: Optional[str | bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> urllib.response.addinfourl:
        if not path.startswith("/"):
            raise OnyxConfigError("path must start with /")
        url = f"{self.base_url}{path}"
        hdrs = self.headers(headers)
        payload = body.encode("utf-8") if isinstance(body, str) else body
        req = urllib.request.Request(url, data=payload, headers=hdrs, method=method)
        try:
            return urllib.request.urlopen(req)
        except urllib.error.HTTPError as err:
            raw_bytes = err.read()
            raw_text = raw_bytes.decode(err.headers.get_content_charset() or "utf-8", errors="replace")
            raise OnyxHTTPError(str(raw_text or err), err.code, err.reason, raw_text, raw_text)
        except urllib.error.URLError as err:
            raise OnyxHTTPError(str(err), None, None, None, None)


class AsyncHttpClient:
    """Async wrapper around HttpClient using thread executors (stdlib only)."""

    def __init__(self, sync_client: HttpClient):
        self._sync = sync_client

    async def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        extra_headers: Optional[Dict[str, str]] = None,
        *,
        wire_format: WireFormat | str = WireFormat.JSON,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.request(method, path, body, extra_headers, wire_format=wire_format),
        )
