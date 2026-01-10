"""Error types surfaced by the Onyx Database SDK."""

from __future__ import annotations

from typing import Any, Optional


class OnyxError(Exception):
    """Base error for the SDK."""


class OnyxConfigError(OnyxError):
    """Raised when configuration is missing or invalid."""


class OnyxHTTPError(OnyxError):
    """Raised for non-2xx HTTP responses."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        status_text: Optional[str] = None,
        data: Any = None,
        raw: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.status_text = status_text
        self.data = data
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"OnyxHTTPError(status={self.status}, message={self.args[0]!r})"


class OnyxUnauthorizedError(OnyxHTTPError):
    """401/403 unauthorized or forbidden."""


class OnyxNotFoundError(OnyxHTTPError):
    """404 not found."""


class OnyxRateLimitedError(OnyxHTTPError):
    """429 too many requests."""


class OnyxClientError(OnyxHTTPError):
    """4xx client errors not covered by a more specific type."""


class OnyxServerError(OnyxHTTPError):
    """5xx server errors."""


class OnyxTimeoutError(OnyxHTTPError):
    """Request timed out."""
