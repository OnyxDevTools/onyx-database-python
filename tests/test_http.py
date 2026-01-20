import json
import urllib.error
import unittest

from onyx_database.http import HttpClient, serialize_dates
from onyx_database.errors import (
    OnyxUnauthorizedError,
    OnyxNotFoundError,
    OnyxRateLimitedError,
    OnyxServerError,
    OnyxClientError,
    OnyxTimeoutError,
)
import datetime


class FakeHttp(HttpClient):
    """HttpClient with injectable responses for testing."""

    def __init__(self, responses, **kwargs):
        super().__init__(
            "https://api.example.com",
            "key",
            "secret",
            request_timeout_seconds=kwargs.get("request_timeout_seconds"),
            max_retries=kwargs.get("max_retries"),
            retry_backoff_seconds=kwargs.get("retry_backoff_seconds", 0),
        )
        self.responses = list(responses)
        self.calls = 0

    def _do_request(self, method, url, headers, payload):
        self.calls += 1
        if not self.responses:
            raise RuntimeError("no more responses")
        item = self.responses.pop(0)
        if isinstance(item, urllib.error.URLError):
            msg = str(item.reason or item)
            if "timed out" in msg.lower():
                raise OnyxTimeoutError(msg, None, None, None, None)
            raise item
        if isinstance(item, Exception):
            raise item
        return item


class HttpTests(unittest.TestCase):
    def test_datetime_serialization_to_rfc3339_millis(self):
        now = datetime.datetime(2024, 1, 2, 3, 4, 5, 678901, tzinfo=datetime.timezone.utc)
        serialized = serialize_dates(now)
        self.assertEqual(serialized, "2024-01-02T03:04:05.678Z")

    def test_retries_on_500_then_succeeds(self):
        responses = [
            (500, "err", {"Content-Type": "application/json"}, json.dumps({"error": {"message": "boom"}})),
            (200, "ok", {"Content-Type": "application/json"}, json.dumps({"ok": True})),
        ]
        http = FakeHttp(responses, max_retries=2, retry_backoff_seconds=0)
        res = http.request("GET", "/data/test")
        self.assertEqual(res.get("ok"), True)
        self.assertEqual(http.calls, 2)

    def test_status_maps_to_specific_errors(self):
        cases = [
            (401, OnyxUnauthorizedError),
            (403, OnyxUnauthorizedError),
            (404, OnyxNotFoundError),
            (429, OnyxRateLimitedError),
            (500, OnyxServerError),
            (400, OnyxClientError),
        ]
        for status, err_cls in cases:
            http = FakeHttp(
                [(status, "err", {"Content-Type": "application/json"}, json.dumps({"error": {"message": "x"}}))],
                max_retries=1,
            )
            with self.assertRaises(err_cls):
                http.request("GET", "/data/test")

    def test_timeout_raises_timeout_error(self):
        err = urllib.error.URLError("timed out")
        http = FakeHttp([err], request_timeout_seconds=0.01, max_retries=1, retry_backoff_seconds=0)
        with self.assertRaises(OnyxTimeoutError):
            http.request("GET", "/data/test")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
