import asyncio
import json
import unittest

from onyx_database.http import HttpClient, AsyncHttpClient
from onyx_database.query_results_async import AsyncQueryResults


class FakeHttp(HttpClient):
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

    def _do_request(self, method, url, headers, payload):
        if not self.responses:
            raise RuntimeError("no more responses")
        item = self.responses.pop(0)
        return item


class AsyncTests(unittest.TestCase):
    def test_async_http_client(self):
        responses = [
            (200, "ok", {"Content-Type": "application/json"}, json.dumps({"ok": True})),
        ]
        sync = FakeHttp(responses)
        async_client = AsyncHttpClient(sync)

        async def run():
            res = await async_client.request("GET", "/data/test")
            self.assertEqual(res.get("ok"), True)

        asyncio.run(run())

    def test_async_query_results_iteration(self):
        async def run():
            results = AsyncQueryResults([1, 2, 3], next_page=None, fetcher=None)
            collected = []
            async for item in results:
                collected.append(item)
            self.assertEqual(collected, [1, 2, 3])

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
