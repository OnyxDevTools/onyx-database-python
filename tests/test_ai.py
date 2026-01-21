import json
import os
import unittest

from onyx_database.config import (
    DEFAULT_AI_BASE_URL,
    DEFAULT_AI_MODEL,
    clear_config_cache,
    resolve_config,
)
from onyx_database.http import HttpClient
from onyx_database.onyx import OnyxDatabase


class FakeStream:
    def __init__(self, lines):
        self._lines = iter(lines)
        self.closed = False

    def readline(self):
        try:
            return next(self._lines)
        except StopIteration:
            return b""

    def close(self):
        self.closed = True


class FakeAiHttp(HttpClient):
    def __init__(self, stream=None, responses=None):
        super().__init__("https://ai.test", "key", "secret")
        self.stream = stream
        self.responses = list(responses or [])
        self.calls = []
        self.last_open_stream = None

    def request(self, method, path, body=None, extra_headers=None):
        self.calls.append((method, path, body, extra_headers))
        if self.responses:
            return self.responses.pop(0)
        return {"ok": True}

    def open_stream(self, path, *, method="PUT", body=None, headers=None):
        self.last_open_stream = {"path": path, "method": method, "body": body, "headers": headers}
        self.calls.append((method, path, body, headers))
        return self.stream


class AiConfigTests(unittest.TestCase):
    def setUp(self):
        self.prev_ai_base = os.environ.pop("ONYX_AI_BASE_URL", None)
        self.prev_default_model = os.environ.pop("ONYX_DEFAULT_MODEL", None)
        clear_config_cache()

    def tearDown(self):
        clear_config_cache()
        if self.prev_ai_base is None:
            os.environ.pop("ONYX_AI_BASE_URL", None)
        else:
            os.environ["ONYX_AI_BASE_URL"] = self.prev_ai_base
        if self.prev_default_model is None:
            os.environ.pop("ONYX_DEFAULT_MODEL", None)
        else:
            os.environ["ONYX_DEFAULT_MODEL"] = self.prev_default_model

    def test_ai_base_url_defaults(self):
        cfg = resolve_config({"databaseId": "db1", "apiKey": "k", "apiSecret": "s"})
        self.assertEqual(cfg.ai_base_url, DEFAULT_AI_BASE_URL)

    def test_ai_base_url_from_env(self):
        os.environ["ONYX_AI_BASE_URL"] = "https://ai.example.com/"
        clear_config_cache()
        cfg = resolve_config({"databaseId": "db1", "apiKey": "k", "apiSecret": "s"})
        self.assertEqual(cfg.ai_base_url, "https://ai.example.com")

    def test_default_model_defaults(self):
        cfg = resolve_config({"databaseId": "db1", "apiKey": "k", "apiSecret": "s"})
        self.assertEqual(cfg.default_model, DEFAULT_AI_MODEL)

    def test_default_model_from_env(self):
        os.environ["ONYX_DEFAULT_MODEL"] = "onyx-chat"
        clear_config_cache()
        cfg = resolve_config({"databaseId": "db1", "apiKey": "k", "apiSecret": "s"})
        self.assertEqual(cfg.default_model, "onyx-chat")


class AiClientTests(unittest.TestCase):
    def setUp(self):
        clear_config_cache()
        self.db = OnyxDatabase(
            {
                "base_url": "https://api.test",
                "ai_base_url": "https://ai.test",
                "database_id": "db1",
                "api_key": "k",
                "api_secret": "s",
            }
        )

    def test_chat_shorthand_returns_content(self):
        fake_ai = FakeAiHttp(
            responses=[{"choices": [{"message": {"content": "hello"}}]}]
        )
        self.db._ai_http = fake_ai
        res = self.db.chat("hi")
        self.assertEqual(res, "hello")
        self.assertEqual(fake_ai.calls[0][1], "/v1/chat/completions?databaseId=db1")
        body = fake_ai.calls[0][2]
        self.assertEqual(body["model"], "onyx")
        self.assertEqual(body["messages"][0]["role"], "user")

    def test_chat_shorthand_raw_returns_response(self):
        fake_ai = FakeAiHttp(
            responses=[{"choices": [{"message": {"content": "hello"}}]}]
        )
        self.db._ai_http = fake_ai
        res = self.db.chat("hi", raw=True)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["choices"][0]["message"]["content"], "hello")

    def test_chat_client_create_returns_response(self):
        fake_ai = FakeAiHttp(
            responses=[{"choices": [{"message": {"content": "hello"}}]}]
        )
        self.db._ai_http = fake_ai
        client = self.db.chat()
        res = client.create(
            {"model": "onyx-chat", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(res["choices"][0]["message"]["content"], "hello")

    def test_streaming_chat_builds_query_and_returns_chunks(self):
        stream = FakeStream(
            [
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
                b"data: [DONE]\n",
            ]
        )
        fake_ai = FakeAiHttp(stream=stream)
        self.db._ai_http = fake_ai

        chunks = list(
            self.db.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="onyx-chat",
                stream=True,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "hello")
        self.assertIsNotNone(fake_ai.last_open_stream)
        self.assertEqual(fake_ai.last_open_stream["method"], "POST")
        self.assertEqual(fake_ai.last_open_stream["path"], "/v1/chat/completions?databaseId=db1")
        body = json.loads(fake_ai.last_open_stream["body"])
        self.assertTrue(body.get("stream"))

    def test_get_models_uses_ai_client(self):
        fake_ai = FakeAiHttp(responses=[{"object": "list", "data": []}])
        self.db._ai_http = fake_ai
        res = self.db.get_models()
        self.assertEqual(res.get("object"), "list")
        self.assertEqual(fake_ai.calls[0][1], "/v1/models")

    def test_ai_namespace_get_models_uses_ai_client(self):
        fake_ai = FakeAiHttp(responses=[{"object": "list", "data": []}])
        self.db._ai_http = fake_ai
        res = self.db.ai.get_models()
        self.assertEqual(res.get("object"), "list")
        self.assertEqual(fake_ai.calls[0][1], "/v1/models")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
