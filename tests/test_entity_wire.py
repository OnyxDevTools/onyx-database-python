import asyncio
import datetime
import io
import json
import threading
import unittest
from collections import UserDict
from pathlib import Path
from unittest.mock import patch

import msgpack

from onyx_database import entity_wire
from onyx_database.config import WireFormat, clear_config_cache, resolve_config
from onyx_database.entity_wire import (
    MESSAGE_PACK_ACCEPT,
    MESSAGE_PACK_MEDIA_TYPE,
    EntityWireError,
    iter_unpack_entities,
    pack_entity,
    unpack_entity,
)
from onyx_database.errors import OnyxClientError, OnyxNotFoundError, OnyxServerError
from onyx_database.http import AsyncHttpClient, HttpClient
from onyx_database.onyx import OnyxDatabase
from onyx_database.onyx_async import OnyxDatabaseAsync
from onyx_database.stream import open_entity_stream

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingHttp(HttpClient):
    def __init__(self, responses):
        super().__init__("https://api.example.com", "key", "secret", max_retries=1)
        self.responses = list(responses)
        self.requests = []

    def _do_request(self, method, url, headers, payload):
        self.requests.append((method, url, headers, payload))
        return self.responses.pop(0)


class MessagePackResponse(io.BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.headers = {"Content-Type": MESSAGE_PACK_MEDIA_TYPE}


class EntityWireCodecTests(unittest.TestCase):
    def test_canonical_cross_client_golden_fixture(self):
        fixture = json.loads((FIXTURES / "entity-wire-v1.json").read_text(encoding="utf-8"))
        expected_hex = (FIXTURES / "entity-wire-v1.msgpack.hex").read_text(encoding="ascii").strip()

        payload = pack_entity(fixture)

        self.assertEqual(payload.hex(), expected_hex)
        self.assertEqual(unpack_entity(bytes.fromhex(expected_hex)), fixture)
        compact_json = json.dumps(fixture, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertLess(len(payload), len(compact_json))

    def test_recursive_dates_and_cycles_follow_entity_json_semantics(self):
        root = {
            "when": datetime.datetime(2024, 1, 2, 3, 4, 5, 678901, tzinfo=datetime.UTC),
            "items": [1, True, None, {"nested": "✓"}],
        }
        root["self"] = root

        decoded = unpack_entity(pack_entity(root))

        self.assertEqual(decoded["when"], "2024-01-02T03:04:05.678Z")
        self.assertEqual(decoded["self"], {"cyclicReference": "detected"})
        self.assertEqual(decoded["items"][3]["nested"], "✓")

    def test_rejects_non_portable_values(self):
        bad_values = [
            {1: "non-string key"},
            {"binary": b"bytes"},
            {"tooLarge": 2**63},
            {"notFinite": float("nan")},
        ]
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(EntityWireError):
                pack_entity(value)

        with self.assertRaises(EntityWireError):
            unpack_entity(msgpack.packb(msgpack.ExtType(1, b"x")))
        with self.assertRaises(EntityWireError):
            unpack_entity(pack_entity({"ok": True}) + pack_entity({"extra": True}))

    def test_rejects_excessive_depth(self):
        value = None
        for _ in range(130):
            value = [value]
        with self.assertRaises(EntityWireError):
            pack_entity(value)

    def test_map_keys_count_toward_node_limit_in_all_codec_paths(self):
        value = {"one": None, "two": None}
        normalized_mapping = UserDict(value)
        payload = msgpack.packb(value, use_bin_type=True)

        # root map + two keys + two values = five wire nodes
        with patch.object(entity_wire, "MAX_NODES", 5):
            self.assertEqual(unpack_entity(pack_entity(value)), value)
            self.assertEqual(unpack_entity(pack_entity(normalized_mapping)), value)
            self.assertEqual(unpack_entity(payload), value)

        with patch.object(entity_wire, "MAX_NODES", 4):
            with self.assertRaises(EntityWireError):
                pack_entity(value)  # already-native validation path
            with self.assertRaises(EntityWireError):
                pack_entity(normalized_mapping)  # generic normalization path
            with self.assertRaises(EntityWireError):
                unpack_entity(payload)  # decoded validation path

    def test_duplicate_map_keys_are_rejected_before_node_validation(self):
        # map(2): "a": 1, "a": 2. The five wire nodes previously collapsed
        # to a three-node dict before Python's recursive validator saw them.
        payload = bytes.fromhex("82a16101a16102")

        with patch.object(entity_wire, "MAX_NODES", 4):
            with self.assertRaisesRegex(EntityWireError, "duplicate key"):
                unpack_entity(payload)
            with self.assertRaisesRegex(EntityWireError, "duplicate key"):
                list(iter_unpack_entities(io.BytesIO(payload)))

    def test_concatenated_stream_skips_no_values_in_codec(self):
        payload = pack_entity(None) + pack_entity({"action": "CREATE", "entity": {"id": "1"}})
        self.assertEqual(
            list(iter_unpack_entities(io.BytesIO(payload))),
            [None, {"action": "CREATE", "entity": {"id": "1"}}],
        )


class EntityWireTransportTests(unittest.TestCase):
    def test_messagepack_request_headers_body_and_response(self):
        response = {"records": [{"id": "one"}], "nextPage": None}
        http = RecordingHttp(
            [(200, "OK", {"content-type": f"{MESSAGE_PACK_MEDIA_TYPE}; charset=binary"}, pack_entity(response))]
        )

        result = http.request("PUT", "/data/db/query/User", {"type": "SelectQuery"}, wire_format="msgpack")

        self.assertEqual(result, response)
        _, _, headers, payload = http.requests[0]
        self.assertEqual(headers["Accept"], MESSAGE_PACK_ACCEPT)
        self.assertEqual(headers["Content-Type"], MESSAGE_PACK_MEDIA_TYPE)
        self.assertEqual(unpack_entity(payload), {"type": "SelectQuery"})

    def test_messagepack_get_has_accept_but_no_content_type(self):
        http = RecordingHttp([(200, "OK", {"Content-Type": MESSAGE_PACK_MEDIA_TYPE}, pack_entity({"id": "one"}))])

        result = http.request("GET", "/data/db/User/one", wire_format=WireFormat.MESSAGE_PACK)

        self.assertEqual(result["id"], "one")
        headers = http.requests[0][2]
        self.assertEqual(headers["Accept"], MESSAGE_PACK_ACCEPT)
        self.assertNotIn("Content-Type", headers)
        self.assertIsNone(http.requests[0][3])

    def test_binary_negotiation_decodes_actual_json_success_and_error(self):
        http = RecordingHttp(
            [
                (200, "OK", {"Content-Type": "application/json"}, b'{"fallback":true}'),
                (400, "Bad Request", {"Content-Type": "application/json"}, b'{"error":{"message":"bad query"}}'),
            ]
        )

        self.assertEqual(
            http.request("GET", "/data/db/User/one", wire_format="msgpack"),
            {"fallback": True},
        )
        with self.assertRaises(OnyxClientError) as context:
            http.request("PUT", "/data/db/query/update/User", {"updates": []}, wire_format="msgpack")
        self.assertEqual(context.exception.data["error"]["message"], "bad query")

    def test_json_remains_the_transport_default(self):
        http = RecordingHttp([(200, "OK", {"Content-Type": "application/json"}, b'{"ok":true}')])

        result = http.request("PUT", "/schemas/db", {"name": "schema"})

        self.assertEqual(result, {"ok": True})
        headers = http.requests[0][2]
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(http.requests[0][3]), {"name": "schema"})

    def test_async_http_uses_same_binary_transport(self):
        response = {"records": [], "nextPage": None}
        sync_http = RecordingHttp(
            [(200, "OK", {"Content-Type": MESSAGE_PACK_MEDIA_TYPE}, pack_entity(response))]
        )

        async def run():
            client = AsyncHttpClient(sync_http)
            result = await client.request(
                "PUT",
                "/data/db/query/User",
                {"type": "SelectQuery"},
                wire_format=WireFormat.MESSAGE_PACK,
            )
            self.assertEqual(result, response)

        asyncio.run(run())


class EntityRouteOptInTests(unittest.TestCase):
    def setUp(self):
        clear_config_cache()

    def tearDown(self):
        clear_config_cache()

    def test_config_defaults_to_json_and_accepts_both_messagepack_spellings(self):
        base = {"database_id": "db", "api_key": "key", "api_secret": "secret"}
        self.assertIs(resolve_config(base).wire_format, WireFormat.JSON)
        clear_config_cache()
        self.assertIs(resolve_config({**base, "wireFormat": "message-pack"}).wire_format, WireFormat.MESSAGE_PACK)
        clear_config_cache()
        self.assertIs(resolve_config({**base, "wire_format": WireFormat.MESSAGE_PACK}).wire_format, WireFormat.MESSAGE_PACK)

    def test_sync_entity_routes_opt_in_but_documents_remain_json(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "wire_format": "msgpack",
            }
        )

        class FakeRequests:
            def __init__(self):
                self.calls = []

            def request(self, method, path, body=None, extra_headers=None, *, wire_format=WireFormat.JSON):
                self.calls.append((method, path, body, extra_headers, wire_format))
                if "/query/count/" in path:
                    return 1
                if "/query/" in path and "/query/update/" not in path and "/query/delete/" not in path:
                    return {"records": [], "nextPage": None}
                return {"id": "one"}

        fake = FakeRequests()
        db._http = fake
        db.create("User", {"id": "created"})
        db.save("User", {"id": "one"})
        db.find_by_id("User", "one")
        db.delete("User", "one")
        db.count("User", {"type": "SelectQuery"}, None)
        db.query_page("User", {"type": "SelectQuery"}, {})
        db.delete_by_query("User", {"type": "SelectQuery"}, None)
        db.update("User", {"type": "UpdateQuery"}, None)
        db.save_document({"documentId": "doc"})

        self.assertEqual(fake.calls[0][0:3], ("POST", "/data/db/User", {"id": "created"}))
        self.assertTrue(all(call[4] is WireFormat.MESSAGE_PACK for call in fake.calls[:8]))
        self.assertIs(fake.calls[8][4], WireFormat.JSON)

    def test_sync_schema_publish_stays_json_and_preserves_native_index_metadata(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "wire_format": "msgpack",
            }
        )

        class FakeRequests:
            def __init__(self):
                self.calls = []

            def request(
                self,
                method,
                path,
                body=None,
                extra_headers=None,
                *,
                wire_format=WireFormat.JSON,
            ):
                self.calls.append((method, path, body, extra_headers, wire_format))
                return {"valid": True}

        schema = {
            "databaseId": "placeholder",
            "entities": [
                {
                    "name": "ActiveDocumentChunk",
                    "type": "SEARCHABLE",
                    "entityText": "generated text must not replace structured metadata",
                    "indexes": [
                        {"name": "corpusId", "type": "DEFAULT"},
                        {"name": "title", "type": "VECTOR"},
                        {"name": "content", "type": "VECTOR"},
                    ],
                }
            ],
        }
        fake = FakeRequests()
        db._http = fake

        db.validate_schema(schema)
        db.update_schema(schema, publish=True)

        self.assertEqual(
            [
                ("POST", "/schemas/db/validate"),
                ("PUT", "/schemas/db?publish=true"),
            ],
            [(call[0], call[1]) for call in fake.calls],
        )
        self.assertTrue(all(call[4] is WireFormat.JSON for call in fake.calls))
        for call in fake.calls:
            entity = call[2]["entities"][0]
            self.assertNotIn("entityText", entity)
            self.assertEqual("SEARCHABLE", entity["type"])
            self.assertEqual(
                {
                    "corpusId": "DEFAULT",
                    "title": "VECTOR",
                    "content": "VECTOR",
                },
                {index["name"]: index["type"] for index in entity["indexes"]},
            )

    def test_sync_atomic_create_rejects_batches_and_old_servers_fail_closed(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
            }
        )
        http = RecordingHttp(
            [(404, "Not Found", {"Content-Type": "application/json"}, b'{"error":{"message":"missing"}}')]
        )
        db._http = http

        with self.assertRaisesRegex(EntityWireError, "exactly one entity object"):
            db.create("User", [{"id": "batch"}])
        self.assertEqual([], http.requests)

        with self.assertRaises(OnyxNotFoundError):
            db.create("User", {"id": "created"})
        self.assertEqual(1, len(http.requests))
        self.assertEqual("POST", http.requests[0][0])

    def test_sync_fenced_save_delete_and_update_use_exact_single_post_wire(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "wire_format": "msgpack",
            }
        )

        class FakeRequests:
            def __init__(self):
                self.calls = []

            def request(self, method, path, body=None, extra_headers=None, *, wire_format=WireFormat.JSON):
                self.calls.append((method, path, body, extra_headers, wire_format))
                return {"applied": True, "affected": len(body.get("entities", [None]))}

        guard = {
            "table": "Lease",
            "id": "head-1",
            "partition": "corpus-a",
            "expected": {"generation": 7, "owner": "worker-a"},
        }
        filters = {
            "conditionType": "SingleCondition",
            "criteria": {"field": "revision", "operator": "EQUAL", "value": "old"},
        }
        fake = FakeRequests()
        db._http = fake

        self.assertEqual(
            {"applied": True, "affected": 1},
            db.fenced_save("Chunk", {"id": "one", "partition": "corpus-a"}, guard=guard),
        )
        db.fenced_delete_where("Chunk", partition="corpus-a", filters=filters, guard=guard)
        db.fenced_update_where(
            "Head",
            partition="corpus-a",
            filters={
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "id",
                            "operator": "EQUAL",
                            "value": "head-1",
                        },
                    },
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "status",
                            "operator": "EQUAL",
                            "value": "STAGING",
                        },
                    },
                ],
            },
            updates={"status": "ACTIVE", "stagedRevisionId": ""},
            guard=guard,
        )

        self.assertEqual(("POST", "/data/db/Chunk/fenced"), fake.calls[0][0:2])
        self.assertEqual(
            {"guard": guard, "operation": "SAVE", "entities": [{"id": "one", "partition": "corpus-a"}]},
            fake.calls[0][2],
        )
        self.assertEqual(
            {"guard": guard, "operation": "DELETE", "partition": "corpus-a", "filters": filters},
            fake.calls[1][2],
        )
        self.assertEqual(
            {
                "guard": guard,
                "operation": "UPDATE",
                "partition": "corpus-a",
                "filters": {
                    "conditionType": "CompoundCondition",
                    "operator": "AND",
                    "conditions": [
                        {
                            "conditionType": "SingleCondition",
                            "criteria": {
                                "field": "id",
                                "operator": "EQUAL",
                                "value": "head-1",
                            },
                        },
                        {
                            "conditionType": "SingleCondition",
                            "criteria": {
                                "field": "status",
                                "operator": "EQUAL",
                                "value": "STAGING",
                            },
                        },
                    ],
                },
                "updates": {"status": "ACTIVE", "stagedRevisionId": ""},
            },
            fake.calls[2][2],
        )
        self.assertTrue(all(call[4] is WireFormat.MESSAGE_PACK for call in fake.calls))

    def test_sync_fenced_validation_and_old_server_fail_closed_without_retry(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "max_retries": 5,
            }
        )
        guard = {"table": "Lease", "id": "one", "expected": {"generation": 1}}
        http = RecordingHttp(
            [(500, "Server Error", {"Content-Type": "application/json"}, b'{"error":{"message":"old server"}}')]
        )
        http.max_retries = 5
        db._http = http

        with self.assertRaisesRegex(EntityWireError, "between 1 and 500"):
            db.fenced_save("Chunk", [], guard=guard)
        with self.assertRaisesRegex(EntityWireError, "between 1 and 500"):
            db.fenced_save("Chunk", [{"id": str(i)} for i in range(501)], guard=guard)
        with self.assertRaisesRegex(EntityWireError, "QueryCondition"):
            db.fenced_delete_where(
                "Chunk",
                partition="corpus-a",
                filters={"revision": "old"},
                guard=guard,
            )
        with self.assertRaisesRegex(EntityWireError, "concrete partition"):
            db.fenced_update_where(
                "Head",
                partition="ALL",
                filters={
                    "conditionType": "SingleCondition",
                    "criteria": {"field": "id", "operator": "EQUAL", "value": "one"},
                },
                updates={"status": "ACTIVE"},
                guard=guard,
            )
        with self.assertRaisesRegex(EntityWireError, "QueryCondition"):
            db.fenced_update_where(
                "Head",
                partition="corpus-a",
                filters={"id": "one"},
                updates={"status": "ACTIVE"},
                guard=guard,
            )
        with self.assertRaisesRegex(EntityWireError, "non-empty object"):
            db.fenced_update_where(
                "Head",
                partition="corpus-a",
                filters={
                    "conditionType": "SingleCondition",
                    "criteria": {"field": "id", "operator": "EQUAL", "value": "one"},
                },
                updates={},
                guard=guard,
            )
        with self.assertRaisesRegex(EntityWireError, "field names"):
            db.fenced_update_where(
                "Head",
                partition="corpus-a",
                filters={
                    "conditionType": "SingleCondition",
                    "criteria": {"field": "id", "operator": "EQUAL", "value": "one"},
                },
                updates={"": "ACTIVE"},
                guard=guard,
            )
        self.assertEqual([], http.requests)

        with self.assertRaises(OnyxServerError):
            db.fenced_update_where(
                "Head",
                partition="corpus-a",
                filters={
                    "conditionType": "SingleCondition",
                    "criteria": {"field": "id", "operator": "EQUAL", "value": "one"},
                },
                updates={"status": "ACTIVE"},
                guard=guard,
            )
        self.assertEqual(1, len(http.requests))
        self.assertEqual("POST", http.requests[0][0])

    def test_async_entity_route_opts_in(self):
        db = OnyxDatabaseAsync(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "wire_format": "msgpack",
            }
        )

        class FakeAsyncRequests:
            def __init__(self):
                self.calls = []

            async def request(self, method, path, body=None, extra_headers=None, *, wire_format=WireFormat.JSON):
                self.calls.append((method, path, body, extra_headers, wire_format))
                return {"id": "one"}

        async def run():
            fake = FakeAsyncRequests()
            db._http = fake
            await db.create("User", {"id": "created"})
            await db.save("User", {"id": "one"})
            self.assertIs(fake.calls[0][4], WireFormat.MESSAGE_PACK)
            self.assertEqual(fake.calls[0][0:3], ("POST", "/data/db/User", {"id": "created"}))
            self.assertIs(fake.calls[1][4], WireFormat.MESSAGE_PACK)

        asyncio.run(run())

    def test_async_atomic_create_rejects_batches_and_old_servers_fail_closed(self):
        db = OnyxDatabaseAsync(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
            }
        )

        class FakeAsyncRequests:
            def __init__(self):
                self.calls = []

            async def request(self, method, path, body=None, extra_headers=None, *, wire_format=WireFormat.JSON):
                self.calls.append((method, path, body, extra_headers, wire_format))
                raise OnyxNotFoundError("missing", 404, "Not Found")

        async def run():
            fake = FakeAsyncRequests()
            db._http = fake
            with self.assertRaisesRegex(EntityWireError, "exactly one entity object"):
                await db.create("User", [{"id": "batch"}])
            self.assertEqual([], fake.calls)

            with self.assertRaises(OnyxNotFoundError):
                await db.create("User", {"id": "created"})
            self.assertEqual(1, len(fake.calls))
            self.assertEqual("POST", fake.calls[0][0])

        asyncio.run(run())

    def test_async_fenced_save_delete_and_update_have_wire_parity(self):
        db = OnyxDatabaseAsync(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
            }
        )
        guard = {"table": "Lease", "id": "one", "expected": {"generation": 1}}
        filters = {
            "conditionType": "SingleCondition",
            "criteria": {"field": "id", "operator": "EQUAL", "value": "one"},
        }

        class FakeAsyncRequests:
            def __init__(self):
                self.calls = []

            async def request(self, method, path, body=None, extra_headers=None, *, wire_format=WireFormat.JSON):
                self.calls.append((method, path, body, extra_headers, wire_format))
                return {"applied": True, "affected": 1}

        async def run():
            fake = FakeAsyncRequests()
            db._http = fake
            saved = await db.fenced_save("Chunk", {"id": "one"}, guard=guard)
            deleted = await db.fenced_delete_where(
                "Chunk",
                partition="corpus-a",
                filters=filters,
                guard=guard,
            )
            updated = await db.fenced_update_where(
                "Head",
                partition="corpus-a",
                filters=filters,
                updates={"status": "ACTIVE"},
                guard=guard,
            )
            self.assertEqual({"applied": True, "affected": 1}, saved)
            self.assertEqual({"applied": True, "affected": 1}, deleted)
            self.assertEqual({"applied": True, "affected": 1}, updated)
            self.assertEqual(["POST", "POST", "POST"], [call[0] for call in fake.calls])
            self.assertEqual(
                [
                    "/data/db/Chunk/fenced",
                    "/data/db/Chunk/fenced",
                    "/data/db/Head/fenced",
                ],
                [call[1] for call in fake.calls],
            )
            self.assertEqual("UPDATE", fake.calls[2][2]["operation"])
            self.assertEqual({"status": "ACTIVE"}, fake.calls[2][2]["updates"])

        asyncio.run(run())

    def test_messagepack_change_stream_skips_nil_sentinel(self):
        db = OnyxDatabase(
            {
                "base_url": "https://api.example.com",
                "database_id": "db",
                "api_key": "key",
                "api_secret": "secret",
                "wire_format": "msgpack",
            }
        )
        received = []
        delivered = threading.Event()
        payload = pack_entity(None) + pack_entity({"action": "CREATE", "entity": {"id": "one"}})
        opened = []

        def open_stream(path, *, method="PUT", body=None, headers=None):
            opened.append((path, method, body, headers))
            return MessagePackResponse(payload)

        db._http.open_stream = open_stream
        handle = db.stream(
            "User",
            {"type": "SelectQuery"},
            include_query_results=True,
            keep_alive=False,
            handlers={"on_item_added": lambda item: (received.append(item), delivered.set())},
        )
        self.assertTrue(delivered.wait(1))
        handle["cancel"]()

        self.assertEqual(received[0], {"id": "one"})
        self.assertEqual(unpack_entity(opened[0][2]), {"type": "SelectQuery"})
        self.assertEqual(opened[0][3]["Accept"], MESSAGE_PACK_ACCEPT)

    def test_entity_stream_uses_json_fallback_without_messagepack_content_type(self):
        payload = b'{"action":"CREATE","entity":{"id":"json-fallback"}}\n'

        for content_type in (None, "application/json; charset=utf-8"):
            with self.subTest(content_type=content_type):
                received = []
                delivered = threading.Event()

                def opener(selected_content_type=content_type):
                    response = io.BytesIO(payload)
                    response.headers = {}
                    if selected_content_type is not None:
                        response.headers["Content-Type"] = selected_content_type
                    return response

                def on_item_added(item, target=received, signal=delivered):
                    target.append(item)
                    signal.set()

                handle = open_entity_stream(
                    opener,
                    handlers={"on_item_added": on_item_added},
                )
                self.assertTrue(delivered.wait(1))
                handle["cancel"]()
                self.assertEqual(received[0], {"id": "json-fallback"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
