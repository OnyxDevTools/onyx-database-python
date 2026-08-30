import asyncio
import unittest
from typing import Any

from onyx_database import SchemaUpsertRequest
from onyx_database.config import clear_config_cache
from onyx_database.onyx import OnyxDatabase
from onyx_database.onyx_async import OnyxDatabaseAsync


def _client_config():
    return {
        "base_url": "https://api.example.com",
        "database_id": "db",
        "api_key": "key",
        "api_secret": "secret",
    }


class RecordingSchemaHttp:
    def __init__(self, get_response=None):
        self.get_response = get_response
        self.calls = []

    def request(self, method, path, body=None, *args, **kwargs):
        self.calls.append((method, path, body))
        if method == "GET":
            return self.get_response
        return body


class RecordingAsyncSchemaHttp:
    def __init__(self, get_response=None):
        self.get_response = get_response
        self.calls = []

    async def request(self, method, path, body=None, *args, **kwargs):
        self.calls.append((method, path, body))
        if method == "GET":
            return self.get_response
        return body


def _searchable_schema(search_support="SEMANTIC") -> dict[str, Any]:
    return {
        "revisionDescription": "Configure article search",
        "entities": [
            {
                "name": "Article",
                "type": "SEARCHABLE",
                "searchSupport": search_support,
                "entityText": "generated source",
                "attributes": [
                    {"name": "id", "type": "String", "isNullable": False},
                    {"name": "body", "type": "String", "isNullable": False},
                ],
                "customExtension": {"preserved": True},
            }
        ],
    }


class SchemaSearchSupportTests(unittest.TestCase):
    def setUp(self):
        clear_config_cache()

    def tearDown(self):
        clear_config_cache()

    def test_typed_schema_request_exposes_search_support(self):
        schema: SchemaUpsertRequest = {
            "entities": [
                {
                    "name": "Article",
                    "type": "SEARCHABLE",
                    "searchSupport": "LEXICAL",
                }
            ]
        }
        self.assertEqual("LEXICAL", schema["entities"][0]["searchSupport"])

    def test_sync_validate_update_and_get_preserve_search_support(self):
        schema = _searchable_schema()
        remote_schema = _searchable_schema("LEXICAL")
        http = RecordingSchemaHttp(remote_schema)
        db = OnyxDatabase(_client_config())
        db._http = http

        validated = db.validate_schema(schema)
        updated = db.update_schema(schema, publish=True)
        fetched = db.get_schema()

        self.assertEqual("SEMANTIC", validated["entities"][0]["searchSupport"])
        self.assertEqual("SEMANTIC", updated["entities"][0]["searchSupport"])
        self.assertEqual("db", updated["databaseId"])
        self.assertEqual("LEXICAL", fetched["entities"][0]["searchSupport"])
        self.assertNotIn("entityText", validated["entities"][0])
        self.assertNotIn("entityText", updated["entities"][0])
        self.assertNotIn("entityText", fetched["entities"][0])
        self.assertTrue(validated["entities"][0]["customExtension"]["preserved"])
        self.assertIn("entityText", schema["entities"][0])
        self.assertIn("entityText", remote_schema["entities"][0])
        self.assertEqual(
            [
                ("POST", "/schemas/db/validate"),
                ("PUT", "/schemas/db?publish=true"),
                ("GET", "/schemas/db"),
            ],
            [(method, path) for method, path, _ in http.calls],
        )

    def test_async_validate_update_and_get_preserve_search_support(self):
        async def run():
            schema = _searchable_schema("BOTH")
            remote_schema = _searchable_schema("SEMANTIC")
            http = RecordingAsyncSchemaHttp(remote_schema)
            db = OnyxDatabaseAsync(_client_config())
            db._http = http

            validated = await db.validate_schema(schema)
            updated = await db.update_schema(schema, publish=True)
            fetched = await db.get_schema()

            self.assertEqual("BOTH", validated["entities"][0]["searchSupport"])
            self.assertEqual("BOTH", updated["entities"][0]["searchSupport"])
            self.assertEqual("db", updated["databaseId"])
            self.assertEqual("SEMANTIC", fetched["entities"][0]["searchSupport"])
            self.assertNotIn("entityText", validated["entities"][0])
            self.assertNotIn("entityText", updated["entities"][0])
            self.assertNotIn("entityText", fetched["entities"][0])
            self.assertIn("entityText", schema["entities"][0])
            self.assertIn("entityText", remote_schema["entities"][0])

        asyncio.run(run())

    def test_sync_diff_defaults_missing_search_support_to_both(self):
        remote = {
            "entities": [
                {"name": "Article", "type": "SEARCHABLE", "attributes": []},
                {"name": "Account", "attributes": []},
            ]
        }
        http = RecordingSchemaHttp(remote)
        db = OnyxDatabase(_client_config())
        db._http = http

        compatible = db.diff_schema(
            {
                "entities": [
                    {
                        "name": "Article",
                        "type": "SEARCHABLE",
                        "searchSupport": "BOTH",
                        "attributes": [],
                    },
                    {"name": "Account", "type": "DEFAULT", "attributes": []},
                ]
            }
        )
        self.assertEqual([], compatible["changed_tables"])

        support_changed = db.diff_schema(
            {
                "entities": [
                    {
                        "name": "Article",
                        "type": "SEARCHABLE",
                        "searchSupport": "LEXICAL",
                        "attributes": [],
                    },
                    {"name": "Account", "attributes": []},
                ]
            }
        )
        self.assertEqual(
            {
                "name": "Article",
                "searchSupport": {"from": "BOTH", "to": "LEXICAL"},
            },
            support_changed["changed_tables"][0],
        )

    def test_diff_normalizes_empty_optional_entity_fields(self):
        remote = {
            "entities": [
                {
                    "name": "Article",
                    "type": "SEARCHABLE",
                    "attributes": [],
                }
            ]
        }
        http = RecordingSchemaHttp(remote)
        db = OnyxDatabase(_client_config())
        db._http = http

        diff = db.diff_schema(
            {
                "entities": [
                    {
                        "name": "Article",
                        "type": "SEARCHABLE",
                        "searchSupport": "BOTH",
                        "partition": "   ",
                        "identifier": None,
                        "attributes": [],
                        "indexes": [],
                        "resolvers": [],
                        "triggers": [],
                    }
                ]
            }
        )

        self.assertEqual([], diff["changed_tables"])

    def test_async_diff_detects_entity_type_from_effective_defaults(self):
        async def run():
            remote = {
                "entities": [
                    {
                        "name": "Article",
                        "type": "DEFAULT",
                        "attributes": [],
                    }
                ]
            }
            http = RecordingAsyncSchemaHttp(remote)
            db = OnyxDatabaseAsync(_client_config())
            db._http = http

            ignored = await db.diff_schema(
                {
                    "entities": [
                        {
                            "name": "Article",
                            "type": "DEFAULT",
                            "searchSupport": "BOTH",
                            "attributes": [],
                        }
                    ]
                }
            )
            self.assertEqual([], ignored["changed_tables"])

            changed = await db.diff_schema(
                {
                    "entities": [
                        {
                            "name": "Article",
                            "type": "SEARCHABLE",
                            "searchSupport": "SEMANTIC",
                            "attributes": [],
                        }
                    ]
                }
            )
            self.assertEqual(
                {
                    "name": "Article",
                    "type": {"from": "DEFAULT", "to": "SEARCHABLE"},
                    "searchSupport": {"from": "BOTH", "to": "SEMANTIC"},
                },
                changed["changed_tables"][0],
            )

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
