import asyncio
import math
import unittest
from unittest.mock import AsyncMock, Mock

from onyx_database.helpers.conditions import (
    is_null,
    not_between,
    not_null,
    search as search_condition,
)
from onyx_database.onyx import OnyxDatabase
from onyx_database.onyx_async import OnyxDatabaseAsync
from onyx_database.query_builder import QueryBuilder
from onyx_database.query_builder_async import AsyncQueryBuilder


class DummyExec:
    pass


class SearchTests(unittest.TestCase):
    def test_null_operators_omit_value_from_wire_query(self):
        qb = QueryBuilder(DummyExec(), table="Table")
        qb.where(is_null("releasedAt")).and_(not_null("createdAt"))

        conditions = qb.to_update_query()["conditions"]["conditions"]

        self.assertEqual(
            conditions[0],
            {
                "conditionType": "SingleCondition",
                "criteria": {"field": "releasedAt", "operator": "IS_NULL"},
            },
        )
        self.assertEqual(
            conditions[1],
            {
                "conditionType": "SingleCondition",
                "criteria": {"field": "createdAt", "operator": "NOT_NULL"},
            },
        )

    def test_query_builder_search_condition(self):
        qb = QueryBuilder(DummyExec(), table="Table")
        qb.search("Text", 4.4)
        query = qb.to_query_object()

        self.assertEqual(query["table"], "Table")
        criteria = query["conditions"]["criteria"]
        self.assertEqual(criteria["field"], "__full_text__")
        self.assertEqual(criteria["operator"], "MATCHES")
        self.assertEqual(criteria["value"], {"queryText": "Text", "minScore": 4.4})

    def test_no_options_search_retains_legacy_wire(self):
        for builder_type in (QueryBuilder, AsyncQueryBuilder):
            with self.subTest(builder=builder_type.__name__):
                criteria = (
                    builder_type(DummyExec(), table="Table")
                    .search("Text")
                    .to_query_object()["conditions"]["criteria"]
                )

                self.assertEqual(criteria["operator"], "MATCHES")
                self.assertEqual(
                    criteria["value"],
                    {"queryText": "Text", "minScore": None},
                )

    def test_legacy_search_keyword_score_retains_exact_wire_shape(self):
        for builder_type in (QueryBuilder, AsyncQueryBuilder):
            with self.subTest(builder=builder_type.__name__):
                criteria = (
                    builder_type(DummyExec(), table="Table")
                    .search("Text", min_score=4.4)
                    .to_query_object()["conditions"]["criteria"]
                )

                self.assertEqual(criteria["operator"], "MATCHES")
                self.assertEqual(
                    criteria["value"], {"queryText": "Text", "minScore": 4.4}
                )

    def test_high_level_search_options_mapping_emits_canonical_wire(self):
        criteria = (
            QueryBuilder(DummyExec(), table="Table")
            .search(
                "how do i calculate cost per horse",
                {
                    "mode": "lexical",
                    "min_score": 0.4,
                    "max_candidates": 500,
                },
            )
            .to_query_object()["conditions"]["criteria"]
        )

        self.assertEqual(criteria["field"], "__full_text__")
        self.assertEqual(criteria["operator"], "SEARCH")
        self.assertEqual(
            criteria["value"],
            {
                "text": "how do i calculate cost per horse",
                "mode": "lexical",
                "match": "any",
                "minScore": 0.4,
                "maxCandidates": 500,
            },
        )

    def test_empty_options_and_mode_less_keywords_use_search_defaults(self):
        empty_options = search_condition("text", {})["value"]
        mode_less_keywords = search_condition(
            "text", match="all", max_candidates=25
        )["value"]

        self.assertEqual(empty_options["mode"], "hybrid")
        self.assertEqual(empty_options["match"], "any")
        self.assertEqual(empty_options["maxCandidates"], 1_000)
        self.assertEqual(mode_less_keywords["mode"], "hybrid")
        self.assertEqual(mode_less_keywords["match"], "all")
        self.assertEqual(mode_less_keywords["maxCandidates"], 25)

    def test_high_level_search_keywords_cover_all_modes_and_async_parity(self):
        builders = (
            QueryBuilder(DummyExec(), table="Table"),
            AsyncQueryBuilder(DummyExec(), table="Table"),
        )

        for builder, mode in zip(builders, ("semantic", "hybrid")):
            with self.subTest(builder=type(builder).__name__, mode=mode):
                criteria = (
                    builder.search(
                        "expense for each animal",
                        mode=mode,
                        match="all",
                        min_score=0.25,
                        max_candidates=75,
                    )
                    .to_query_object()["conditions"]["criteria"]
                )
                self.assertEqual(criteria["operator"], "SEARCH")
                self.assertEqual(
                    criteria["value"],
                    {
                        "text": "expense for each animal",
                        "mode": mode,
                        "match": "all",
                        "minScore": 0.25,
                        "maxCandidates": 75,
                    },
                )

    def test_high_level_search_is_composable_but_read_only(self):
        search_first = (
            QueryBuilder(DummyExec(), table="Table")
            .search("cost per horse", mode="hybrid")
            .where({"field": "active", "operator": "EQUAL", "value": True})
        )
        filter_first = (
            QueryBuilder(DummyExec(), table="Table")
            .where({"field": "active", "operator": "EQUAL", "value": True})
            .search("cost per horse", mode="semantic")
        )

        for builder in (search_first, filter_first):
            with self.subTest(order=builder.to_query_object()["conditions"]):
                conditions = builder.to_query_object()["conditions"]
                self.assertEqual(conditions["operator"], "AND")
                self.assertEqual(len(conditions["conditions"]), 2)
                with self.assertRaisesRegex(ValueError, "SEARCH is read-only"):
                    builder.delete()

        update_builder = QueryBuilder(DummyExec(), table="Table").search(
            "cost per horse", mode="lexical"
        )
        update_builder.set_updates({"active": False})
        with self.assertRaisesRegex(ValueError, "SEARCH is read-only"):
            update_builder.update()

        async_builder = AsyncQueryBuilder(DummyExec(), table="Table").where(
            search_condition("cost per horse", mode="semantic")
        )
        with self.assertRaisesRegex(ValueError, "SEARCH is read-only"):
            asyncio.run(async_builder.delete())

    def test_search_composition_is_recursive_and_order_independent(self):
        structured = {
            "conditionType": "CompoundCondition",
            "operator": "OR",
            "conditions": [
                {
                    "conditionType": "SingleCondition",
                    "criteria": {
                        "field": "active",
                        "operator": "EQUAL",
                        "value": True,
                    },
                },
                {
                    "conditionType": "SingleCondition",
                    "criteria": {
                        "field": "tenantId",
                        "operator": "EQUAL",
                        "value": "tenant-a",
                    },
                },
            ],
        }
        for builder_type in (QueryBuilder, AsyncQueryBuilder):
            with self.subTest(builder=builder_type.__name__, case="structured"):
                builder_type(DummyExec(), table="Table").search(
                    "cost per horse", mode="semantic"
                ).and_(structured)

            two_searches = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "__full_text__",
                            "operator": "SEARCH",
                            "value": {},
                        },
                    },
                    {
                        "conditionType": "CompoundCondition",
                        "operator": "OR",
                        "conditions": [
                            {
                                "conditionType": "SingleCondition",
                                "criteria": {
                                    "field": "__full_text__",
                                    "operator": "SEARCH",
                                    "value": {},
                                },
                            }
                        ],
                    },
                ],
            }
            with (
                self.subTest(builder=builder_type.__name__, case="two-searches"),
                self.assertRaisesRegex(ValueError, "at most once"),
            ):
                builder_type(DummyExec(), table="Table").where(two_searches)

            search_and_legacy = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "__full_text__",
                            "operator": "SEARCH",
                            "value": {},
                        },
                    },
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "__full_text__",
                            "operator": "MATCHES",
                            "value": "legacy",
                        },
                    },
                ],
            }
            with (
                self.subTest(builder=builder_type.__name__, case="raw-mixed"),
                self.assertRaisesRegex(ValueError, "another __full_text__"),
            ):
                builder_type(DummyExec(), table="Table").where(search_and_legacy)

            wrong_target = {
                "conditionType": "CompoundCondition",
                "operator": "AND",
                "conditions": [
                    {
                        "conditionType": "SingleCondition",
                        "criteria": {
                            "field": "body",
                            "operator": "SEARCH",
                            "value": {},
                        },
                    }
                ],
            }
            with (
                self.subTest(builder=builder_type.__name__, case="wrong-target"),
                self.assertRaisesRegex(ValueError, "must target __full_text__"),
            ):
                builder_type(DummyExec(), table="Table").where(wrong_target)

            for first_high_level in (False, True):
                with (
                    self.subTest(
                        builder=builder_type.__name__,
                        case="call-order",
                        first_high_level=first_high_level,
                    ),
                    self.assertRaisesRegex(ValueError, "another __full_text__"),
                ):
                    builder = builder_type(DummyExec(), table="Table")
                    if first_high_level:
                        builder.search("natural language", mode="lexical").search(
                            "legacy"
                        )
                    else:
                        builder.search("legacy").search(
                            "natural language", mode="lexical"
                        )

            with (
                self.subTest(builder=builder_type.__name__, case="second-search"),
                self.assertRaisesRegex(ValueError, "at most once"),
            ):
                builder_type(DummyExec(), table="Table").search(
                    "first", mode="semantic"
                ).search("second", mode="hybrid")

    def test_live_streams_reject_search_and_candidate_admission_locally(self):
        candidate_builders = [
            QueryBuilder(DummyExec(), table="Table").search(
                "natural language", mode="hybrid"
            ),
            QueryBuilder(DummyExec(), table="Table").approximate_search("lexical"),
            QueryBuilder(DummyExec(), table="Table").hnsw_candidates(
                {"calibrationId": 1, "vector": [1]}
            ),
            QueryBuilder(DummyExec(), table="Table").approximate_candidates(
                "tenantId", "tenant-a"
            ),
        ]

        for builder in candidate_builders:
            with (
                self.subTest(operator=builder.to_query_object()["conditions"]),
                self.assertRaisesRegex(ValueError, "live query streams"),
            ):
                builder.on_item(lambda *_: None).stream()

    def test_database_wide_search_does_not_inherit_default_partition(self):
        config = {
            "base_url": "https://api.example.com",
            "database_id": "db",
            "api_key": "key",
            "api_secret": "secret",
            "partition": "tenant-a",
        }
        sync_db = OnyxDatabase(dict(config))
        sync_db._entity_request = Mock(
            return_value={"records": [], "nextPage": None}
        )
        sync_query = sync_db.search("needle", mode="lexical")
        self.assertIsNone(sync_query.to_query_object()["partition"])
        sync_query.list()
        self.assertNotIn("partition=", sync_db._entity_request.call_args.args[1])
        sync_db._entity_request.return_value = 0
        sync_query.count()
        self.assertNotIn("partition=", sync_db._entity_request.call_args.args[1])

        table_query = sync_db.from_table("Document")
        self.assertEqual(table_query.to_query_object()["partition"], "tenant-a")

        async def run_async_search():
            async_db = OnyxDatabaseAsync(dict(config))
            async_db._entity_request = AsyncMock(
                return_value={"records": [], "nextPage": None}
            )
            async_query = async_db.search("needle", mode="semantic")
            self.assertIsNone(async_query.to_query_object()["partition"])
            await async_query.list()
            self.assertNotIn(
                "partition=", async_db._entity_request.call_args.args[1]
            )
            async_db._entity_request.return_value = 0
            await async_query.count()
            self.assertNotIn(
                "partition=", async_db._entity_request.call_args.args[1]
            )

        asyncio.run(run_async_search())

    def test_high_level_search_validation_and_option_conflicts(self):
        invalid_calls = [
            lambda: search_condition(" ", {"mode": "lexical"}),
            lambda: search_condition("text", mode="vectors"),
            lambda: search_condition("text", mode="lexical", match="some"),
            lambda: search_condition("text", mode="semantic", min_score=math.inf),
            lambda: search_condition("text", {"min_score": math.nan}),
            lambda: search_condition("text", mode="semantic", min_score=-0.01),
            lambda: search_condition("text", mode="semantic", min_score=1.01),
            lambda: search_condition("text", {"min_score": -0.01}),
            lambda: search_condition("text", {"minScore": 1.01}),
            lambda: search_condition("text", mode="semantic", max_candidates=0),
            lambda: search_condition("text", mode="semantic", max_candidates=5_001),
            lambda: search_condition("text", {"mode": "semantic", "unknown": 1}),
            lambda: search_condition(
                "text", {"min_score": 0.2, "minScore": 0.2}
            ),
            lambda: search_condition(
                "text", {"mode": "semantic"}, mode="semantic"
            ),
            lambda: search_condition(
                "text", {"mode": "semantic"}, max_candidates=50
            ),
            lambda: search_condition(
                "text", mode="semantic", require_all_terms=False
            ),
        ]

        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_hybrid_requires_two_candidates_but_single_channel_modes_allow_one(self):
        with self.assertRaisesRegex(ValueError, "at least 2 for hybrid"):
            search_condition("text", mode="hybrid", max_candidates=1)

        for mode in ("lexical", "semantic"):
            with self.subTest(mode=mode):
                value = search_condition(
                    "text", mode=mode, max_candidates=1
                )["value"]
                self.assertEqual(value["maxCandidates"], 1)

    def test_search_combines_with_existing_conditions_and_null_score(self):
        qb = QueryBuilder(DummyExec(), table="Table")
        qb.where({"field": "status", "operator": "EQUAL", "value": "active"})
        qb.search("text")
        query = qb.to_query_object()

        conditions = query["conditions"]
        self.assertEqual(conditions["operator"], "AND")
        self.assertEqual(conditions["conditions"][0]["criteria"]["field"], "status")
        search_criteria = conditions["conditions"][1]["criteria"]
        self.assertEqual(search_criteria["field"], "__full_text__")
        self.assertEqual(search_criteria["operator"], "MATCHES")
        self.assertIsNone(search_criteria["value"]["minScore"])

    def test_db_search_sets_table_all(self):
        db = OnyxDatabase(
            {"base_url": "https://api.example.com", "database_id": "db", "api_key": "key", "api_secret": "secret"}
        )
        qb = db.search("needle")
        query = qb.to_query_object()

        self.assertEqual(query["table"], "ALL")
        criteria = query["conditions"]["criteria"]
        self.assertEqual(criteria["operator"], "MATCHES")
        self.assertEqual(criteria["value"], {"queryText": "needle", "minScore": None})

    def test_sync_and_async_database_search_support_high_level_options(self):
        config = {
            "base_url": "https://api.example.com",
            "database_id": "db",
            "api_key": "key",
            "api_secret": "secret",
        }

        sync_query = OnyxDatabase(dict(config)).search(
            "needle", {"mode": "hybrid", "maxCandidates": 25}
        )
        async_query = OnyxDatabaseAsync(dict(config)).search(
            "needle", mode="semantic"
        )

        for query, mode, maximum in (
            (sync_query, "hybrid", 25),
            (async_query, "semantic", 1_000),
        ):
            criteria = query.to_query_object()["conditions"]["criteria"]
            self.assertEqual(query.to_query_object()["table"], "ALL")
            self.assertEqual(criteria["operator"], "SEARCH")
            self.assertEqual(criteria["value"]["mode"], mode)
            self.assertEqual(criteria["value"]["match"], "any")
            self.assertIsNone(criteria["value"]["minScore"])
            self.assertEqual(criteria["value"]["maxCandidates"], maximum)


class SearchHelperTests(unittest.TestCase):
    def test_not_between_helper_shape(self):
        self.assertEqual(
            not_between("score", 10, 20),
            {
                "field": "score",
                "operator": "NOT_BETWEEN",
                "value": [10, 20],
            },
        )

    def test_search_helper_shape(self):
        cond = search_condition("text", 1.5)
        self.assertEqual(cond["field"], "__full_text__")
        self.assertEqual(cond["operator"], "MATCHES")
        self.assertEqual(cond["value"], {"queryText": "text", "minScore": 1.5})

    def test_search_helper_accepts_camel_case_mapping_aliases(self):
        cond = search_condition(
            "text",
            {
                "mode": "lexical",
                "match": "all",
                "minScore": 0.5,
                "maxCandidates": 12,
            },
        )

        self.assertEqual(
            cond["value"],
            {
                "text": "text",
                "mode": "lexical",
                "match": "all",
                "minScore": 0.5,
                "maxCandidates": 12,
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
