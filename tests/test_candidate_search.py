import asyncio
import math
import unittest

from onyx_database import (
    MAX_HNSW_EF_SEARCH,
    MAX_VECTOR_SEARCH_CANDIDATES,
    approximate_candidates,
    approximate_index_candidate_query,
    approximate_search,
    hnsw_candidates,
    hnsw_search_query,
    search,
    semantic_vector_signature,
    vector_search_query,
)
from onyx_database.onyx import OnyxDatabase
from onyx_database.onyx_async import OnyxDatabaseAsync
from onyx_database.query_builder import QueryBuilder
from onyx_database.query_builder_async import AsyncQueryBuilder


class DummyExec:
    pass


def semantic_signature():
    return semantic_vector_signature(
        calibration_id=-9_223_372_036_854_775_808,
        bucket_id=5,
        cells=[1, 2],
        cell_counts=[2, 3],
        fingerprint=[-1],
    )


class CandidateSearchWireTests(unittest.TestCase):
    def test_semantic_signature_is_lossless_and_computes_bands(self):
        signature = semantic_signature()

        self.assertEqual(signature["calibrationId"], "-9223372036854775808")
        self.assertEqual(signature["bucketId"], 5)
        self.assertEqual(signature["cells"], [1, 2])
        self.assertEqual(signature["cellCounts"], [2, 3])
        self.assertEqual(signature["fingerprint"], ["0xffffffffffffffff"])
        self.assertEqual(signature["bands"], ["0x000000000000ffff"] * 4)
        self.assertEqual(signature["boundaryConfidence"], 0.0)

    def test_semantic_signature_accepts_wire_and_snake_case_mappings(self):
        signature = semantic_vector_signature(
            {
                "calibration_id": "17",
                "bucketId": 1,
                "cells": [1],
                "cell_counts": [2],
                "fingerprintWords": ["0x0123456789abcdef"],
                "semanticBands": [0xCDEF, 0x89AB, 0x4567, 0x0123],
                "boundary_confidence": 0.25,
            }
        )

        self.assertEqual(signature["calibrationId"], "17")
        self.assertEqual(
            signature["bands"],
            [
                "0x000000000000cdef",
                "0x00000000000089ab",
                "0x0000000000004567",
                "0x0000000000000123",
            ],
        )
        self.assertEqual(signature["boundaryConfidence"], 0.25)

    def test_vector_query_emits_every_optional_field_with_defaults(self):
        query = vector_search_query({"text": "storm warning"})

        self.assertEqual(
            query,
            {
                "text": "storm warning",
                "semantic": None,
                "minScore": None,
                "nearbyBucketRadius": 1,
                "maxCandidates": 1_000,
                "requireAllTerms": True,
            },
        )

    def test_vector_query_supports_semantic_and_hybrid_parameters(self):
        signature = semantic_signature()
        query = vector_search_query(
            text="hybrid",
            semantic=signature,
            min_score=0.2,
            nearby_bucket_radius=3,
            max_candidates=125,
            require_all_terms=False,
        )

        self.assertEqual(query["semantic"], signature)
        self.assertEqual(query["minScore"], 0.2)
        self.assertEqual(query["nearbyBucketRadius"], 3)
        self.assertEqual(query["maxCandidates"], 125)
        self.assertFalse(query["requireAllTerms"])

    def test_semantic_signature_mapping_is_a_semantic_only_query_convenience(self):
        query = vector_search_query(semantic_signature())

        self.assertIsNone(query["text"])
        self.assertEqual(query["semantic"], semantic_signature())

    def test_hnsw_query_emits_lossless_id_and_derived_ef_search(self):
        query = hnsw_search_query(
            calibration_id="9223372036854775807",
            vector=[3, 4],
            max_candidates=1_250,
            min_score=-0.5,
        )

        self.assertEqual(
            query,
            {
                "calibrationId": "9223372036854775807",
                "vector": [3.0, 4.0],
                "maxCandidates": 1_250,
                "efSearch": 1_250,
                "minScore": -0.5,
                "formatVersion": 1,
            },
        )

    def test_approximate_index_query_normalizes_scalar_and_sequence(self):
        self.assertEqual(
            approximate_index_candidate_query("tenant-a", 17),
            {"values": ["tenant-a"], "maxCandidates": 17},
        )
        self.assertEqual(
            approximate_index_candidate_query(["a", "b"]),
            {"values": ["a", "b"], "maxCandidates": 1_000},
        )

    def test_condition_helpers_emit_native_operators(self):
        self.assertEqual(
            approximate_search(
                "storm warning", max_candidates=32, require_all_terms=False
            ),
            {
                "field": "__full_text__",
                "operator": "SEARCH_CANDIDATES",
                "value": {
                    "text": "storm warning",
                    "semantic": None,
                    "minScore": None,
                    "nearbyBucketRadius": 1,
                    "maxCandidates": 32,
                    "requireAllTerms": False,
                },
            },
        )
        self.assertEqual(
            hnsw_candidates({"calibrationId": 7, "vector": [1]})["operator"],
            "HNSW_CANDIDATES",
        )
        self.assertEqual(
            approximate_candidates("tenantId", ["a", "b"])["operator"], "CANDIDATES"
        )
        self.assertEqual(
            search(vector_search_query(text="text"))["operator"], "MATCHES"
        )

    def test_approximate_search_accepts_a_typed_options_mapping(self):
        condition = approximate_search(
            "storm",
            {
                "minScore": 0.2,
                "maxCandidates": 25,
                "requireAllTerms": False,
            },
        )

        self.assertEqual(condition["value"]["minScore"], 0.2)
        self.assertEqual(condition["value"]["maxCandidates"], 25)
        self.assertFalse(condition["value"]["requireAllTerms"])


class CandidateSearchValidationTests(unittest.TestCase):
    def test_semantic_signature_rejects_invalid_routing_metadata(self):
        valid = {
            "calibrationId": 1,
            "bucketId": 5,
            "cells": [1, 2],
            "cellCounts": [2, 3],
            "fingerprint": [1],
        }
        invalid = [
            {**valid, "calibrationId": 0},
            {**valid, "calibrationId": 1 << 63},
            {**valid, "bucketId": 4},
            {**valid, "bucketId": 1 << 31},
            {**valid, "cells": []},
            {**valid, "cells": [1], "cellCounts": [2, 3]},
            {**valid, "cells": [2, 2]},
            {**valid, "cellCounts": [46_341, 46_341], "cells": [0, 0], "bucketId": 0},
            {**valid, "fingerprint": []},
            {**valid, "fingerprint": [1, 2, 3, 4, 5]},
            {**valid, "bands": [0, 0, 0, 0]},
            {**valid, "boundaryConfidence": math.nan},
            {**valid, "boundaryConfidence": 1.1},
        ]

        for signature in invalid:
            with (
                self.subTest(signature=signature),
                self.assertRaises((TypeError, ValueError)),
            ):
                semantic_vector_signature(signature)

    def test_vector_query_rejects_invalid_optional_parameters(self):
        invalid = [
            {},
            {"text": ""},
            {"text": "text", "minScore": math.inf},
            {"text": "text", "nearbyBucketRadius": -1},
            {"text": "text", "nearbyBucketRadius": 1 << 31},
            {"text": "text", "maxCandidates": 0},
            {"text": "text", "maxCandidates": MAX_VECTOR_SEARCH_CANDIDATES + 1},
            {"text": "text", "requireAllTerms": 1},
        ]

        for query in invalid:
            with self.subTest(query=query), self.assertRaises((TypeError, ValueError)):
                vector_search_query(query)

    def test_hnsw_query_rejects_unbounded_or_invalid_values(self):
        invalid = [
            {"calibrationId": 0, "vector": [1]},
            {"calibrationId": 1 << 63, "vector": [1]},
            {"calibrationId": 1, "vector": []},
            {"calibrationId": 1, "vector": [0, 0]},
            {"calibrationId": 1, "vector": [math.nan]},
            {"calibrationId": 1, "vector": [1], "maxCandidates": 0},
            {"calibrationId": 1, "vector": [1], "maxCandidates": 10, "efSearch": 9},
            {"calibrationId": 1, "vector": [1], "efSearch": MAX_HNSW_EF_SEARCH + 1},
            {"calibrationId": 1, "vector": [1], "minScore": 1.1},
            {"calibrationId": 1, "vector": [1], "formatVersion": 2},
        ]

        for query in invalid:
            with self.subTest(query=query), self.assertRaises((TypeError, ValueError)):
                hnsw_search_query(query)

    def test_approximate_index_query_rejects_invalid_routes_and_bounds(self):
        invalid = [([], 1_000), ([None], 1_000), ([1], 0), ([1] * 5_001, 5_000)]

        for values, maximum in invalid:
            with (
                self.subTest(values=len(values), maximum=maximum),
                self.assertRaises((TypeError, ValueError)),
            ):
                approximate_index_candidate_query(values, maximum)

    def test_float32_overflow_is_rejected_across_float_fields(self):
        overflow = 1e100

        with self.assertRaisesRegex(ValueError, "Float32"):
            vector_search_query(text="text", min_score=overflow)
        with self.assertRaisesRegex(ValueError, "Float32"):
            semantic_vector_signature(
                {
                    "calibrationId": 1,
                    "bucketId": 1,
                    "cells": [1],
                    "cellCounts": [2],
                    "fingerprint": [1],
                    "boundaryConfidence": overflow,
                }
            )
        with self.assertRaisesRegex(ValueError, "Float32"):
            hnsw_search_query(calibration_id=1, vector=[overflow])
        with self.assertRaisesRegex(ValueError, "Float32"):
            hnsw_search_query(calibration_id=1, vector=[1], min_score=overflow)

    def test_hnsw_norm_rejects_values_that_all_underflow_in_float32(self):
        with self.assertRaisesRegex(ValueError, "non-zero finite norm"):
            hnsw_search_query(calibration_id=1, vector=[2.0**-150, -(2.0**-150)])

    def test_smallest_nonzero_float32_is_valid_and_wire_precision_is_preserved(self):
        smallest_float32 = 2.0**-149
        precise_wire_value = smallest_float32 * 1.1
        rounds_to_one = 1.0 + 2.0**-25

        smallest = hnsw_search_query(
            calibration_id=1,
            vector=[smallest_float32],
        )
        precise = hnsw_search_query(
            calibration_id=1,
            vector=[precise_wire_value],
            min_score=rounds_to_one,
        )
        signature = semantic_vector_signature(
            calibration_id=1,
            bucket_id=1,
            cells=[1],
            cell_counts=[2],
            fingerprint=[1],
            boundary_confidence=rounds_to_one,
        )

        self.assertEqual(smallest["vector"], [smallest_float32])
        self.assertEqual(precise["vector"], [precise_wire_value])
        self.assertNotEqual(precise["vector"][0], smallest_float32)
        self.assertEqual(precise["minScore"], rounds_to_one)
        self.assertEqual(signature["boundaryConfidence"], rounds_to_one)


class CandidateSearchBuilderTests(unittest.TestCase):
    def test_table_builder_accepts_vector_query_mapping(self):
        criteria = (
            QueryBuilder(DummyExec(), table="Document")
            .search(
                {
                    "text": "storm",
                    "minScore": 0.25,
                    "nearbyBucketRadius": 3,
                    "maxCandidates": 75,
                    "requireAllTerms": False,
                }
            )
            .to_query_object()["conditions"]["criteria"]
        )

        self.assertEqual(criteria["operator"], "MATCHES")
        self.assertEqual(
            criteria["value"],
            {
                "text": "storm",
                "semantic": None,
                "minScore": 0.25,
                "nearbyBucketRadius": 3,
                "maxCandidates": 75,
                "requireAllTerms": False,
            },
        )

    def test_table_builder_accepts_pythonic_vector_options(self):
        value = (
            QueryBuilder(DummyExec(), table="Document")
            .search("storm", max_candidates=50, require_all_terms=False)
            .to_query_object()["conditions"]["criteria"]["value"]
        )

        self.assertEqual(value["text"], "storm")
        self.assertEqual(value["maxCandidates"], 50)
        self.assertFalse(value["requireAllTerms"])

    def test_database_wide_search_remains_text_only(self):
        config = {
            "base_url": "https://api.example.com",
            "database_id": "db",
            "api_key": "key",
            "api_secret": "secret",
        }

        for database in (OnyxDatabase(config), OnyxDatabaseAsync(config)):
            with (
                self.subTest(database=type(database).__name__),
                self.assertRaises(TypeError),
            ):
                database.search(vector_search_query(text="not-supported"))

    def test_sync_builder_serializes_all_candidate_channels(self):
        lexical = (
            QueryBuilder(DummyExec(), table="Document")
            .approximate_search(
                "storm", min_score=0.1, max_candidates=20, require_all_terms=False
            )
            .to_query_object()["conditions"]["criteria"]
        )
        hnsw = (
            QueryBuilder(DummyExec(), table="Document")
            .hnsw_candidates(
                {
                    "calibrationId": -7,
                    "vector": [0.25, -0.5],
                    "maxCandidates": 5,
                    "efSearch": 10,
                }
            )
            .to_query_object()["conditions"]["criteria"]
        )
        indexed = (
            QueryBuilder(DummyExec(), table="Document")
            .approximate_candidates("tenantId", ["a", "b"], 8)
            .to_query_object()["conditions"]["criteria"]
        )

        self.assertEqual(lexical["operator"], "SEARCH_CANDIDATES")
        self.assertEqual(lexical["value"]["maxCandidates"], 20)
        self.assertEqual(hnsw["operator"], "HNSW_CANDIDATES")
        self.assertEqual(hnsw["value"]["calibrationId"], "-7")
        self.assertEqual(indexed["operator"], "CANDIDATES")
        self.assertEqual(indexed["value"], {"values": ["a", "b"], "maxCandidates": 8})

    def test_async_builder_has_the_same_wire_contract(self):
        query = (
            AsyncQueryBuilder(DummyExec(), table="Document")
            .search(
                vector_search_query(
                    text="hybrid",
                    semantic=semantic_signature(),
                    nearby_bucket_radius=2,
                    max_candidates=40,
                )
            )
            .to_query_object()
        )

        criteria = query["conditions"]["criteria"]
        self.assertEqual(criteria["operator"], "MATCHES")
        self.assertEqual(criteria["value"]["nearbyBucketRadius"], 2)
        self.assertEqual(criteria["value"]["maxCandidates"], 40)
        self.assertEqual(
            criteria["value"]["semantic"]["calibrationId"], "-9223372036854775808"
        )

    def test_async_builder_serializes_all_candidate_channels(self):
        lexical = (
            AsyncQueryBuilder(DummyExec(), table="Document")
            .approximate_search("storm", max_candidates=20)
            .to_query_object()["conditions"]["criteria"]
        )
        hnsw = (
            AsyncQueryBuilder(DummyExec(), table="Document")
            .hnsw_candidates({"calibrationId": -7, "vector": [1]})
            .to_query_object()["conditions"]["criteria"]
        )
        indexed = (
            AsyncQueryBuilder(DummyExec(), table="Document")
            .approximate_candidates("tenantId", "a", 8)
            .to_query_object()["conditions"]["criteria"]
        )

        self.assertEqual(lexical["operator"], "SEARCH_CANDIDATES")
        self.assertEqual(lexical["value"]["maxCandidates"], 20)
        self.assertEqual(hnsw["operator"], "HNSW_CANDIDATES")
        self.assertEqual(hnsw["value"]["calibrationId"], "-7")
        self.assertEqual(indexed["operator"], "CANDIDATES")
        self.assertEqual(indexed["value"], {"values": ["a"], "maxCandidates": 8})

    def test_candidate_builder_methods_reject_an_existing_root(self):
        builders = [
            lambda builder: builder.approximate_search("text"),
            lambda builder: builder.hnsw_candidates(
                {"calibrationId": 1, "vector": [1]}
            ),
            lambda builder: builder.approximate_candidates("tenantId", "a"),
        ]

        for builder_call in builders:
            for builder_type in (QueryBuilder, AsyncQueryBuilder):
                with self.subTest(builder=builder_type.__name__, call=builder_call):
                    builder = builder_type(DummyExec(), table="Document").where(
                        {"field": "active", "operator": "EQUAL", "value": True}
                    )
                    with self.assertRaises(ValueError):
                        builder_call(builder)

    def test_candidate_roots_reject_later_condition_composition(self):
        candidate_calls = [
            lambda builder: builder.approximate_search("text"),
            lambda builder: builder.hnsw_candidates(
                {"calibrationId": 1, "vector": [1]}
            ),
            lambda builder: builder.approximate_candidates("tenantId", "a"),
        ]
        compose_calls = [
            lambda builder: builder.where(
                {"field": "active", "operator": "EQUAL", "value": True}
            ),
            lambda builder: builder.and_(
                {"field": "active", "operator": "EQUAL", "value": True}
            ),
            lambda builder: builder.or_(
                {"field": "active", "operator": "EQUAL", "value": True}
            ),
            lambda builder: builder.search("another query"),
        ]

        for candidate_call in candidate_calls:
            for compose_call in compose_calls:
                for builder_type in (QueryBuilder, AsyncQueryBuilder):
                    with self.subTest(
                        builder=builder_type.__name__,
                        candidate=candidate_call,
                        composition=compose_call,
                    ):
                        builder = candidate_call(
                            builder_type(DummyExec(), table="Document")
                        )
                        with self.assertRaises(ValueError):
                            compose_call(builder)

    def test_candidate_roots_reject_update_and_delete_execution(self):
        candidate_calls = [
            lambda builder: builder.approximate_search("text"),
            lambda builder: builder.hnsw_candidates(
                {"calibrationId": 1, "vector": [1]}
            ),
            lambda builder: builder.approximate_candidates("tenantId", "a"),
        ]

        for candidate_call in candidate_calls:
            sync_builder = candidate_call(QueryBuilder(DummyExec(), table="Document"))
            with (
                self.subTest(
                    candidate=candidate_call, builder="sync", operation="delete"
                ),
                self.assertRaisesRegex(ValueError, "read-only"),
            ):
                sync_builder.delete()

            sync_builder = candidate_call(QueryBuilder(DummyExec(), table="Document"))
            sync_builder.set_updates({"active": False})
            with (
                self.subTest(
                    candidate=candidate_call, builder="sync", operation="update"
                ),
                self.assertRaisesRegex(ValueError, "read-only"),
            ):
                sync_builder.update()

            async_builder = candidate_call(
                AsyncQueryBuilder(DummyExec(), table="Document")
            )
            with (
                self.subTest(
                    candidate=candidate_call, builder="async", operation="delete"
                ),
                self.assertRaisesRegex(ValueError, "read-only"),
            ):
                asyncio.run(async_builder.delete())

            async_builder = candidate_call(
                AsyncQueryBuilder(DummyExec(), table="Document")
            )
            async_builder.set_updates({"active": False})
            with (
                self.subTest(
                    candidate=candidate_call, builder="async", operation="update"
                ),
                self.assertRaisesRegex(ValueError, "read-only"),
            ):
                asyncio.run(async_builder.update())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
