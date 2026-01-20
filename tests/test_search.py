import unittest

from onyx_database.helpers.conditions import search as search_condition
from onyx_database.onyx import OnyxDatabase
from onyx_database.query_builder import QueryBuilder


class DummyExec:
    pass


class SearchTests(unittest.TestCase):
    def test_query_builder_search_condition(self):
        qb = QueryBuilder(DummyExec(), table="Table")
        qb.search("Text", 4.4)
        query = qb.to_query_object()

        self.assertEqual(query["table"], "Table")
        criteria = query["conditions"]["criteria"]
        self.assertEqual(criteria["field"], "__full_text__")
        self.assertEqual(criteria["operator"], "MATCHES")
        self.assertEqual(criteria["value"], {"queryText": "Text", "minScore": 4.4})

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
        self.assertIsNone(search_criteria["value"]["minScore"])

    def test_db_search_sets_table_all(self):
        db = OnyxDatabase(
            {"base_url": "https://api.example.com", "database_id": "db", "api_key": "key", "api_secret": "secret"}
        )
        qb = db.search("needle")
        query = qb.to_query_object()

        self.assertEqual(query["table"], "ALL")
        self.assertEqual(query["conditions"]["criteria"]["value"]["queryText"], "needle")


class SearchHelperTests(unittest.TestCase):
    def test_search_helper_shape(self):
        cond = search_condition("text", 1.5)
        self.assertEqual(cond["field"], "__full_text__")
        self.assertEqual(cond["operator"], "MATCHES")
        self.assertEqual(cond["value"], {"queryText": "text", "minScore": 1.5})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
