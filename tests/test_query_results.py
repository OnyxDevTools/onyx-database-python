import unittest

from onyx_database.query_results import QueryResults
from onyx_database.query_results_async import AsyncQueryResults


class Dummy:
    def __init__(self, val):
        self.val = val


class QueryResultsTests(unittest.TestCase):
    def test_values_with_dicts_and_models(self):
        qr = QueryResults([{"val": 1}, Dummy(2), {"val": 3}])
        self.assertEqual(qr.values("val"), [1, 2, 3])

    def test_async_values(self):
        aq = AsyncQueryResults([{"val": 4}, Dummy(5)])
        self.assertEqual(aq.values("val"), [4, 5])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
