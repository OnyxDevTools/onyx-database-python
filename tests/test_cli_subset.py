import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from onyx_database import cli


EXAMPLE_SCHEMA = {
    "databaseId": "db",
    "entities": [
        {"name": "User", "attributes": [{"name": "id", "type": "String"}]},
        {"name": "Role", "attributes": [{"name": "id", "type": "String"}]},
    ],
}


class CliSubsetTests(unittest.TestCase):
    def test_gen_tables_subset_prints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            schema_path.write_text(json.dumps(EXAMPLE_SCHEMA), encoding="utf-8")

            with mock.patch("onyx_database.cli._load_json", return_value=EXAMPLE_SCHEMA), mock.patch("onyx_database.cli.onyx.init") as init_mock:
                init_mock.return_value = mock.Mock(get_schema=mock.Mock(return_value=EXAMPLE_SCHEMA))
                with mock.patch("builtins.print") as print_mock:
                    cli.main(["gen", "--source", "file", "--schema", str(schema_path), "--tables", "User"])
                    # Ensure we printed JSON subset
                    printed = "".join(call.args[0] for call in print_mock.call_args_list if isinstance(call.args[0], str) and call.args[0].startswith("{"))
                    self.assertIn("User", printed)
                    self.assertNotIn("Role", printed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
