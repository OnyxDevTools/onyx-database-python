import tempfile
import importlib.util
import sys
import unittest
from pathlib import Path

from onyx_database.codegen import generate_models

try:
    import pydantic  # noqa: F401
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False


EXAMPLE_SCHEMA = {
    "databaseId": "test",
    "entities": [
        {
            "name": "Thing",
            "identifier": {"name": "id", "generator": "None", "type": "String"},
            "attributes": [
                {"name": "id", "type": "String"},
                {"name": "age", "type": "Int", "isNullable": True},
            ],
        }
    ],
}


class CodegenPydanticTests(unittest.TestCase):
    @unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
    def test_generate_pydantic_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "onyx"
            generate_models(EXAMPLE_SCHEMA, out, package="onyx", models_mode="pydantic")

            sys.path.insert(0, tmpdir)
            try:
                spec = importlib.util.spec_from_file_location("onyx.models", out / "models.py")
                module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                assert spec and spec.loader
                spec.loader.exec_module(module)  # type: ignore[arg-type]
                Thing = getattr(module, "Thing")
                obj = Thing(id="t1", age=5)
                self.assertEqual(obj.id, "t1")
                self.assertEqual(obj.age, 5)
                # extra fields allowed
                obj2 = Thing(id="t2", age=None, extra_field="ok")
                self.assertEqual(obj2.extra_field, "ok")
            finally:
                sys.path = [p for p in sys.path if p != tmpdir]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
