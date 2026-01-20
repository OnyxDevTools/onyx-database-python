from onyx_database import onyx
from onyx import SCHEMA

db = onyx.init(schema=SCHEMA)

schema = db.get_schema()
entities = schema.get("entities", []) if isinstance(schema, dict) else []
if not entities:
    raise RuntimeError("Schema returned no entities")
print("current schema tables:", [e.get("name") for e in entities])

print("example: completed")
