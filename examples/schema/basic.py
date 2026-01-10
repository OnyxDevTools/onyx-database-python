from onyx_database import onyx
from onyx import SCHEMA

db = onyx.init(schema=SCHEMA)

schema = db.get_schema()
print("current schema tables:", [e.get("name") for e in schema.get("entities", [])])

print("example: completed")
