from onyx_database import onyx
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = db.from_table(tables.User).limit(10).list()

if not users:
    raise RuntimeError("List returned no users")

print(f"listed {len(users)} users")
print("example: completed")
