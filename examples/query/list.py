from onyx_database import onyx
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = db.from_table(tables.User).limit(10).list()

if users is None:
    raise RuntimeError("List returned None")

if len(users) and len(users) != 10:
    raise RuntimeError("List retuerned unexpected number of users: {len(users)}")

print(f"listed {len(users)} users")
print("example: completed")
