from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = (
    db.from_table(tables.User)
    .select("id", "email")
    .where(eq("isActive", True))
    .limit(5)
    # You can also call `.list(model=User)` if you prefer a model wrapper for the selected fields.
    .list()
)

if not users:
    raise RuntimeError("No users returned from select example")

print([(u.get("id"), u.get("email")) for u in users])
print("example: completed")
