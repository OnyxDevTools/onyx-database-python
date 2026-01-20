from onyx_database import onyx, eq, contains
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .or_(contains("email", "admin"))
    .limit(10)
    .list()
)

if not users:
    raise RuntimeError("Compound query returned no users")

for u in users:
    if not (u.isActive or ("admin" in u.email)):
        raise RuntimeError(f"User {u.id} did not satisfy compound condition")

print([u.email for u in users])
print("example: completed")
