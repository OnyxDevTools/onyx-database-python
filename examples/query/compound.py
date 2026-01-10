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

print([u.email for u in users])
print("example: completed")
