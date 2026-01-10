from onyx_database import onyx, eq, contains, asc
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

active_users = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .and_where(contains("email", "@example.com"))
    .order_by(asc("createdAt"))
    .limit(25)
    .list()
)

first = active_users[0]
print(first.email)
print("example: completed")
