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

if not active_users:
    raise RuntimeError("No active users found (inference example)")

first = active_users[0]
if "@example.com" not in first.email:
    raise RuntimeError("First user email did not match filter (inference example)")
print(first.email)
print("example: completed")
