from onyx_database import onyx, asc, desc
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = (
    db.from_table(tables.User)
    .order_by(asc("username"), desc("createdAt"))
    .limit(10)
    .list()
)

if not users:
    raise RuntimeError("No users returned in order_by example")

usernames = [u.username for u in users]
if usernames != sorted(usernames):
    raise RuntimeError("Usernames were not ordered ascending as expected")

print(usernames)
print("example: completed")
