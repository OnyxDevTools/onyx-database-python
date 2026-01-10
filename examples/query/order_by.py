from onyx_database import onyx, asc, desc
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

users = (
    db.from_table(tables.User)
    .order_by(asc("username"), desc("createdAt"))
    .limit(10)
    .list()
)

print([u.username for u in users])
print("example: completed")
