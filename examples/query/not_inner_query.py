from onyx_database import onyx, not_within, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

admin_ids = (
    db.from_table(tables.UserRole)
    .where(eq("roleId", "role-admin"))
    .select("userId")
)

users = (
    db.from_table(tables.User)
    .where(not_within("id", admin_ids))
    .limit(10)
    .list()
)

print("users without admin role:", [u.id for u in users])
print("example: completed")
