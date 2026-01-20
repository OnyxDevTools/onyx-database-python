import uuid
from datetime import datetime, timezone

from onyx_database import onyx, not_within, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

# Ensure a non-admin user exists
user_id = f"non_admin_{uuid.uuid4().hex[:8]}"
now = datetime.now(timezone.utc)
db.save(
    tables.User,
    {
        "id": user_id,
        "email": f"{user_id}@example.com",
        "username": user_id,
        "isActive": True,
        "createdAt": now,
        "updatedAt": now,
    },
)

admin_ids = (
    db.from_table(tables.UserRole)
    .where(eq("roleId", "role-admin"))
    .select("userId")
)

users = (
    db.from_table(tables.User)
    .where(eq("id", user_id))
    .where(not_within("id", admin_ids))
    .limit(10)
    .list()
)

if not any(getattr(u, "id", None) == user_id for u in users):
    raise RuntimeError("Non-admin user not returned by not_within query")
if any(getattr(u, "id", None) == "admin-user-1" for u in users):
    raise RuntimeError("Admin user should not appear in not_within results")

print("users without admin role:", [u.id for u in users])
print("example: completed")
