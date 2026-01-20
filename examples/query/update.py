import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

user_id = f"update_{uuid.uuid4().hex[:8]}"
now = datetime.now(timezone.utc)
db.save(
    tables.User,
    {
        "id": user_id,
        "email": f"{user_id}@example.com",
        "username": user_id,
        "isActive": False,
        "createdAt": now,
        "updatedAt": now,
    },
)

updated = (
    db.from_table(tables.User)
    .where(eq("id", user_id))
    .set_updates({"isActive": True})
    .update()
)

fresh = db.find_by_id(tables.User, user_id)
if not fresh or getattr(fresh, "isActive", None) is not True:
    raise RuntimeError("Update did not flip isActive to True")

print("updated:", updated)
print("example: completed")
