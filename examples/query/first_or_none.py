import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

user_id = f"first_or_none_{uuid.uuid4().hex[:8]}"
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

maybe_user = (
    db.from_table(tables.User)
      .where(eq("email", f"{user_id}@example.com"))
      .first_or_none()
)

if maybe_user:
    print("Found:", maybe_user.id)
else:
    raise RuntimeError("No user found")

print("example: completed")
