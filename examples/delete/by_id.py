import uuid
from datetime import datetime, timezone

from onyx_database import onyx
from onyx import SCHEMA, tables

db = onyx.init(schema=SCHEMA)

# Create then delete a record so the example is self-contained.
user_id = f"user_delete_example_{uuid.uuid4().hex[:8]}"
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

deleted = db.delete(tables.User, user_id)
if not deleted:
    raise RuntimeError("Delete did not confirm success")

print("example: completed")
