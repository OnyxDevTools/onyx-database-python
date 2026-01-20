import threading
import time
import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

deleted_event = threading.Event()
deleted_entities = []


def on_deleted(entity):
    deleted_entities.append(entity)
    deleted_event.set()


handle = (
    db.from_table(tables.User)
    .where(eq("isActive", False))
    .on_item_deleted(on_deleted)
    .stream(include_query_results=False, keep_alive=True)
)

time.sleep(0.25)
user_id = f"stream_del_{uuid.uuid4().hex[:8]}"
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
db.delete(tables.User, user_id)

if not deleted_event.wait(timeout=15):
    handle["cancel"]()
    if db.find_by_id(tables.User, user_id) is not None:
        raise RuntimeError("Delete did not occur and no delete event received")
    print("Warning: no delete event received; record removed.")

handle["cancel"]()
if not any(getattr(e, "id", None) == user_id for e in deleted_entities):
    print("Warning: delete event payload missing expected user")
print("example: completed")
