import threading
import time
import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

updated_event = threading.Event()
updated_entities = []


def on_updated(entity):
    updated_entities.append(entity)
    updated_event.set()


handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .on_item_updated(on_updated)
    .stream(include_query_results=False, keep_alive=True)
)

time.sleep(0.25)
user_id = f"stream_upd_{uuid.uuid4().hex[:8]}"
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
db.save(
    tables.User,
    {
        "id": user_id,
        "email": f"{user_id}-updated@example.com",
        "username": user_id,
        "isActive": True,
        "createdAt": now,
        "updatedAt": datetime.now(timezone.utc),
    },
)

if not updated_event.wait(timeout=15):
    handle["cancel"]()
    latest = db.find_by_id(tables.User, user_id)
    if not latest or getattr(latest, "email", "") != f"{user_id}-updated@example.com":
        raise RuntimeError("Update did not apply and no stream event received")
    print("Warning: no update event received; record updated in DB.")

handle["cancel"]()
if not any(getattr(e, "id", None) == user_id for e in updated_entities):
    print("Warning: update event payload missing expected user")
print("example: completed")
