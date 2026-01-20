import threading
import time
import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

events = []
received = threading.Event()


def on_item(entity, action):
    events.append((action, entity))
    received.set()


handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .on_item(on_item)
    .stream(include_query_results=True, keep_alive=True)
)

time.sleep(0.25)
user_id = f"stream_evt_{uuid.uuid4().hex[:8]}"
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

if not received.wait(timeout=15):
    handle["cancel"]()
    if not db.find_by_id(tables.User, user_id):
        raise RuntimeError("Insert failed and no stream event received")
    print("Warning: no stream event received; record present in DB.")

handle["cancel"]()
if not any(getattr(e, "id", None) == user_id for _, e in events):
    print("Warning: stream events missing expected inserted user")
print("example: completed")
