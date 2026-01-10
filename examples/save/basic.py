from datetime import datetime, timezone

from onyx_database import onyx
from onyx import tables, SCHEMA, User

db = onyx.init(schema=SCHEMA)

# Upsert a single user (typed)
user = User(
    id="user_123",
    username="alice",
    email="alice@example.com",
    isActive=True,
    createdAt=datetime.now(timezone.utc),
    updatedAt=datetime.now(timezone.utc),
    asdf="placeholder",
)
db.save(tables.User, vars(user))

# Batch upsert users
users = [
    User(
        id="user_124",
        username="bob",
        email="bob@example.com",
        isActive=True,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
        asdf="placeholder",
    ),
    User(
        id="user_125",
        username="carol",
        email="carol@example.com",
        isActive=False,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
        asdf="placeholder",
    ),
]

db.save(tables.User, [vars(u) for u in users])

fetched = db.find_by_id(tables.User, "user_123")
if not fetched or getattr(fetched, "email", None) != "alice@example.com":
    raise RuntimeError("Failed to verify saved user_123")

fetched_bob = db.find_by_id(tables.User, "user_124")
fetched_carol = db.find_by_id(tables.User, "user_125")
if not fetched_bob or not fetched_carol:
    raise RuntimeError("Failed to verify batch-saved users")

print("example: completed")
