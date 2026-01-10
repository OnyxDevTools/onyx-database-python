from datetime import datetime, timezone

from onyx_database import onyx
from onyx import tables, SCHEMA, User

db = onyx.init(schema=SCHEMA)

large_user_array = [
    vars(
        User(
            id=f"user_{i}",
            username=f"user_{i}",
            email=f"user{i}@example.com",
            isActive=True,
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc),
            asdf="placeholder",
        )
    )
for i in range(500)
]

db.batch_save(tables.User, large_user_array, batch_size=100)

first_id = large_user_array[0]["id"]
last_id = large_user_array[-1]["id"]
check_first = db.find_by_id(tables.User, first_id)
check_last = db.find_by_id(tables.User, last_id)

if not check_first or not check_last:
    raise RuntimeError("Batch save verification failed (first/last user not found)")

print("saved", len(large_user_array), "users")
print("example: completed")
