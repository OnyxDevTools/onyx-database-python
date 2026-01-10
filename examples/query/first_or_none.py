from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

maybe_user = (
    db.from_table(tables.User)
      .where(eq("email", "alice@example.com"))
      .first_or_none()
)

if maybe_user:
    print("Found:", maybe_user.id)
else:
    raise RuntimeError("No user found")

print("example: completed")
