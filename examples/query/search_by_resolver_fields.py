from onyx_database import onyx, contains
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

user = (
    db.from_table(tables.User)
    .resolve("roles")
    .where(contains("email", "@example.com"))
    .first_or_none()
)

if not user:
    raise RuntimeError("No user matched resolver search example")

print("matched user:", user.email)
print("example: completed")
