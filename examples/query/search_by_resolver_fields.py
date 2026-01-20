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

roles = getattr(user, "roles", []) or []
if not roles:
    raise RuntimeError("Resolver did not return roles for matched user")

print("matched user:", user.email)
print("example: completed")
