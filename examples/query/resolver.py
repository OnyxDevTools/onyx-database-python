from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

user = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .resolve("roles")
    .first_or_none()
)

if not user:
    raise RuntimeError("No user found for resolver example")

roles = getattr(user, "roles", []) or []
if not roles:
    raise RuntimeError("Resolver did not return roles for user")

print("user roles:", roles)
print("example: completed")
