from onyx_database import onyx, eq
from onyx import tables, SCHEMA, Role, UserProfile

db = onyx.init(schema=SCHEMA)

user = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .resolve(("roles", Role))
    .resolve(("profile", UserProfile))
    .first_or_none()
)

if not user:
    raise RuntimeError("No user found for resolver example")

roles = getattr(user, "roles", []) or []
profile = getattr(user, "profile", None)

if not roles:
    raise RuntimeError("Expected roles resolver to return at least one role")
if any(not isinstance(r, Role) for r in roles):
    raise RuntimeError("Roles resolver did not return typed Role instances")

if profile is None:
    raise RuntimeError("Expected profile resolver to return a profile")
if not isinstance(profile, UserProfile):
    raise RuntimeError("Profile resolver did not return a typed UserProfile")

print("user roles:", roles)
print("example: completed")
