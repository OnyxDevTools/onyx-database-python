from onyx_database import onyx
from onyx import tables, SCHEMA, User

db = onyx.init(schema=SCHEMA)

# Ensure a user exists for the example
new_user = User(
    id="user_123",
    username="johndoe",
    email="name@example.com",
    isActive=True,
)
db.save(tables.User, new_user)

user = db.find_by_id(tables.User, "user_123", resolvers=["roles.permissions"])

if user:
    print("User:", user.username, "roles:", user.roles)
else:
    print("User not found")
    raise RuntimeError("User not found in find_by_id example")

print("example: completed")
