from datetime import datetime, timezone

from onyx_database import onyx
from onyx import SCHEMA, tables, User, UserRole

db = onyx.init(schema=SCHEMA)

db.cascade("userRoles:UserRole(userId, id)").save(
    tables.User,
    vars(
        User(
            id="user_126",
            username="dana",
            email="dana@example.com",
            isActive=True,
            createdAt=datetime.now(timezone.utc),
            updatedAt=datetime.now(timezone.utc),
            asdf="placeholder",
            userRoles=[
                vars(
                    UserRole(
                        id="ur_1",
                        userId="user_126",
                        roleId="role_admin",
                        createdAt=datetime.now(timezone.utc),
                    )
                ),
                vars(
                    UserRole(
                        id="ur_2",
                        userId="user_126",
                        roleId="role_editor",
                        createdAt=datetime.now(timezone.utc),
                    )
                ),
            ],
        )
    ),
)

fetched_user = db.find_by_id(tables.User, "user_126", resolvers=["userRoles"])
if not fetched_user or not getattr(fetched_user, "userRoles", None):
    raise RuntimeError("Cascade save failed: userRoles resolver missing")

print("saved user with cascaded userRoles")
print("example: completed")
