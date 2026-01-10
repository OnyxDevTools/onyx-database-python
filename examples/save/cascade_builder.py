from datetime import datetime, timezone

from onyx_database import onyx
from onyx import SCHEMA, tables, Role, Permission

db = onyx.init(schema=SCHEMA)

# The Python SDK exposes cascade via the string syntax:
#   field:Type(target, source)
# This mirrors the TypeScript builder example while keeping usage simple.
permissions_graph = "permissions:Permission(roleId, id)"

role = Role(
    id="role_editor",
    name="Editor",
    description="",
    isSystem=False,
    createdAt=datetime.now(timezone.utc),
    updatedAt=datetime.now(timezone.utc),
    permissions=[
        vars(
            Permission(
                id="perm_edit_content",
                name="user.edit",
                description="Edit content",
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc),
            )
        ),
        vars(
            Permission(
                id="perm_publish_content",
                name="user.publish",
                description="Publish content",
                createdAt=datetime.now(timezone.utc),
                updatedAt=datetime.now(timezone.utc),
            )
        ),
    ],
)

db.cascade(permissions_graph).save(tables.Role, vars(role))

fetched_role = db.find_by_id(tables.Role, "role_editor")
perm_ids = ["perm_edit_content", "perm_publish_content"]
fetched_perms = [db.find_by_id(tables.Permission, pid) for pid in perm_ids]
if not fetched_role or any(p is None for p in fetched_perms):
    raise RuntimeError("Cascade builder save failed: role or permissions missing")

print("saved role with permissions cascade")
print("example: completed")
