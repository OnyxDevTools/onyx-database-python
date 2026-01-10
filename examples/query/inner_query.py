from onyx_database import onyx, within, eq
from onyx import tables, SCHEMA, Role, RolePermission

db = onyx.init(schema=SCHEMA)

users_with_admin = (
    db.from_table(tables.User)
    .resolve(("roles", Role))
    .where(
        within(
            "id",
            db.select("userId")
            .from_table(tables.UserRole)
            .where(eq("roleId", "role-admin")),
        )
    )
    .list()
)

if not users_with_admin:
    raise RuntimeError("Expected at least one user with admin role via inner query")

roles_with_permission = (
    db.from_table(tables.Role)
    .resolve(("rolePermissions", RolePermission))
    .where(
        within(
            "id",
            db.from_table(tables.RolePermission).where(eq("permissionId", "perm-manage-users")),
        )
    )
    .list()
)

if not roles_with_permission:
    raise RuntimeError("Expected at least one role that references perm-manage-users via inner query")

def _to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

# Verify resolvers align with the inner query results
for user in users_with_admin:
    role_resolver = getattr(user, "roles", None) if hasattr(user, "roles") else user.get("roles") if isinstance(user, dict) else None
    role_ids = [getattr(r, "id", None) if hasattr(r, "id") else r.get("id") if isinstance(r, dict) else None for r in _to_list(role_resolver)]
    if "role-admin" not in role_ids:
        raise RuntimeError(f"User {getattr(user, 'id', None)} missing role-admin in resolver data")

for role in roles_with_permission:
    rps = getattr(role, "rolePermissions", None) if hasattr(role, "rolePermissions") else role.get("rolePermissions") if isinstance(role, dict) else None
    permission_ids = [
        getattr(rp, "permissionId", None) if hasattr(rp, "permissionId") else rp.get("permissionId") if isinstance(rp, dict) else None
        for rp in _to_list(rps)
    ]
    if "perm-manage-users" not in permission_ids:
        raise RuntimeError(f"Role {getattr(role, 'id', None)} missing perm-manage-users in resolver data")

print("users with admin role:", [u.id for u in users_with_admin])
print("roles referencing perm-manage-users:", [r.id for r in roles_with_permission])
print("example: completed")
