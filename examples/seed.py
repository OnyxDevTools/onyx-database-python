from __future__ import annotations

import uuid
from datetime import datetime, timezone

from onyx_database import onyx
from onyx import SCHEMA, tables, Role, Permission, RolePermission, User, UserProfile, UserRole


def seed() -> None:
    db = onyx.init(schema=SCHEMA)

    now = datetime.now(timezone.utc)

    role = Role(
        id="role-admin",
        name="Admin",
        description="Administrators with full access",
        isSystem=False,
        createdAt=now,
        updatedAt=now,
    )
    db.save(tables.Role, vars(role))

    perm_write = Permission(
        id="perm-manage-users",
        name="user.write",
        description="Create, update, and delete users",
        createdAt=now,
        updatedAt=now,
    )
    perm_read = Permission(
        id="perm-user-read",
        name="user.read",
        description="Read users",
        createdAt=now,
        updatedAt=now,
    )
    db.save(tables.Permission, [vars(perm_write), vars(perm_read)])

    db.save(
        tables.RolePermission,
        [
            vars(
                RolePermission(
                    id=role.id,  # align with inner-query example that compares Role.id to RolePermission.id
                    roleId=role.id,
                    permissionId=perm_write.id,
                    createdAt=now,
                )
            ),
            vars(
                RolePermission(
                    id=f"{role.id}-read",
                    roleId=role.id,
                    permissionId=perm_read.id,
                    createdAt=now,
                )
            ),
        ],
    )

    user = User(
        id="admin-user-1",
        username="admin-user-1",
        email="admin@example.com",
        isActive=True,
        lastLoginAt=None,
        deletedAt=None,
        createdAt=now,
        updatedAt=now,
        asdf="placeholder",
    )
    db.save(tables.User, vars(user))

    db.save(
        tables.UserProfile,
        vars(
            UserProfile(
                id=str(uuid.uuid4()),
                userId=user.id,
                firstName="Example",
                lastName="Admin",
                bio="Seeded admin profile",
                age=42,
                createdAt=now,
                updatedAt=now,
            )
        ),
    )

    db.save(
        tables.UserRole,
        vars(
            UserRole(
                id=str(uuid.uuid4()),
                userId=user.id,
                roleId=role.id,
                createdAt=now,
            )
        ),
    )

    print("Seeded admin user, role, permissions, and profile.")
    fetched = db.find_by_id(tables.User, user.id)
    if not fetched or getattr(fetched, "email", None) != "admin@example.com":
        raise RuntimeError("Seed verification failed for admin user")
    fetched_role = db.find_by_id(tables.Role, role.id)
    if not fetched_role:
        raise RuntimeError("Seed verification failed for admin role")
    print("example: completed")


if __name__ == "__main__":
    seed()
