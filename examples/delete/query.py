from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

deleted_count = (
    db.from_table(tables.User)
    .where(eq("isActive", False))
    .delete()
)

if not isinstance(deleted_count, int):
    raise RuntimeError(f"Unexpected delete response: {deleted_count}")

print("example: completed")
