from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

updated = (
    db.from_table(tables.User)
    .where(eq("isActive", False))
    .set_updates({"isActive": True})
    .update()
)

print("updated:", updated)
print("example: completed")
