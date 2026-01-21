from onyx_database import onyx, format
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

rows = (
    db.from_table(tables.User)
    .select("id", format("createdAt", "M-dd-yyyy"))
    .limit(3)
    .list()
)

if not rows:
    raise RuntimeError("No users returned from format example")

formatted_date = rows[0].get("format(createdAt, 'M-dd-yyyy')")
if formatted_date is None:
    raise RuntimeError("Expected formatted createdAt value")

print("formatted createdAt:", formatted_date)
print("example: completed")
