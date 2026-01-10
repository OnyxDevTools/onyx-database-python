from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .stream(include_query_results=False)
)

# immediately close
handle["cancel"]()
print("example: completed")
