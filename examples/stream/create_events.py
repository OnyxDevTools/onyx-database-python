from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .on_item_added(lambda u: print("USER ADDED", u))
    .stream(include_query_results=False)
)

# ... run workload ...

handle["cancel"]()
print("example: completed")
