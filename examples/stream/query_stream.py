from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .on_item(lambda entity, action: print("STREAM EVENT", action, entity))
    .stream(include_query_results=True, keep_alive=True)
)

# ... run workload ...

handle["cancel"]()
print("example: completed")
