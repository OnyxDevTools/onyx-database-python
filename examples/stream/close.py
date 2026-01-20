from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

handle = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .stream(include_query_results=False)
)

cancel_fn = handle.get("cancel")
if not cancel_fn or not callable(cancel_fn):
    raise RuntimeError("Stream handle did not include a callable cancel")

cancel_fn()
print("example: completed")
