from onyx_database import onyx, avg
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

stats = (
    db.select(avg("age"))
    .from_table(tables.UserProfile)
    .list()
)

if not stats:
    raise RuntimeError("Expected at least one aggregate result")

avg_age = stats[0].get("avg(age)")
if avg_age is None:
    raise RuntimeError("Average age was not returned")

print("avg age:", avg_age)
print("example: completed")
