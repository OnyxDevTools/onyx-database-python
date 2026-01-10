from collections import Counter

from onyx_database import onyx, count
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

stats = (
    db.from_table(tables.User)
    .select("isActive", count("id"))
    .group_by("isActive")
    .list()
)

if not stats:
    raise RuntimeError("Expected grouped aggregate results by isActive")

missing_count = [g for g in stats if g.get("count(id)") is None]
if missing_count:
    raise RuntimeError("Missing count(id) in grouped aggregate results")

counts = Counter()
for group in stats:
    counts[bool(group.get("isActive"))] += int(group.get("count(id)") or 0)
print("aggregate groups:", stats)
print("active count:", counts[True], "inactive count:", counts[False])
print("example: completed")
