import uuid
from datetime import UTC, datetime

from onyx import SCHEMA, tables

from onyx_database import onyx

db = onyx.init(schema=SCHEMA)

suffix = uuid.uuid4().hex[:8]
now = datetime.now(UTC)

records = [
    {
        "id": f"user_search_all_{suffix}_pm",
        "email": f"{suffix}.pm@example.com",
        "username": f"{suffix}-product manager remote",
        "isActive": True,
        "createdAt": now,
        "updatedAt": now,
    },
    {
        "id": f"user_search_all_{suffix}_ux",
        "email": f"{suffix}.ux@example.com",
        "username": f"{suffix}-ux designer hybrid",
        "isActive": True,
        "createdAt": now,
        "updatedAt": now,
    },
]

# Seed data discoverable via db.search (table = ALL)
db.save(tables.User, records)


def require_found(results, target_id: str, label: str):
    def _get_id(item):
        if isinstance(item, dict):
            return item.get("id")
        return getattr(item, "id", None)

    if not any(_get_id(r) == target_id for r in results):
        raise RuntimeError(f"{label} did not return expected record {target_id}")


# Native OR query across all tables (email wildcard + phrase)
all_tables_query = f'("{suffix}-product manager" AND remote) OR email:{suffix}.ux*'
all_hits = db.search(all_tables_query).list()
require_found(all_hits, records[0]["id"], f"db.search phrase branch ({all_tables_query})")
require_found(all_hits, records[1]["id"], f"db.search wildcard branch ({all_tables_query})")

print(
    f"Native all-table search ({all_tables_query}) matched:",
    [getattr(r, "username", r.get("username")) for r in all_hits],
)
print("example: completed")
