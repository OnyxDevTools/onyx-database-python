import uuid
from datetime import datetime, timezone

from onyx_database import onyx
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

suffix = uuid.uuid4().hex[:8]
now = datetime.now(timezone.utc)

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


# Lucene OR query across all tables (email wildcard + phrase)
lucene_all_query = f'("{suffix}-product manager" AND remote) OR email:{suffix}.ux*'
all_hits = db.search(lucene_all_query).list()
require_found(all_hits, records[0]["id"], f"db.search phrase branch ({lucene_all_query})")
require_found(all_hits, records[1]["id"], f"db.search wildcard branch ({lucene_all_query})")

print(f"Lucene ALL search ({lucene_all_query}) matched:", [getattr(r, 'username', r.get('username')) for r in all_hits])
print("example: completed")
