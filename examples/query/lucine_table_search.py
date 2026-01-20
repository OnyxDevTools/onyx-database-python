import uuid
from datetime import datetime, timezone

from onyx_database import onyx, search, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

unique_suffix = uuid.uuid4().hex[:8]
now = datetime.now(timezone.utc)

records = [
    {
        "id": f"user_search_{unique_suffix}_eng",
        "email": f"eng-{unique_suffix}@example.com",
        "username": f"{unique_suffix}-product engineer remote",
        "isActive": True,
        "createdAt": now,
        "updatedAt": now,
    },
    {
        "id": f"user_search_{unique_suffix}_ds",
        "email": f"ds-{unique_suffix}@example.com",
        "username": f"{unique_suffix}-data scientist onsite",
        "isActive": False,
        "createdAt": now,
        "updatedAt": now,
    },
]

db.save(tables.User, records)


def require_found(results, target_id, label: str):
    def _get_id(item):
        if isinstance(item, dict):
            return item.get("id")
        return getattr(item, "id", None)

    if not any(_get_id(r) == target_id for r in results):
        raise RuntimeError(f"{label} did not return expected record {target_id}")


# Lucene phrase + boolean search within the Users table
lucene_query = '"product engineer" AND remote'
user_hits = db.from_table(tables.User).search(lucene_query, 0).list()
require_found(user_hits, records[0]["id"], f'table search: {lucene_query}')

# Lucene AND combined with a structured filter
data_query = "data AND scientist"
active_hits = (
    db.from_table(tables.User)
    .where(search(data_query, 0))
    .and_(eq("isActive", False))
    .list()
)
require_found(active_hits, records[1]["id"], f'lucene + filter: {data_query} AND isActive:false')

print(f"Lucene table search ({lucene_query}) matched:", [getattr(r, 'username', r.get('username')) for r in user_hits])
print(f"Lucene + filter ({data_query} AND isActive:false) matched:", [getattr(r, 'username', r.get('username')) for r in active_hits])
print("example: completed")
