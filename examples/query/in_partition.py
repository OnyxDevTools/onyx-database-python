import uuid
from datetime import datetime, timezone

from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(partition="tenantA", schema=SCHEMA)

log_id = f"audit_{uuid.uuid4().hex[:8]}"
db.save(
    tables.AuditLog,
    {
        "id": log_id,
        "tenantId": "tenantA",
        "partition": "tenantA",
        "action": "CREATE",
        "resource": "User",
        "status": "SUCCESS",
        "dateTime": datetime.now(timezone.utc),
    },
)

logs = (
    db.from_table(tables.AuditLog)
    .where(eq("tenantId", "tenantA"))
    .in_partition("tenantA")
    .limit(10)
    .list()
)

if not any(getattr(log_item, "id", None) == log_id for log_item in logs):
    raise RuntimeError("Partitioned query did not return inserted audit log")

print("tenantA logs:", [log_item.id for log_item in logs])
print("example: completed")
