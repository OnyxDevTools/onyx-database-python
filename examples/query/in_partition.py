from onyx_database import onyx, eq
from onyx import tables, SCHEMA

db = onyx.init(partition="tenantA", schema=SCHEMA)

logs = (
    db.from_table(tables.AuditLog)
    .where(eq("tenantId", "tenantA"))
    .in_partition("tenantA")
    .limit(10)
    .list()
)

print("tenantA logs:", [l.id for l in logs])
print("example: completed")
