from onyx_database import onyx
from onyx import SCHEMA

db = onyx.init(schema=SCHEMA)

deleted = db.delete("User", "user_125")
if not deleted:
    raise RuntimeError("Delete did not confirm success")

print("example: completed")
