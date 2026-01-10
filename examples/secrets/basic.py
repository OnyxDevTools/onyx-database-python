from onyx_database import onyx
from onyx import SCHEMA

db = onyx.init(schema=SCHEMA)

secrets = db.list_secrets()
print("secrets:", secrets)

db.put_secret(
    "api-key",
    {
        "value": "super-secret",
        "purpose": "Access to external API",
    },
)

secret = db.get_secret("api-key")
print("secret value:", secret.get("value") if isinstance(secret, dict) else secret)

db.delete_secret("api-key")

print("example: completed")
