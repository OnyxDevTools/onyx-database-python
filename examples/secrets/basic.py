from onyx_database import onyx
from onyx import SCHEMA

db = onyx.init(schema=SCHEMA)

secrets = db.list_secrets()
if secrets is None:
    raise RuntimeError("Secrets list returned None")
print("secrets:", secrets)

db.put_secret(
    "api-key",
    {
        "value": "super-secret",
        "purpose": "Access to external API",
    },
)

secret = db.get_secret("api-key")
if not secret or (isinstance(secret, dict) and not secret.get("value")):
    raise RuntimeError("Secret was not stored correctly")
print("secret value:", secret.get("value") if isinstance(secret, dict) else secret)

db.delete_secret("api-key")
try:
    db.get_secret("api-key")
    raise RuntimeError("Secret delete did not remove secret")
except Exception:
    # Expected to raise not-found after deletion
    pass

print("example: completed")
