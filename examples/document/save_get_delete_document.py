from onyx_database import onyx
from onyx import SCHEMA
import base64

db = onyx.init(schema=SCHEMA)

payload = base64.b64encode(b"hello").decode("ascii")
doc = {
    "documentId": "hello.txt",
    "path": "/docs/hello.txt",
    "mimeType": "text/plain",
    "content": payload,  # base64 content
}

db.save_document(doc)

content = db.get_document("hello.txt")
if not content:
    raise RuntimeError("Failed to retrieve saved document")

db.delete_document("hello.txt")
try:
    db.get_document("hello.txt")
    raise RuntimeError("Document still retrievable after delete")
except Exception:
    pass

print("example: completed")
