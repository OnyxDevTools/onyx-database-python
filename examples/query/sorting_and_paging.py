from onyx_database import onyx, asc, eq
from onyx import tables, SCHEMA

db = onyx.init(schema=SCHEMA)

first_page = (
    db.from_table(tables.User)
    .where(eq("isActive", True))
    .order_by(asc("createdAt"))
    .limit(25)
    .page()
)

items = list(first_page.get("records", []))
next_page = first_page.get("nextPage") or first_page.get("next_page")

while next_page:
    page = (
        db.from_table(tables.User)
        .where(eq("isActive", True))
        .order_by(asc("createdAt"))
        .limit(25)
        .page(next_page=next_page)
    )
    items.extend(page.get("records", []))
    next_page = page.get("nextPage") or page.get("next_page")

print(f"fetched {len(items)} active users")
print("example: completed")
