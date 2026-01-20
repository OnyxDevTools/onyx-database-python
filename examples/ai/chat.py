from onyx_database import onyx


def main():
    db = onyx.init()

    # Basic chat completion
    resp = db.chat(
        messages=[
            {
                "role": "user",
                "content": "Give me a quick onboarding blurb. Respond in one short sentence.",
            }
        ],
        model="onyx-chat",
    )
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("Chat completion did not return content")
    print("Assistant:", choices[0]["message"]["content"])

    # Optional: scope to a database for grounding/billing
    scoped = db.chat(
        messages=[
            {
                "role": "user",
                "content": "List top 3 products by revenue in one short sentence.",
            }
        ],
        model="onyx-chat",
        database_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    scoped_choices = scoped.get("choices", []) if isinstance(scoped, dict) else []
    if not scoped_choices or not scoped_choices[0].get("message", {}).get("content"):
        raise RuntimeError("Scoped chat completion did not return content")
    print("Scoped:", scoped_choices[0]["message"]["content"])

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
