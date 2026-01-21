from onyx_database import onyx


def main():
    db = onyx.init()

    reply = db.chat("Give me a quick onboarding blurb. Respond in one short sentence.")
    if not reply:
        raise RuntimeError("Chat completion did not return content")
    print("Assistant:", reply)

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
