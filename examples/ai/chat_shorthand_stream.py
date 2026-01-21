from onyx_database import onyx


def main():
    db = onyx.init()

    chunks = []
    print("Streaming response:")
    for chunk in db.chat(
        "List three product highlights in one short sentence.",
        stream=True,
    ):
        delta = chunk["choices"][0].get("delta", {})
        if delta.get("content"):
            print(delta["content"], end="", flush=True)
            chunks.append(delta["content"])
    print()

    if not chunks:
        raise RuntimeError("Streaming chat did not return any content chunks")

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
