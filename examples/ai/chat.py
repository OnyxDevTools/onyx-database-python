from onyx_database import onyx


def main():
    db = onyx.init()

    # Basic chat completion (full request)
    resp = db.ai.chat(
        {
            "model": "onyx-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "Give me a quick onboarding blurb. Respond in one short sentence.",
                }
            ],
        },
    )
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise RuntimeError("Chat completion did not return content")
    print("Assistant:", choices[0]["message"]["content"])

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
