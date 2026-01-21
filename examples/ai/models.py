from onyx_database import onyx


def main():
    db = onyx.init()

    # List models
    models = db.ai.get_models()
    data = models.get("data", []) if isinstance(models, dict) else []
    if not data:
        raise RuntimeError("Model list was empty")
    print("Models:", [m["id"] for m in data])

    # Retrieve one model
    model_id = data[0].get("id", "onyx-chat")
    model = db.ai.get_model(model_id)
    if not model or model.get("id") != model_id:
        raise RuntimeError(f"Retrieve model failed for {model_id}")
    print("Model detail:", model)

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
