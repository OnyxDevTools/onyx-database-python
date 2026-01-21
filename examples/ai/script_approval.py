from onyx_database import onyx


def main():
    db = onyx.init()

    approval = db.ai.request_script_approval("db.save({ 'table': 'User', 'id': '123' })")
    normalized = approval.get("normalizedScript") if isinstance(approval, dict) else None
    if not normalized:
        raise RuntimeError("Expected normalizedScript in approval response")
    if approval.get("requiresApproval"):
        findings = approval.get("findings")
        if not findings:
            raise RuntimeError("Approval required but findings were empty")
        print("Approval required:", findings)
    else:
        print("No approval needed. Fingerprint:", normalized)

    print("example: completed")


if __name__ == "__main__":  # pragma: no cover
    main()
