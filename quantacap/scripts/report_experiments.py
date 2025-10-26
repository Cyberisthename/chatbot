import json
import os
import time

from quantacap.core.adapter_store import list_adapters, load_adapter


def main():
    rows = []
    for adapter_id in list_adapters():
        record = load_adapter(adapter_id)
        data = record.get("data", {})
        rows.append(
            {
                "id": adapter_id,
                "ts": record.get("ts"),
                "keys": list(data.keys()),
                "meta": data.get("meta", {}),
            }
        )
    out = {"generated_at": time.time(), "adapters": rows}
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/report.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote artifacts/report.json")


if __name__ == "__main__":
    main()
