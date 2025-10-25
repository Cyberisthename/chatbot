import json, os, time

ROOT = ".adapters"


def create_adapter(id, data):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f"{id}.json")
    payload = {"id": id, "ts": time.time(), "data": data}
    json.dump(payload, open(path, "w"), indent=2)
    return path


def load_adapter(id):
    path = os.path.join(ROOT, f"{id}.json")
    return json.load(open(path))


