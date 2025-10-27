"""Bundle generated artifacts into a single ZIP archive."""

from __future__ import annotations

import os
import time
import zipfile


ART = "artifacts"
OUT = os.path.join(ART, "quion_experiment.zip")


def main() -> None:
    os.makedirs(ART, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(ART):
            for name in files:
                if name.endswith(".zip") and name == os.path.basename(OUT):
                    continue
                path = os.path.join(root, name)
                arcname = os.path.relpath(path, ART)
                zf.write(path, arcname=arcname)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"wrote {OUT} at {timestamp}")


if __name__ == "__main__":
    main()
