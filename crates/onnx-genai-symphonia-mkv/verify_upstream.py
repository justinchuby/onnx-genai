"""Verify the packaged immutable upstream archive; no network or Cargo needed."""

import hashlib
import io
import json
import tarfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    receipt = json.loads((root / "UPSTREAM.json").read_text())
    archive = (root / "upstream/symphonia-format-mkv-0.6.1.crate").read_bytes()
    assert hashlib.sha256(archive).hexdigest() == receipt["archive_sha256"]

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for entry in receipt["files"]:
            path = entry["path"]
            original = source.extractfile("symphonia-format-mkv-0.6.1/" + path).read()
            assert hashlib.sha256(original).hexdigest() == entry["upstream_sha256"], (
                path
            )
            local = (root / path).read_bytes()
            status = "unchanged " if original == local else "modified  "
            print(status + path)

    print("verified immutable upstream source archive and all copied-file receipts")


if __name__ == "__main__":
    main()
