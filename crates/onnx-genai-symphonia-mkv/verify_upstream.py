"""Verify the packaged immutable upstream archive; no network or Cargo needed."""
import hashlib, io, json, pathlib, tarfile
root = pathlib.Path(__file__).resolve().parent
receipt = json.loads((root / "UPSTREAM.json").read_text())
archive = (root / "upstream/symphonia-format-mkv-0.6.1.crate").read_bytes()
assert hashlib.sha256(archive).hexdigest() == receipt["archive_sha256"]
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
    for entry in receipt["files"]:
        path = entry["path"]
        original = source.extractfile("symphonia-format-mkv-0.6.1/" + path).read()
        assert hashlib.sha256(original).hexdigest() == entry["upstream_sha256"], path
        local = (root / path).read_bytes()
        print(("unchanged " if original == local else "modified  ") + path)
print("verified immutable upstream source archive and all copied-file receipts")
