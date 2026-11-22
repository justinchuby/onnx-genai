#!/usr/bin/env python3
"""Rebuild the pinned official Llama Chat assets. Requires git, Node >=20, npm.

The checked-in capability patch is applied before building. Runtime/Cargo builds never run this script or
access npm. Review source/asset/license diffs and run compatibility tests after
an upstream version change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
COMMIT = "8d60873dfb8619f2b5ed31adfbab07cccb40db5c"
REPOSITORY = Path(__file__).resolve().parents[1]
DESTINATION = REPOSITORY / "crates/onnx-genai-server/vendor/llama-chat"
PATCH = REPOSITORY / "scripts/llama_chat/capabilities.patch"


def run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True)


def install_build(source, destination):
    app = source / "ui/apps/chat"
    run("git", "apply", "--reverse", "--check", str(PATCH), cwd=source)
    if not (app / "dist/index.html").is_file():
        raise SystemExit(f"No built UI at {app / 'dist'}; run npm ci and npm run build first")
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / "assets"
    if assets.exists():
        shutil.rmtree(assets)  # Only replace this script's generated vendor directory.
    shutil.copytree(app / "dist", assets)
    shutil.copyfile(app / "LICENSE", destination / "LICENSE")
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    (destination / "UPSTREAM").write_text(f"{UPSTREAM}\n{version}\nui/apps/chat\nBuilt with npm ci && npm run build; scripts/llama_chat/capabilities.patch applied.\n")
    provenance = {"repository": UPSTREAM, "commit": version,
                  "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                  "package_lock_sha256": hashlib.sha256((app / "package-lock.json").read_bytes()).hexdigest()}
    (destination / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    notices = []
    # Include installed dependency notices conservatively (including build-only
    # packages), plus the upstream UI/voice libraries and bundled font licenses.
    for root in [app / "node_modules", source / "ui/lib", app / "static"]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name.lower().startswith(("license", "licence", "copying", "ofl")):
                relative = path.relative_to(source)
                notices.append(f"\n{'=' * 72}\n{relative}\n{'=' * 72}\n{path.read_text(errors='replace')}\n")
    (destination / "THIRD_PARTY_NOTICES.txt").write_text("Notices from the upstream build and asset dependencies.\n" + "".join(notices))
    manifest = {p.relative_to(assets).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(assets.rglob("*")) if p.is_file()}
    (destination / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-source", type=Path, help="Use an already-built checkout of the pinned commit")
    parser.add_argument("--check", action="store_true", help="Verify vendored asset names and SHA-256 hashes without network or npm")
    args = parser.parse_args()
    if args.check:
        assets = DESTINATION / "assets"
        actual = {p.relative_to(assets).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(assets.rglob("*")) if p.is_file()}
        expected = json.loads((DESTINATION / "SHA256SUMS.json").read_text())
        provenance = json.loads((DESTINATION / "PROVENANCE.json").read_text())
        if provenance["commit"] != COMMIT or provenance["patch_sha256"] != hashlib.sha256(PATCH.read_bytes()).hexdigest():
            raise SystemExit("UI provenance differs from the pinned commit/capability patch; rebuild and review")
        if actual != expected:
            raise SystemExit("Vendored Llama Chat assets differ from SHA256SUMS.json; rebuild the explicit upstream pin and review the asset diff")
        print(f"Verified {len(actual)} pinned Llama Chat assets")
        return
    if args.built_source:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.built_source, text=True).strip()
        if actual != COMMIT:
            raise SystemExit(f"Expected upstream {COMMIT}, found {actual}; update the explicit pin before refreshing")
        install_build(args.built_source.resolve(), DESTINATION)
        return
    with tempfile.TemporaryDirectory(prefix="nxrt-llama-chat-") as temp:
        source = Path(temp)
        run("git", "init", cwd=source)
        run("git", "remote", "add", "origin", UPSTREAM, cwd=source)
        run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=source)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
        run("git", "apply", str(PATCH), cwd=source)
        app = source / "ui/apps/chat"
        run("npm", "ci", "--no-audit", "--no-fund", cwd=app)
        run("npm", "run", "build", cwd=app)
        install_build(source, DESTINATION)


if __name__ == "__main__":
    main()
