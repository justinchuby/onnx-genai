"""Inspect GenAI wheel/sdist MPL notices and corresponding source without extraction.

Python 3.11+. Call with --source-root <matching checkout> and archive paths, or
--directory <wheel/sdist output directory>. This validates artifacts, not inference.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from contextlib import contextmanager
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import tomllib

PACKAGE = "onnx-genai-symphonia-mkv"
FIXED = (
    "Cargo.toml",
    "build.rs",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "CHANGES.md",
    "UPSTREAM.json",
    "verify_upstream.py",
    "verify_distribution.py",
)


def source_context(root: Path):
    nested = root / "crates" / PACKAGE
    if nested.is_dir():
        return nested, tomllib.loads((root / "Cargo.toml").read_text())["workspace"]
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    if manifest.get("package", {}).get("name") != PACKAGE:
        raise ValueError(
            f"{root} is neither the matching workspace nor reader source package"
        )
    return root, {"package": manifest["package"]}


def expected_source(root: Path) -> dict[str, bytes]:
    crate, _workspace = source_context(root)
    result = {name: (crate / name).read_bytes() for name in FIXED}
    for directory in ("src", "upstream", "tests"):
        for path in sorted((crate / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                if path.is_symlink():
                    raise ValueError(
                        f"canonical covered source {path} is a symlink; restore a regular source file"
                    )
                result[path.relative_to(crate).as_posix()] = path.read_bytes()
    if (crate / "Cargo.toml.orig").is_file():
        result["Cargo.toml.orig"] = (crate / "Cargo.toml.orig").read_bytes()
    if (
        crate != root
        and result["NOTICE.md"] != (root / "THIRD_PARTY_NOTICES.md").read_bytes()
    ):
        raise ValueError(
            "reader NOTICE.md differs from root THIRD_PARTY_NOTICES.md; synchronize the binary redistribution notice"
        )
    return result


class Artifact:
    """Index names only; never load unrelated compiled libraries/model fixtures."""

    def __init__(self, path: Path, archive):
        self.path = path
        self.wheel = path.suffix == ".whl"
        self.archive = archive
        members = self.archive.infolist() if self.wheel else self.archive.getmembers()
        self.members = {}
        for member in members:
            name = member.filename if self.wheel else member.name
            if member.is_dir() if self.wheel else member.isdir():
                continue
            parts = PurePosixPath(name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or "\\" in name
                or name in self.members
            ):
                raise ValueError(
                    f"artifact {path.name} has unsafe/ambiguous member {name!r}; rebuild with safe unique archive paths"
                )
            if not self.wheel and not member.isfile():
                raise ValueError(
                    f"artifact {path.name} has a nonregular source member {name!r}; package regular source files"
                )
            self.members[name] = member

    def __iter__(self):
        return iter(self.members)

    def __getitem__(self, name):
        member = self.members[name]
        if self.wheel:
            return self.archive.read(member)
        with self.archive.extractfile(member) as source:
            return source.read()

    def get(self, name):
        return self[name] if name in self.members else None


@contextmanager
def open_artifact(path: Path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            yield Artifact(path, archive)
    else:
        with tarfile.open(path, "r:gz") as archive:
            yield Artifact(path, archive)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency(value):
    return {"version": value} if isinstance(value, str) else value


def validate_manifest(data: bytes, expected: bytes, workspace: dict) -> None:
    """Maturin may resolve workspace fields when writing dependency sdists."""
    actual = tomllib.loads(data.decode())
    original = tomllib.loads(expected.decode())
    for key, value in original["package"].items():
        if isinstance(value, dict) and value.get("workspace") is True:
            value = workspace["package"][key]
        found = actual["package"].get(key)
        if isinstance(found, dict) and found.get("workspace") is True:
            found = workspace["package"][key]
        if found != value:
            raise ValueError(
                f"source Cargo.toml package.{key} differs from corresponding source; rebuild from the matching revision"
            )
    if {
        key: dependency(value) for key, value in actual.get("dependencies", {}).items()
    } != {
        key: dependency(value)
        for key, value in original.get("dependencies", {}).items()
    }:
        raise ValueError(
            "source Cargo.toml dependencies differ from corresponding source; rebuild the matching reader package"
        )

    for section in (
        "build-dependencies",
        "dev-dependencies",
        "features",
        "target",
        "patch",
        "replace",
    ):
        if actual.get(section, {}) != original.get(section, {}):
            raise ValueError(
                f"source Cargo.toml {section} differs from corresponding source; preserve the matching build configuration"
            )


def validate(path: Path, root: Path) -> dict:
    expected = expected_source(root)
    _crate, workspace = source_context(root)
    version = workspace["package"]["version"]
    with open_artifact(path) as files:
        if path.suffix == ".whl":
            metadata = [name for name in files if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(
                    f"wheel {path.name} must contain exactly one METADATA file"
                )
            distribution = BytesParser().parsebytes(files[metadata[0]])["Name"]
            modules = {
                "onnx-genai": "onnx_genai",
                "onnx-genai-server": "onnx_genai_server",
            }
            if distribution not in modules:
                raise ValueError(
                    f"unsupported distribution {distribution!r}; inspect GenAI binding/CLI artifacts only"
                )
            prefix = f"{modules[distribution]}/_third_party/{PACKAGE}-{version}/"
            marker = prefix + "SOURCE_VERSION"
            if files.get(marker) != f"{PACKAGE} {version}\n".encode():
                raise ValueError(
                    f"wheel {path.name} lacks version-bound MPL source at {marker}; rebuild with the dependency OUT_DIR include"
                )
        else:
            candidates = [
                name.removesuffix("UPSTREAM.json")
                for name in files
                if name.endswith("/UPSTREAM.json")
                and files[name] == expected["UPSTREAM.json"]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"sdist {path.name} must contain exactly one canonical maintained reader source; found {len(candidates)}"
                )
            prefix = candidates[0]
        checked = {}
        for name, content in expected.items():
            included = files.get(prefix + name)
            if included is None:
                raise ValueError(
                    f"artifact {path.name} omits corresponding source {prefix + name}; include the complete reader package"
                )
            if name == "Cargo.toml" and included != content:
                validate_manifest(included, content, workspace)
                if (
                    path.suffix == ".whl"
                    and files.get(prefix + "Cargo.toml.orig") != content
                ):
                    raise ValueError(
                        "normalized wheel source manifest lacks the exact authored Cargo.toml.orig; stage both registry manifests"
                    )
            elif included != content:
                raise ValueError(
                    f"artifact {path.name} has stale/changed covered source {prefix + name}; rebuild from the matching canonical tree"
                )
            checked[name] = hashlib.sha256(included).hexdigest()
        return {
            "artifact": path.name,
            "sha256": file_hash(path),
            "source_prefix": prefix,
            "covered_files_sha256": checked,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("archives", type=Path, nargs="*")
    args = parser.parse_args()
    archives = list(args.archives)
    if args.directory:
        archives += sorted(args.directory.glob("*.whl")) + sorted(
            args.directory.glob("*.tar.gz")
        )
    if not archives:
        parser.error(
            "no wheel/sdist artifacts selected; provide archive paths or a populated --directory"
        )
    import json

    for path in archives:
        try:
            print(
                json.dumps(validate(path, args.source_root.resolve()), sort_keys=True)
            )
        except (
            ValueError,
            OSError,
            KeyError,
            zipfile.BadZipFile,
            tarfile.TarError,
        ) as error:
            parser.exit(1, f"MPL artifact validation failed for {path}: {error}\n")


if __name__ == "__main__":
    main()
