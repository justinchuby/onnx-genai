#!/usr/bin/env python3
"""Validate annotated inference metadata across a Hugging Face collection.

Requires PyYAML and huggingface_hub. Downloads stay under the repository's
``target/`` directory unless ``--workspace`` selects another local path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, get_token, hf_hub_download

DEFAULT_COLLECTION = "justinchuby/onnx-genai-inference-metadata-examples"
DEFAULT_VALIDATOR = "cargo run -q -p onnx-genai-metadata --bin validate_metadata --"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    token: str | None,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=local_dir,
            token=token,
        )
    )


def _provenance_status(
    provenance_path: Path | None, files: dict[str, Path]
) -> dict[str, Any]:
    if provenance_path is None:
        return {"present": False, "hashes_verified": []}

    provenance = json.loads(provenance_path.read_text())
    manifest = provenance.get("files")
    manifest_by_path = (
        {entry["path"]: entry for entry in manifest}
        if isinstance(manifest, list)
        else {}
    )
    if provenance.get("hash_policy") and "README.md" in manifest_by_path:
        required = {
            "inference_metadata.yaml",
            "inference_metadata.annotated.yaml",
            "README.md",
        }
        missing = sorted(required - manifest_by_path.keys())
        if missing:
            raise ValueError(
                f"{provenance_path}: exhaustive distribution manifest is missing {missing}"
            )
    verified = []
    for filename in (
        "inference_metadata.yaml",
        "inference_metadata.annotated.yaml",
        "README.md",
    ):
        entry = manifest_by_path.get(filename)
        if entry is None:
            continue
        path = files[filename]
        if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha256(
            path
        ):
            raise ValueError(f"{provenance_path}: stale hash for {filename}")
        verified.append(filename)
    return {
        "present": True,
        "metadata_source_repository": provenance.get("metadata_source_repository"),
        "metadata_source_commit": provenance.get("metadata_source_commit"),
        "hashes_verified": verified,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("target/hf-metadata-annotation-validation"),
    )
    parser.add_argument(
        "--validator-command",
        default=DEFAULT_VALIDATOR,
        help="Command prefix; metadata paths are appended after --metadata-only --shape.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    token = get_token()
    api = HfApi(token=token)
    collection = api.get_collection(args.collection)
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    validation_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    non_model_items: list[dict[str, Any]] = []

    for item in sorted(collection.items, key=lambda value: value.position):
        if item.item_type != "model":
            non_model_items.append(
                {
                    "position": item.position,
                    "repo": item.item_id,
                    "type": item.item_type,
                }
            )
            continue

        repo_id = item.item_id
        info = api.model_info(repo_id=repo_id, revision="main", files_metadata=True)
        revision = info.sha
        repo_files = {sibling.rfilename for sibling in info.siblings or []}
        required = {
            "inference_metadata.yaml",
            "inference_metadata.annotated.yaml",
            "README.md",
        }
        missing = sorted(required - repo_files)
        if missing:
            raise ValueError(f"{repo_id}@{revision}: missing {missing}")

        local_dir = workspace / repo_id.replace("/", "--")
        files = {
            filename: _download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
                token=token,
            )
            for filename in sorted(required)
        }
        provenance_path = (
            _download(
                repo_id=repo_id,
                filename="provenance.json",
                revision=revision,
                local_dir=local_dir,
                token=token,
            )
            if "provenance.json" in repo_files
            else None
        )

        canonical = yaml.safe_load(files["inference_metadata.yaml"].read_text())
        annotated = yaml.safe_load(
            files["inference_metadata.annotated.yaml"].read_text()
        )
        if canonical != annotated:
            raise ValueError(
                f"{repo_id}@{revision}: canonical and annotated metadata differ after parsing"
            )
        if "inference_metadata.annotated.yaml" not in files["README.md"].read_text():
            raise ValueError(
                f"{repo_id}@{revision}: README does not link the annotation"
            )

        validation_paths.extend(
            [
                files["inference_metadata.yaml"],
                files["inference_metadata.annotated.yaml"],
            ]
        )
        rows.append(
            {
                "position": item.position,
                "repo": repo_id,
                "revision": revision,
                "semantic_equivalence": True,
                "provenance": _provenance_status(provenance_path, files),
            }
        )

    command = [
        *shlex.split(args.validator_command),
        "--metadata-only",
        "--shape",
        *(str(path) for path in validation_paths),
    ]
    subprocess.run(command, check=True)

    report = {
        "collection": collection.slug,
        "item_count": len(collection.items),
        "model_count": len(rows),
        "non_model_items": non_model_items,
        "validated_metadata_files": len(validation_paths),
        "models": rows,
    }
    report_path = workspace / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"validated {len(rows)} model repositories / {len(validation_paths)} metadata files; "
        f"report: {report_path}"
    )


if __name__ == "__main__":
    main()
