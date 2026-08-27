#!/usr/bin/env python3
"""Statically audit ONNX inference-metadata packages in a Hugging Face collection."""

from __future__ import annotations

import argparse
import io
import json
import posixpath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import onnx
    import yaml
except ImportError as error:  # pragma: no cover - dependency failure is user-facing
    raise SystemExit(
        "audit_hf_metadata_collection.py requires PyYAML and onnx "
        "(for example: python -m pip install pyyaml onnx)"
    ) from error


DEFAULT_COLLECTION = "justinchuby/onnx-genai-inference-metadata-examples"
DEFAULT_MAX_ONNX_BYTES = 64 * 1024 * 1024
MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
MAX_STRUCTURED_REQUEST_ASSET_BYTES = 8 * 1024 * 1024
REQUEST_ASSET_SUFFIXES = {
    ".bin",
    ".flac",
    ".jpeg",
    ".jpg",
    ".json",
    ".mp3",
    ".mp4",
    ".npz",
    ".png",
    ".webm",
    ".wav",
}
MEDIA_SUFFIXES = {
    "audio": {".flac", ".mp3", ".wav"},
    "image": {".jpeg", ".jpg", ".png"},
    "video": {".mp4", ".webm"},
}
MEDIA_TERMS = {
    "audio": {"audio", "speech", "waveform"},
    "image": {"image", "images", "pixel", "pixels", "photo", "picture"},
    "video": {"frame", "frames", "video"},
}
TOKENIZER_BASENAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "preprocessor_config.json",
    "processor_config.json",
    "processor_config.yaml",
    "processor_config.yml",
}


class AuditError(RuntimeError):
    pass


def fetch_bytes(url: str, *, retries: int = 3, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "onnx-genai-hf-static-audit/1"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise AuditError(f"GET {url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url))


def api_url(repo_type: str, repo_id: str) -> str:
    collection = "datasets" if repo_type == "dataset" else "models"
    return f"https://huggingface.co/api/{collection}/{repo_id}?blobs=true"


def resolve_url(repo_type: str, repo_id: str, revision: str, path: str) -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    quoted_path = "/".join(
        urllib.parse.quote(part, safe="") for part in path.split("/")
    )
    return (
        f"https://huggingface.co/{prefix}{repo_id}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/{quoted_path}?download=true"
    )


def tensor_shape(value_info: Any) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def value_info(value: Any) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
        "shape": tensor_shape(value),
    }


def parse_onnx(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    external_data: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        entries = {entry.key: entry.value for entry in initializer.external_data}
        if initializer.data_location == onnx.TensorProto.EXTERNAL or entries:
            item: dict[str, Any] = {"tensor": initializer.name, **entries}
            for key in ("offset", "length"):
                if key in item:
                    try:
                        item[key] = int(item[key])
                    except ValueError:
                        pass
            external_data.append(item)
    return {
        "ir_version": model.ir_version,
        "opsets": {
            (opset.domain or "ai.onnx"): opset.version for opset in model.opset_import
        },
        "inputs": [
            value_info(value)
            for value in model.graph.input
            if value.name not in initializer_names
        ],
        "outputs": [value_info(value) for value in model.graph.output],
        "external_data": external_data,
    }


def role_name(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    role = spec.get("role")
    if isinstance(role, str):
        return role
    if isinstance(role, dict):
        nested = role.get("role")
        return nested if isinstance(nested, str) else None
    return None


def workflow_inputs(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for name, spec in (workflow.get("inputs") or {}).items():
        if not isinstance(spec, dict):
            continue
        source = spec.get("source") or {}
        source_kind = source.get("kind") if isinstance(source, dict) else None
        inputs.append(
            {
                "name": name,
                "role": role_name(spec),
                "source_kind": source_kind,
                "source_name": source.get("name") if isinstance(source, dict) else None,
                "required": bool(spec.get("required")),
                "has_default": "default" in source,
            }
        )
    return inputs


def request_media_kind(input_spec: dict[str, Any]) -> str | None:
    role = str(input_spec.get("role") or "").lower()
    role_terms = normalized_terms(role)
    if role != "media" and not any(
        role_terms & terms for terms in MEDIA_TERMS.values()
    ):
        return None
    text = f"{input_spec.get('name', '')} {role}".lower()
    for kind, terms in MEDIA_TERMS.items():
        if any(term in text for term in terms):
            return kind
    return None


def request_asset_requirements(
    inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requirements = []
    for input_spec in inputs:
        if input_spec.get("source_kind") != "request":
            continue
        media_kind = request_media_kind(input_spec)
        if media_kind is not None:
            requirements.append(
                {
                    "input": input_spec["name"],
                    "role": input_spec.get("role"),
                    "required": input_spec.get("required", False),
                    "media_kind": media_kind,
                }
            )
    return requirements


def request_asset_candidates(files: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        path
        for path in files
        if (
            "request" in path.lower()
            or "input" in path.lower()
            or path.lower().startswith("evidence/")
        )
        and Path(path).suffix.lower() in REQUEST_ASSET_SUFFIXES
        and not path.lower().endswith((".onnx", ".onnx.data"))
    )


def normalized_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def npz_member_terms(data: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return set().union(*(normalized_terms(name) for name in archive.namelist()))


def request_asset_evidence(
    repo_type: str,
    repo_id: str,
    revision: str,
    files: dict[str, dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = request_asset_candidates(files)
    structured_terms: dict[str, set[str]] = {}
    inspection_errors: list[str] = []
    for path in candidates:
        if Path(path).suffix.lower() != ".npz":
            continue
        size = files[path]["size"]
        if size > MAX_STRUCTURED_REQUEST_ASSET_BYTES:
            inspection_errors.append(
                f"{path}: {size} bytes exceeds structured asset inspection cap "
                f"{MAX_STRUCTURED_REQUEST_ASSET_BYTES}"
            )
            continue
        try:
            structured_terms[path] = npz_member_terms(
                fetch_bytes(resolve_url(repo_type, repo_id, revision, path))
            )
        except (AuditError, zipfile.BadZipFile) as error:
            inspection_errors.append(f"{path}: could not inspect NPZ members: {error}")

    matches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for requirement in requirements:
        kind = requirement["media_kind"]
        kind_terms = MEDIA_TERMS[kind]
        requirement_matches = []
        for path in candidates:
            suffix = Path(path).suffix.lower()
            path_terms = normalized_terms(path)
            reason = None
            if suffix in MEDIA_SUFFIXES[kind]:
                reason = f"{kind} file in a request/input evidence path"
            elif suffix == ".bin" and path_terms & kind_terms:
                reason = f"raw tensor filename identifies {kind} content"
            elif suffix == ".npz" and structured_terms.get(path, set()) & kind_terms:
                members = sorted(structured_terms[path] & kind_terms)
                reason = f"NPZ member names identify {kind} content ({', '.join(members)})"
            if reason is not None:
                requirement_matches.append({"path": path, "reason": reason})
        record = {**requirement, "matches": requirement_matches}
        if requirement_matches:
            matches.append(record)
        else:
            unmatched.append(record)
    return {
        "candidates": candidates,
        "matched_requirements": matches,
        "unmatched_requirements": unmatched,
        "inspection_errors": inspection_errors,
    }


def component_artifacts(workflow: dict[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name, component in (workflow.get("components") or {}).items():
        if not isinstance(component, dict):
            continue
        implementation = component.get("implementation") or {}
        if implementation.get("kind") == "onnx":
            artifact = implementation.get("artifact")
            if isinstance(artifact, str):
                artifacts[name] = artifact
    return artifacts


def token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def metadata_special_tokens(metadata: dict[str, Any]) -> dict[str, list[int]]:
    tokenizer = (metadata.get("package") or {}).get("tokenizer") or {}
    special_tokens = tokenizer.get("special_tokens") or {}
    result: dict[str, list[int]] = {}
    for name, value in special_tokens.items():
        if isinstance(value, int):
            result[name] = [value]
        elif isinstance(value, list) and all(isinstance(item, int) for item in value):
            result[name] = value
    return result


def inspect_tokenizer(
    repo_type: str,
    repo_id: str,
    revision: str,
    files: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    inspection: dict[str, Any] = {
        "parsed_files": [],
        "metadata_special_token_ids": metadata_special_tokens(metadata or {}),
        "resolved_special_token_ids": {},
        "findings": [],
    }
    parsed: dict[str, Any] = {}
    candidates = [
        path
        for path in files
        if posixpath.basename(path)
        in {
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "vocab.txt",
        }
        and files[path]["size"] <= MAX_TOKENIZER_BYTES
    ]
    for path in sorted(candidates):
        try:
            data = fetch_bytes(resolve_url(repo_type, repo_id, revision, path))
            if path.endswith(".json"):
                parsed[path] = json.loads(data)
            elif path.endswith("vocab.txt"):
                parsed[path] = data.decode("utf-8").splitlines()
            inspection["parsed_files"].append(path)
        except (
            AuditError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            inspection["findings"].append(f"{path}: could not parse: {error}")

    token_to_id: dict[str, int] = {}
    for path, document in parsed.items():
        basename = posixpath.basename(path)
        if basename == "tokenizer.json" and isinstance(document, dict):
            for token in document.get("added_tokens", []):
                if isinstance(token, dict):
                    content, token_id = token.get("content"), token.get("id")
                    if isinstance(content, str) and isinstance(token_id, int):
                        token_to_id[content] = token_id
            vocab = (document.get("model") or {}).get("vocab") or {}
            if isinstance(vocab, dict):
                token_to_id.update(
                    {
                        token: token_id
                        for token, token_id in vocab.items()
                        if isinstance(token, str) and isinstance(token_id, int)
                    }
                )
        elif basename == "vocab.json" and isinstance(document, dict):
            token_to_id.update(
                {
                    token: token_id
                    for token, token_id in document.items()
                    if isinstance(token, str) and isinstance(token_id, int)
                }
            )
        elif basename == "vocab.txt" and isinstance(document, list):
            token_to_id.update({token: index for index, token in enumerate(document)})

    config_documents = [
        document
        for path, document in parsed.items()
        if posixpath.basename(path) in {"config.json", "tokenizer_config.json"}
        and isinstance(document, dict)
    ]
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        for document in config_documents:
            content = token_content(document.get(name))
            if content is not None and content in token_to_id:
                inspection["resolved_special_token_ids"].setdefault(
                    f"{name}_id", []
                ).append(token_to_id[content])
    for values in inspection["resolved_special_token_ids"].values():
        values[:] = sorted(set(values))

    for name, expected in inspection["metadata_special_token_ids"].items():
        resolved = inspection["resolved_special_token_ids"].get(name)
        if resolved and not set(expected).issubset(resolved):
            inspection["findings"].append(
                f"{name}: metadata declares {expected}, tokenizer resolves {resolved}"
            )
        for document in config_documents:
            configured = document.get(name)
            if isinstance(configured, int):
                configured = [configured]
            if (
                isinstance(configured, list)
                and all(isinstance(item, int) for item in configured)
                and not set(expected).issubset(configured)
            ):
                inspection["findings"].append(
                    f"{name}: metadata declares {expected}, config declares {configured}"
                )
    return inspection


def declared_ports(component: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    ports = component.get("ports") or {}
    inputs = set((ports.get("inputs") or {}).keys())
    outputs = set((ports.get("outputs") or {}).keys())
    roles = set((ports.get("roles") or {}).keys())
    return inputs, outputs, roles


def compare_component_abi(
    component: dict[str, Any], graph: dict[str, Any]
) -> list[str]:
    physical_inputs = {value["name"]: value for value in graph["inputs"]}
    physical_outputs = {value["name"]: value for value in graph["outputs"]}
    declared_inputs, declared_outputs, role_ports = declared_ports(component)
    findings: list[str] = []
    graph_input_names = set(physical_inputs)
    graph_output_names = set(physical_outputs)

    def present_or_aggregate(port: str, names: set[str]) -> bool:
        return port in names or any(name.startswith(f"{port}.") for name in names)

    for port in sorted(declared_inputs):
        if not present_or_aggregate(port, graph_input_names):
            findings.append(f"declared input {port!r} is absent from graph inputs")
    for port in sorted(declared_outputs):
        if not present_or_aggregate(port, graph_output_names):
            findings.append(f"declared output {port!r} is absent from graph outputs")
    for port in sorted(role_ports):
        if not present_or_aggregate(port, graph_input_names | graph_output_names):
            findings.append(f"role port {port!r} is absent from graph I/O")

    dtype_aliases = {
        "float": "float32",
        "double": "float64",
        "float16": "float16",
        "bfloat16": "bfloat16",
    }
    ports = component.get("ports") or {}
    for direction, physical in (
        ("inputs", physical_inputs),
        ("outputs", physical_outputs),
    ):
        for port, contract in (ports.get(direction) or {}).items():
            if port not in physical or not isinstance(contract, dict):
                continue
            actual = physical[port]
            expected_dtype = contract.get("dtype")
            observed_dtype = actual.get("dtype")
            actual_dtype = dtype_aliases.get(observed_dtype, observed_dtype)
            if expected_dtype and actual_dtype and expected_dtype != actual_dtype:
                findings.append(
                    f"{direction[:-1]} {port!r} dtype is {actual_dtype}, "
                    f"metadata declares {expected_dtype}"
                )
            expected_rank = contract.get("rank")
            actual_shape = actual.get("shape")
            if (
                expected_rank is not None
                and actual_shape is not None
                and expected_rank != len(actual_shape)
            ):
                findings.append(
                    f"{direction[:-1]} {port!r} rank is {len(actual_shape)}, "
                    f"metadata declares {expected_rank}"
                )
                continue
            expected_shape = contract.get("shape")
            if isinstance(expected_shape, list) and actual_shape is not None:
                for axis, (expected, observed) in enumerate(
                    zip(expected_shape, actual_shape)
                ):
                    if (
                        isinstance(expected, int)
                        and isinstance(observed, int)
                        and expected != observed
                    ):
                        findings.append(
                            f"{direction[:-1]} {port!r} axis {axis} is {observed}, "
                            f"metadata declares {expected}"
                        )
    return findings


def external_data_findings(
    graph_path: str, graph: dict[str, Any], files: dict[str, dict[str, Any]]
) -> list[str]:
    findings: list[str] = []
    graph_dir = posixpath.dirname(graph_path)
    by_location: dict[str, int] = {}
    for entry in graph["external_data"]:
        location = entry.get("location")
        if not isinstance(location, str):
            findings.append(
                f"{graph_path}: tensor {entry['tensor']!r} has no external-data location"
            )
            continue
        resolved = posixpath.normpath(posixpath.join(graph_dir, location))
        required = int(entry.get("offset", 0)) + int(entry.get("length", 0))
        by_location[resolved] = max(by_location.get(resolved, 0), required)
    for location, required in sorted(by_location.items()):
        file = files.get(location)
        if file is None:
            findings.append(f"{graph_path}: missing external data {location!r}")
        elif required and file["size"] < required:
            findings.append(
                f"{graph_path}: external data {location!r} is {file['size']} bytes, "
                f"but tensors require at least {required}"
            )
    return findings


def classify(item: dict[str, Any]) -> str:
    if item["repo_type"] == "dataset":
        return "metadata-only/example"
    if (
        item["metadata"]["status"] != "parsed"
        or item["missing_artifacts"]
        or item["missing_external_data"]
    ):
        return "broken/missing generated asset"
    inspections = item["artifact_inspection"]
    if not inspections:
        return "broken/missing generated asset"
    if item["onnx"]["skipped"] or item["onnx"]["errors"]:
        return "unverified/incomplete ONNX inspection"
    if any(
        inspection["graph_status"] != "parsed"
        or inspection["abi_status"] in {"not_declared", "not_inspected"}
        for inspection in inspections
    ):
        return "unverified/incomplete ONNX inspection"
    if any(inspection["abi_status"] == "unsupported" for inspection in inspections):
        return "unsupported metadata/ONNX ABI"
    if item["request_asset_evidence"]["unmatched_requirements"]:
        return "needs external model/media/request input by design"
    return "fully self-contained runnable"


def classification_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    metadata = item.get("metadata", {})
    if metadata.get("status") != "parsed":
        reasons.append(f"metadata status is {metadata.get('status')}")
    if item.get("missing_artifacts"):
        reasons.append(
            "missing metadata-referenced artifacts: "
            + ", ".join(item["missing_artifacts"])
        )
    if item.get("missing_external_data"):
        reasons.extend(item["missing_external_data"])
    for inspection in item.get("artifact_inspection", []):
        label = f"{inspection['component']} ({inspection['path']})"
        status = inspection["graph_status"]
        if status != "parsed":
            reasons.append(f"{label}: graph status is {status}")
        abi_status = inspection["abi_status"]
        if abi_status == "not_declared":
            reasons.append(f"{label}: metadata declares no component port ABI")
        elif abi_status == "not_inspected":
            reasons.append(f"{label}: ABI was not inspected")
        elif abi_status == "unsupported":
            reasons.extend(
                f"{label}: {finding}" for finding in inspection["abi_findings"]
            )
    for requirement in item.get("request_asset_evidence", {}).get(
        "unmatched_requirements", []
    ):
        reasons.append(
            f"{requirement['input']}: no relevant bundled "
            f"{requirement['media_kind']} asset was identified"
        )
    return reasons


def artifact_inspection_records(
    artifacts: dict[str, str], components: dict[str, Any], files: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for component_name, path in artifacts.items():
        component = components.get(component_name) or {}
        declared_inputs, declared_outputs, role_ports = declared_ports(component)
        records.append(
            {
                "component": component_name,
                "path": path,
                "graph_status": "pending" if path in files else "missing",
                "graph_error": None,
                "abi_status": (
                    "not_inspected"
                    if declared_inputs or declared_outputs or role_ports
                    else "not_declared"
                ),
                "abi_findings": [],
                "external_data_status": "not_inspected",
            }
        )
    return records


def set_graph_status(
    records: list[dict[str, Any]], path: str, status: str, error: str | None = None
) -> None:
    for record in records:
        if record["path"] == path:
            record["graph_status"] = status
            record["graph_error"] = error


def complete_artifact_inspection(
    records: list[dict[str, Any]],
    path: str,
    graph: dict[str, Any],
    components: dict[str, Any],
    external_findings: list[str],
) -> list[str]:
    abi_findings: list[str] = []
    for record in records:
        if record["path"] != path:
            continue
        record["graph_status"] = "parsed"
        record["external_data_status"] = (
            "missing_or_short" if external_findings else "passed"
        )
        if record["abi_status"] == "not_declared":
            continue
        findings = compare_component_abi(components[record["component"]], graph)
        record["abi_findings"] = findings
        record["abi_status"] = "unsupported" if findings else "passed"
        abi_findings.extend(f"{path}: {finding}" for finding in findings)
    return abi_findings


def validate_classifiable_item(item: dict[str, Any]) -> None:
    valid_statuses = {
        "missing",
        "parsed",
        "parse_error",
        "skipped_oversized",
        "unsupported_artifact",
    }
    invalid = [
        record["graph_status"]
        for record in item["artifact_inspection"]
        if record["graph_status"] not in valid_statuses
    ]
    if invalid:
        raise AuditError(f"internal artifact inspection status remained pending: {invalid}")


def summary_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["classification"] for item in items).items()))


def static_validation_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    inspections = [
        inspection
        for item in items
        for inspection in item.get("artifact_inspection", [])
    ]
    return {
        "onnx_files_total": sum(
            len(item.get("onnx", {}).get("files", [])) for item in items
        ),
        "onnx_files_parsed": sum(
            len(item.get("onnx", {}).get("parsed", [])) for item in items
        ),
        "onnx_files_skipped": sum(
            len(item.get("onnx", {}).get("skipped", [])) for item in items
        ),
        "onnx_parse_errors": sum(
            len(item.get("onnx", {}).get("errors", [])) for item in items
        ),
        "abi_findings": sum(len(item.get("abi_findings", [])) for item in items),
        "missing_external_data_findings": sum(
            len(item.get("missing_external_data", [])) for item in items
        ),
        "referenced_onnx_components": len(inspections),
        "referenced_graphs_parsed": sum(
            inspection["graph_status"] == "parsed" for inspection in inspections
        ),
        "abi_passed": sum(
            inspection["abi_status"] == "passed" for inspection in inspections
        ),
        "abi_not_declared": sum(
            inspection["abi_status"] == "not_declared" for inspection in inspections
        ),
        "abi_not_inspected": sum(
            inspection["abi_status"] == "not_inspected" for inspection in inspections
        ),
        "abi_unsupported": sum(
            inspection["abi_status"] == "unsupported" for inspection in inspections
        ),
        "media_requirements_matched": sum(
            len(
                item.get("request_asset_evidence", {}).get(
                    "matched_requirements", []
                )
            )
            for item in items
        ),
        "media_requirements_unmatched": sum(
            len(
                item.get("request_asset_evidence", {}).get(
                    "unmatched_requirements", []
                )
            )
            for item in items
        ),
    }


def audit_repo(
    collection_item: dict[str, Any],
    *,
    max_onnx_bytes: int,
    metadata_dir: Path | None,
) -> dict[str, Any]:
    repo_id = collection_item["id"]
    repo_type = collection_item.get("repoType") or collection_item["type"]
    info = fetch_json(api_url(repo_type, repo_id))
    revision = info["sha"]
    files = {
        sibling["rfilename"]: {
            "size": sibling.get("size", 0),
            "blob_id": sibling.get("blobId"),
            "lfs": sibling.get("lfs"),
        }
        for sibling in info.get("siblings", [])
    }
    total_bytes = sum(file["size"] for file in files.values())
    lfs_files = {path: file for path, file in files.items() if file["lfs"]}
    result: dict[str, Any] = {
        "repo": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "source": api_url(repo_type, repo_id),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "lfs_file_count": len(lfs_files),
        "lfs_bytes": sum(file["lfs"]["size"] for file in lfs_files.values()),
        "largest_files": [
            {"path": path, "size": file["size"], "lfs": bool(file["lfs"])}
            for path, file in sorted(
                files.items(), key=lambda pair: pair[1]["size"], reverse=True
            )[:8]
        ],
        "files": files,
        "tokenizer_files": sorted(
            path for path in files if posixpath.basename(path) in TOKENIZER_BASENAMES
        ),
        "tokenizer_inspection": {},
        "bundled_request_assets": request_asset_candidates(files),
        "request_asset_evidence": {
            "candidates": [],
            "matched_requirements": [],
            "unmatched_requirements": [],
            "inspection_errors": [],
        },
        "metadata": {
            "path": None,
            "status": "missing",
            "error": None,
            "workflow_inputs": [],
            "component_artifacts": {},
        },
        "missing_artifacts": [],
        "artifact_inspection": [],
        "onnx": {"files": [], "parsed": [], "skipped": [], "errors": []},
        "abi_findings": [],
        "missing_external_data": [],
    }
    metadata_path = next(
        (
            candidate
            for candidate in ("inference_metadata.yaml", "inference_metadata.yml")
            if candidate in files
        ),
        None,
    )
    workflow: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    if metadata_path:
        result["metadata"]["path"] = metadata_path
        try:
            metadata_bytes = fetch_bytes(
                resolve_url(repo_type, repo_id, revision, metadata_path)
            )
            metadata = yaml.safe_load(metadata_bytes)
            if not isinstance(metadata, dict):
                raise AuditError("metadata root is not a mapping")
            workflow = (metadata.get("pipeline") or {}).get("workflow") or {}
            if not isinstance(workflow, dict):
                raise AuditError("pipeline.workflow is not a mapping")
            artifacts = component_artifacts(workflow)
            result["metadata"].update(
                {
                    "status": "parsed",
                    "workflow_inputs": workflow_inputs(workflow),
                    "component_artifacts": artifacts,
                }
            )
            result["missing_artifacts"] = sorted(
                artifact for artifact in artifacts.values() if artifact not in files
            )
            if metadata_dir is not None:
                destination = metadata_dir / repo_id.replace("/", "__") / metadata_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(metadata_bytes)
        except (AuditError, urllib.error.URLError, yaml.YAMLError) as error:
            result["metadata"]["status"] = "invalid"
            result["metadata"]["error"] = str(error)

    components = (workflow.get("components") or {}) if workflow else {}
    result["artifact_inspection"] = artifact_inspection_records(
        result["metadata"]["component_artifacts"], components, files
    )
    result["request_asset_evidence"] = request_asset_evidence(
        repo_type,
        repo_id,
        revision,
        files,
        request_asset_requirements(result["metadata"]["workflow_inputs"]),
    )
    result["tokenizer_inspection"] = inspect_tokenizer(
        repo_type, repo_id, revision, files, metadata
    )
    onnx_paths = sorted(path for path in files if path.lower().endswith(".onnx"))
    result["onnx"]["files"] = onnx_paths
    for path in onnx_paths:
        size = files[path]["size"]
        if size > max_onnx_bytes:
            set_graph_status(
                result["artifact_inspection"], path, "skipped_oversized"
            )
            result["onnx"]["skipped"].append(
                {
                    "path": path,
                    "size": size,
                    "reason": f"exceeds --max-onnx-bytes={max_onnx_bytes}",
                }
            )
            continue
        try:
            graph = parse_onnx(
                fetch_bytes(resolve_url(repo_type, repo_id, revision, path))
            )
            result["onnx"]["parsed"].append({"path": path, "size": size, **graph})
            external_findings = external_data_findings(path, graph, files)
            result["missing_external_data"].extend(external_findings)
            result["abi_findings"].extend(
                complete_artifact_inspection(
                    result["artifact_inspection"],
                    path,
                    graph,
                    components,
                    external_findings,
                )
            )
        except Exception as error:  # noqa: BLE001 - ONNX parse errors vary by build
            set_graph_status(
                result["artifact_inspection"], path, "parse_error", str(error)
            )
            result["onnx"]["errors"].append({"path": path, "error": str(error)})

    for inspection in result["artifact_inspection"]:
        if inspection["graph_status"] == "pending":
            inspection["graph_status"] = "unsupported_artifact"
            inspection["graph_error"] = (
                "metadata declares an ONNX implementation whose artifact path "
                "does not end in .onnx"
            )
    validate_classifiable_item(result)
    result["classification"] = classify(result)
    result["classification_reasons"] = classification_reasons(result)
    result["limitations"] = []
    if result["onnx"]["skipped"]:
        result["limitations"].append(
            f"{len(result['onnx']['skipped'])} ONNX file(s) exceeded the byte cap"
        )
    if result["onnx"]["errors"]:
        result["limitations"].append(
            f"{len(result['onnx']['errors'])} ONNX file(s) could not be parsed"
        )
    undeclared = [
        inspection
        for inspection in result["artifact_inspection"]
        if inspection["abi_status"] == "not_declared"
    ]
    if undeclared:
        result["limitations"].append(
            f"{len(undeclared)} metadata-referenced ONNX component(s) have no "
            "declared port ABI"
        )
    if result["abi_findings"]:
        result["limitations"].append(
            f"{len(result['abi_findings'])} metadata/ONNX ABI finding(s)"
        )
    unmatched_assets = result["request_asset_evidence"]["unmatched_requirements"]
    if unmatched_assets:
        result["limitations"].append(
            f"{len(unmatched_assets)} media request input(s) have no relevant "
            "bundled asset"
        )
    result["limitations"].extend(
        result["request_asset_evidence"]["inspection_errors"]
    )
    return result


def audit_collection(
    collection: str, *, max_onnx_bytes: int, metadata_dir: Path | None
) -> dict[str, Any]:
    collection_data = fetch_json(f"https://huggingface.co/api/collections/{collection}")
    items = []
    for index, collection_item in enumerate(collection_data.get("items", []), 1):
        print(
            f"[{index}/{len(collection_data['items'])}] {collection_item['id']}",
            file=sys.stderr,
        )
        try:
            items.append(
                audit_repo(
                    collection_item,
                    max_onnx_bytes=max_onnx_bytes,
                    metadata_dir=metadata_dir,
                )
            )
        except Exception as error:  # noqa: BLE001 - preserve per-repo audit progress
            items.append(
                {
                    "repo": collection_item["id"],
                    "repo_type": collection_item.get("repoType")
                    or collection_item["type"],
                    "revision": None,
                    "source": None,
                    "classification": "unverified/audit failure",
                    "classification_reasons": [str(error)],
                    "fatal_error": str(error),
                }
            )
    return {
        "collection": collection,
        "collection_source": f"https://huggingface.co/api/collections/{collection}",
        "collection_last_updated": collection_data.get("lastUpdated"),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "audit_scope": (
            "static package, ONNX parse, external-data, and declared ABI validation; "
            "no model generation or runtime smoke execution"
        ),
        "python_version": sys.version.split()[0],
        "onnx_version": onnx.__version__,
        "max_onnx_bytes": max_onnx_bytes,
        "items": items,
        "classification_counts": summary_counts(items),
        "static_validation_summary": static_validation_summary(items),
    }


def self_test() -> None:
    workflow = {
        "inputs": {
            "request.input_ids": {
                "source": {"kind": "request"},
                "role": {"kind": "runtime", "role": "prompt_tokens"},
            },
            "request.image": {
                "source": {"kind": "request"},
                "role": {"kind": "runtime", "role": "image"},
            },
        },
        "components": {
            "model": {"implementation": {"kind": "onnx", "artifact": "model.onnx"}}
        },
    }
    assert component_artifacts(workflow) == {"model": "model.onnx"}
    assert [item["role"] for item in workflow_inputs(workflow)] == [
        "prompt_tokens",
        "image",
    ]
    graph = {"inputs": [{"name": "x"}], "outputs": [{"name": "y"}]}
    component = {"ports": {"inputs": {"missing": {}}, "outputs": {"y": {}}}}
    assert compare_component_abi(component, graph) == [
        "declared input 'missing' is absent from graph inputs"
    ]

    image_requirement = request_asset_requirements(workflow_inputs(workflow))
    unrelated = request_asset_evidence(
        "model",
        "example/repo",
        "revision",
        {"request.json": {"size": 2}},
        image_requirement,
    )
    assert unrelated["unmatched_requirements"][0]["input"] == "request.image"
    relevant = request_asset_evidence(
        "model",
        "example/repo",
        "revision",
        {"request.jpg": {"size": 2}},
        image_requirement,
    )
    assert relevant["matched_requirements"][0]["matches"][0]["path"] == "request.jpg"

    def classifiable_item() -> dict[str, Any]:
        return {
            "repo_type": "model",
            "metadata": {"status": "parsed", "workflow_inputs": []},
            "missing_artifacts": [],
            "missing_external_data": [],
            "onnx": {"skipped": [], "errors": []},
            "artifact_inspection": [
                {
                    "component": "model",
                    "path": "model.onnx",
                    "graph_status": "parsed",
                    "abi_status": "passed",
                    "abi_findings": [],
                }
            ],
            "request_asset_evidence": {"unmatched_requirements": []},
        }

    item = classifiable_item()
    assert classify(item) == "fully self-contained runnable"
    item["artifact_inspection"][0]["graph_status"] = "skipped_oversized"
    item["artifact_inspection"][0]["abi_status"] = "not_inspected"
    assert classify(item) == "unverified/incomplete ONNX inspection"
    item = classifiable_item()
    item["artifact_inspection"][0]["abi_status"] = "not_declared"
    assert classify(item) == "unverified/incomplete ONNX inspection"
    item = classifiable_item()
    item["artifact_inspection"][0]["abi_status"] = "unsupported"
    item["artifact_inspection"][0]["abi_findings"] = ["missing input"]
    assert classify(item) == "unsupported metadata/ONNX ABI"
    item = classifiable_item()
    item["missing_external_data"] = ["missing model.onnx.data"]
    assert classify(item) == "broken/missing generated asset"
    item = classifiable_item()
    item["request_asset_evidence"]["unmatched_requirements"] = image_requirement
    assert classify(item) == "needs external model/media/request input by design"
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--max-onnx-bytes", type=int, default=DEFAULT_MAX_ONNX_BYTES)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    result = audit_collection(
        arguments.collection,
        max_onnx_bytes=arguments.max_onnx_bytes,
        metadata_dir=arguments.metadata_dir,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
