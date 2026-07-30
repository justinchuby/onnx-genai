#!/usr/bin/env python3
"""Copy the tokenizer/generation companion files Mobius's onnx-genai target omits.

``mobius build --runtime onnx-genai`` writes exactly one tokenizer artifact:
``tokenizer.json`` (see ``_write_hf_tokenizer`` in
``mobius/integrations/onnx_genai/auto_export.py``). The ``ort-genai`` target
copies the rest, but the two targets are mutually exclusive, so a static-cache
build - which *must* use ``onnx-genai`` to get ``inference_metadata.yaml`` -
silently ends up without them.

That is not cosmetic. Two runtime behaviours depend on the missing files:

* **Stop tokens.** ``load_eos_token_ids``
  (``crates/onnx-genai-ort/src/tokenizer.rs:103``) reads ``generation_config.json``
  then ``tokenizer_config.json``, and otherwise falls back to a fixed list
  (``<|endoftext|>``, ``</s>``, ``<eos>``, ``[EOS]``). Qwen2.5-Instruct's real
  stop token is ``<|im_end|>`` and appears **only** in ``tokenizer_config.json``.
  The fallback ``<|endoftext|>`` does exist in Qwen's vocabulary, so nothing
  errors - generation simply never stops on its own.
* **Chat template.** The server's ``load_chat_template``
  (``crates/onnx-genai-server/src/state.rs:314``) reads ``chat_template`` out of
  ``tokenizer_config.json``. Without it, chat completions cannot be formatted.

Files are taken from the local HuggingFace cache when present, so this normally
performs no network I/O. Existing files are never overwritten: whatever the
exporter wrote is authoritative.

Usage:
    write_tokenizer_assets.py MODEL_ID_OR_DIR OUTPUT_DIR
    write_tokenizer_assets.py MODEL_ID_OR_DIR OUTPUT_DIR --check
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# tokenizer.json is deliberately absent: the exporter writes its own fast-format
# copy and that one is what inference_metadata.yaml references.
COMPANION_FILES = (
    "tokenizer_config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

# Without this the model loads but will not stop generating, so its absence is
# an error rather than a warning.
REQUIRED_FILES = ("tokenizer_config.json",)


def resolve_source_dir(model_id: str) -> Path:
    """Return a local directory holding the source model's tokenizer files."""
    local = Path(model_id)
    if local.is_dir():
        return local

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "error: huggingface_hub is required to fetch tokenizer files; "
            "install Mobius and its dependencies"
        )

    # allow_patterns keeps this to a few KB, and the files are already cached
    # when the export that just ran downloaded them.
    return Path(snapshot_download(model_id, allow_patterns=list(COMPANION_FILES)))


def copy_companions(source_dir: Path, output_dir: Path, dry_run: bool) -> tuple[list[str], list[str]]:
    """Copy missing companion files. Returns (copied, already_present)."""
    copied: list[str] = []
    present: list[str] = []

    for name in COMPANION_FILES:
        destination = output_dir / name
        if destination.is_file():
            present.append(name)
            continue
        candidate = source_dir / name
        if not candidate.is_file():
            continue
        if not dry_run:
            shutil.copy2(candidate, destination)
        copied.append(name)

    return copied, present


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy tokenizer/generation companion files into a built model directory."
    )
    parser.add_argument("model_id", help="HuggingFace model id, or a local model directory")
    parser.add_argument("output_dir", type=Path, help="Built model package directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would be copied without writing anything.",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"error: no such directory: {args.output_dir}", file=sys.stderr)
        return 1

    try:
        source_dir = resolve_source_dir(args.model_id)
    except Exception as error:  # network, auth, unknown model id
        print(
            f"error: could not obtain tokenizer files for {args.model_id!r}: {error}",
            file=sys.stderr,
        )
        return 1

    copied, present = copy_companions(source_dir, args.output_dir, dry_run=args.check)

    verb = "would copy" if args.check else "copied"
    if copied:
        print(f"{verb} into {args.output_dir}: {', '.join(sorted(copied))}")
    if present:
        print(f"already present: {', '.join(sorted(present))}")

    missing_required = [
        name
        for name in REQUIRED_FILES
        if not (args.output_dir / name).is_file() and name not in copied
    ]
    if missing_required:
        print(
            f"error: {args.model_id} provides no {', '.join(missing_required)}. "
            "Without it the runtime cannot resolve this model's stop token or chat "
            "template, so it would load but never stop generating.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
