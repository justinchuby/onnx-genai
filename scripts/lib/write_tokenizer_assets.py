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

Because nothing is overwritten, *presence* of a file is not evidence that this
build produced it. A ``tokenizer_config.json`` left behind by an earlier,
different export satisfies any existence check while naming the wrong stop token
- which is precisely how this repository's known-good scatter model came to be
correct only by accident (see the FORCE guard in ``scripts/build_qwen.sh``). So
this script finishes by resolving the stop token the *runtime* would resolve and
failing when there is none, or when a file it did not write disagrees with the
source model about what it is.

Usage:
    write_tokenizer_assets.py MODEL_ID_OR_DIR OUTPUT_DIR
    write_tokenizer_assets.py MODEL_ID_OR_DIR OUTPUT_DIR --check
"""

from __future__ import annotations

import argparse
import json
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
# an error rather than a warning. Absence is not the only failure mode, though:
# see stop_signal() for the stale-leftover case an existence check cannot see.
REQUIRED_FILES = ("tokenizer_config.json",)


def load_json(path: Path | None) -> object | None:
    """Return parsed JSON, or None when the file is absent or unreadable."""
    if path is None or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def stop_signal(files: dict[str, Path]) -> tuple[str, str] | None:
    """Return (stop token, filename) the runtime would resolve, or None.

    Mirrors ``load_eos_token_ids`` in ``crates/onnx-genai-ort/src/tokenizer.rs``,
    which reads ``generation_config.json`` first and falls back to
    ``tokenizer_config.json``. Mirroring the real resolution order is the point:
    a check that asks "does the file exist" answers a different question than
    "will this model stop", and only the second one is what we ship.

    None means nothing here declares a stop token, so generation would run to
    the length cap on every request.
    """
    generation = load_json(files.get("generation_config.json"))
    if isinstance(generation, dict):
        eos_id = generation.get("eos_token_id")
        # bool is a subclass of int; `"eos_token_id": true` is not a token id.
        if (isinstance(eos_id, int) and not isinstance(eos_id, bool)) or (
            isinstance(eos_id, list) and eos_id
        ):
            return str(eos_id), "generation_config.json"

    tokenizer = load_json(files.get("tokenizer_config.json"))
    if isinstance(tokenizer, dict):
        eos = tokenizer.get("eos_token")
        if isinstance(eos, dict):
            eos = eos.get("content")
        if isinstance(eos, str) and eos:
            return eos, "tokenizer_config.json"

    return None


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

    # The files a real run leaves in place. Under --check nothing was written,
    # so fall back to the source for what would have been copied - otherwise the
    # dry run would report on a directory that no run ever produces.
    effective: dict[str, Path] = {}
    for name in COMPANION_FILES:
        destination = args.output_dir / name
        if destination.is_file():
            effective[name] = destination
        elif name in copied:
            effective[name] = source_dir / name

    resolved = stop_signal(effective)
    if resolved is None:
        print(
            f"error: {args.output_dir} declares no stop token. The runtime reads "
            "generation_config.json then tokenizer_config.json and neither names one "
            "here, so this model would load, batch, and generate to the length cap on "
            "every request.",
            file=sys.stderr,
        )
        return 1

    token, signal_file = resolved
    print(f"stop token: {token} (from {signal_file})")

    # A file that was already in place did not come from this build, and this
    # script never overwrites it. If it disagrees with the source model about the
    # stop token it is a leftover from a different export, and the model stops on
    # the wrong token or not at all. Compare same-file signals only: a numeric id
    # from generation_config.json and a literal from tokenizer_config.json are
    # not comparable, and a false alarm here would train people to ignore it.
    if signal_file in present:
        source_resolved = stop_signal(
            {name: source_dir / name for name in COMPANION_FILES}
        )
        if (
            source_resolved is not None
            and source_resolved[1] == signal_file
            and source_resolved[0] != token
        ):
            print(
                f"error: {signal_file} already in {args.output_dir} declares stop token "
                f"{token!r}, but {args.model_id} declares {source_resolved[0]!r}. It was "
                "kept because this script never overwrites, so it is a leftover from an "
                "earlier, different export rather than a product of this build. Delete it "
                "and re-run so this build writes its own.",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
