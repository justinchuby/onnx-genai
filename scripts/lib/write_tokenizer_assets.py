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
failing when there is none, when a declared token is absent from the tokenizer's
vocabulary (the runtime drops those silently), or when a file it did not write
disagrees with the source model about what the stop token is.

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
# see runtime_stop_ids() for the stale-leftover and unresolvable-token cases an
# existence check cannot see.
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


def token_to_id(tokenizer_json: object | None, token: str) -> int | None:
    """Resolve a token literal to its id, as ``tokenizers::token_to_id`` does.

    Added tokens are checked first and that ordering is load-bearing:
    Qwen2.5-Instruct's ``<|im_end|>`` lives ONLY in ``added_tokens`` and is
    absent from ``model.vocab``, so a lookup that consults the vocabulary alone
    would report the one stop token we care about as unresolvable.
    """
    if not isinstance(tokenizer_json, dict):
        return None

    for entry in tokenizer_json.get("added_tokens") or ():
        if isinstance(entry, dict) and entry.get("content") == token:
            ident = entry.get("id")
            if isinstance(ident, int) and not isinstance(ident, bool):
                return ident

    model = tokenizer_json.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if isinstance(vocab, dict):
        ident = vocab.get(token)
        if isinstance(ident, int) and not isinstance(ident, bool):
            return ident

    return None


def collect_generation_eos_ids(value: object, ids: list[int]) -> None:
    """Mirror ``collect_generation_eos_ids``: a number, or arrays of numbers."""
    # bool is a subclass of int; `"eos_token_id": true` is not a token id.
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if value not in ids:
            ids.append(value)
    elif isinstance(value, list):
        for item in value:
            collect_generation_eos_ids(item, ids)


def eos_token_string(value: object) -> str | None:
    """Mirror ``eos_token_string``: a string, or an object with ``content``."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def declared_stop_tokens(files: dict[str, Path]) -> list[tuple[str, str]]:
    """Return [(filename, declaration)] for every stop token the model declares.

    This is what the config files SAY. It is not what the runtime can USE - see
    runtime_stop_ids for that, and do not confuse the two.
    """
    declared: list[tuple[str, str]] = []

    generation = load_json(files.get("generation_config.json"))
    if isinstance(generation, dict):
        ids: list[int] = []
        collect_generation_eos_ids(generation.get("eos_token_id"), ids)
        if ids:
            declared.append(("generation_config.json", str(ids)))

    tokenizer_config = load_json(files.get("tokenizer_config.json"))
    if isinstance(tokenizer_config, dict):
        token = eos_token_string(tokenizer_config.get("eos_token"))
        if token is not None:
            declared.append(("tokenizer_config.json", token))

    return declared


def runtime_stop_ids(files: dict[str, Path]) -> tuple[list[int], list[str]]:
    """Return (ids the runtime will collect, token literals it silently drops).

    Faithful mirror of ``load_eos_token_ids`` in
    ``crates/onnx-genai-ort/src/tokenizer.rs``. Two properties of that function
    are easy to get wrong, and getting either wrong produces a FALSE PASS:

    * It **unions** generation_config.json and tokenizer_config.json into one
      id list. It does not prefer one and stop. A checker that returns on the
      first hit never inspects the second file - and since every current build
      writes generation_config.json, such a checker would skip
      tokenizer_config.json entirely, which is the file this script exists for.
    * A token literal absent from the tokenizer's vocabulary is **silently
      dropped** by ``token_to_id``. Reading a plausible literal out of JSON is
      NOT evidence the runtime can use it.

    An empty id list means the runtime falls back to its hardcoded guesses, so
    the model does not declare a stop token it can actually resolve.
    """
    ids: list[int] = []
    dropped: list[str] = []

    generation = load_json(files.get("generation_config.json"))
    if isinstance(generation, dict):
        collect_generation_eos_ids(generation.get("eos_token_id"), ids)

    tokenizer_config = load_json(files.get("tokenizer_config.json"))
    if isinstance(tokenizer_config, dict):
        token = eos_token_string(tokenizer_config.get("eos_token"))
        if token is not None:
            ident = token_to_id(load_json(files.get("tokenizer.json")), token)
            if ident is None:
                dropped.append(token)
            elif ident not in ids:
                ids.append(ident)

    return ids, dropped


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

    # tokenizer.json belongs to the exporter, not to this script, but the
    # runtime needs it to turn a token literal into an id - so the check needs
    # to read it too, or it cannot tell a usable declaration from a decorative
    # one.
    tokenizer_json = args.output_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        effective["tokenizer.json"] = tokenizer_json

    declared = declared_stop_tokens(effective)
    if not declared:
        print(
            f"error: {args.output_dir} declares no stop token. The runtime reads "
            "generation_config.json and tokenizer_config.json and neither names one "
            "here, so this model would load, batch, and generate to the length cap on "
            "every request.",
            file=sys.stderr,
        )
        return 1

    for source_file, declaration in declared:
        print(f"stop token: {declaration} (declared in {source_file})")

    if tokenizer_json.is_file():
        ids, dropped = runtime_stop_ids(effective)
        if dropped:
            print(
                "error: tokenizer_config.json declares stop token "
                f"{', '.join(repr(token) for token in dropped)}, which is not in this "
                "model's tokenizer.json. The runtime resolves literals through the "
                "tokenizer and SILENTLY DROPS what it cannot find, so this declaration "
                "has no effect and generation would fall back to a guess.",
                file=sys.stderr,
            )
            return 1
        if not ids:
            print(
                f"error: nothing in {args.output_dir} resolves to a stop token id, so "
                "the runtime would fall back to its hardcoded guesses.",
                file=sys.stderr,
            )
            return 1
        print(f"runtime stop ids: {ids}")
    else:
        # Counted and loud: a narrow pass has to announce its own narrowness.
        print(
            "note: no tokenizer.json here, so the stop token was checked as DECLARED "
            "but NOT as resolvable against the vocabulary."
        )

    # A companion already in place did not come from this build, and this script
    # never overwrites it. Check EVERY such file against the source, not just
    # the first that declared something: the runtime UNIONS these files, so a
    # stale one is still live even when another file also declares a token.
    if present:
        source_declared = dict(
            declared_stop_tokens({name: source_dir / name for name in COMPANION_FILES})
        )
        for source_file, declaration in declared:
            if source_file not in present:
                continue
            expected = source_declared.get(source_file)
            # Same-file comparison only: a numeric id from generation_config.json
            # and a literal from tokenizer_config.json are not comparable, and a
            # false alarm would train people to ignore this.
            if expected is not None and expected != declaration:
                print(
                    f"error: {source_file} already in {args.output_dir} declares stop "
                    f"token {declaration!r}, but {args.model_id} declares {expected!r}. "
                    "It was kept because this script never overwrites, so it is a "
                    "leftover from an earlier, different export rather than a product "
                    "of this build. Delete it and re-run so this build writes its own.",
                    file=sys.stderr,
                )
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
