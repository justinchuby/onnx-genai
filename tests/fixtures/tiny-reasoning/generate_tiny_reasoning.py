#!/usr/bin/env python3
"""Generate the tiny *reasoning* decoder-only fixture.

Run from the onnx-genai repo root with only the Python standard library::

    python tests/fixtures/tiny-reasoning/generate_tiny_reasoning.py

Unlike ``tests/fixtures/tiny-llm/generate_tiny_llm.py`` this generator needs no
Mobius/torch: the reasoning fixture is a thin, fully regenerable derivation of
the already-committed ``tiny-llm`` graph. It reuses that model's ONNX graph
verbatim and its tokenizer with a single vocab entry renamed (see below), and
layers four things on top that turn a plain tiny LLM into a reasoning one:

1. a **chat template** (``tokenizer_config.json``) that opens a ``<think>``
   reasoning span right after the assistant generation prompt, exactly the way a
   real reasoning model's template does. The runtime detects the reasoning
   convention from this template alone -- never from a model name (RULES.md §2);
2. **generation defaults** (``inference_metadata.yaml`` ``generation`` block)
   declaring ``do_sample: true`` so the fixture also exercises the sampling
   resolution order landed in #385/#392 (explicit flag > model-declared >
   greedy fallback);
3. a deliberately **low context** (inherited ``max_sequence_length`` of 16, the
   size of the graph's position table) so context exhaustion is reached in a few
   tokens and is cheap to hit in CI;
4. a **reachable close**: the one vocab entry tiny-llm's greedy decode reaches
   only on the ``quick``/``fox``/``dog`` prompts is renamed to ``</think>`` (see
   CLOSE_TOKEN_* below), so those prompts close the span and commit while every
   other prompt degenerates -- one model, two reachable greedy outcomes.

Why this reproduces the bug shape -- and why it can also succeed. The fixture
must express *both* invariant outcomes on one model, or a regression that drops
every turn (a broken commit path, a mis-set finish reason) would pass unnoticed
against a fixture that can only ever drop. So the closing ``</think>`` delimiter
is made *reachable but not on every path*:

* the tiny-llm graph's greedy decode is deterministic and prompt-sensitive. On
  most prompts it settles into an attractor whose token ids never include id 22;
  on the prompts "quick"/"fox"/"dog" (and only those) greedy emits id 22 at the
  third position, immediately followed by a real word (id 12, "lazy").
* this generator renames that single vocab entry -- id 22, ``tok22`` in tiny-llm
  -- to ``</think>``. It changes *decoded text only*, never a weight or an emitted
  id, so the greedy attractors are untouched: the "quick" family now greedily
  emits ``... </think> lazy`` and *closes the span with a non-empty answer*,
  while every other prompt still degenerates because its attractor never reaches
  id 22.

The result is one model with two reachable greedy outcomes -- degenerate-and-drop
on most prompts, close-and-commit on the "quick" family -- plus a declared
``do_sample`` regime that avoids the loop. That is Justin's "DeepSeek repeats its
thinking and won't stop" defect reduced to a CPU/ORT assertion (a context stop
here stands in for the CUDA KV-capacity stop the full-size model hit), with the
positive path kept live so a drop assertion means something by contrast. See
docs/research/testing/00-integration-stress-design.md.

The id-22 choice is empirical: it is grounded in the *current* tiny-llm weights.
If that graph is ever regenerated the greedy attractors can move, and the close
token must be re-derived (regenerate, then probe which prompts emit which ids).
The tests assert on properties, not token strings, precisely so they survive a
close token moving to a different word -- but the two *reachable outcomes* are a
property of these weights and must be re-verified after any graph change.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# The chat template opens the reasoning span for the model: it appends the
# opening <think> after the assistant generation prompt, so generation begins
# *inside* the span and the model only ever has to emit the close. This matches
# how published reasoning templates are written and is the exact template the
# repl_e2e reasoning contract test expects.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}"
)

# Author-declared generation defaults. do_sample=true is precisely the knob a
# reasoning model ships because greedy decoding makes it loop; the runtime must
# honor it unless the caller forces --greedy. Kept as literal YAML text so the
# generator needs no YAML dependency.
GENERATION_BLOCK = (
    "# Author-declared generation defaults. do_sample=true is what a reasoning\n"
    "# model ships *because* greedy decoding degenerates; the runtime honors it\n"
    "# unless the caller forces --greedy (sampling resolution order, #385/#392).\n"
    "generation:\n"
    "  do_sample: true\n"
    "  temperature: 0.6\n"
    "  top_k: 20\n"
)

# Files reused verbatim from the tiny-llm fixture. The tokenizer is *not* here:
# it is copied then remapped (see CLOSE_TOKEN_*) so the reasoning span has a
# reachable close, so it must be regenerated whenever tiny-llm's is.
INHERITED_FILES = ("model.onnx.textproto",)

# The reachable-close remap. tiny-llm's greedy decode emits id 22 only on the
# "quick"/"fox"/"dog" prompts, at a position immediately followed by a real word,
# so renaming that one vocab entry to the closing delimiter makes those prompts
# close their reasoning span with a non-empty answer while every other prompt
# still degenerates. Renaming a vocab key changes decoded text only -- never a
# weight or an emitted id -- so the greedy attractors are preserved. Empirically
# grounded in the current tiny-llm weights; re-derive if that graph changes.
CLOSE_TOKEN_SOURCE = "tok22"
CLOSE_TOKEN = "</think>"


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=here.parent / "tiny-llm",
        help="Directory of the tiny-llm fixture this one derives from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here,
        help="Directory to receive the generated reasoning fixture.",
    )
    args = parser.parse_args()

    source_dir: Path = args.source_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in INHERITED_FILES:
        shutil.copyfile(source_dir / name, output_dir / name)

    # tokenizer.json: copy, then rename the one vocab entry the "quick" family's
    # greedy attractor reaches so that decodes to the closing delimiter. This is
    # what gives the fixture a reachable close (and thus a positive, committed
    # outcome) without touching the graph, weights, or any emitted token id.
    tokenizer = json.loads(
        (source_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    vocab = tokenizer["model"]["vocab"]
    if CLOSE_TOKEN_SOURCE not in vocab:
        raise SystemExit(
            f"expected {CLOSE_TOKEN_SOURCE!r} in the source vocab to rename to "
            f"{CLOSE_TOKEN!r}; the source tokenizer changed, re-derive the close token"
        )
    if CLOSE_TOKEN in vocab:
        raise SystemExit(f"{CLOSE_TOKEN!r} already in the source vocab; nothing to rename")
    # Preserve id and ordering: rename in place rather than pop+insert.
    tokenizer["model"]["vocab"] = {
        (CLOSE_TOKEN if key == CLOSE_TOKEN_SOURCE else key): value
        for key, value in vocab.items()
    }
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # tokenizer_config.json: the reasoning chat template.
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": CHAT_TEMPLATE}) + "\n",
        encoding="utf-8",
    )

    # inference_metadata.yaml: inherit the source model I/O contract and low
    # context, then append the author-declared generation defaults.
    source_metadata = (source_dir / "inference_metadata.yaml").read_text(
        encoding="utf-8"
    )
    if not source_metadata.endswith("\n"):
        source_metadata += "\n"
    (output_dir / "inference_metadata.yaml").write_text(
        source_metadata + "\n" + GENERATION_BLOCK,
        encoding="utf-8",
    )

    manifest = {
        "generator": "tests/fixtures/tiny-reasoning/generate_tiny_reasoning.py",
        "derived_from": "tests/fixtures/tiny-llm",
        "description": (
            "Tiny reasoning LLM: tiny-llm's graph + a <think> chat template + "
            "declared do_sample, with one vocab entry renamed to </think> so the "
            "reasoning span has a reachable close. Greedy degenerates on most "
            "prompts (no answer, exchange dropped) but closes and commits a "
            "non-empty answer on the 'quick'/'fox'/'dog' family, so CI can assert "
            "both halves of the reasoning-progress invariant on CPU: the "
            "degenerate drop and the non-empty-close commit. The 'quick' close "
            "also lands exactly on </think> at a 3-token budget, giving CI the "
            "closed-but-empty boundary that pins the non-empty-committed guard on "
            "the closed path."
        ),
        "reasoning_open_delimiter": "<think>",
        "reasoning_close_delimiter": "</think>",
        "close_token_source": CLOSE_TOKEN_SOURCE,
        "closes_under_greedy": "on the quick/fox/dog prompts only (empirical)",
        "generation_defaults": {"do_sample": True, "temperature": 0.6, "top_k": 20},
        "files": {
            name: (output_dir / name).stat().st_size
            for name in (
                "model.onnx.textproto",
                "tokenizer.json",
                "tokenizer_config.json",
                "inference_metadata.yaml",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
