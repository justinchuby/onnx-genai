#!/usr/bin/env python3
"""Generate the tiny *reasoning* decoder-only fixture.

Run from the onnx-genai repo root with only the Python standard library::

    python tests/fixtures/tiny-reasoning/generate_tiny_reasoning.py

Unlike ``tests/fixtures/tiny-llm/generate_tiny_llm.py`` this generator needs no
Mobius/torch: the reasoning fixture is a thin, fully regenerable derivation of
the already-committed ``tiny-llm`` graph. It reuses that model's ONNX graph and
tokenizer verbatim and layers three things on top that turn a plain tiny LLM
into a reasoning one:

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
   tokens and is cheap to hit in CI.

Why this reproduces the bug shape. The tiny tokenizer's vocabulary contains no
token that decodes to the closing ``</think>`` delimiter, so the reasoning span
opened by the template can never be closed by generated tokens. Under greedy
decoding the model therefore degenerates deterministically: it stays inside the
reasoning span, produces no answer, and hits the low context/token budget, at
which point the REPL drops the exchange rather than committing an empty turn.
That is Justin's "DeepSeek repeats its thinking and won't stop" defect reduced
to a CPU/ORT assertion (a context stop here stands in for the CUDA KV-capacity
stop the full-size model hit). See docs/research/testing/00-integration-stress-design.md.
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

# Files reused verbatim from the tiny-llm fixture.
INHERITED_FILES = ("model.onnx.textproto", "tokenizer.json")


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
            "declared do_sample. Degenerates under greedy (reasoning never "
            "closes) so CI can assert the reasoning-progress invariant on CPU."
        ),
        "reasoning_open_delimiter": "<think>",
        "reasoning_close_delimiter": "</think>",
        "closes_under_greedy": False,
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
