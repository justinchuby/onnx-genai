#!/usr/bin/env python3
"""Generate the deterministic tokenizer for the tiny tool-call workflow fixture.

The checked-in ONNX graph is the deterministic tiny-llm graph. This tokenizer
maps its fixed output tokens to two tagged-json calls and then an ordinary
answer, making the HTTP request → calls → results → answer path hermetic. It
also retains deterministic observer-only tokens used by the engine protocol
boundary tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing


HERE = Path(__file__).resolve().parent


def tokenizer() -> Tokenizer:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|assistant|>": 4,
        '<tool_call>{"id":"call_weather","name":"weather","arguments":{"city":"Paris"}}</tool_call>': 5,
        "Results accepted.": 7,
        '<tool_call>{"id":"call_time","name":"time","arguments":{"timezone":"UTC"}}</tool_call>': 15,
        '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>': 22,
        "<tool-eos>": 26,
        "ordinary": 27,
        "assistant": 28,
        "text": 29,
    }
    vocab.update(
        {
            f"unused-{index}": index
            for index in range(4, 32)
            if index not in {4, 5, 7, 15, 22, 26, 27, 28, 29}
        }
    )
    result = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    result.pre_tokenizer = WhitespaceSplit()
    result.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 2), ("<eos>", 3)],
    )
    result.add_special_tokens(["<bos>", "<eos>", "<tool-eos>"])
    return result


def main() -> None:
    (HERE / "tokenizer.json").write_text(tokenizer().to_str(pretty=True) + "\n")
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tokenizer.json"] = (HERE / "tokenizer.json").stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
