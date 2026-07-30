#!/usr/bin/env python3
"""Tests for the stop-token guard in ``write_tokenizer_assets.py``.

``runtime_stop_ids`` is a HAND-WRITTEN PYTHON MIRROR of ``load_eos_token_ids``
in ``crates/onnx-genai-ort/src/tokenizer.rs``. The two cannot share code across
the Rust/Python boundary, so the mirror can silently drift from its original --
and a guard that resolves the stop token differently from the runtime is worse
than no guard, because it certifies a model that does not stop.

So this file has two halves, and the second is the load-bearing one:

* ``RuntimeStopIdsBehaviour`` pins what the mirror DOES, including the two
  false-pass traps its own docstring names.
* ``MirrorsTheRustRuntime`` reads ``tokenizer.rs`` and FAILS WHEN THE TWO
  DISAGREE -- a new config file, a renamed key, or an early return added to the
  Rust turns this red instead of leaving a stale mirror quietly certifying
  models. Every check there carries an anti-vacuity floor, because a drifted
  regex that matches nothing reports a spotless mirror in the same green as a
  correct one.

Run: ``python3 -m unittest discover -s scripts/lib -p 'test_*.py'``
"""

from __future__ import annotations

import io
import json
import re
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_tokenizer_assets import (  # noqa: E402
    collect_generation_eos_ids,
    declared_stop_tokens,
    eos_token_string,
    main,
    runtime_stop_ids,
    token_to_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_TOKENIZER = REPO_ROOT / "crates" / "onnx-genai-ort" / "src" / "tokenizer.rs"

# Qwen2.5-Instruct's real stop token lives ONLY in added_tokens, which is the
# case that motivated this whole script.
IM_END = "<|im_end|>"


def write_model_dir(directory: Path, **files: object) -> dict[str, Path]:
    """Write JSON fixtures and return the {name: path} mapping the guard takes."""
    mapping: dict[str, Path] = {}
    for stem, payload in files.items():
        name = f"{stem}.json"
        path = directory / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        mapping[name] = path
    return mapping


def tokenizer_json(added: dict[str, int] | None = None, vocab: dict[str, int] | None = None) -> dict:
    return {
        "added_tokens": [{"content": token, "id": ident} for token, ident in (added or {}).items()],
        "model": {"vocab": dict(vocab or {})},
    }


class RuntimeStopIdsBehaviour(unittest.TestCase):
    """The two traps ``runtime_stop_ids``'s docstring names as false passes."""

    def test_unions_both_files_instead_of_stopping_at_the_first(self):
        # THE TRAP: every current build writes generation_config.json, so a
        # checker that returns on the first hit would never open
        # tokenizer_config.json -- the file this script exists to supply.
        with TemporaryDirectory() as tmp:
            files = write_model_dir(
                Path(tmp),
                generation_config={"eos_token_id": 151643},
                tokenizer_config={"eos_token": IM_END},
                tokenizer=tokenizer_json(added={IM_END: 151645}),
            )
            ids, dropped = runtime_stop_ids(files)

        self.assertEqual(dropped, [])
        self.assertEqual(
            ids,
            [151643, 151645],
            "the runtime UNIONS both config files; a mirror that preferred one and "
            "stopped would miss the stop token this script exists to install",
        )

    def test_a_token_absent_from_the_vocabulary_is_reported_not_counted(self):
        # The runtime resolves literals through the tokenizer and silently drops
        # what it cannot find. A plausible literal in JSON is not evidence.
        with TemporaryDirectory() as tmp:
            files = write_model_dir(
                Path(tmp),
                tokenizer_config={"eos_token": IM_END},
                tokenizer=tokenizer_json(vocab={"<|endoftext|>": 151643}),
            )
            ids, dropped = runtime_stop_ids(files)

        self.assertEqual(ids, [])
        self.assertEqual(dropped, [IM_END])

    def test_added_tokens_are_consulted_before_the_vocabulary(self):
        # Load-bearing ordering: <|im_end|> exists ONLY in added_tokens. A lookup
        # that consulted model.vocab alone would call it unresolvable.
        resolved = token_to_id(tokenizer_json(added={IM_END: 151645}), IM_END)
        self.assertEqual(resolved, 151645)

    def test_a_boolean_is_not_a_token_id(self):
        # bool subclasses int in Python but not in Rust; the mirror must not
        # inherit a truthiness bug the original cannot have.
        ids: list[int] = []
        collect_generation_eos_ids(True, ids)
        self.assertEqual(ids, [])

    def test_nested_arrays_of_ids_are_flattened(self):
        ids: list[int] = []
        collect_generation_eos_ids([1, [2, [3]], 1], ids)
        self.assertEqual(ids, [1, 2, 3], "ids are collected recursively and deduped")

    def test_eos_token_may_be_an_object_with_content(self):
        self.assertEqual(eos_token_string({"content": IM_END}), IM_END)
        self.assertEqual(eos_token_string(IM_END), IM_END)
        self.assertIsNone(eos_token_string(""))
        self.assertIsNone(eos_token_string({"content": ""}))

    def test_a_model_declaring_nothing_resolves_to_nothing(self):
        # An empty id list is what sends the runtime to its hardcoded guesses,
        # which is the silent never-stops failure this guard exists to catch.
        with TemporaryDirectory() as tmp:
            files = write_model_dir(Path(tmp), tokenizer=tokenizer_json(vocab={"a": 1}))
            ids, dropped = runtime_stop_ids(files)

        self.assertEqual((ids, dropped), ([], []))

    def test_a_stale_companion_is_visible_as_a_per_file_declaration(self):
        # The stale-leftover case: presence satisfies an existence check while
        # naming a different stop token than the source model. declared_stop_tokens
        # must keep the two files separate so they can be compared like-for-like.
        with TemporaryDirectory() as tmp:
            files = write_model_dir(
                Path(tmp),
                generation_config={"eos_token_id": [151643]},
                tokenizer_config={"eos_token": "<|endoftext|>"},
            )
            declared = dict(declared_stop_tokens(files))

        self.assertEqual(declared["tokenizer_config.json"], "<|endoftext|>")
        self.assertEqual(declared["generation_config.json"], "[151643]")


class MirrorsTheRustRuntime(unittest.TestCase):
    """Fails when the Python mirror and the Rust original disagree.

    The precedence cannot be shared across the language boundary, so it is
    pinned here instead. These read the Rust source rather than executing it:
    that is a real limit, stated rather than hidden -- it catches a changed
    SHAPE (a new file, a renamed key, an added early return), not a changed
    semantic within an unchanged shape.
    """

    @classmethod
    def setUpClass(cls):
        cls.source = RUST_TOKENIZER.read_text(encoding="utf-8")
        match = re.search(r"\nfn load_eos_token_ids\b.*?\n\}\n", cls.source, re.DOTALL)
        cls.body = match.group(0) if match else ""

    def test_the_original_is_findable_so_a_green_here_means_something(self):
        # ANTI-VACUITY. Every assertion below scans self.body. If the function
        # were renamed or the file moved, an empty body would satisfy the
        # "no early return" check and report a faithful mirror.
        self.assertTrue(RUST_TOKENIZER.is_file(), f"{RUST_TOKENIZER} is gone")
        self.assertGreater(
            len(self.body),
            200,
            "could not extract load_eos_token_ids from tokenizer.rs — this mirror "
            "test is scoring an empty string and every check below is vacuous",
        )

    def test_consults_exactly_the_config_files_the_mirror_consults(self):
        found = set(re.findall(r'join\("([a-z_]+\.json)"\)', self.body))
        self.assertEqual(
            found,
            {"generation_config.json", "tokenizer_config.json"},
            "the Rust runtime reads a different set of config files than "
            "runtime_stop_ids() mirrors — update the mirror, not this test",
        )

    def test_reads_exactly_the_json_keys_the_mirror_reads(self):
        self.assertIn('get("eos_token_id")', self.body)
        self.assertIn('get("eos_token")', self.body)

    def test_does_not_short_circuit_between_the_two_files(self):
        # The union property, asserted structurally. `?` may return on an IO or
        # parse ERROR; what must not exist is a value-path early return, which
        # would make the second file conditional on the first.
        returns = re.findall(r"\breturn\b", self.body)
        self.assertEqual(
            returns,
            [],
            "load_eos_token_ids gained an early return, so it may no longer union "
            "both config files. runtime_stop_ids() still unions them and would now "
            "certify a stop token the runtime never resolves",
        )

    def test_the_documented_fallback_literals_still_match_the_runtime(self):
        # write_tokenizer_assets.py's module docstring quotes this list as the
        # reason an absent stop token fails silently. Prose drifts from code.
        literals = re.findall(r'"(<\|endoftext\|>|</s>|<eos>|\[EOS\])"', self.body)
        self.assertEqual(
            literals,
            ["<|endoftext|>", "</s>", "<eos>", "[EOS]"],
            "the runtime's hardcoded fallback list changed; the module docstring "
            "in write_tokenizer_assets.py quotes it and is now wrong",
        )

    def test_the_cited_source_file_still_defines_the_cited_symbol(self):
        # write_tokenizer_assets.py cites tokenizer.rs by path. Deliberately NOT
        # pinning the line number: a positional citation rots on every edit above
        # it, and this repository has been deleting those all night.
        module = (Path(__file__).parent / "write_tokenizer_assets.py").read_text(encoding="utf-8")
        self.assertIn("crates/onnx-genai-ort/src/tokenizer.rs", module)
        self.assertIn("fn load_eos_token_ids", self.source)


class StaleCompanionFailsTheBuild(unittest.TestCase):
    """End-to-end: the leftover-companion case, driven through ``main()``.

    A local directory short-circuits ``resolve_source_dir``, so this runs with
    no network and no third-party import. This is the case the guard was built
    for: presence satisfies an existence check while the file names a DIFFERENT
    stop token than the source model, because the script never overwrites.
    """

    def drive(self, source: Path, output: Path) -> tuple[int, str]:
        argv = sys.argv
        sys.argv = ["write_tokenizer_assets.py", str(source), str(output)]
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = main()
        finally:
            sys.argv = argv
        return code, out.getvalue() + err.getvalue()

    def build(self, tmp: str, source_token: str, output_token: str | None) -> tuple[Path, Path]:
        source, output = Path(tmp) / "src", Path(tmp) / "out"
        source.mkdir()
        output.mkdir()
        write_model_dir(source, tokenizer_config={"eos_token": source_token})
        if output_token is not None:
            write_model_dir(output, tokenizer_config={"eos_token": output_token})
        return source, output

    def test_a_leftover_declaring_a_different_stop_token_fails_the_build(self):
        with TemporaryDirectory() as tmp:
            source, output = self.build(tmp, IM_END, "<|endoftext|>")
            code, text = self.drive(source, output)

        self.assertEqual(code, 1, f"a stale companion was accepted:\n{text}")
        self.assertIn(IM_END, text, "the error must name what the SOURCE declares")
        self.assertIn("<|endoftext|>", text, "and what the LEFTOVER declares")

    def test_a_companion_that_agrees_with_the_source_passes(self):
        # THE CONTROL THAT MUST STAY GREEN. Without it, a guard that failed on
        # every pre-existing file would satisfy the test above and break every
        # rebuild -- and a guard that reddens on correct work gets switched off.
        with TemporaryDirectory() as tmp:
            source, output = self.build(tmp, IM_END, IM_END)
            code, text = self.drive(source, output)

        self.assertEqual(code, 0, f"an agreeing companion was rejected:\n{text}")

    def test_a_model_that_declares_no_stop_token_at_all_fails(self):
        with TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "src", Path(tmp) / "out"
            source.mkdir()
            output.mkdir()
            write_model_dir(source, tokenizer_config={})
            write_model_dir(output, tokenizer_config={})
            code, text = self.drive(source, output)

        self.assertEqual(code, 1, f"a model with no stop token was accepted:\n{text}")
        self.assertIn("declares no stop token", text)


if __name__ == "__main__":
    unittest.main()
