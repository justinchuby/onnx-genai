#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark the native ORT GenAI path for the paired Muse workflow test.

Local copy of the canonical `muse-native-harness/scripts/benchmark_muse_native.py`.
The canonical artifact is immutable and hash-pinned, so the two fixes it needs to
run live here instead:

* the canonical prompt check compares `tokenizer.encode(...)` against the
  canonical ID list with `!=`. Recent ORT GenAI returns a numpy array from
  `encode`, and `array != list` evaluates elementwise, so `if` on the result
  raises `ValueError: The truth value of an array with more than one element is
  ambiguous` — the exact-prompt gate fails open into a crash instead of
  comparing. `prompt_ids_match` compares element by element, and `--self-test`
  covers it without a GPU or the model package.
* the release gate is asserted from the config rather than assumed.

Run the regression check with:

    python scripts/benchmark_muse_native_local.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ["ORT_ENABLE_CUDNN_FLASH_ATTENTION"] = "0"

# Nothing below is imported at module scope: `--self-test` checks pure logic and
# must run on a machine with no GPU, no CUDA toolkit and no runtime installed.

_CUDART: Any = None

# Most specific first. A machine usually ships only the versioned soname; the
# bare name exists solely in a development toolkit install.
_CUDART_CANDIDATES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "cudart64_13.dll",
    "cudart64_12.dll",
)


def _load_runtimes() -> tuple[Any, Any]:
    """Import the runtimes, and only then. Returns `(onnxruntime, genai)`."""
    import onnxruntime as ort
    import onnxruntime_genai as og

    return ort, og


def _cudart() -> Any:
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    attempts = []
    for name in _CUDART_CANDIDATES:
        try:
            _CUDART = ctypes.CDLL(name)
            return _CUDART
        except OSError as error:  # noqa: PERF203 - the message names every attempt
            attempts.append(f"{name}: {error}")
    raise RuntimeError("could not load the CUDA runtime; tried " + "; ".join(attempts))


def _synchronize_cuda() -> None:
    status = _cudart().cudaDeviceSynchronize()
    if status:
        raise RuntimeError(f"cudaDeviceSynchronize failed with status {status}")


MINIMUM_RUNS = 3


def harness_checkout_is_clean() -> bool | None:
    """Whether *this* checkout has uncommitted changes, or `None` if unknown."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() == ""


def evaluate_gates(
    *,
    tokens: list[int],
    reference_tokens: list[int] | None,
    stable_across_runs: bool,
) -> dict[str, Any]:
    """Report each release gate this harness can decide, and say so when it cannot.

    The point is that every entry is *measured here*. A gate this harness has no
    way to observe is reported as `not_evaluated` rather than assumed, so a
    reader cannot mistake a native-only run for a paired result.
    """
    if reference_tokens is None:
        parity: Any = "not_evaluated: no reference token sequence was supplied"
    else:
        parity = tokens == reference_tokens
    return {
        # Native throughput alone cannot produce a ratio: there is no workflow
        # arm in this process.
        "min_throughput_ratio": "not_evaluated: this harness measures the native "
        "runtime only, so it cannot produce a paired ratio",
        "exact_token_parity": parity,
        "token_output_stable_across_runs": stable_across_runs,
        # CUDA Graph use is a property of the runtime's internals; nothing this
        # harness can read from Python proves capture happened.
        "cuda_graph_required": "not_evaluated: capture counters are not exposed to "
        "this harness",
        # This is a property of the *runtime under test*, which is an installed
        # wheel that need not come from this checkout at all. Reporting the
        # harness checkout's cleanliness in its place would be the same
        # asserted-not-measured mistake the other gates were changed to avoid.
        "clean_landed_runtime": "not_evaluated: the runtime under test is an "
        "installed package whose provenance this harness cannot establish",
        "harness_checkout_clean": harness_checkout_is_clean(),
        "upload_eligible": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(tokenizer: Any, text: str, *, include_image: bool) -> str:
    content: Any = [{"type": "image"}, {"type": "text", "text": text}]
    if not include_image:
        content = text
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)


def expanded_input_length(inputs: Any) -> int:
    """Number of input ids the *processor* produced.

    A multimodal processor expands each image placeholder into many ids, so the
    tokenizer's count is not the sequence the generator starts from. Search
    options are lengths of that sequence, so they must use this, not the
    tokenizer's count; getting it wrong silently shortens or lengthens decode.
    """
    ids = inputs["input_ids"]
    shape = getattr(ids, "shape", None)
    # `onnxruntime_genai.Tensor.shape` is a *method*; numpy's is an attribute.
    # Reading it the wrong way is not a graceful failure - subscripting a bound
    # method raises - and it is exactly the type the real processor returns.
    if callable(shape):
        shape = shape()
    if shape is not None:
        return int(shape[-1])
    row = ids[0] if len(ids) and hasattr(ids[0], "__len__") else ids
    return len(row)


def _run_once(
    model: Any,
    og: Any,
    processor: Any,
    prompt: str,
    image: Any,
    *,
    max_new_tokens: int,
    decode_skip: int,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    inputs = processor(prompt, images=image) if image is not None else processor(prompt)
    input_length = expanded_input_length(inputs)
    started = time.perf_counter()
    params = og.GeneratorParams(model)
    search_options: dict[str, Any] = {
        "max_length": input_length + max_new_tokens,
        "min_length": input_length + max_new_tokens,
        "do_sample": bool(sampling["do_sample"]),
        "temperature": float(sampling["temperature"]),
        "top_k": int(sampling["top_k"]),
        "top_p": float(sampling["top_p"]),
    }
    # Demanding a seed and then not passing it would leave a "reproducible"
    # sampling run unseeded.
    if sampling.get("seed") is not None:
        search_options["random_seed"] = int(sampling["seed"])
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    times: list[float] = []
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        generator.generate_next_token()
        _synchronize_cuda()
        times.append(time.perf_counter() - started)
        tokens.append(int(generator.get_next_tokens()[0]))
    if len(times) <= decode_skip:
        raise ValueError("max_new_tokens must exceed decode_skip")
    decode_seconds = times[-1] - times[decode_skip - 1]
    decode_tokens = len(times) - decode_skip
    return {
        "token_ids": tokens,
        "input_length": input_length,
        "ttft_ms": times[0] * 1000,
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "throughput_tok_s": decode_tokens / decode_seconds,
    }


def prompt_ids_match(encoded: Any, canonical: Any) -> bool:
    """Whether two token-ID sequences are exactly equal, element by element.

    `tokenizer.encode` may return a numpy array. `array != list` is an
    elementwise comparison whose truth value raises, so the sequences are
    compared as plain integer lists and never with a bare `!=`.
    """
    left = [int(token) for token in encoded]
    right = [int(token) for token in canonical]
    return left == right


def _self_test(*, require_numpy: bool = False) -> int:
    """Regression coverage for the prompt-equality check. No GPU, no model."""
    try:
        import numpy as np
    except ImportError:  # the pure-logic cases below do not need it
        np = None  # type: ignore[assignment]

    cases: list[tuple[str, Any, Any, bool]] = [
        ("equal lists", [1, 2, 3], [1, 2, 3], True),
        ("unequal lists", [1, 2, 3], [1, 2, 4], False),
        ("shorter encoding", [1, 2], [1, 2, 3], False),
        ("longer encoding", [1, 2, 3, 4], [1, 2, 3], False),
        ("empty", [], [], True),
    ]
    # The regression this guard exists for is a *numpy* encoding, which is what
    # `tokenizer.encode` returns. numpy is optional so the pure-logic cases can
    # run on a CI lane that has only the system interpreter; when it is present
    # the real cases run and are counted.
    if np is not None:
        cases += [
            ("equal numpy vs list", np.array([1, 2, 3], dtype=np.int32), [1, 2, 3], True),
            ("unequal numpy vs list", np.array([1, 2, 4], dtype=np.int32), [1, 2, 3], False),
            (
                "numpy differing only in the last position",
                np.array([5, 6, 7], dtype=np.int64),
                [5, 6, 8],
                False,
            ),
            ("numpy vs numpy", np.array([9, 9]), np.array([9, 9]), True),
            ("single element numpy", np.array([7]), [7], True),
        ]
    failures = 0
    for name, encoded, canonical, expected in cases:
        try:
            actual = prompt_ids_match(encoded, canonical)
        except Exception as error:  # noqa: BLE001 - the bug this guards against
            print(f"FAIL {name}: raised {type(error).__name__}: {error}")
            failures += 1
            continue
        if actual is not expected:
            print(f"FAIL {name}: expected {expected}, got {actual}")
            failures += 1
    # A bare `!=` on the numpy case is exactly the defect being guarded, but
    # numpy is not required to run this self-test: CI lanes that only have the
    # system interpreter must still be able to check the pure logic.
    if np is None and require_numpy:
        print(
            "FAIL: numpy is required for this self-test to gate the regression it exists "
            "for; install it or drop --require-numpy"
        )
        return 1
    if np is None:
        print("note: numpy is absent, skipping the array-inequality guard")
    else:
        try:
            bool(np.array([1, 2, 3]) != [1, 2, 3])
        except ValueError:
            pass
        else:
            print("FAIL: numpy array inequality no longer raises; the guard needs revisiting")
            failures += 1
    # The processor's expansion, not the tokenizer's count, is what the search
    # options must use.
    class _AttributeShape:
        """numpy-like: `shape` is a data attribute."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class _MethodShape:
        """`onnxruntime_genai.Tensor`-like: `shape` is a bound method."""

        def __init__(self, shape: list[int]) -> None:
            self._shape = shape

        def shape(self) -> list[int]:
            return self._shape

    length_cases = [
        ("2-D numpy-like", {"input_ids": _AttributeShape((1, 1543))}, 1543),
        ("1-D numpy-like", {"input_ids": _AttributeShape((77,))}, 77),
        ("2-D genai-like (shape is a method)", {"input_ids": _MethodShape([1, 1543])}, 1543),
        ("1-D genai-like (shape is a method)", {"input_ids": _MethodShape([77])}, 77),
        ("list of rows", {"input_ids": [[5, 6, 7, 8]]}, 4),
        ("flat list", {"input_ids": [5, 6, 7]}, 3),
    ]
    for name, inputs, expected in length_cases:
        actual = expanded_input_length(inputs)
        if actual != expected:
            print(f"FAIL expanded_input_length {name}: expected {expected}, got {actual}")
            failures += 1

    # Gates this harness cannot observe must say so, not default to a pass.
    gates = evaluate_gates(tokens=[1, 2], reference_tokens=None, stable_across_runs=True)
    if not str(gates["min_throughput_ratio"]).startswith("not_evaluated"):
        print("FAIL: a native-only run must not report a paired throughput ratio")
        failures += 1
    if not str(gates["cuda_graph_required"]).startswith("not_evaluated"):
        print("FAIL: CUDA Graph use must not be claimed without evidence")
        failures += 1
    if not str(gates["exact_token_parity"]).startswith("not_evaluated"):
        print("FAIL: parity must not be claimed without a reference sequence")
        failures += 1
    if not str(gates["clean_landed_runtime"]).startswith("not_evaluated"):
        print("FAIL: runtime provenance must not be claimed from the harness checkout")
        failures += 1
    if gates["upload_eligible"] is not False:
        print("FAIL: a native-only run must never imply upload eligibility")
        failures += 1
    matched = evaluate_gates(tokens=[1, 2], reference_tokens=[1, 2], stable_across_runs=True)
    if matched["exact_token_parity"] is not True:
        print("FAIL: parity against a matching reference must be reported as true")
        failures += 1
    differed = evaluate_gates(tokens=[1, 2], reference_tokens=[1, 3], stable_across_runs=True)
    if differed["exact_token_parity"] is not False:
        print("FAIL: parity against a differing reference must be reported as false")
        failures += 1

    checks = len(cases) + len(length_cases) + 7
    if failures:
        print(f"{failures} self-test case(s) failed")
        return 1
    print(f"self-test: {checks} cases passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the prompt-equality regression cases and exit. Needs no GPU or model.",
    )
    parser.add_argument(
        "--require-numpy",
        action="store_true",
        help="Fail if numpy is unavailable. The regression this self-test guards is a "
        "numpy array compared against a list, so a gate that silently skips those "
        "cases would not be a gate.",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/native_decode_example.json"),
        help="Workload description. The default is the checked-in example, which "
        "documents every field and is validated the same way a real one is.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return _self_test(require_numpy=args.require_numpy)
    if args.model is None or args.output is None:
        parser.error("--model and --output are required unless --self-test is given")
    if not args.config.is_file():
        parser.error(f"no benchmark config at {args.config}")
    config = json.loads(args.config.read_text())
    workload = config["workload"]
    sampling = config["sampling"]

    # The workload has to be self-consistent before anything is loaded, so a
    # bad config fails in a second rather than after a warmup.
    max_new_tokens = int(workload["max_new_tokens"])
    decode_skip = int(workload["decode_skip"])
    runs_requested = int(workload["runs"])
    if not 1 <= decode_skip < max_new_tokens:
        raise ValueError(
            f"decode_skip must be at least 1 and below max_new_tokens, got {decode_skip} "
            f"with max_new_tokens={max_new_tokens}"
        )
    if runs_requested < MINIMUM_RUNS:
        raise ValueError(
            f"a reported median needs at least {MINIMUM_RUNS} runs, config asks for "
            f"{runs_requested}"
        )
    if bool(sampling["do_sample"]) and sampling.get("seed") is None:
        raise ValueError(
            "sampling runs are only reproducible with an explicit seed; set sampling.seed, "
            "or set do_sample=false for greedy decoding"
        )

    # Every artifact this run depends on, checked before the first warmup.
    missing = [
        str(path)
        for path in (
            args.model,
            args.model / "genai_config.json",
            Path(workload["prompt_ids_file"]),
            *([Path(workload["image"])] if workload.get("image") else []),
            *(
                [Path(workload["reference_token_ids_file"])]
                if workload.get("reference_token_ids_file")
                else []
            ),
        )
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("missing benchmark artifacts: " + ", ".join(missing))

    if config["runtime"]["cudnn_flash_attention"]:
        raise ValueError("paired Muse benchmark requires cuDNN Flash Attention disabled")

    ort, og = _load_runtimes()
    if ort.__version__ != config["runtime"]["onnxruntime"]:
        raise RuntimeError(
            f"ORT version mismatch: {ort.__version__} != {config['runtime']['onnxruntime']}"
        )
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {providers}")

    model = og.Model(str(args.model))
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    image_path = workload.get("image")
    image = og.Images.open(image_path) if image_path else None
    prompt = (
        _prompt(tokenizer, workload["prompt"], include_image=True)
        if image is not None
        else workload["rendered_prompt"]
    )

    # Two different quantities, checked separately. The canonical prompt IDs are
    # a property of the *tokenizer* and pin the text; the sequence the generator
    # actually starts from is the processor's expansion, which for an image
    # prompt is much longer.
    encoded_prompt = tokenizer.encode(prompt)
    prompt_ids = json.loads(Path(workload["prompt_ids_file"]).read_text())
    if not prompt_ids_match(encoded_prompt, prompt_ids):
        raise RuntimeError("native tokenizer output differs from the canonical prompt IDs")
    prompt_tokens = len(encoded_prompt)
    if prompt_tokens != int(workload["prompt_tokens"]):
        raise RuntimeError(
            f"prompt token count mismatch: {prompt_tokens} != {workload['prompt_tokens']}"
        )

    for _ in range(int(workload["warmups"])):
        _run_once(
            model,
            og,
            processor,
            prompt,
            image,
            max_new_tokens=max_new_tokens,
            decode_skip=decode_skip,
            sampling=sampling,
        )

    runs = [
        _run_once(
            model,
            og,
            processor,
            prompt,
            image,
            max_new_tokens=max_new_tokens,
            decode_skip=decode_skip,
            sampling=sampling,
        )
        for _ in range(runs_requested)
    ]
    reference = runs[0]["token_ids"]
    stable = all(run["token_ids"] == reference for run in runs[1:])
    if not stable:
        raise RuntimeError("native token output changed across measured runs")

    # Compare against a stored native reference when the config names one, so
    # "exact_token_parity" is a measurement rather than a claim.
    reference_file = workload.get("reference_token_ids_file")
    reference_tokens = (
        json.loads(Path(reference_file).read_text()) if reference_file else None
    )
    gates = evaluate_gates(
        tokens=reference,
        reference_tokens=reference_tokens,
        stable_across_runs=stable,
    )
    if gates["exact_token_parity"] is False:
        raise RuntimeError("native output differs from the stored reference token sequence")

    record = {
        "kind": "native",
        "config": config,
        "environment": {
            "onnxruntime": ort.__version__,
            "onnxruntime_genai": getattr(og, "__version__", "unknown"),
            "providers": providers,
        },
        "package": {
            # Optional: a generic GenAI package has no workflow metadata.
            "metadata_sha256": (
                _sha256(args.model / "inference_metadata.yaml")
                if (args.model / "inference_metadata.yaml").is_file()
                else None
            ),
            "genai_config_sha256": _sha256(args.model / "genai_config.json"),
        },
        "release_gates": gates,
        "prompt_tokens": prompt_tokens,
        "input_length": runs[0]["input_length"],
        "metrics": {
            "ttft_ms": statistics.median(run["ttft_ms"] for run in runs),
            "throughput_tok_s": statistics.median(run["throughput_tok_s"] for run in runs),
            "ttft_ms_runs": [run["ttft_ms"] for run in runs],
            "throughput_tok_s_runs": [run["throughput_tok_s"] for run in runs],
            # A median with no spread beside it hides a contended machine.
            "ttft_ms_range": [
                min(run["ttft_ms"] for run in runs),
                max(run["ttft_ms"] for run in runs),
            ],
            "throughput_tok_s_range": [
                min(run["throughput_tok_s"] for run in runs),
                max(run["throughput_tok_s"] for run in runs),
            ],
        },
        "token_ids": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
