# Execution-provider conformance

This is the first, deliberately narrow EP conformance slice. It drives nxrt's
real loader, optimizer, session, and CPU execution provider and compares its
outputs with `onnx.reference.ReferenceEvaluator`. It is a baseline, not a claim
of exhaustive ONNX conformance.

## Architecture

`crates/onnx-runtime-session/examples/conformance_runner.rs` is a dependency-free
Rust process bridge:

1. load an ONNX model with `InferenceSession::load`;
2. read any graph inputs from `input_N.nxrt`;
3. run the session (currently the default `onnx-runtime-ep-cpu`);
4. write ordered outputs to `output_N.nxrt`; and
5. print one machine-readable status: `OK`, `UNSUPPORTED_OP`, or `ERROR`.

The `.nxrt` tensor interchange is intentionally small: an eight-byte magic,
ONNX dtype byte, rank, `u64` dimensions, payload length, and contiguous
little-endian bytes. It avoids adding a Rust NPY/JSON dependency and supports
future models with runtime inputs, although the current `onnx-tests` generators
embed their drawn arrays as initializers.

`conformance/run_onnx_tests.py` draws deterministic, small, non-empty float32
examples from the external `cbourjau/onnx-tests` Hypothesis strategies, runs the
ONNX reference evaluator, invokes the Rust bridge, and classifies each case:

- `PASS`: shapes, dtypes, and values agree (`rtol=1e-4`, `atol=1e-5`);
- `UNSUPPORTED`: the session returned its unsupported-op error;
- `MISMATCH`: nxrt ran but returned different values; or
- `ERROR`: generation, parsing, execution, or process failure.

Spox adds output-only `Identity` wrappers to generated graphs. The driver
removes those wrappers before execution so an unimplemented `Identity` does not
hide the target operator's result. Eighteen CPU-op cases and the `Abs` and
`Sigmoid` negative cases come directly from `onnx-tests`. Focused ONNX models
cover seven CPU registrations not suitable for this first generator slice:
`LayerNormalization`, `Shape`, `Constant`, and `Gemm` have no corresponding
suite generator; `ReduceMean` and `Unsqueeze` currently exercise nxrt's legacy
attribute-form kernels; and the upstream `Gather` strategy can raise while
constructing an empty-axis case before the driver's non-empty predicate applies.
`Conv` is the third focused negative case.

## Run

The external suite must remain outside this repository.

```bash
git clone https://github.com/cbourjau/onnx-tests /home/justinchu/onnx-tests
python3 -m pip install --user "hypothesis>=6.130.4" "spox>=0.16" "onnx>=1.21"

source "$HOME/.cargo/env"
cargo build -p onnx-runtime-session --example conformance_runner --offline
python3 conformance/run_onnx_tests.py \
  --onnx-tests /home/justinchu/onnx-tests
```

Scratch models and tensor files are written under ignored `target/ep-conformance`.
Use `--json PATH` only when a disposable machine-readable report is wanted.

## Results

Run on 2026-07-14 against `cbourjau/onnx-tests` commit `856e89b`, with nxrt
branch `squad/nxrt-ep-conformance`.

| Result | Cases |
|---|---:|
| PASS | 25 |
| UNSUPPORTED | 3 |
| MISMATCH | 0 |
| ERROR | 0 |
| **Total** | **28** |

| Status | Operators |
|---|---|
| PASS | MatMul, Add, Relu, Reshape, Transpose, Gather, LayerNormalization, Sub, Mul, Div, Pow, Min, Max, Sqrt, Erf, Tanh, Cast, ReduceMean, Softmax, Shape, Unsqueeze, Expand, Slice, Constant, Gemm |
| UNSUPPORTED | Abs, Conv, Sigmoid |
| MISMATCH | None |
| ERROR | None |

Each operator has one baseline case. This proves the end-to-end harness,
enumerates every default-domain op in `PHASE1_OPS`, and proves unsupported
classification; it does not yet cover all dtypes, shapes, attributes, opsets,
optional inputs, or contrib-domain fused kernels.

## Diagnostic finding

The runner preserves the current nxrt diagnostic:

```text
op type not supported by any available EP: Abs
```

and adds that the selected EP has no registered kernel. The underlying session
message is understandable but does **not** yet satisfy `RULES.md` section 1:
it omits the node name, domain, opset, selected/available EPs, and a concrete
remediation (for example, choose an EP that registers the op or implement the
kernel). Improving `SessionError::UnsupportedOp` to carry and display that
context is a follow-up exposed by this harness.

## Scale-up plan

1. Add an nxrt runtime adapter to the suite's existing `RUN_CANDIDATE`/pytest
   path, backed by this same process bridge, so its normal parametrization,
   xfail/skip files, shrinking, and repeated Hypothesis examples produce the
   full `ai.onnx` matrix.
2. Stop stripping `Identity` once the CPU EP implements it; until then, keep the
   transformation narrowly limited to output-only wrappers.
3. Add explicit profiles per EP: supported opsets, dtypes, shape constraints,
   tolerances, and expected unsupported cases. Exercise multiple examples and
   retain minimized mismatch reproducers outside git/through CI artifacts.
4. Extend modern `ReduceMean`/`Unsqueeze` coverage and use the suite's `Gather`
   generator after its zero-axis construction failure is fixed. Add the four
   missing suite generators upstream.
5. Include contrib-domain fused registrations in a separate optimizer/fusion
   profile; they are not `ai.onnx` operators.
6. Add `--ep cpu|cuda` after `InferenceSession` exposes explicit provider
   selection. The current construction path hard-codes CPU auto-detection, so
   selecting `onnx-runtime-ep-cuda` is not yet an honest runner option. On an
   H200 CI worker, build the CUDA EP, select it through that API, reuse the same
   model/tensor protocol and Python comparisons, and publish a separate
   per-CUDA-EP report rather than silently falling back to CPU.
