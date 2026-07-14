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

## Next steps

1. Add per-EP expected-failure profiles and CI around the Python adapter
   described below, while retaining unfiltered runs as the coverage baseline.
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
6. Build a CUDA-enabled wheel on an H200 CI worker, select
   `CUDAExecutionProvider` explicitly through the adapter, and publish a
   separate CUDA report rather than silently falling back to CPU.

## Upstream onnx-tests (cbourjau)

This is a separate, broader result from the 28-case bespoke process runner
above. The upstream repository's normal `tests/` tree was run directly with
pytest through
`crates/onnx-runtime-python/tests/nxrt_runtime.py::run_nxrt`. The adapter uses
the Python binding, explicitly selects `CPUExecutionProvider`, and returns
ordered NumPy outputs using the suite's runtime contract.

### Run and headline

Run on 2026-07-14 against upstream commit
`856e89bcd3d2ee3cb31c7bf88b5706dec00eba5c`, using the suite's default 100
Hypothesis examples:

```bash
export PATH=/home/justinchu/.conda/envs/onnx/bin:$PATH
cd /path/outside/onnx-tests-upstream
python -m pip install -e . \
  "hypothesis>=6.130.4" "spox>=0.16" "ndonnx>=0.10.1" \
  "pytest-xdist>=3.8,<4"
PYTHONPATH=/path/to/onnx-genai/crates/onnx-runtime-python/tests \
RUN_CANDIDATE=nxrt_runtime.run_nxrt \
python -m pytest tests -q -n 8 --dist loadfile \
  --junitxml=nxrt-junit.xml
```

| Measure | Pass | Fail | Skip | Total |
|---|---:|---:|---:|---:|
| dtype/opset pytest cases | 158 | 1,038 | 2 | 1,198 |
| operator names with that overall status | 0 | 112 | 0 | 112 |

An operator is counted as failed in the second row if any of its dtype/opset
cases failed. Seventeen operators had at least one passing case: Add, Cast,
Div, Expand, Identity, MatMul, Max, Min, Mul, Pow, Relu, Reshape, Softmax,
Sqrt, Sub, Tanh, and Transpose. The two skips are the suite's undefined
bool-to-string and string-to-bool Cast cases.

### Failure coverage

| Failure class | Cases | Operators |
|---|---:|---|
| No CPU kernel registered | 784 | 90 operators listed below |
| Registered kernel is float32-only | 167 | Add, Div, Erf, Gather, MatMul, Max, Min, Mul, Pow, Relu, Reshape, Softmax, Sqrt, Sub, Tanh, Transpose |
| Kernel execution gap | 35 | Cast (Float16/Uint64 conversions), Slice (negative-axis and empty-output handling) |
| String tensor/binding gap | 29 | Cast, Expand, Gather, Identity, Reshape, Slice, Transpose |
| Modern input signature unsupported | 20 | ReduceMean (axes input), Unsqueeze (axes input) |
| Value mismatch | 2 | Erf near zero (absolute error `1e-9` with upstream `atol=0`); Slice |
| Upstream generator failure | 1 | Gather can generate an empty axis, then constructs `max_value=-1 < min_value=0` |

The operators with a registered or partially working path break down as
follows (counts are dtype/opset pytest cases):

| Operator | Pass | Fail | Skip | Failure reason |
|---|---:|---:|---:|---|
| Add | 1 | 10 | 0 | non-float32 dtypes |
| Cast | 120 | 47 | 2 | Float16/Uint64 conversions and strings |
| Div | 1 | 10 | 0 | non-float32 dtypes |
| Erf | 0 | 3 | 0 | float16/64 unsupported; float32 near-zero mismatch |
| Expand | 12 | 1 | 0 | strings |
| Gather | 0 | 13 | 0 | non-float32/string dtypes; upstream empty-axis generator error |
| Identity | 12 | 1 | 0 | Python string output conversion |
| MatMul | 1 | 6 | 0 | non-float32 dtypes |
| Max | 1 | 10 | 0 | non-float32 dtypes |
| Min | 1 | 10 | 0 | non-float32 dtypes |
| Mul | 1 | 10 | 0 | non-float32 dtypes |
| Pow | 1 | 54 | 0 | non-float32 input combinations |
| ReduceMean | 0 | 7 | 0 | opset-18 axes input |
| Relu | 1 | 6 | 0 | non-float32 dtypes |
| Reshape | 1 | 12 | 0 | non-float32/string dtypes |
| Slice | 0 | 13 | 0 | negative axis, empty output, value mismatch, strings |
| Softmax | 1 | 2 | 0 | float16/64 unsupported |
| Sqrt | 1 | 2 | 0 | float16/64 unsupported |
| Sub | 1 | 10 | 0 | non-float32 dtypes |
| Tanh | 1 | 2 | 0 | float16/64 unsupported |
| Transpose | 1 | 12 | 0 | non-float32/string dtypes |
| Unsqueeze | 0 | 13 | 0 | opset-13 axes input |

The 90 fully unsupported operator names are:

Abs, Acos, Acosh, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool,
BitShift, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, Ceil, Celu, Clip,
Compress, Concat, ConstantOfShape, Conv, ConvInteger, Cos, Cosh, CumSum,
DepthToSpace, Det, Einsum, Elu, Equal, Exp, EyeLike, Flatten, Floor,
GatherElements, Gelu, Greater, GreaterOrEqual, HardSigmoid, HardSwish, Hardmax,
IsInf, IsNaN, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, Mean, Mish, Mod,
Neg, NonZero, Not, Or, PRelu, Pad, Reciprocal, ReduceL1, ReduceL2,
ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMin, ReduceProd, ReduceSum,
ReduceSumSquare, Round, Selu, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Softplus,
Softsign, SpaceToDepth, Split, Squeeze, Sum, Tan, ThresholdedRelu, Tile, Trilu,
Unique, Where, and Xor.

The result is intentionally not normalized with expected-failure files: it is
the current ep-cpu coverage picture. `conformance/summarize_junit.py` prints
the reproducible per-operator pass/fail/skip table from the JUnit report.
