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

### Op-coverage wave (2026-07-14)

Registered 25 new ai.onnx operators as dedicated ep-cpu kernels (new files
`unary_math.rs`, `reduce_ops.rs`, `concat.rs`, `movement_ops.rs`, `logical.rs`,
`where_op.rs`). Before/after measured in the same environment (upstream commit
`856e89bcd3d2ee3cb31c7bf88b5706dec00eba5c`, 100 Hypothesis examples, `-n 8`):

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| dtype/opset pytest cases passing | 158 | 228 | +70 |

Per-operator additions (all previously "no CPU kernel registered"):

| Operator | Pass | Fail | Notes |
|---|---:|---:|---|
| Size | 13 | 0 | byte-agnostic, all dtypes |
| Concat | 12 | 1 | byte-agnostic; 1 fail = size-0 upstream ref bug |
| Squeeze | 12 | 1 | byte-agnostic (axes-as-input, opset-13) |
| Where | 12 | 1 | byte-agnostic broadcasting select |
| ReduceMax | 2 | 17 | f32; axes-as-input/attr, keepdims, noop |
| ReduceMin | 2 | 17 | f32 |
| ReduceSum | 1 | 6 | f32 |
| ReduceProd | 1 | 6 | f32 |
| ReduceL2 | 1 | 6 | f32 |
| ReduceSumSquare | 1 | 6 | f32 |
| Abs | 1 | 10 | f32 (non-f32 dtypes = Luv's wave) |
| Neg | 1 | 6 | f32 |
| Sign | 1 | 10 | f32; ONNX sign(0)=0, sign(NaN)=NaN |
| Reciprocal, Exp, Log, Sin, Cos, Ceil, Floor, Round, Sigmoid | 1 each | 2 each | f32; Round = round-half-to-even |
| Not | 1 | 0 | bool |
| Softplus | 0 | 3 | f32 kernel is numerically stable; upstream naive ref overflows to inf on extreme input (e.g. softplus(89)), an inherent reference-overflow mismatch like Add's 1/11 |
| Flatten | 0 | 13 | kernel correct; upstream `op_flatten.py` reference crashes on size-0 arrays |

Kernels are f32 (plus bool for Not/Where condition; byte-agnostic for
Concat/Squeeze/Size/Where). f16/f64/int dtype coverage is a separate wave.
Output shapes are supplied by `onnx-runtime-shape-inference`; these kernels
write into pre-shaped output views.

### Wave 6 CPU kernels (2026-07-15)

Added dedicated CPU kernels for **ScatterElements** (opsets 11/16), **OneHot** (opset 9),
and **Trilu** (opset 14), raising the ep-cpu registry from **132 to 136** entries. Trilu
accepts its optional scalar int64 `k` input (including negative values) and defaults to zero;
`upper` remains an attribute. The ep-cpu unit suite has **349** passing tests.

## ONNX backend test (`onnx.backend.test`)

### CPU node-model coverage (2026-07-19)

A fresh unfiltered ONNX 1.22.0 node-model run increased CPU coverage from the
previously recorded **921/1,765** passing cases to **936/1,765** (829
failures). CUDA remains unsupported by the CPU-only adapter, so all 1,765 CUDA
variants are skipped. The run used the pure-Rust wheel (`maturin develop
--release --no-default-features`) built from commit `39edb76`.

| Scope | Pass | Fail | Skip | Total |
|---|---:|---:|---:|---:|
| CPU node cases | 936 | 829 | 0 | 1,765 |
| CUDA variants (unsupported device) | 0 | 0 | 1,765 | 1,765 |
| **Collected node cases** | **936** | **829** | **1,765** | **3,530** |

Measured CPU node-model history:

- **2026-07-14:** 360 passed / 1,405 failed / 1,765 skipped.
- **2026-07-19:** 875 passed / 890 failed / 1,765 skipped.
- **2026-07-19:** 921 passed / 844 failed / 1,765 skipped.
- **2026-07-19:** 936 passed / 829 failed / 1,765 skipped.

The latest +15 cases come from the ReduceLogSumExp opset-18 fix,
ReduceMax/ReduceMin bool support, ReduceSum empty-set handling, and new Selu,
ThresholdedRelu, and LpNormalization kernels.

The largest current failing test-name/operator families are CastLike (96),
SoftmaxCrossEntropyLoss/SCE (68), Attention (66), Cast (52), and Resize (39).
These are coverage counts from the unfiltered suite, not expected failures;
newly implemented kernels turn red cases green without maintaining an xfail
list.

Two representative current failures confirm that the count measures nxrt
operator gaps rather than adapter wiring errors:

- `test_unique_length_1_cpu` fails while preparing the session because no CPU
  EP kernel is registered for `ai.onnx::Unique` at opset 11.
- `test_upsample_nearest_cpu` fails while preparing the session because no CPU
  EP kernel is registered for `ai.onnx::Upsample` at opset 9.

### Scan, window, bitwise, and Hardmax wave (2026-07-19)

This +46-case wave fixed **CumSum** (including int32, exclusive/reverse, and
negative-axis cases), added **CumProd**, **BitwiseAnd**, **BitwiseOr**,
**BitwiseXor**, **BitwiseNot**, and **Hardmax** (including fp16/bf16), plus
the **HannWindow**, **HammingWindow**, and **BlackmanWindow** kernels. The
window `*_expanded` reference-decomposition variants still fail because they
exercise a decomposition graph rather than the base window kernel.

### Previous CPU node-model coverage (2026-07-19)

The preceding unfiltered ONNX 1.22.0 node-model run increased CPU coverage from
the 2026-07-14 **360/1,765** passing cases to **875/1,765** (890 failures).
CUDA remained unsupported by the CPU-only adapter, so all 1,765 CUDA variants
were skipped. The run used the pure-Rust wheel (`maturin develop --release
--no-default-features`) built from commit `080f0ba`.

| Scope | Pass | Fail | Skip | Total |
|---|---:|---:|---:|---:|
| CPU node cases | 875 | 890 | 0 | 1,765 |
| CUDA variants (unsupported device) | 0 | 0 | 1,765 | 1,765 |
| **Collected node cases** | **875** | **890** | **1,765** | **3,530** |

The exact pass/fail/skip test-name set is recorded in
`crates/onnx-runtime-python/conformance/onnx_backend_node_results.txt`. Re-run
with:

```bash
cd /path/to/onnx-genai/crates/onnx-runtime-python
source .venv/bin/activate
maturin develop --release --no-default-features
python -m pytest tests/test_onnx_backend.py -q \
  --junitxml=../../target/onnx-backend-test/junit.xml
```

### CPU unary and activation coverage (2026-07-14)

Hythe added default-domain CPU registrations for `Acos`, `Acosh`, `Asin`,
`Asinh`, `Atan`, `Atanh`, `Cosh`, `Sinh`, `Tan`, `Elu`, `LeakyRelu`, and
`HardSigmoid`. The trigonometric kernels use the Rust `f32` intrinsics; the
activation kernels honor ONNX's `alpha`/`beta` defaults and attributes.

Using the command in the Run section after building and installing the local
wheel, the CPU node-case result increased from the recorded **130/1,765** to
**360/1,765** passing cases (1,405 failures). CUDA remained 1,765 skipped
cases. This comparison uses ONNX 1.22.0 and the unfiltered node-model group.

Run on **2026-07-14** with ONNX 1.22.0 against the nxrt wheel built from commit
`f2dd92d`. The runner exposes the official node-model group only, avoiding
offline model downloads. ONNX generates both CPU and CUDA variants; nxrt's
CPU-only backend executes the 1,765 CPU cases and skips all 1,765 CUDA cases.

## Sequence ops — copy-free, race-free container type (2026-07-14)

Added the ONNX *sequence-of-tensors* runtime value type and its core ops to the
session executor (not EP kernels — a `Kernel` sees only tensor views):
`SequenceEmpty`, `SequenceConstruct`, `SequenceInsert`, `SequenceErase`,
`SequenceAt`, `SequenceLength`, `SplitToSequence`, `ConcatFromSequence`. Routed
by `is_sequence_op`, mirroring the `If`/`Loop`/`Scan` control-flow path, so there
is **no** EP kernel-registry entry to conflict with tensor-op registration.

**Design (see `docs/OPERATORS.md` §6.9):**

- **No copy.** Elements are `Arc`-shared immutable `SeqTensor`s. Insert/erase/
  construct produce a new list that *shares* the surviving element `Arc`s
  (persistent-structure update); `SequenceAt` returns the shared element and
  backs its output tensor value with the same allocation (`seq_elem_values`),
  read zero-copy by downstream kernels. Only boundary crossings copy: the
  tensor→sequence entry (one move into the element `Arc`) and the single-alloc
  `Split`/`Concat` data movement.
- **No race.** Immutable elements shared read-only through `Arc`, no interior
  mutability → concurrent readers cannot race. `SequenceValue: Send + Sync`.

**Tests (all green):**

- 15 unit tests in `sequence.rs`, including the **no-copy proof**
  (`at_returns_shared_handle_no_copy` asserts `Arc::ptr_eq` + data-pointer
  equality between the inserted element and the `SequenceAt` result after
  intervening insert/erase), an `Arc`-strong-count sharing test, a **concurrency
  smoke test** (8 threads reading one shared sequence), and a `Send + Sync`
  assertion.
- 13 end-to-end integration tests in `tests/sequence_ops.rs` exercising each op
  through the public `InferenceSession` surface (construct/at, negative index,
  insert position + default append, erase, length, split (default/keepdims=0/
  explicit sizes), concat (existing axis + `new_axis=1` stack), a no-copy
  round-trip feeding `Identity`, and actionable out-of-bounds / sequence-graph-
  output error paths).

Deferred follow-up: `SplitToSequence` currently emits single-alloc contiguous
slices; making split outputs true strided *views* over the input (on top of the
zero-copy strided-view foundation) is a future optimization.
