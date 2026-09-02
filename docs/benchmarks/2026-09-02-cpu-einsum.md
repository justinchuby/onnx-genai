# Native CPU Einsum — synthetic benchmark handoff

## Scope

`crates/onnx-runtime-ep-cpu/benches/einsum.rs` exercises the canonical CPU
Einsum lowering on synthetic shapes. It is not a model benchmark: the repository
model census currently contains no real Einsum node.

The native arm is timed. The generic f64 evaluator is run once per case as a
correctness oracle and is deliberately not a Criterion arm: it is a diagnostic
loop, not a replaceable native baseline. Relative factors against that oracle
are therefore not speedups and are withdrawn from this document.

The zero-node real-model census is also the ceiling on the present claim: these
rows establish synthetic kernel behavior only. They do not predict model
throughput, end-to-end latency, or the frequency of a class that does not occur
in the repository's real-model corpus.

BFloat16 is absent from the harness because the canonical ONNX Einsum
opset-12 type constraint does not permit it. This PR does not expand the schema.
The native CPU kernel executes Float32 and Float16 only.

## Synthetic case inventory

| class / equation | shape, dtype, layout |
|---|---|
| small GEMM `ik,kj->ij` | `4x16 · 16x4`, f32, contiguous |
| GEMM `ik,kj->ij` | `32x256 · 256x256`, f32, contiguous |
| large GEMM `ik,kj->ij` | `64x512 · 512x256`, f32, contiguous |
| transpose-required `ik,jk->ij` | `32x256 · 128x256`, f32, strided B view |
| broadcast BMM `...mk,...kn->...mn` | `[4,16,128] · [1,128,64]`, f32 |
| GEMM `ik,kj->ij` | `32x256 · 256x128`, f16 |
| reduction `ij->i` | `[512,512]`, f32, contiguous suffix reduction |
| elementwise `ij,ij->ij` | two `[512,512]` f32 tensors |
| flattened GEMM + output permutation `abxy,xycd->dcab` | `[4,4,8,8] · [8,8,4,4]`, f32 |
| diagonal copy fallback `ii->i` | `[1024,1024]`, f32 |
| zero-copy permutation metadata `abc->bca` | `[32,64,128]`, f32 strided output view |
| equivalent native control | MatMul `32x256 · 256x256`, f32, the exact `gemm_friendly` input buffers and layout |

Kernel construction/planning occurs outside every timed loop. The harness
reports native and diagnostic-oracle setup separately, warmed allocation
counts/bytes, reusable Einsum workspace, process CPU time, foreign CPU load,
SMT-sibling load, CPU-frequency samples, physical-core affinity, and two-ended
host-lock provenance.

Every optimized validation dispatch and its diagnostic oracle consume the same
input tensors. The harness prints a shared-input hash, requires at least one
nonzero oracle element, reports native/oracle hashes and maximum absolute error,
and refuses when the observed native route differs from the expected canonical
route. The view-only row proves the zero-copy route through the returned input
index and exact strides.

## Governed invocation

Run the exact census and complete Criterion target from the repository root.
The checked-in runbook owns the host lock for the whole Cargo child process,
sets the admission gate and physical-core budget, and requires an absolute
Cargo target directory:

```bash
EINSUM_TARGET="$PWD/target-einsum-evidence-$(git rev-parse --short=12 HEAD)"
crates/onnx-runtime-ep-cpu/benches/run_einsum.sh \
  census cpu-bench 16 "$EINSUM_TARGET"
crates/onnx-runtime-ep-cpu/benches/run_einsum.sh \
  run cpu-bench 16 "$EINSUM_TARGET"
```

The census must report `selector census passed: 12/12` before the timed run.
The runbook's lock wraps the actual `cargo bench --bench einsum` invocation so
compilation, warmup, every case, and every control are covered by one custody
interval. Merely reading the lock from inside the benchmark does not establish
custody. Choose a physical-core budget valid for the machine and a fresh target
directory; a completed prior run is rejected rather than overwritten.

## Fail-closed evidence contract

The benchmark refuses a publishable run unless all of these hold:

1. `hostlock.sh provenance` says `HELD`, `declared=yes`, flag-sourced owner,
   `lock_scope=box`, `contended=no`, satisfied admission gate, no takeover, and
   no live legacy-path holder;
2. tracked worktree state is clean and exact commit/tree/branch plus rustc are
   printed;
3. `ONNX_GENAI_CPU_DECODE_THREADS` is explicit and the realized process mask
   contains exactly one logical CPU for each distinct physical package/core;
4. the CPU model and `scaling_cur_freq` are readable for every allowed CPU;
5. a fresh `CARGO_TARGET_DIR` preserves the complete Criterion raw tree;
6. every absolute, interleaved, null, Criterion, and full-sweep window remains
   protected, has complete process-own-time accounting, `foreign_pct <= 5`,
   `sibling_peak_pct <= 25`, and at most 20% median-frequency drift;
7. every reported arm has at least three independent repetitions, absolute
   full range is at most 15%, and control full range is at most 10%;
8. the two identical MatMul A/A arms differ by at most 3% in median.

The target has 12 exact Criterion selectors with `sample_size=10`. In addition,
it emits three raw absolute repetitions per synthetic case and deterministic
ABBA/BAAB blocks with six repetitions per arm for:

- the only comparative row: contiguous f32 `ik,kj->ij` Einsum versus existing
  native MatMul with identical buffers, shape, dtype, layout, and output; and
- an A/A null made from two identical native MatMul kernels.

The comparison is reported as wrapper/path overhead relative to the equivalent
native MatMul control, not as a speedup over an oracle. No other class has a
replaceable baseline in this PR, so those rows remain absolute.

Complete Criterion `sample.json`, `estimates.json`, `benchmark.json`, and
`tukey.json` files remain under the dedicated target directory. The stdout raw
rows carry setup, validation, route, allocation, per-repetition, summary,
contention, frequency, and custody evidence needed to interpret them.

No speedup claim is made by this revision.

## Coverage ceiling and remaining work

The kernel intentionally declines N-way coupled contractions, mixed
operand-local-reduction contractions, and reduced-ellipsis contractions.
View-only permutation and diagonal extraction use `view_outputs`; direct/plugin
calls retain a copy fallback. Binary contractions lower through the existing
MatMul implementation, and layouts that cannot collapse to a view use bounded
Float32 materialization.

Potential follow-up work includes reducing direct MatMul panel allocations and
precomputing shape-specialized layout descriptors. Those are hypotheses, not
measured gains.
