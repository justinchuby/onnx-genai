# Native CPU Einsum — synthetic benchmark handoff

## Scope

`crates/onnx-runtime-ep-cpu/benches/einsum.rs` exercises the canonical CPU
Einsum lowering on synthetic shapes. It is not a model benchmark: the repository
model census currently contains no real Einsum node.

The native arm is timed. The generic f64 evaluator is run once per case as a
correctness oracle and is deliberately not a Criterion arm: it is a diagnostic
loop, not a replaceable native baseline. Relative factors against that oracle
are therefore not speedups and are withdrawn from this document.

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
| unaffected control | MatMul `32x256 · 256x256`, f32 |

Kernel construction/planning occurs outside the timed loop. The harness reports
native setup time, warmed allocation counts/bytes, reusable Einsum workspace,
process CPU time, foreign CPU load, SMT-sibling load, and host-lock provenance.
Those diagnostics remain available for a measured follow-up without turning
the oracle into a comparison baseline.

## Governed invocation

Run the complete Criterion target from the repository root under the host lock:

```bash
scripts/hostlock.sh run --owner your_name --reason "CPU Einsum synthetic Criterion sweep" -- \
  cargo bench -p onnx-runtime-ep-cpu --bench einsum -- --noplot
```

The lock must wrap the `cargo bench` invocation so compilation, warmup, every
case, and the MatMul control are covered by one custody interval. Merely reading
the lock from inside the benchmark does not establish custody. Replace
`your_name` with a stable identifier accepted by `hostlock.sh`.

## Measurement handoff

Sebastian owns the replacement performance evidence. A publishable result should:

1. record the exact revision SHA and `scripts/hostlock.sh provenance`;
2. preserve the exact `--bench einsum` selector and prove it lists a nonzero,
   expected benchmark census before timing;
3. report absolute native times per arm with raw Criterion artifacts;
4. use an actual replaceable implementation as any comparative baseline;
5. keep the f64 evaluator labelled only as a correctness oracle;
6. report the MatMul control and contention fields beside every run.

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
