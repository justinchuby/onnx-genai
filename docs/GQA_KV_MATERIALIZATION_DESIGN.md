# CPU GroupQueryAttention — KV materialization design

Author: Roy (CPU-kernel performance)
Status: Tier A implemented; Tier B deferred (documented only)
Kernel: `crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs`

## 1. Problem

The pure-Rust CPU `GroupQueryAttention` kernel re-materializes the *entire*
KV history on every decode step, producing O(S²) copy traffic across a
generation instead of the O(S) that the attention math itself requires.

Concretely, on the pre-change kernel (line ranges relative to the unchanged
`origin/main` file `d7a0819`):

1. **Past is materialized once into an owned struct.** `Bhsd::from_cache`
   (lines 157–170) calls `to_dense_f32_widen(...).into_owned()`, forcing a full
   owned `Vec<f32>` clone of `past_key` and `past_value` even when the cache
   input is already contiguous `f32` (the common decode case, where
   `to_dense_f32_widen` can borrow with zero allocation).

2. **A fresh full present_k/present_v is allocated every call.** Lines 531–533
   `vec![0.0; …]` allocate the entire present K and V each step.

3. **Every past element is copied a second time with a scalar `.at()` loop.**
   Lines 534–543 iterate `b × kv_heads × past_len × cache_dim` and copy one f32
   at a time via `Bhsd::at`, recomputing a 4-D flat index (with bounds checks)
   per element. The current token(s) are then appended by the same scalar
   pattern at lines 545–554.

4. **The full present buffers are copied a third time into the outputs.** Lines
   634–645 write the whole `present_k`/`present_v` out again.

So past K/V data is touched at least three times per step (owned clone →
scalar per-element copy into present → output write), and each of those passes
is O(S). The **prefix concat** at 534–543 is the O(S)-per-step defect that
compounds to **O(S²)** over a generation.

Things that are correctly O(1) or necessarily O(S) and are **not** the defect
(left untouched):

- RoPE rotates only the *current* q/k before concat (lines 509–524) → O(1) at
  decode. Not the defect.
- The attention math — score dot products (575–588) and the V-weighted
  reduction (602–610) — is inherently O(S) per step and is *required*. Not
  touched.

## 2. Cost model

For Qwen2.5-0.5B (B=1, `kv_num_heads=2`, `D=64`, f32) the prefix-concat pass
alone moves, per layer per step:

```
kv_heads · D · 4 bytes · 2 (K and V) = 2 · 64 · 4 · 2 = 1024 bytes per past row (S)
⇒ 1024 · S bytes / layer / step
× 24 layers          ⇒ 24,576 · S bytes / step   (~3 MiB at S = 128)
```

On top of that, the redundant owned clone in `from_cache` (item 1) plus the
double output pass (item 4) roughly **triples** the avoidable cache-copy
traffic beyond the single necessary materialization.

At S = 512 the concat pass alone is ~12 MiB/step; while `memcpy` at
~10–15 GB/s handles that in well under a millisecond, the *scalar* `.at()`
variant pays per-element index arithmetic and bounds checks and cannot reach
`memcpy` bandwidth, and the extra owned clone doubles the resident traffic.

## 3. Two-tier fix

### Tier A — implemented here (contained, kernel-local)

Goal: eliminate the redundant per-element scalar copies **and** the double
materialization of the past, **without** changing the SSA / disjoint-output
execution contract (the kernel still writes a fresh, fully-populated
`present_k`/`present_v` output each call).

Two changes:

1. **Borrow the past instead of cloning it.** A new `PastCache<'a>` struct
   (kernel lines ~245–271) holds a `Cow<'a, [f32]>` from `to_dense_f32_widen`
   rather than calling `.into_owned()`. When the cache input is contiguous
   `f32` — the standard CPU decode path — the dense past data is *borrowed*
   directly from the input tensor, removing one full O(S) allocation + copy per
   K and per V, per layer, per step. (`Bhsd` is retained unchanged for Q/K/V,
   which are always freshly built owned tensors.)

2. **Contiguous slice copies instead of scalar `.at()` loops.** Both
   `present_k`/`present_v` and the dense past/current caches use BNSH layout, so
   for a fixed `(b, h)` the `[s, d]` block is a single contiguous run in every
   buffer. The fill (kernel lines ~551–577) now copies the whole past prefix
   with one `copy_from_slice` per (b, h), then appends the current token(s) with
   a second `copy_from_slice`, replacing the `b × h × s × d` scalar loops. This
   lets the compiler lower the copies to `memcpy`.

   Index derivation for fixed `(b, h)` with `head = b·kv_num_heads + h`:
   - present dst prefix: `[head·present_seq·D, head·present_seq·D + past_len·D)`
   - past src prefix:    `[head·past_seq·D,    head·past_seq·D    + past_len·D)`
   - present dst append: starts at `head·present_seq·D + past_len·D`, length `k.seq·D`
   - current src append: `[head·k.seq·D, head·k.seq·D + k.seq·D)`

   All four ranges are contiguous and stay within head `h`'s block because
   `past_len ≤ past_seq` (validated at kernel input) and `D = cache_dim = q.dim`
   is shared by past/current/present. This is correct for both prefill (`k.seq
   = S`) and decode (`k.seq = 1`) with no special-casing, since the append copy
   is length `k.seq·D` in either case.

The double output pass (item 4 above) is left as-is: it is part of the SSA
contract (the engine binds the present K/V as owned outputs), so removing it
belongs to Tier B.

### Tier B — deferred (cross-cutting, document only; NOT implemented)

The residual O(S²) comes from allocating and writing a full `present_k`/
`present_v` every step. The real fix is a **shared append-only KV buffer**:

- Cache shape `[B, N_kv, max_len, D]` allocated once for the sequence.
- Each decode step appends one token at `past_seq` (an O(1) write of `D` f32),
  and attention reads `[0, past_seq + 1)` directly from that buffer — no
  per-step concat, no fresh present allocation, no output copy.

This requires kernel input/output **aliasing** (present K/V alias past K/V) plus
engine-side KV-lifecycle / binding changes so the runtime hands the kernel a
persistent, growable buffer. The ORT-backed runner already models this via
`SharedBuffer` metadata (see `docs/KV_INSERTION_DESIGN.md:48-57`), but the
**pure-Rust CPU kernel/runtime path does not** — it always materializes
present outputs. Wiring that through is out of scope for this kernel-local
slice.

Additionally, sliding-window / attention-sink eviction (`local_window_size`)
breaks the naive append-only model once `past_seq` reaches `max_len`: it needs
an explicit **compaction / ring-buffer / paged** policy to decide what to evict
and how to keep the live window contiguous. That policy is deferred with Tier B.

## 4. Correctness invariants (preserved)

- Rotate the **new** k only; current-position handling lives at kernel lines
  471–505, using `past_seq` for positions. RoPE untouched.
- GQA head map `kvh = qh / (num_heads / kv_num_heads)` (score/reduction loops).
- Causal / local-window read range (`local_start..=causal_limit`).
- BNSH layout `[B, kv_num_heads, S, D]`; present index
  `((b·kvh + h)·present_seq + s)·cache_dim + d`.
- `decode_fast_write` path gated on `q.seq == 1 && k.seq == 1`.
- Output shapes, attention math, softmax (f64 exp → f32), and softcap unchanged.
- Prefill (`M > 1`) semantics unchanged; the slice-copy fill is correct for
  both prefill and decode without a special case.

## 5. Benchmark results

Model: `qwen2.5-0.5b-int4-onnx` (int4 MatMulNBits, 24 layers), `--ep cpu`,
prompt `"The capital of France is"`, via `profile_native`. Machine is shared,
so per-run variance is ±3–4%.

### Numeric parity (bit-identical) — CRITICAL, PASS

`--tokens 32 --warmups 1 --runs 2`, generated token IDs **identical** before
and after:

```
[12095, 13, 1084, 374, 279, 7772, 3283, 304, 279, 3146, 323, 279, 6722, 315,
 9625, 13, 1084, 374, 1083, 279, 1429, 94451, 3283, 304, 279, 3146, 13, 1084,
 374, 279, 7772, 3283]
```

First token `12095` = "Paris" — coherent, and every ID matches the pre-change
baseline exactly. The change is a pure data-movement refactor; numerics are
unaffected.

### Decode throughput (tok/s, higher is better)

Matched binaries built from the same tree (baseline = kernel file stashed),
run interleaved to share machine load:

| Context (`--tokens`) | Baseline (tok/s)    | Tier A (tok/s)      | Δ           |
|----------------------|---------------------|---------------------|-------------|
| 256 (7 samples each) | 21.39 avg           | 21.43 avg           | +0.2%       |
| 512 (2 samples each) | ~19.05 avg          | ~18.93 avg          | −0.6%       |

**Interpretation.** The end-to-end decode delta is **within measurement noise**
(±3–4% run-to-run on this shared host). This is expected and consistent with the
prior investigation (`docs/DECODE_PERF_INVESTIGATION.md`): int4 MatMulNBits GEMV
is ~83% of decode and *all* of GQA is ~14.7%, of which the KV copy is only a
fraction. A few MiB of `memcpy` per step is sub-millisecond against a ~46 ms
step, so the copy-traffic reduction is real but below the throughput noise floor
at these context lengths and B=1.

**Why keep it.** Tier A is a *deterministic* reduction in avoidable work with
bit-identical output: it removes one full owned clone of the past cache per
step, and replaces O(S·D) scalar indexed loads (with per-element index math and
bounds checks) with `memcpy`. Its benefit scales with context length, batch
size, and KV-head count, and it is the natural precondition for Tier B (shared
append-only KV), where the O(S²) term is actually eliminated.

### Test status

- `cargo test -p onnx-runtime-ep-cpu`: **633 passed / 1 ignored** (== baseline),
  all `group_query_attention` tests green.
- 3 ignored e2e wiring tests (deepseek_e2e, glm_tiny_quant_e2e,
  glm_tiny_qmoe_e2e): **all pass**.
- `cargo build --release -p onnx-runtime-ep-cpu`: clean, 0 warnings on touched
  code.
