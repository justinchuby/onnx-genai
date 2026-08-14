### 2026-08-14: Marlin int4 numerics gate — f64 oracle + justified tolerance envelope + current-path baseline (Pris, #957)

**By:** Pris (Tester), branch `squad/marlin-numerics`

**What:** Landed a reusable f64 dequant→GEMM **numerics gate** for `com.microsoft::MatMulNBits`
int4 that any GEMM implementation (today's `gemm_f16_tiled` prefill / decode GEMV path **and**
Deckard's forthcoming Marlin int4 GEMM) must pass before shipping. New self-contained integration
test file `crates/onnx-runtime-ep-cuda/tests/matmul_nbits_marlin_numerics.rs` (no edits to
`matmul_nbits.rs`, so it does not conflict with `squad/marlin-kernel`).

- **Oracle (ground truth):** dequantize packed int4 to f64 as `(code - zero_point) * scale`
  (scale pre-rounded to its fp16/f32 storage value) and accumulate `sum_k a·w` in f64. Both
  candidate and oracle consume the **same fp16-rounded activations** and the **same rounded
  scale**, so the residual isolates only accumulation precision + fp16 output rounding — never
  shared input quantization (mirrors the in-crate GEMV `run_parity_dims_block` convention).
- **Coverage:** group sizes {16,32,64,128}; symmetric AND asymmetric-zp; M∈{1,2,4,8,16,32};
  realistic glm-4-9b (K/N = 4096/4096, 4096/256, 4096/13696, 13696/4096) + Qwen2.5-1.5B
  (1536/8960, 8960/1536) attention+MLP projections; fp16 & fp32 scales.
- **Marlin-ready interface:** the gate drives the op purely through **op semantics** (ONNX
  `MatMulNBits` node + `ExecutionProvider::get_kernel`/`execute`) — never the internal weight
  layout. Once Marlin is wired into the same dispatch, `run_matmul_nbits_f16` validates it with
  ZERO changes; alternatively Deckard/Chew feed any candidate output slice into
  `Int4Problem::parity`.

**Justified tolerance envelope** (single source of truth = `Envelope::for_output(max_out)`):
- **Absolute (primary, physical):** `abs_bound = max(max_out * 4e-3, 4e-3)`. fp16 output ULP is
  `2^-11 ≈ 4.9e-4` of magnitude; weights dequant through fp16 (another ULP); the K-reduction is
  fp32 and a Marlin partial-sum **relayout** re-associates it, adding `~K·eps_f32 < 2e-3` rel drift
  at the deepest K≈13696. `4e-3` ≈ 8 fp16 ULP of headroom covers both the tiled baseline and a
  re-associated Marlin reduction.
- **Relative (secondary, conditioning-aware):** `rel_bound = 5e-2` against a denominator floored
  at `max(1e-1, 3e-2 * max_out)`. Outputs far below the operator's peak are cancellation-dominated
  (`|Σ a·w| ≪ Σ|a·w|`); their fp16 round-off is inherently large in *relative* terms while the
  *absolute* error is one fp16 ULP of the peak — so they are governed by the abs bound. The `3%`
  floor keeps ~3× margin on the worst measured cancellation column.

**CURRENT-PATH BASELINE (measured on H200, GPU7, feature `gpu-tests`):**
- Group-size sweep (K=4096, N=896, all {group×M×zp×scale-dtype}): **max_abs = 6.50e-2,
  max_rel = 2.71e-3, max_out = 1.67e2** → abs error ≈ **3.9e-4 · max_out (≈1 fp16 ULP)**.
- Projection shapes (glm-4 + Qwen2.5, bs=32, M∈{1,8}, sym+asym): **max_abs = 2.53e-1,
  max_rel = 8.14e-3, max_out = 5.27e2** → abs error ≈ **4.8e-4 · max_out (≈1 fp16 ULP)**.
- ⇒ The tiled/GEMV path sits ~8–10× inside the asserted abs bound and ~6× inside the rel bound.
  **Marlin must land in the SAME envelope** (abs ≲ `max_out·4e-3`, rel ≲ `5e-2` with the 3%
  conditioning floor). If Marlin's measured error is comparable to the ~1-ULP baseline it is a
  clean pass; anything approaching the bound warrants a conditioning review before sign-off.

**Why:** Marlin's weight relayout reorders partial sums, so its output is NOT byte-exact vs the
tiled kernel and cannot be diffed bit-for-bit — the only defensible reference is a high-precision
f64 oracle with a justified tolerance. This gate gives Chew (numerics reviewer) an apples-to-apples,
op-level sign-off criterion and gives Deckard a red/green target while iterating.

**How Deckard/Chew invoke it:**
`source .cudaenv.sh && CUDA_VISIBLE_DEVICES=<idle-gpu> cargo test -p onnx-runtime-ep-cuda \
--features gpu-tests --test matmul_nbits_marlin_numerics -- --nocapture --test-threads=1`.
Pure-CPU oracle self-consistency tests (5) run with no GPU and no feature. GPU baseline tests are
`#[ignore]` unless `gpu-tests` is set (CPU-only CI stays green).

**NOTE (harness lesson, not a kernel bug):** an early un-synchronized run showed a transient
catastrophic divergence (abs 81.8) at one config deep in the sweep; root cause was a missing
post-`execute` `runtime.synchronize()` + reading a non-pre-zeroed output buffer (stale device-pool
memory). Fixed by pre-zeroing the output and synchronizing after execute. With that, the current
int4 path passes the full matrix — no kernel defect. Deckard's Marlin driver should follow the same
sync/zero discipline.
