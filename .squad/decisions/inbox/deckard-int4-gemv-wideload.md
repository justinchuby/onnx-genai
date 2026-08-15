# Decision drop — wide-load (128-bit `uint4`) int4 M=1 decode GEMV

**Author:** Deckard (Systems Dev, CUDA / decode-performance)
**Branch:** `squad/int4-gemv-wideload` (off main `b24e961e`)
**Program:** high-MLP wide-load int4 M=1 decode GEMV — raise DRAM bandwidth toward
ORT's 2.42 TB/s via 128-bit vectorized weight loads + software-pipelined in-flight
loads. Stacks on merged #978 (LOP3 + split-K) and #981 (block SkipRMSNorm).

## Diagnosis this closes (from the ORT-gap spike)
Head-to-head on the identical glm gate_up weights / geometry / occupancy, ORT streams
at **2.42 TB/s (50% DRAM)** vs our **0.92 TB/s (19% DRAM)**. Same math (dequant→fp16
FMA), same tiling, same occupancy → the wall was **narrow-load issue / low
memory-level-parallelism**, NOT dequant compute and NOT occupancy starvation. The narrow
loop issues one 32-bit weight load (8 nibbles) then immediately dequant+FMA — a dependent
chain with ~1 load in flight per lane, so DRAM sits at 19%.

## Mechanism shipped
New shared device helper `gemv_int4_wide_lane_dot` (matmul_nbits.rs ~L1463):
- Each lane owns **32 contiguous nibbles via ONE `uint4` (128-bit) load** — 4× bytes/instr,
  4× fewer load instructions.
- **Depth-2 software pipeline**: issue the next `uint4` before consuming the current one →
  ≥2 synchronous wide loads in flight, hiding the ~10-cyc Long-Scoreboard latency at
  ~constant register footprint (no cp.async — proven pure overhead at M=1, #980; no scalar
  `#pragma unroll` of LOP3 — register bloat → occupancy cliff).
- Reuses the proven `dot_int4x8_f16_sub` (fp32-accumulate LOP3 dequant) ×4 on the four
  sub-words per `uint4`; per-lane ascending-K accumulation order preserved.

Two glm kernels wired: `matmul_nbits_gemv_f16_general_bs_wide` and `..._splitk_wide`.
Dispatch selects wide in the `block_size != 32` general_bs / split-K arm via
`use_gemv_wideload(bits, block_size, k)` — **default-on**, guard `bits==4 && block_size%32==0
&& k%32==0`, env `ONNX_GENAI_GEMV_WIDELOAD=0` forces narrow for A/B (like the split-K toggle).

## Portability / capture
Portable, **default-on, no arch guard, no opt-in** — 128-bit loads are baseline; no SM80
intrinsic. Static launch grid, no host divergence across replays → capture-safe. `<SM80`/CPU
never reach this path (int4 CUDA GEMV only); byte behavior of the narrow path is untouched
when the env forces narrow.

## Numerics
The 32-wide lane interleave regroups the fp32 partial sums (same class as split-K, which
ships default-on) → **near-equal, not bit-exact** at the fp32 level, but empirically
**greedy tokens are BYTE-IDENTICAL**. Gated by the f64 dequant→GEMM oracle (within tolerance,
incl. asymmetric zp) + greedy-token equality.

## Results — glm-4-9b-int4 (block-128), H200, `--steady --tokens 160 --decode-skip 40 --runs 3`

| Path | decode tok/s | greedy tokens |
|------|-------------:|---------------|
| narrow (`ONNX_GENAI_GEMV_WIDELOAD=0`) | 140.69 | baseline |
| **wide (default)** | **192.38** | **BYTE-IDENTICAL** |

**+36.7%** e2e decode (all-idle host; earlier contended run measured +35.1%, 185.19 vs 137.16).
ncu on the gate_up matrix: **65.3µs → 43.2µs**, DRAM **19.1% → 29.2%**, **0.92 → 1.40 TB/s**
(1.51× on the kernel). Partial capture of the ORT gap — real, banked; not yet full 2.42 TB/s.

Cumulative glm decode this session: 97.5 → 112.4 (#978) → 137.8 (#981) → **192.4** (this).
glm native-vs-ORT gap: **2.57× → ~1.30×** (ORT 250.3).

## Honest NO-GO — qwen block-32 fused variants (reverted, not shipped)
qwen2.5-14b-int4 (block-32, asymmetric zp): wide on the fused gate_up / scales_f16 path was
measured **flat** (149.57 vs 148.10, tokens identical) → reverted; block-32 keeps its tuned
narrow fp16-accumulate entries. Root cause (ncu):
- **gate_up** is **compute/SM-bound, not DRAM-bound**: wide RAISED occupancy 61→83% and
  dropped regs 40→32 but was slightly SLOWER (57.8→61.5µs) — the fp16 half2 narrow accumulate
  is already efficient; the fp32 wide path adds compute without a bandwidth deficit to fill.
- **down_c2** is latency/grid-starved (11.6% DRAM, 16% SM) — not a wide candidate at all.

Wide-load is a **glm-class block-128 large-N GEMV** lever, not a universal one. block-32's
smaller N/K-per-block simply isn't bandwidth-starved. Not forcing a flat/negative change.

## Gates
- f64 oracle `matmul_nbits_marlin_numerics`: **7/7 PASS** (`--features cuda,gpu-tests`).
- glm greedy tokens: **byte-identical** wide vs narrow (diff-verified).
- qwen: unchanged (narrow path) → byte-identical by construction.
- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib`: **0 warnings** (mine).
  native-backend gate (`onnx-genai-engine`): only the 2 pre-existing `u64→u64` casts.

## Reproduce
```bash
source .cudaenv.sh
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
M=/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda
# wide (default) vs narrow
CUDA_VISIBLE_DEVICES=<idle> ./target/release/profile_native --model $M --ep cuda --steady --tokens 160 --decode-skip 40 --warmups 1 --runs 3
CUDA_VISIBLE_DEVICES=<idle> ONNX_GENAI_GEMV_WIDELOAD=0 ./target/release/profile_native --model $M --ep cuda --steady --tokens 160 --decode-skip 40 --warmups 1 --runs 3
# f64 oracle
cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests --test matmul_nbits_marlin_numerics
```

## Handoff
Ready for Sebastian's decisive native-vs-ORT gate on GPU7 (glm +36.7%, 192 tok/s vs ORT
250.3) and Chew (numerics) + Gaff (Rule 11 / capture) review. This is base-decode's second
act — it compounds under the speculative-verify drafts (they run the same M=1 int4 GEMV).
