### 2026-07-24: Fixed-signature CUDA capture closes the DeepSeek mask island
**By:** Rachael
**What:** CUDA `CumSum`, `Unsqueeze`, and `Slice` now warm and retain their exact fixed decode signature, skip runtime metadata D2H during graph recording, and avoid capture-time synchronization/allocation. `Slice` retains its device metadata buffers. General broadcasting `Where` now captures after its dtype/broadcast geometry has warmed because its condition and metadata are already device-resident.
**Why:** DeepSeek-V2-Lite fixed-capacity decode keeps mask geometry stable while mask values remain device-sourced. On both block-32 and block-128 exports, the mask-island seams fell from `Unsqueeze=4, Slice=1, CumSum=1, Where=1` to zero. Listed seam nodes fell 275→268 (the remaining 268 are Reshape work owned separately); segmented eager boundaries fell 246→241 as adjacent captured regions merged.

Verification:
- Both DeepSeek exports produced `[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]` three independent times (`" Paris.\nThe currency of France is the Euro.\n"`).
- Both exports reported measured CUDA graphs `captures=1, replays=9, fallbacks=0`.
- Qwen2.5-0.5B remained coherent and capture-clean: one segment, zero seams, measured `captures=1, replays=13, fallbacks=0`.
- Phi-4-mini on idle GPU 1 produced the same 16-token sequence three times and reported `captures=2, replays=26, fallbacks=0`.
- CUDA EP lib tests: 205 passed; session MLAS lib tests: 65 passed; CUDA clippy with warnings denied passed; construction GPU tests: 18 passed; targeted CumSum GPU test passed.

The implementation necessarily changes generic CUDA movement/elementwise kernels rather than model-specific Attention code. Leon/Sebastian should review the warmed fixed-signature contract, especially the established assumption (shared with Reshape) that runtime shape/axis/bound metadata stays invariant across captured replays.
