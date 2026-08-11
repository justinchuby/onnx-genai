# Cohaagen finding: QMoE decode next lever after capture fixes

Date: 2026-08-10
Worktree: `/home/justinchu/onnx-genai-wt-moeprof`
Commit: `a3856b6a06be8092477a9fa8398f4feb74562ba9`
GPU: H200, `CUDA_VISIBLE_DEVICES=1`

## Measurement

Command required `ONNX_GENAI_CUDA_KV_MAX_LEN=262144` because the pipeline decoder directory does not carry the parent `model.max_sequence_length`; without it, native CUDA resolves an unbounded KV cap and fails the mask reservation overflow guard.

Clean steady-state median: **11.563 ms/token** over 3 runs (128 tokens, warmups=1, decode_skip=8).
Capture segmentation with `ONNX_GENAI_LOG_CAPTURE_SEGMENTS=1`: **14 captured segments, 13 eager seams**. Residual seams are 3 `Squeeze` copy-path seams and 10 `GroupQueryAttention` seams due growing KV/total-sequence symbols.

## Attribution

Per-op measured blocks rank: `MatMulNBits` 13.6%, `QMoE` 12.5%, `GroupQueryAttention` 10.0%, `Add` 8.3%, `Cast` 7.4%, `Softplus` 6.7%, `ReduceSumSquare` 6.0%, `LinearAttention` 5.6%, `RMSNormalization` 4.6%, `SkipSimplifiedLayerNormalization` 4.5%.

Nsight Systems kernel grouping over 256 decode forwards:
- QMoE route/linear/activate/combine: **32.8%**, 2.84 ms/token of GPU kernel time.
- MatMulNBits dense GEMV: **23.5%**, 2.05 ms/token.
- Elementwise/copy/activation: **22.2%**, 1.94 ms/token.
- Norm/RMS components: **11.9%**, 1.02 ms/token.
- LinearAttention: **6.2%**, 0.54 ms/token.

Top individual kernels: `matmul_nbits_gemv_f16_scales_f16_zp_splitk` 20.9%; `qmoe_linear_f32` 14.3%; `qmoe_linear_f16` 11.0%; `skip_rmsnorm_f16_warp_half4` 6.7%; `linear_attention_f16` 6.2%; `qmoe_route` 5.9%.

## Conclusion

The leading norm-fusion hypothesis is not the top lever on current main: norms are real but only about **1.0 ms/token / 11.9%** of GPU kernel time. The dominant model-level lever is QMoE decode, especially the two expert linear kernels plus routing. NCU on `qmoe_linear_f32` shows low DRAM throughput (~3.5% peak) and high barrier/no-eligible stalls (~46%, CTA barrier ~48% of issue cycles), so this is latency/synchronization-bound rather than bandwidth-bound. NCU on the top MatMulNBits GEMV likewise shows low bandwidth (DRAM ~17% peak) and L1TEX scoreboard stalls, consistent with M=1 GEMV latency behavior.

Recommended next work: design and validate a QMoE decode-specialized expert linear path that reduces block-level barrier/imbalance or groups the k=8 active expert work without reordering fp32 accumulations. This is numerically high-risk and should not be shipped without the 35B oracle lock and an attention-only model gate.
