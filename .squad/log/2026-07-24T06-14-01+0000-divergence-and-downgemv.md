# Qwen1.5B divergence and int4 down-GEMV

- Qwen2.5-1.5B native token 1909 at decode index 26 is more accurate than ORT-CUDA token 821: native fp32 GEMV agrees with fp32 deployed-int4 dequantization and fp32 unquantized Transformers references, while ORT fp16 accumulation flips the near-tie.
- `c4690bf7` locks the result in a CUDA-model-gated native engine regression test.
- `720fa032` removes redundant activation shared-memory staging from the bit-exact int4 down-projection GEMV; CUDA gate was 206/0 and an NVRTC staged-reference test proves exact-byte parity.
- Known TODO: `profile_native --backend ort` decode errors on Qwen1.5B unless ORT CUDA graph is disabled (`ort_value must contain a constructed tensor`).
- Pending: Marsten-12's independent Qwen-7B end-to-end perf-delta confirmation for the down-GEMV change.
