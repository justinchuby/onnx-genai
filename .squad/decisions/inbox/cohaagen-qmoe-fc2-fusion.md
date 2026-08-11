# Cohaagen: QMoE FC2/down+combine decode fusion experiment

Date: 2026-08-11
Branch: `squad/moe-fc2-combine-fusion`
GPU: H200, `CUDA_VISIBLE_DEVICES=1`

## Design tried

I implemented the safe remainder of the decode-path fusion for `rows == 1 && routes <= 16`: fuse FC2/down with weighted combine. The experimental `qmoe_down_combine_*` kernel kept the existing #765 FC1/SwiGLU fused activation global scratch, then for each output feature computed each selected expert's FC2 GEMV in slot order, applied the router weight, and wrote the final combined output. This removed the separate `qmoe_combine_*` launch and the `route_output` global scratch write/read.

I did **not** attempt a single full FC1+SwiGLU+FC2 kernel because the current CTA shape computes one `(route, intermediate)` FC1 reduction per CTA. Keeping all `inter` activations private while preserving FC2 parallelism would require either cross-CTA shared state (not available), recomputing gate/up for each hidden output (prohibitively expensive), or collapsing a whole route/expert into one/few CTAs (massive under-parallelization for hidden=2048). That is a larger persistent/tiled design, not a surgical extension.

## Correctness

The experimental down+combine fusion was deterministic and passed validation:

- New rows==1 fused-path tests: `qmoe_decode_fused_swiglu_down_combine_matches_cpu` and `qmoe_decode_fused_silu_fc3_down_combine_matches_cpu` passed.
- Full QMoE GPU suite with the experiment: 31/31 passed.
- 35B oracle lock passed: teacher-forced token 33803 held; measured `logprob(33803)-logprob(5342) = 0.09375`.

## Profile result

`profile_native --pipeline --steady --warmups 1 --runs 3 --tokens 128` with `ONNX_GENAI_PROFILE_OPS=1`:

| Build | median ms/token | QMoE op | MatMulNBits op | LinearAttention op | RMSNorm op |
| --- | ---: | ---: | ---: | ---: | ---: |
| #765 tip baseline | 11.105 | 67.7-68.0 ms, ~12.1-13.2% | 69.3-79.6 ms, ~13.5-14.1% | 29.3-30.7 ms, ~5.4-5.7% | 24.2-26.8 ms, ~4.7% |
| down+combine experiment | 11.096 | 71.0-71.3 ms, ~13.6-13.8% | 70.1-72.2 ms, ~13.5-13.9% | 29.1-29.3 ms, ~5.6% | 24.9-25.0 ms, ~4.8% |

Wall-clock delta was only ~0.08% (11.105 -> 11.096 ms/token), far below the 1% ship gate and within shared-host noise. Per-op QMoE time did not improve; it slightly increased in the steady profiled passes, likely because the fused down+combine kernel serializes the route loop per output CTA and adds synchronization inside a heavier kernel, offsetting the saved combine launch/scratch traffic.

## Decision

Do **not** ship this FC2/down+combine fusion. Although it is oracle-stable and test-clean, it does not clear the performance gate and adds a new reduction kernel on a numerically sensitive path. The remaining meaningful lever is not route-output scratch removal alone; it needs a bigger tiled/persistent QMoE decode design that can keep FC1 activations local while still exposing enough FC2 output-feature parallelism.
