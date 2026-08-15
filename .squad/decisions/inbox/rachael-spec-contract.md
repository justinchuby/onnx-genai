# Rachael — captured speculative decode contract

Date: 2026-08-15

## Finding
Captured verify accepted draft tokens using the M>1 verify row argmax, while the binding contract is M=1 greedy. A reproduced qwen accept-path mismatch with Marlin verify enabled and the M=1 row probe showed row 2 at past=20/pos=22 accepted draft 18493 from the verify row, while M=1 greedy selected 9370. The verify row was a narrow fp16/logit flip: Marlin top=18493 (16.921875), second=9370 (16.890625), margin=0.03125; M=1 top=9370 (16.90625), second=18493 (16.90625).

## Decision
Until the M>1 CUDA verify kernels can be made argmax-identical to the M=1 GEMV path, captured-spec acceptance now derives its verify rows from the M=1 decode kernel sequence. That makes every accepted draft and bonus token use the same argmax contract as plain greedy. This is deliberately correctness-first; it preserves opt-in behavior and leaves the Marlin numeric-match optimization as follow-up work.

## Proof run
- qwen `/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx`, degenerate `哈`×20 and normal prompt, W=5..9: byte-identical to plain greedy with accepted>0. Acceptance examples: degenerate W=5..9 accepted 56 each; normal W=5/6/7/8/9 accepted 128/134/138/140/143.
- glm `/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`, generic and repetitive W=6: byte-identical to plain greedy with accepted>0. Acceptance: generic accepted=13; repetitive accepted=96.
- `cargo fmt --check` clean.
- `cargo clippy -p onnx-genai-engine --features cuda,native-backend --all-targets -- -D warnings` is clean except the known pre-existing `platform_capacity.rs:247/249` unnecessary `u64` casts.
