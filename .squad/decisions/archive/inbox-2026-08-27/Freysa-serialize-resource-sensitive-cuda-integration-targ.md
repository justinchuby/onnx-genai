### 2026-08-26T05-08-13: Serialize resource-sensitive CUDA integration targets with target-local locks
**By:** Freysa
**What:** Serialize resource-sensitive CUDA integration targets with target-local locks
**References:** fix/gpu-suite-parallel-isolation, crates/onnx-runtime-ep-cuda/tests/multi_head_attention_gpu.rs, crates/onnx-runtime-ep-cuda/tests/dft_gpu.rs, crates/onnx-runtime-ep-cuda/tests/stft_gpu.rs, .github/scripts/verify_cuda_test_honesty.py
**Why:** ### 2026-08-25: CUDA GPU suite isolation is per integration-test process
**By:** Freysa
**What:** MHA, DFT, and STFT each use one target-local process-wide mutex, acquired as the first statement of every test. The CUDA honesty source scan structurally enforces the helper and every acquisition.
**Why:** Rust integration targets are separate binaries/processes, so a shared source helper cannot serialize across them. The races occur among libtest threads inside each affected binary: overlapping MHA/STFT EP VMM reservations and DFT/STFT process-global telemetry deltas. One mutex per target covers EP construction, execution, counter reads, assertions, and EP teardown; poisoned locks warn and recover to prevent cascade failures.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
