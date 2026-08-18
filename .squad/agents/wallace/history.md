# Wallace — History (compacted 2026-07-29)

**Role:** CUDA EP GEMV/kernel author and reviewer for the Rust runtime — sub-4-bit and IQ decode kernels, GQA/RMSNorm/SiLU parity, and native-serving safety. Verify bit-exactness against CPU, keep kernels correct and fast across all supported SM architectures, and honor reviewer lockouts.

## Durable lessons
- CUDA architecture strings must derive from the live selected device capability (SM60–SM120), never hardcoded; retain the native-CUBIN fallback for when the driver rejects a PTX ISA (e.g. CUDA 13.3 PTX ISA 9.3 on driver 580.105.08). All CUDA kernel work must remain correct and fast across supported SMs, not only `sm_90`.
- IQ super-block M=1 GEMV is bit-exact vs CPU for IQ4_XS/IQ2_XXS/IQ3_XXS/IQ2_XS/IQ2_S/IQ3_S/IQ1_S/IQ1_M; M>1 and unknown formats must fall back. Shared `IQ1S_GRID` hash is `0x6703ed863501ae2e`.
- Native CUDA serving must fail closed: Roy's CUDA-only `559c46f` was rejected/locked out because a real 144-BQMM model failed mid-serving without CPU fallback; Deckard's `fa30410` (startup failure with heterogeneous-placement guidance) is canonical.
- CUDA op order must match CPU's branch-stable SiLU/RMSNorm; a small reduction-order drift (token-16 `1.9073486e-5`) is accepted only because exact emulation costs ~8.4%. Stale CUDA tests must assert the real unsupported error so they cannot pass if the dtype later becomes accepted.
- Repository-wide serialized custom-operator domain is `pkg.nxrt`. Session EP claim planning preserves omitted optional inputs as `DataType::Undefined` (`848ad87`).
- CI covers all 27 offline crates with warnings-as-errors and native Windows ARM64; wave-2 native fp16 CUDA decode reached 663–672 tok/s on H200 vs ORT GenAI 657 with zero fallbacks.

## Recent work (current wave, ~2026-07-28)
## 2026-07-28T17:40:00+0000
Approved PR #365 after four rounds, including recursive scanning and uniform structural identity.

Full pre-compaction history in `history-archive.md`.
## 2026-08-18T01:35Z — V2-Lite MoE measurement + graph-capture scope

- Measured corrected DeepSeek-V2-Lite int4 MoE: native CUDA eager median ~55.6 tok/s; graph flag currently no-ops because capture declines on `attention_mask_consumers_are_capacity_aware`; ORT CUDA lacks QMoE kernels and falls back to CPU experts at ~0.20 tok/s.
- Scoped V2-Lite graph-capture unlock as GO: topology-gated capacity policy for additive-mask-builder → capacity-form `Attention[3]`, with GLM-5.2 negative guard intact; post-implementation Wallace owns byte-identity/perf A/B.
