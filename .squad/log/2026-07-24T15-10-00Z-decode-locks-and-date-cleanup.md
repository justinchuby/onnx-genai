# Decode locks and date cleanup — 2026-07-24T15:10:00Z

The native-CUDA decode-correctness suite now uses shared `common/decode_lock.rs` coverage for Qwen2.5-0.5B/1.5B/7B, Qwen3-0.6B, Phi-4-mini, and DeepSeek-R1-1.5B, each comparing native and ORT output bit-exactly through 64 tokens. Phi's documentation now reports a 36–43% native lead with GEMV as the primary bottleneck, and benchmark filenames/dates were normalized to real dates. This reconciliation processed 13 inbox notes; the one note dated before the seven-day retention cutoff was archived.
