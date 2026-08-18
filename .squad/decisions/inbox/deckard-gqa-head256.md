### 2026-08-18: head_size=256 fused GQA decode kernel (qwen3.5-2b)
**By:** Deckard
**What:** GO. Extended the f32 fused split-K GQA decode fast path from head_dim<=128 to head_dim<=256 (GQA_MAX_DPL 4->8, GQA_MAX_HEAD_SIZE/MAX_HEAD_DIM 128->256). qwen3.5-2b-text decode A/B on an idle H200 (tokens=128, warmups=2, runs=5, --steady --decode-skip 1, ONGPU_ARGMAX=1), native backend, medians of 5:
  - BEFORE (gqa_attention_reference_f32): 102.31 tok/s (9.774 ms/token)
  - AFTER  (fused head256):              170.47 tok/s (5.866 ms/token)
  - Speedup: 1.67x decode throughput.
nsys: gqa_attention_reference_f32 share dropped 31.2% -> 1.1% (only warmup calls remain; the fused kernel now runs inside the captured CUDA graph).
Correctness: new head_dim=256 parity test (GQA 8/2, cache lengths incl. split-K boundaries 64/128/256) vs f64 CPU reference -> max_abs=1.79e-7, max_rel=3.59e-7, identical magnitude to the head_dim<=128 test; passes the same 1e-3/5e-3 tolerance. head 64/128 unchanged (no regression).
**Why:** head_dim=256 previously fell back to the serial reference kernel (nsys #1 decode hotspot, 31.2%). No register spill / occupancy loss materialized; register pressure roughly doubled but the win is large. Byte-identical-eligible; widens native's lead over ORT on this model.
