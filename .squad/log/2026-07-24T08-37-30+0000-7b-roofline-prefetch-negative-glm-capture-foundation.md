# 7B roofline, prefetch negative, GLM capture foundation — 2026-07-24T08-37-30+0000

- Gaff cleared bf16 RoPE/norm; Marsten's regression sweep found no covered-model token regression and a 1.10–1.58× native/ORT lead.
- The 7B gate/up prefetch A/B is closed: it reduced Long-Scoreboard cycles but regressed throughput 1.01%.
- GLM capacity-present CPU-oracle/byte-parity foundation landed; CUDA capture stages and the frozen-v1 ABI remain in progress.
