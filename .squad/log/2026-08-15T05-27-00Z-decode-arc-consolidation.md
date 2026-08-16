# Decode arc consolidation — 2026-08-15T05:27:00Z

Scribe consolidated the decode-vs-ORT state after main advanced to `e8f76c53`.

- **#986 merged:** Deckard's default-on 128-bit `uint4` wide-load int4 M=1 GEMV raised glm base decode **140.7→192.4 tok/s** (+36.7%); Chew approved f64 **7/7** plus glm/qwen byte identity, and Gaff reproduced +35% with capture-clean portable code.
- **#984 closed/superseded:** captured fused verify PR remained rejected for qwen workspace/capture failures and is no longer active.
- **#988 rejected pending contract fix:** Gaff approved the graph-slot/capture fix, but Chew rejected qwen W=9 because captured spec diverged from plain greedy at token[2] (**9370 vs 2810**). Deckard is locked out of revision.
- **Next work:** Deckard launched GEMV-v2 on `squad/int4-gemv-wideload-v2` to move native streaming **1.40→2.42 TB/s** and beat ORT base; Batty owns `squad/spec-decode-w9-contract` as the additive spec-decode contract repair.

Binding metric remains base native decode vs ORT base. After #986, native base is **192.4 tok/s** vs ORT **~250 tok/s**: still ~**1.30× behind**.
