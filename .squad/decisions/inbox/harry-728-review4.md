### 2026-08-07: PR #728 round-4 re-review
**By:** Harry
**Verdict:** REJECT

## Blocking finding

1. **`broadcast_dim` is not the single chokepoint where a growing dependency can disappear from IR symbols.** A valid `Reshape([-1])` over `[seq_kv, 8]` forms the derived expression `seq_kv * 8` (`crates/onnx-runtime-shape-inference/src/handlers/movement/transform.rs:71,120-138`). When shapes are written back, `SymbolInterner::lower` interns that non-bare expression as a fresh `SymbolId` without recording its source symbols (`crates/onnx-runtime-shape-inference/src/context.rs:246-269`). The new record only appends inside the two-bare-symbol broadcast arm (`context.rs:506-518`), so `Graph::symbol_unifications` contains no edge from `seq_kv` to the fresh output symbol. A downstream unary op copies only that fresh shape (`handlers/elementwise.rs:11-16`), while the executor closes only over `symbol_unifications` and performs exact membership (`crates/onnx-runtime-session/src/executor/kernel_cache.rs:334-355,395-412`), admitting the downstream op to capture even though its launch extent grows every decode step.

   I reproduced this on `571ea0d9` with real inference: `past_key` supplied growing `seq_kv`; `Reshape([seq_kv,8], [-1])` produced `[SymbolId(0x80000000)]`; `growing == {seq_kv}` and `symbol_unifications == []`; a downstream `Sigmoid` was classified capture-safe. The focused assertion failed exactly on that classification. `Flatten` has the same unrecorded derived-product path (`transform.rs:154-173`). This is the same stale-geometry/silent-corruption class, not an invalid symbolic-shape scenario.

   The authoritative record must include symbol lineage for lowered derived expressions (or the classifier must conservatively propagate growing dependencies through them), with a downstream-consumer regression for `Reshape([-1])` and preferably `Flatten`.

## Confirmed sound in this revision

- Recording placement is otherwise correct: `1`, equal dims, and symbolic-vs-static return before the distinct-bare-symbol arm; inferred representatives are unchanged. The full shape-inference suite passed.
- `record_unification` does not deduplicate, but each inference run overwrites the graph record and duplicates are bounded by broadcast calls, so this is non-blocking.
- The executor union-find is transitive, input-order-independent, and treats pairs bidirectionally. The Add and MatMul downstream-alias regressions pass.
- `Expand` and symbolic non-concat `Concat` funnel through `broadcast_dim`; removing the old elementwise mirror did not lose broadcast-alias coverage.
- Reconciliation keeps the inferred symbolic dim rather than replacing it with a declared symbolic name; Squeeze/Unsqueeze preserve surviving symbol IDs.
- Unioning a growing symbol with a global `batch` symbol can over-poison many nodes and threaten the 34-segment result, but only conservatively. The GPU growing-set/segment re-measure remains required.

**Revision owner:** Sebastian should own the next revision; Cohaagen, Deckard, Leon, and Batty remain locked out.

**REJECT: derived symbolic dimensions can erase growing-symbol lineage outside `broadcast_dim`, and a reproduced `Reshape([-1])` downstream consumer is still classified capture-safe.**
