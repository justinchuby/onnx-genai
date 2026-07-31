# Harry — History

## 2026-07-29T05:55:00+0000 — PR #388 approved

- Verified schema constraints are consumed by llguidance, and checked constraint precedence and validation behavior.
- Targeted HTTP tests and clippy passed in a scratch worktree.
- Confirmed two unrelated full-suite context-limit failures already occur on origin/main.

## 2026-07-30T09:16:00Z — cfg-gated LoopStatePair hotfix

- Landed PR #441, repairing the cfg-gated `LoopStatePair` import.

## 2026-07-30T15:20:00Z — PR #477 merged (shape-inference container types + Sequence)

- PR #477 merged (Lori APPROVED): shape-inference IR container-type model + Sequence foundation (#449). Additive `ValueType` layer, byte-identical tensor path, 4 Sequence ops, 300 tests. Unblocks the previously-deferred Sequence/Optional/Map/ZipMap propagation.

## 2026-07-30T21:15:00Z — PR #486 merged (#449 inc2 sequence ops)

- PR #486 merged: #449 inc2 — SequenceInsert/SequenceErase/SplitToSequence/ConcatFromSequence + seq↔tensor conversion; op catalog 213→217, shape-inference 258→262.
- In flight: #449 inc3 (#527 in review) and inc4.

## 2026-07-31T00:25:00Z — PR #531 merged (#449 inc4); #534 held

- PR #531 merged: #449 inc4 — SequenceMap + Scan container support + cross-subgraph capture. CLOSED issue #449. Container-type shape inference COMPLETE: additive `ValueType{Tensor|Sequence|Optional|Map}`, byte-identical tensor path guaranteed (gated on non-empty container map). Catalog 217 ops/262 entries. Deferred non-load-bearing: Optional/Map handlers, IR-persistence of `ValueType`.
- PR #534 (server contracts #481/#482 — `build_dirty` Option<bool> present-as-null; truncated predicate uses actual scan size): Melina APPROVED but HELD — targets Justin's active branch `feat/genai-demo-dashboard` (PR #476); code exists only there, not main.
- In flight (harry-5): generalize ORT `clone_value`/`clone_owned` to all POD dtypes (unblocks Bool / gemma-3n audio mask).
