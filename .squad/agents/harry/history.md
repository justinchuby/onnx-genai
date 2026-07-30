# Harry — History

## 2026-07-29T05:55:00+0000 — PR #388 approved

- Verified schema constraints are consumed by llguidance, and checked constraint precedence and validation behavior.
- Targeted HTTP tests and clippy passed in a scratch worktree.
- Confirmed two unrelated full-suite context-limit failures already occur on origin/main.

## 2026-07-30T09:16:00Z — cfg-gated LoopStatePair hotfix

- Landed PR #441, repairing the cfg-gated `LoopStatePair` import.

## 2026-07-30T15:20:00Z — PR #477 merged (shape-inference container types + Sequence)

- PR #477 merged (Lori APPROVED): shape-inference IR container-type model + Sequence foundation (#449). Additive `ValueType` layer, byte-identical tensor path, 4 Sequence ops, 300 tests. Unblocks the previously-deferred Sequence/Optional/Map/ZipMap propagation.
