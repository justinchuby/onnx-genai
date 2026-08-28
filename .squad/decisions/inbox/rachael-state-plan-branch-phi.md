### 2026-08-28: Branch-local terminal writers require structural export
**By:** Rachael
**What:** Terminal-writer analysis now retains the owning control-flow edge, so equal component/port/binding triples from different branch cases still require one declared output phi before session commit.
**Why:** Equal names do not make branch-local SSA values visible outside their case; only the selected phi output is available to commit.
