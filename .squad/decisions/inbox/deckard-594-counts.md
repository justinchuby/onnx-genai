### 2026-08-02: #594 pinned registry-count fix
**By:** Deckard
**What:** Bumped op_rules.rs pinned operator_count/entry_count (and any sibling catalog-count tests) to reflect the new standard-domain LinearAttention shape rule; rebased #594 onto latest main.
**Why:** CI "Test Linux offline crates" failed on expanded_registry_catalog_count_is_pinned (218 actual vs 217 pinned). Legit +1 by design, not a regression.
