# Opset coverage consolidation

- **Agents:** Wallace, Zhora, Cotton, Chew, Nandez, Freysa, Deckard, Sapper, Pris
- **Merged:** `7c06c39` (coverage-wave head)
- **What/why:** Added standard, contrib-fused, and C1 shape coverage across opsets 17–26, plus numerical and optional-slot correctness fixes. This substantially reduces known CPU conformance gaps; kernel↔shape comm-diff, not static schema registration alone, directs further work.
- **Review:** Fused normalization, Range parity, Tile semantics, and Attention revisions cleared review gates.
