### 2026-07-27: Make Unique's data-dependent extent falsifiable
**By:** Pris
**What:** Assert that Unique's data, indices, and counts outputs use a fresh symbolic extent in both flattened and axis modes.
**Why:** A concrete replacement such as `constant(1)` must fail the test; inverse-indices lengths remain derived from the input shape as required by the schema.
