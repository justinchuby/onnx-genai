## 2026-07-17 — Shape-inference axis validation

- Landed `cb30ced`: replaced clamping with checked validation for TopK and ArgReduce, Transpose, Unsqueeze, and Gather; added middle-axis, out-of-range, duplicate, and dynamic coverage.
