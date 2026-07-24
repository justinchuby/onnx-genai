### 2026-07-23: Static Split capture/replay test coverage
**By:** Sebastian
**What:** Reworked the static even `Split` byte-parity integration test to build with concrete input shapes, execute the static kernel, capture it, replay it with changed input, and compare replayed outputs with eager output bytes.
**Why:** The generic `run()` helper supplies empty input shapes and therefore exercises only Split's dynamic path; successful CUDA graph capture is a regression guard for the static no-synchronize path.
