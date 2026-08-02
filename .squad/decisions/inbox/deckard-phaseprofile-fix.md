### 2026-08-02: profile_native bench compile fix
**By:** Deckard
**What:** Restored pub reset_exec_phase_profile (clears phase registry + resets print guard) in onnx-runtime-session; bench bin profile_native --steady compiles again.
**Why:** Function was dropped while caller remained → bench broken on main. Tooling-only, profiler env-gated.
