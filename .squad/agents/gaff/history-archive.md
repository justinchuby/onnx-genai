# Gaff — History Archive

## Archived from live history (Scribe compaction 2026-08-12T00:15:00Z)

### 2026-07-28T17:40:00+0000
Reviewed #364: blocked partial-cache stat divergence, then approved guarded fix.

### 2026-07-29T12:30:00Z — tiny-reasoning-fixture REJECT (PR #410)
Mutation proof: commented out `resolve_sampling_defaults` per-turn, suite stayed green. REJECT issued; Batty locked out; Leon took round 2+. Durable rule: "Assert on what the code did, not a summary of what it should have done." Also confirmed rubber-duck's empty-answer diagnosis.

### 2026-08-11 — env var verifier false-positive fix
CI job 93646716235 failed: `NXRT_ABI` matched from filename reference. Added negative lookahead `(?!\.(?:md|rst|toml)\b)` to ENV_PATTERN. Gate still catches genuinely undocumented variables.

### 2026-08-11 — PR #31973 Review: AVX2 LayerNorm/RMSNorm
40/40 tests pass. Welford pairwise merge correct. Two substantive: S1 unnecessary sum in RMSNorm when MeanOut null, S2 test reference variance formula comment.

### 2026-08-11 — Independent Re-Review of PR #762 (B1-B4 corrective wave)
All four blockers resolved. 245 passed. Shapes correct. Dtypes sourced from graph.

### 2026-08-11 — Re-review PR #762 (post-Sapper, 2ca515eb7)
3 BLOCKING: UAF (dangling ptr from MutexGuard), CopyTensors wrong direction, panic bomb always fires. S4: constructor reads EP name via unconditional panic. Issued rejection.

### 2026-08-11 — Verify #762 CUDA EP Fix (d64a49d59)
B1 (UAF): FIXED (Arc<Mutex>). B3 (copy direction): FIXED. S4 (panic bomb): FIXED. B2 (pointer equality): deferral unjustified — MemoryDevice_GetDeviceId exists in ORT 1.27 bindings. Conditional pass.

### 2026-08-11 — CUDA review wave + upstream PR #31973 review
Sapper rejection (3 blockers + lockout). Nabil B1/B3/S4 genuine fixes confirmed. PR #762 at 31687667a: no blockers. #31973: 40/40 tests, S1 RMSNorm sum, S2 reference comment.

### 2026-08-11 — PR #762 final delta review (bb280c0ea)
Scoped to Rachael test-hardening + Zhora doc pass. Assertions falsifiable. BL1 shapes intact. Helper duplication tech debt noted. Ready to leave draft.
