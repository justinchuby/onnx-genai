## 🦆 Rubber-Duck review — independent verification

**Verdict: GO WITH FINDINGS**

Documentation-only, `cargo fmt` clean, 19/19 `simd_activations` tests pass, and the *central* correction (the old "`sigmoid(-Inf)` leaks `1.5e-8`" claim) is genuinely false and is now fixed — good catch. It is safe to merge. **But** this PR's whole purpose is numerical-doc accuracy, and I independently reproduced **two** quantitative claims it introduces/retains that do **not** match the code they annotate. Both are the same class of error the PR sets out to eliminate, so I'm flagging them before merge.

Host: AMD EPYC 9V74 (`avx avx2 f16c fma`, AVX-512 masked) — matches the PR body. Tooling: rustc 1.97.1, onnxruntime 1.28.0, onnx 1.22.0, CPU EP, `ORT_DISABLE_ALL`, 1 intra-op thread.

---

### ✅ Claims I independently reproduced (confirmed TRUE)

1. **`sigmoid(-Inf)` and `x <= -18` return exactly `0.0` from vendored MLAS.** Temporary `#[cfg(test)]` in `mlas-sys` calling `compute_logistic`, then reverted (`git status --porcelain` clean afterwards):
   - `-Inf → 0x00000000`, `-18.0 → 0x00000000`, `-18.000002 → 0x00000000`, `-17.999998 → 0x33800000`, `-9.0 → 0x39016000`, `+18/+Inf → 0x3F800000` — every value in the PR table matches bit-for-bit.
   - Swept **all 3 670 017** f32 in `[-25, -18]`: **every one returns `0.0`**. The "for every `x <= -18`" claim holds.
2. **ORT 1.28.0 agrees on the endpoints and the two divergences.** Single-node models, opt disabled, 1 thread:
   - `Sigmoid(-Inf) = 0x00000000`; `Sigmoid(-18) = 0`; `Sigmoid(-17.999998) = 0x33800000`.
   - `Tanh(8.442762) = 0x3F800001` (`1.0000001`) — exactly as the pin-site comment says.
   - `Gelu(-Inf, approximate=tanh) = 0xFFC00000` (NaN); `FastGelu(-Inf) = 0xFFC00000`; `QuickGelu(-Inf, alpha=1.702) = 0xFFC00000`. The "ORT returns NaN" divergence is real.
3. **`quick_gelu`/`tanh_gelu(-Inf) → +0.0` is produced by the scalar fallback and the f64 references.** Confirmed by reading the code: `tanh_gelu_scalar`, `quick_gelu_scalar`, `tanh_gelu_ref`, `quick_gelu_ref` each hard-code `if x == f32::NEG_INFINITY { return 0.0; }`. (See NIT below — this is agreement *by construction*, not an independent measurement.)
4. **Spot-checks of "vector matches ORT where the libm scalar fallback does not"** (temp test in `simd_activations`, reverted). Vector = ORT, scalar libm differs, in every requested case:
   - `tanh(9.0)`: vector `0x3F800000` = ORT `0x3F800000`; libm scalar `0x3F7FFFFF` (`0.99999994`).
   - `sigmoid(-17.999998)`: vector `0x33800000` = ORT `0x33800000` = MLAS; libm scalar `0x3282D325`.
   - `quick_gelu(-9.000001)`: vector `0xB6100001` = ORT `0xB6100001`; libm scalar `0xB6066E47`.
5. **NaN payload contract** (temp test + ORT probe): `qNaN 0x7FC01234 → 0x7FC01234` and `sNaN 0x7F800001 → 0x7FC00001`, identical on the **vector path, the scalar path, and ORT**. The new sentence is correct.
6. **Documentation-only.** Every added/removed line in the diff is a `//!` or `//` comment; 1 file, `36 insertions / 11 deletions`, no executable line changed.
7. **`cargo fmt --all -- --check`** exits 0; **`cargo test -p onnx-runtime-ep-cpu --lib simd_activations`** → **19 passed / 0 failed**.

### ⚠️ Claims I could NOT reproduce as stated

- The aggregate counts **"138/140 bit-identical", "exactly 2 divergences", "32 pairs"** were **not** independently reproduced end-to-end — that needs the exact 35-value special set per function, which isn't in the tree. The spot-checks above are fully consistent with them and I have no reason to doubt them, but I did not recompute the totals.
- Findings A and B below are claims I *did* reproduce and found **inaccurate**.

---

### 🟠 MAJOR — Finding A: the tanh-overshoot figures describe a *non-fused* evaluation, not this module's FMA path

The pin-site comment (in the diff) states:

> `p/q` exceeds `1.0` for **26 503** of the f32 values in `[8, 9]`, peaking at `1.0000002` near `|x| = 8.443`.

`p` and `q` there are the `_mm256_fmadd_ps`-computed rationals. I reproduced the sweep two ways over every f32 in `[8, 9]` (Rust, `-C target-cpu=native`; `f32::mul_add` has identical semantics to `_mm256_fmadd_ps`):

| evaluation | count `p/q > 1.0` | span | peak-x (first) | `f(8.442762)` |
|---|---|---|---|---|
| **FMA (what this module compiles to)** | **57 437** | `[8.127431, 8.999997]` | `8.47554` | `1.0000001` (`0x3F800001`) |
| naive non-fused f32 (e.g. numpy) | **26 503** | `[8.052297, 8.999964]` | `8.442762` | `1.0000002` (`0x3F800002`) |

The PR's `26 503`, its span `[8.052297, 8.999964]`, **and** its "peak at `8.442762`" match the **non-fused** row exactly — i.e. the sweep was measured with double-rounding (numpy-style), not with the fused kernel the comment annotates. The module (and MLAS, which is also FMA3) actually overshoots for **~57 437** values, and at `x = 8.442762` the fused rational is `1.0000001`, one ULP *below* the peak; the true peak region is `x ≈ 8.476 … 8.998`. `8.442762` is ORT's probe point, not the module's peak.

The *decision* to pin is correct (both evaluations overshoot, peak `1.0000002`) and "ORT ships the overshoot" is verified. Only the specific count and peak location are attributed to the wrong arithmetic. Suggest re-measuring with `mul_add`/the actual kernel, or dropping the exact count.

### 🟠 MAJOR — Finding B: "`tanh(9)` … round[s] to `1.0f32`, so the substituted limits are the correctly rounded values" is FALSE for tanh

Retained (and reworded) module doc, in the diff:

> `tanh(9) = 1 - 3.0e-8` and `sigmoid(18) = 1 - 1.5e-8` both round to `1.0f32`, so the substituted limits are the correctly rounded values.

- `sigmoid(18) → 1.0f32`: **TRUE** (`1 - sigmoid(18) = 1.523e-8 <` the `2.980e-8` round-up threshold).
- `tanh(9) → 1.0f32`: **FALSE.** `tanh(9) = 0.999999969…`, and `1 - tanh(9) = 3.046e-8 > 2.980e-8`, so it rounds to **`0.99999994` (`0x3F7FFFFF`)**, the largest f32 below 1.0 — *not* `1.0f32`. Verified in float64→float32, and the module's own scalar/f64-ref paths return `0x3F7FFFFF` for `tanh(9)`.

This directly contradicts **this PR's own body**, which correctly states `tanh(±9)` → "libm `0.99999994`". Because the premise is false, the conclusion "the substituted limits are the correctly rounded values" is also false for tanh: for `x ∈ (9.0, ~9.01)` the correctly-rounded `tanh` is `0.99999994`, but the module (like MLAS/ORT) substitutes `1.0` — a deliberate 1-ULP overshoot, not a correctly-rounded value. The old wording ("saturation is strictly more accurate than clamping") had the same false premise; the reword makes the claim stronger, so it's worth fixing here rather than carrying forward. The sigmoid half can stay.

### ⚪ NIT — Finding C: "produced identically by … scalar fallback and its f64 references" is agreement by construction

True, but all four functions contain the *identical* hard-coded `if x == NEG_INFINITY { return 0.0 }` guard, so this is a shared pin, not independent corroboration that `+0.0` is "the limit." Fine to keep; just don't read it as convergent evidence.

---

### Bottom line
Doc-only, tests green, and a net accuracy improvement — **GO WITH FINDINGS**. Please correct Finding A (FMA vs non-FMA sweep numbers) and Finding B (the `tanh(9) → 1.0f32` rounding claim, which contradicts the PR body) before merge, since numerical accuracy is the entire point of this change. No code, security, or behavioural risk.

<sub>All probes run read-only in a throwaway worktree; every temporary `#[cfg(test)]` helper was reverted and the tree confirmed clean (`git status --porcelain`).</sub>
