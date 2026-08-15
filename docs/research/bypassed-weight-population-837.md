# The bypassed weight population is single-read — #837 item 3

- Date: 2026-08-14
- Host: this box (`wt-b2b1` worktree), single RTX-class GPU, WDDM
- Model: `C:\Users\justinchu\dev\models\qwen14b-zp` (16.65 GiB data file,
  ~7.87 GB distinct weight bytes read per decode step)
- Runtime: ONNX Runtime 1.27.0, managed weight streaming
  (`ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1`), default derived budget
  (`resolved_device_budget_bytes = 7,730,940,928`, no explicit override)
- Branch: `squad/transient-staging-837` off `main` (`29d1515a`)

## Question

#837 item 3 asked whether admission — not eviction — leaves the residency
policy 1.97× above its streaming floor, and whether the smallest change that
admits "the right tensors" recovers it. The deciding sub-question: **how often
is each bypassed tensor read per decode step?** A tensor read once per step
cannot be ranked by reuse frequency; one read many times per step would be the
prize (#864).

## Method

Added an off-by-default per-key trace
(`ONNX_GENAI_WEIGHT_PAGING_KEY_TRACE=1`) that attributes every residency-cache
lookup to its weight key over the measurement window: `len`, `hits`,
`retained_page_ins`, `bypass_page_ins`. It is pure process-local accounting
under a `Mutex`, taken only when enabled, and touches no device state, so it
does not perturb the counters it explains. `profile_native` dumps it after the
existing `weight_offload_*` lines. `reads / emitted_tokens` is reads-per-step
(one decode step per emitted token).

`profile_native --steady --tokens 16 --warmups 1 --runs 1 --decode-skip 2
--prompt "Hello"`, solo (`nvidia-smi` verified empty before the run).

## Result 1 — the headline reproduces, byte-identical

| Metric | Value |
|---|---:|
| `htod_bytes_per_token` | 2,349,010,944 (2.349 GB/step) |
| `byte_hit_rate` | 70.16% |
| `bypassed_byte_share` | 44.63% |
| `bypassed_page_in_bytes / step` | 1,048,412,160 (1.048 GB/step) |
| token IDs (16) | `[96347, 3375, 724, 11, 358, 2776, 14589, 311, 6723, 429, 498, 3003, 2581, 6617, 315, 752]` — **matches reference** |

## Result 2 — every weight is read exactly once per step

Across all **867** distinct weight keys — hits, retained page-ins, and bypasses
alike — `reads_per_step = 1.000` (`reads = 16` over 16 steps). There is **no
high-reuse tensor**. A transformer weight is used once per forward pass; the
per-key trace confirms it directly rather than by assumption. This closes the
reuse-reservation candidate: reservation for high-reuse tensors has nothing to
rank on.

This kills **prioritisation by reuse**; it does **not** by itself kill
**retention**. #864's finding was narrower than "once-per-step reads make
residency worthless": copying a weight to VRAM and *evicting it before it is
re-read* buys nothing, because the byte is paid per step either way. A tensor
that stays resident **across** steps is a different case — it is read from VRAM
every step instead of streamed every step, so its bytes are saved every step.
That is precisely why the static-pin question (Result 4) is worth measuring:
uniform reads remove the reuse *ranking*, but retention could still help if the
budget has room to hold more resident bytes than churn currently keeps.

## Result 3 — the gap is 23 large single-read tensors

`bypass_keys = 23` account for the whole 1.048 GB/step (≈90% of the 1.158
GB/step recoverable gap):

| Count | `len` each | Bypass behaviour | Bytes/step |
|---:|---:|---|---:|
| 1 | 371.2 MiB (389,283,840 B) | bypassed all 16 steps | 389.3 MB |
| 1 | 46.4 MiB (48,660,480 B) | bypassed all 16 steps | 48.7 MB |
| 17 | 33.75 MiB (35,389,440 B) | bypassed all 16 steps | 601.6 MB |
| 4 | 4.22 MiB (4,423,680 B) | thrash: retained 8 / bypassed 8 | 8.8 MB |
| | | **total** | **1,048.4 MB** |

The sum matches `bypassed_page_in_bytes / step` to the byte. These are the
largest tensors in the model; they bypass because they do not fit the residual
budget headroom under arrival-order first-fit admission (`!can_fit(bytes)`),
while smaller tensors already hold the space. Meanwhile ~167 mid-size tensors
per step are retained-then-evicted (churn), and ~679 are served as hits.

## Why no admission change recovers this safely

Because reuse is uniformly 1×/step, the resident set that minimises streamed
bytes is simply the **largest-B-bytes** subset — i.e. retain these 23 large
tensors and evict smaller residents to fit them. That is exactly **byte-aware
residency**, which #886/#888 measured to corrupt decode (token-identity
collapse) whenever it engages, and #892/#901 confirmed the cause is the
retain-vs-bypass *flip*, not eviction order (eviction-order tuning is bounded at
~10% of the gap).

### Direct confirmation: byte-aware still corrupts on `29d1515a`

Same box, same prompt, `ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE=1`:

| Metric | Value |
|---|---:|
| token IDs | `[96347, 3375, 724]` — **3 tokens, generation collapsed** |
| decoded text | `"Politiciens"` (garbage) |
| `byte_hit_rate` | 76.64% (the metric *improved* while output broke) |

Byte-identity **failed**. This is a fresh reproduction of #886: the one
admission lever that moves streamed bytes is the one that breaks decode.

## Result 4 — the static pin: the provably-safe retention shape

Byte-aware residency evicts-and-re-admits large stable-slot residents, and #892
localised the corruption to exactly that pattern. That leaves one case
byte-aware admission does not test: **pin the large tensors at first touch and
never evict or re-admit them** — retain once, chosen from the known layer walk,
never entering the eviction population. If the corruption really is about
evict-and-re-admit, a permanent pin should be safe *by construction*.

Two env-gated diagnostic knobs (both off by default, shipped path byte-identical):

- `ONNX_GENAI_WEIGHT_OFFLOAD_PIN_THRESHOLD_BYTES` + `..._PIN_BUDGET_BYTES` —
  pin any tensor `len ≥ threshold`, up to a total budget.
- `ONNX_GENAI_WEIGHT_OFFLOAD_PIN_KEYS` — pin an explicit key allow-list,
  ignoring size/budget. Pinned keys are excluded from every eviction-victim
  selector, so a pinned page is served as a hit across steps and never
  re-enters the evict-and-re-admit population.

### Arm 4a — threshold pin (33 tensors, 1.09 GiB): safe but zero benefit

`PIN_THRESHOLD=30 MB`, `PIN_BUDGET=1.2 GB` pinned the **first 33 large tensors
by arrival order** (`pinned_keys=33 pinned_bytes=1,167,851,520`):

| Metric | Baseline | Threshold pin |
|---|---:|---:|
| token IDs (16) | reference | **matches reference — byte-identical PASS** |
| `htod_bytes_per_token` | 2.349 GB | **2.351 GB (unchanged)** |
| `byte_hit_rate` | 71.8% | 70.13% (marginally worse) |
| `bypassed_byte_share` | 44.63% | 44.59% |
| bypass bytes/step | 1.048 GB | 1.048 GB (unchanged) |

The 21 chronic bypassers (keys 919, 920, 287…) **still bypass all 16 steps** —
they arrive *after* the 1.2 GB budget is consumed by earlier-arriving large
tensors, so the pin never covers them. The pin held tensors LRU would have kept
resident anyway, so streaming is unchanged. Safe, and useless.

### Arm 4b — targeted pin of exactly the 21 chronic bypassers: **corrupts**

`PIN_KEYS` = the 21 bypasser keys from Result 3
(`pinned_keys=21 pinned_bytes=1,048,412,160`, the bypass population to the byte):

| Metric | Value |
|---|---:|
| token IDs | `[96347, 3375, 724]` — **3 tokens, generation collapsed** |
| decoded text | `"Politiciens"` (garbage) |
| `htod_bytes_per_token` | 2.412 GB (measured over the 3 tokens before collapse) |

Byte-identity **failed** — the identical #886 signature. But *why* is not what I
first claimed: the reviewer's control (below) refuted a churn-based explanation,
and bisecting the 21 keys narrowed the trigger to **a single tensor**.

### The counter-control that refuted "conservation" (reviewer)

An earlier draft argued the budget is saturated, so pinning any bypasser
*necessarily* evicts-and-re-admits an equal mass of large residents every step,
and that churn is #892's corrupting pattern. A single control refutes it:
over-pinning far past the slack — `PIN_THRESHOLD=30 MB`, `PIN_BUDGET=2.4 GB` —
pins **67 tensors / 2.371 GB**, forces demonstrably *more* churn than arm 4b
(`htod` 2.349 → 2.626 GB/step, +11.8%; `page_ins_per_token` 321 → 375), and
stays **byte-identical** (full 16 reference IDs). Mass-forced evict-and-re-admit
on a saturated budget does **not** corrupt. The trigger is *which* tensor is
pinned, not how much churn pinning causes.

### Bisection: the trigger is one tensor — the int4 lm_head (key 919)

`PIN_KEYS` is an allow-list, so the 21 keys bisect directly. Each row is a solo
hardware run, byte-identity gated:

| Pinned keys | Count | Result |
|---|---:|---|
| 919,920,287,247,912,914,855,874,232,304 | 10 | **FAIL** (3 tokens) |
| 919,920,287,247,912 | 5 | **FAIL** |
| 919,920 | 2 | **FAIL** |
| **919** | **1** | **FAIL** — 3 tokens `[96347, 3375, 724]` |
| 920 (its int4 scales) | 1 | **PASS** (16 IDs) |
| 287 (a 35 MiB block weight) | 1 | **PASS** (16 IDs) |

Pinning **only key 919** corrupts; pinning any other single tensor tested does
not. Key 919 is **389,283,840 bytes = 152 064 (vocab) × 5 120 (hidden) × 0.5** —
the **int4-quantised vocabulary projection (lm_head / embedding-class weight)**,
the one tensor that directly produces the logits that decide token identity. Key
920 (48,660,480 B = 24 330 240 blockwise scales × 2 B fp16) is its scale tensor;
pinning the scales alone is safe, so the trigger is the **weight matrix itself**,
not its quantisation metadata.

### Reconciliation: why the threshold rule is safe and the bypasser rule is not

The two selection rules differ in exactly one thing that matters: **whether key
919 ends up pinned.** Reproducing the reviewer's 2.4 GB threshold arm with the
key trace on shows key 919 with `retained_page_ins=0 bypass_page_ins=16` — it is
**never pinned**, even with 2.4 GB of budget and 67 tensors pinned. The lm_head
is the *last* large tensor touched each step (the logits projection runs after
every block), so by the time it is first touched the size-ordered pin budget is
already spent on earlier-arriving block weights. The threshold rule is safe
because it **structurally cannot reach the one poisonous tensor**; the explicit
bypasser rule corrupts because its allow-list **contains** key 919. That
difference between the two selection rules *is* the finding.

### The mechanism is narrowed, not yet explained (graph capture ruled out)

The stable-slot machinery (issue #716) exists to give retained weights a stable
VA that a captured CUDA graph can bake — so the obvious hypothesis is a
graph-pointer hazard when the lm_head is promoted from the bypass population
(fresh throwaway VA, freed each step, never baked) into the stable-slot
population. **A control refutes it:** pinning 919 with `ONNX_GENAI_CUDA_GRAPH=0`
(`cuda_graph: enabled=false captures=0 replays=0`) still corrupts. So the
corruption is **not** the CUDA graph. What remains: the first three tokens
`[96347, 3375, 724]` *match the reference* before the collapse, so the retained
lm_head reads correctly for the first steps and only then goes wrong — consistent
with progressive staleness of the retained page's physical content (a pinned page
is filled once and served as a hit thereafter, never re-filled), not with a
first-read address error. This is a **leading hypothesis, not a proven cause**;
the honest state is that the corruption is isolated to one tensor and one
transition (retain-across-steps vs stream-fresh-each-step) but its precise
mechanism is not yet root-caused.

## Conclusion — #886 is isolated to a single tensor

The item-3 measurement stands: reuse is uniformly 1×/step (867/867 keys), the
gap is 23 single-read large tensors summing to 1.048 GB/step to the byte, and no
*shipped* admission change recovers it safely. But the static-pin work turned the
long-standing #886 corruption from "byte-aware retention breaks decode, cause
unisolated" (`MEMORY_ARCHITECTURE.md` §3.5) into a **single-tensor bug**:

- Retaining **key 919 — the int4 lm_head/vocab-projection weight — resident
  across decode steps corrupts token identity.** Pinning it alone reproduces the
  exact #886 3-token collapse; pinning 2.37 GB of *other* large tensors (more
  churn) does not; pinning its scale tensor does not; pinning a block weight does
  not.
- The trigger is **selection-specific, not churn-mass-specific** — which refutes
  the conservation/evict-and-re-admit story an earlier draft told, and means
  #892's "evict-and-re-admit" localisation is at best incomplete: here there is
  no re-admission (the page is pinned, never evicted) and it still corrupts.
- The CUDA graph is **not** the mechanism (corrupts with capture off). The
  mechanism beyond "retaining this one tensor resident is unsafe" is not yet
  proven; the pre-collapse correct tokens point at retained-page staleness as the
  lead.

For item 3 itself the practical answer is unchanged — the gap is only safely
recoverable by the funded **#864 zero-copy hybrid** (never evicts a retained page
for a cold weight; ~8× on Linux #925/#936; WDDM-blocked by the aperture ceiling,
not by admission), so no managed-path admission policy should ship. But the
better deliverable is the isolation: #886 has been hiding behind the lm_head, and
a single-tensor trigger is a bug that can be chased, not a wall. The per-key
trace and the two static-pin knobs (threshold+budget, and the explicit key
allow-list that did the bisection) land as the reusable instruments that produced
this evidence.
