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

Byte-identity **failed** — the identical #886 signature reached by an
independent mechanism.

### Reconciliation: this **confirms** #892, it does not refute it

The two arms differ only in *which* tensors are pinned, and that is the whole
result. The VRAM budget is **saturated**: baseline already serves ~5.5 GB/step
from residency against a ~6.1 GB device budget, with the ~0.6 GB slack being
allocator/KV overhead, not packable weight space (arm 4a proves it — pinning 1.09
GB of extra resident bytes moved `htod` by *nothing*). On a saturated budget,
**pinning a tensor that currently bypasses necessarily evicts an equal byte-mass
of large stable-slot residents to make room, and — since every tensor is read
once per step — those displaced residents are re-admitted next step.** The pin
does not *avoid* evict-and-re-admit; it *relocates* it onto the victims. That is
precisely #892's corrupting pattern, which is why arm 4b reproduces #886 exactly.

Arm 4a is safe only because it pins tensors that were **already resident**, so it
induces no new eviction — and buys nothing for the same reason. There is no
static pin that is both safe (induces no evict-and-re-admit of large residents)
and useful (reduces streamed bytes): on a saturated budget those two properties
are mutually exclusive by conservation of VRAM.

## Conclusion (a definitive negative)

On the managed-streaming (WDDM) path there is **no admission change — not even
the provably-safe retention shape — that recovers item 3's gap without
corrupting decode.** The gap is 23 single-read large tensors. Keeping any of
them resident on a budget that is already full requires evicting-and-re-admitting
an equal mass of large residents every step (because reuse is uniformly 1×/step),
which is the #886/#892 corruption. This closes item 3 as a negative *and*
strengthens #892's localisation rather than overturning it: the corruption is
about evict-and-re-admit of large stable-slot residents, and a static pin cannot
escape that on a saturated budget — it only moves the churn onto the displaced
tensors.

The gap is only safely recoverable by the **#864 zero-copy hybrid** (a resident
hot set + zero-copy cold reads, whose safety hinge never evicts a retained page
for a cold weight, so no large tensor is ever evicted-and-re-admitted). That is
already the funded direction; on Linux it is worth ~8× (#925/#936). On
Windows/WDDM it is blocked by the aperture ceiling, not by admission. The
correct action for item 3 is therefore to **close it as characterised** and not
add a managed-path admission policy.

The per-key trace and the static-pin knobs land as the reusable instruments that
produced this evidence.
