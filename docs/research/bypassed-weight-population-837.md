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

Every candidate in the item's scope — reservation for high-reuse tensors (dead:
no such tensor), size-aware admission that refuses to let small tensors block a
large one, two-pass layer-order admission, a transient staging zone — recovers
the gap only by moving these 23 tensors into the *retained-and-served-as-a-hit-
across-steps* population, which is precisely the corrupting population. A
staging zone that retains bypass traffic across steps **is** retention.

### Direct confirmation: byte-aware still corrupts on `29d1515a`

Same box, same prompt, `ONNX_GENAI_WEIGHT_OFFLOAD_BYTE_AWARE=1`:

| Metric | Value |
|---|---:|
| token IDs | `[96347, 3375, 724]` — **3 tokens, generation collapsed** |
| decoded text | `"Politiciens"` (garbage) |
| `byte_hit_rate` | 76.64% (the metric *improved* while output broke) |

Byte-identity **failed**. This is a fresh reproduction of #886: the one
admission lever that moves streamed bytes is the one that breaks decode.

## Conclusion (a truthful negative)

On the managed-streaming (WDDM) path there is **no safe admission change** that
recovers item 3's gap. The gap is 23 single-read large tensors; keeping them
resident is byte-aware retention, which corrupts decode on this build. Read
count cannot discriminate because every weight is read exactly once per step.

The gap is only safely recoverable by the **#864 zero-copy hybrid** (a resident
hot set + zero-copy cold reads, whose safety hinge never evicts a retained page
for a cold weight, so no large tensor is ever evicted-and-re-admitted). That is
already the funded direction; on Linux it is worth ~8× (#925/#936). On
Windows/WDDM it is blocked by the aperture ceiling, not by admission. The
correct action for item 3 is therefore to **close it as characterised** and not
add a managed-path admission policy.

The per-key trace lands as the reusable instrument that produced this evidence.
