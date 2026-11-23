# Shared CPU view alignment safety

Baseline: `098f7d3f1052746d1600d91de2b436e359a7b2be`, which already fixes the dtype widening/narrowing helpers. This follow-up covers the older shared kernel accessors used by QMoE and other kernels; it is correctness evidence, not a timing report.

`TensorView::validate` checks offset granularity, not backing-pointer alignment. Byte-backed views can therefore be valid without being aligned for f32/i32/i64 references. The old contiguous f32 materializer created an invalid typed slice, and the strided float/index readers and float writer directly dereferenced potentially unaligned pointers. The direct f32 borrowing helper also incorrectly claimed validation proved alignment.

Contiguous f32 materialization and output retain bulk copying through u8 pointers. The destination Vec is initialized before setting its length; f32 accepts all bit patterns. Strided scalar paths use read_unaligned/write_unaligned. True borrowed f32 slices require actual element-origin alignment; empty borrows return an empty slice. ArgMax/ArgMin use their existing generic path for unaligned logits, preserving first/last tie behavior, while aligned input keeps the zero-copy path. The view contract documentation now states the actual alignment boundary.

## Evidence

Under default Miri, inserting the new named regression tests into the old module reproduces Undefined Behavior at its contiguous from_raw_parts call: required alignment 4, found 1. The candidate source is restored in a finally block before positive runs. The same regression then passes, covering f32/i32/i64, contiguous and positive/negative strides, zero/nonzero offsets, NaN payload/signed-zero bits and output canaries. A second test checks rejected unaligned borrowing, empty borrowing, and aligned pointer identity. Two selection tests cover generic unaligned ArgMax/ArgMin with both tie modes and aligned borrowed ArgMax. All four positive default-Miri tests pass; no borrow model or leak checks are disabled.

Full native CPU library: 2026 passed, 20 existing ignored. MLAS reference configuration: 2068 passed, 35 existing ignored. Session library: 296 passed. Strict all-target CPU/session and MLAS Clippy passed. Raw logs are compressed in this directory; source/patch hashes and the Miri receipt pin the tested code. The expected failing Miri run remains separately labelled `miri-before.log.gz`.

The archived scripts capture the executed workflow and absolute local paths. To repeat the red/green check, use baseline 098f7d3f1 with candidate.patch applied (without changing HEAD), adjust the isolated worktree/target paths, and run miri.py through the shared host-lock wrapper. Its baseline comes from HEAD; do not run its expected-negative stage on a newer already-fixed HEAD.

No new EP interface, resource grant or quota callback is introduced. Aligned contiguous copies keep a bulk path, but this report makes no measured performance claim. The private Miri target is removed under the same stable flock after validation; cleanup.json records that cleanup. CUDA behavior and whole-crate Miri soundness are outside these four selected tests.
