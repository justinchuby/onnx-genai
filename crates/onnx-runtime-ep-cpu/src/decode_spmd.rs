//! Persistent SPMD decode pool: one hot worker set joined by a lightweight
//! reusable barrier, replacing the ~141 per-token Rayon fork-join regions.
//!
//! # Why
//!
//! Native M=1 int4 decode issues ~141 `MatMulNBits` projections per token, each
//! run today as a *separate* Rayon parallel region (`par_chunks_mut(..).for_each`
//! -- see [`crate::kernels::matmul_nbits::parallel_output_rows`]). Even with the
//! pool kept hot by [`crate::kernels::matmul_nbits::with_decode_pool_scope`],
//! every projection still pays Rayon's per-region machinery: task publication on
//! the crossbeam deque, work-stealing coordination, `crossbeam-epoch` memory
//! reclamation, and a join latch. Profiling attributes ~27% of the decode step
//! to this fork-join glue, and it is exactly the term that makes >32 cross-socket
//! threads regress.
//!
//! This module keeps a fixed set of worker threads parked-and-hot and drives
//! them with a hand-rolled **broadcast + counting barrier**: one atomic sequence
//! bump publishes the op, workers observe it, run their pre-assigned output-row
//! shard, and decrement a per-node completion counter; the dispatcher spins on
//! those counters. No per-op allocation, no deque, no epoch GC -- just a handful
//! of atomics per projection. An unwind-only completion guard still decrements
//! the counter if a worker panics, poisons the pool, and makes the dispatcher
//! report an actionable panic instead of hanging.
//!
//! # Two-level, NUMA-aware (mirrors `numa-split`)
//!
//! To use both sockets' memory bandwidth without a toxic flat cross-socket
//! barrier, workers are split into per-node groups (16+16 on a 2-node host),
//! each pinned to its node and reading a node-local first-touched weight shard,
//! exactly like [`crate::decode_numa`]. Row-sharding a GEMV is exactly
//! associative -- each output row is an independent dot product over the whole K
//! dimension -- so concatenating the per-worker row slices reproduces the flat
//! result bit-for-bit, with no cross-row/-node reduction. The only cross-socket
//! traffic per op is the dispatcher reading each node's own completion counter
//! (one line per node), not an N-way shared barrier.
//!
//! # Generality (rule 2)
//!
//! Topology is queried at runtime, never hardcoded. On a single-node host, a
//! non-NUMA machine, or a platform without CPU pinning, the pool degrades to one
//! unpinned worker group -- it still replaces the per-op Rayon barrier with the
//! lightweight one, and stays correct.
//!
//! # Auto-calibrated by default, with an env override (rule 5)
//!
//! The pool's activation is **auto-calibrated**. `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL`
//! selects the policy: `=1` forces the pool on (operator override for dedicated
//! hosts), `=0` forces it off (always the flat path), and **unset (the default)
//! is `Auto`** -- a runtime calibrate-and-pick heuristic.
//!
//! Auto exists because this busy-wait barrier *beats* the flat Rayon decode path
//! on a quiet host (int4 decode; e.g. Qwen3-0.6B ~32->74 tok/s, Phi-3.5-mini
//! ~12->30 tok/s) but *regresses* under co-tenant load, where its spinning
//! workers and non-participating dispatcher contend with the neighbours (the
//! flat path degrades gracefully; the barrier does not). There is no portable,
//! reliable "current host load" API across Linux/macOS/Windows and x86_64/aarch64,
//! so instead of *guessing* the host state we *measure* it: because the pool is
//! token-exact (N-tile aligned, PR #110), switching paths never changes the
//! emitted tokens, so Auto can time the *same real decode step* both ways on the
//! live workload and keep the faster one. See [`Calibrator`] for the state
//! machine. The default committed path is the flat path (the safe choice under
//! load), the pool is adopted only when it is measured meaningfully faster
//! (hysteresis margin), and the choice is periodically re-probed so a host that
//! becomes loaded mid-generation falls back within one recalibration window.
//! This makes "never regress vs the flat path under load" a *measured* property
//! rather than a heuristic hope, while still winning out-of-the-box on an idle
//! host. The forced worker count is
//! [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`] (about
//! half the logical CPUs); a `THREADS=0` opt-out leaves the decode path unchanged.
//!
//! # Precedence when forced (`=1`) vs the affinity control
//!
//! When the pool is forced on (`=1`), the decode strategy precedence is, highest
//! first:
//!
//! 1. **`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split`** -- the explicit multi-node
//!    split wins when its two-level layout can be built (the mutually-exclusive
//!    selection vs the forced persistent pool is reported once).
//! 2. **Forced persistent SPMD** (`=1`) -- its own per-node pinning applies.
//! 3. **Flat Rayon + auto-`compact`** legacy path -- reached by `=0` (Off) and by
//!    the `Auto` default whenever calibration has the flat path committed, which
//!    also honors any explicit `ONNX_GENAI_CPU_DECODE_AFFINITY` via
//!    [`crate::decode_affinity::plan_decode_affinity`] as before. Under `Auto`,
//!    an explicit `numa-split` affinity likewise takes precedence over calibration
//!    (the user picked a specific strategy), so Auto calibrates the persistent
//!    SPMD pool against the flat path only.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::decode_affinity::{NodeShard, NumaTopology};
use crate::kernels::matmul_nbits::output_chunk_len;

/// Environment switch selecting the persistent SPMD decode pool policy:
/// `=1` forces the pool on, `=0` forces it off (flat path), and **unset (the
/// default) is `Auto`** -- a runtime calibrate-and-pick heuristic (see
/// [`Calibrator`]). The pool beats the flat path on a quiet host but regresses
/// under co-tenant load, so Auto times the same token-exact decode step both
/// ways on the live workload, keeps the flat path committed by default (the safe
/// choice under load), and adopts the pool only when it measures meaningfully
/// faster -- re-probing periodically so it falls back if the host becomes loaded.
/// See `.squad/decisions.md` (Voight 2026-07-24; Hudson 2026-07-24 auto-enable).
pub const PERSISTENT_POOL_ENV: &str = "ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL";

/// Spin iterations a worker or the dispatcher busy-waits before yielding /
/// parking. Sized so back-to-back decode projections (microseconds apart) are
/// always caught while spinning; only genuinely idle gaps fall through to park.
const SPIN_BEFORE_YIELD: u32 = 1 << 12;
const YIELD_BEFORE_PARK: u32 = 1 << 6;
/// Bounded park so a (rare, off-hot-path) lost wakeup self-heals rather than
/// hanging. It is only a backstop: dispatch wakes parked workers with an explicit
/// `unpark` (SeqCst-paired below), so this timeout never fires on the hot path.
/// An idle worker re-parks on each timeout WITHOUT re-running the spin cycle (see
/// `worker_loop`), so a longer timeout simply means fewer idle futex wakeups; it
/// is kept modest so a theoretical lost wakeup still self-heals quickly.
const PARK_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(50);

/// Cache-line pad so per-node completion counters and per-worker park flags do
/// not false-share (which would reintroduce cross-socket coherency traffic).
#[repr(align(128))]
struct Padded<T>(T);

/// A type-erased decode job: run the shard for the given global worker index.
/// The data pointer is only dereferenced between [`SharedState::publish`] and the
/// matching dispatcher wait, so the borrowed closure always outlives its use.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

/// State shared between the dispatcher (the engine thread running the forward)
/// and the persistent worker threads.
struct SharedState {
    /// Bumped once per dispatched op; workers wait for it to change.
    sequence: Padded<AtomicUsize>,
    /// The current op, published before `sequence` bumps and read after the
    /// bump is observed (release/acquire pairing on `sequence`).
    job: UnsafeCell<Option<Job>>,
    /// Outstanding worker acknowledgements for the current op, one counter per
    /// node so the dispatcher only reads each node's own (mostly node-local)
    /// line instead of an N-way shared barrier.
    node_pending: Vec<Padded<AtomicUsize>>,
    /// Per-worker park flags: the dispatcher only issues an `unpark` syscall to
    /// workers actually parked, so the hot back-to-back path costs zero syscalls.
    parked: Vec<Padded<AtomicBool>>,
    /// The node each global worker index belongs to (drives which pending
    /// counter it decrements).
    worker_node: Vec<usize>,
    /// Count of workers that have entered their loop and are ready to receive
    /// ops. `build` blocks until this reaches `total_workers` so no dispatch can
    /// race a not-yet-started worker (which would miss the op and hang the
    /// barrier).
    ready: AtomicUsize,
    /// Nonzero after a worker panics while running an op (`worker_index + 1`).
    /// A poisoned pool rejects this and every later dispatch instead of hanging
    /// forever waiting for a worker that has unwound.
    poisoned_worker: AtomicUsize,
    shutdown: AtomicBool,
}

// SAFETY: `job` is a raw pointer guarded by the publish/observe protocol on
// `sequence`; it is only read by workers while the dispatcher blocks in
// `dispatch`, so the pointee outlives every access. All other fields are atomics.
unsafe impl Sync for SharedState {}
unsafe impl Send for SharedState {}

impl SharedState {
    /// Publish `job` for `node_pending[node] = counts[node]` workers and wake any
    /// parked worker. Must be paired with [`SharedState::wait`].
    fn publish(&self, job: Job, counts: &[usize], handles: &[JoinHandle<()>]) {
        // Publish the job pointer, then the per-node counts, before the sequence
        // bump makes them visible to workers.
        unsafe {
            *self.job.get() = Some(job);
        }
        for (counter, &count) in self.node_pending.iter().zip(counts) {
            counter.0.store(count, Ordering::Release);
        }
        // SeqCst so this bump and the parked-flag read below share one total
        // order with the worker's SeqCst park guard (store parked, then load
        // sequence): that pairing is what guarantees a parking worker is always
        // either seen here (and unparked) or observes this bump itself and skips
        // parking -- no lost wakeup. Off the hot path (one atomic per op).
        self.sequence.0.fetch_add(1, Ordering::SeqCst);
        // Wake only workers that actually parked (SeqCst load pairs with the
        // worker's SeqCst park-guard store to avoid a lost wakeup; on the hot
        // path every flag is false so this issues no syscalls).
        for (index, parked) in self.parked.iter().enumerate() {
            if parked.0.load(Ordering::SeqCst) {
                handles[index].thread().unpark();
            }
        }
    }

    /// Spin-wait until every node's workers have finished the published op.
    fn wait(&self) {
        let mut spins = 0u32;
        loop {
            let done = self
                .node_pending
                .iter()
                .all(|counter| counter.0.load(Ordering::Acquire) == 0);
            if done {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins >= SPIN_BEFORE_YIELD {
                thread::yield_now();
            }
        }
    }

    fn panic_if_poisoned(&self) {
        let poisoned = self.poisoned_worker.load(Ordering::Acquire);
        if poisoned != 0 {
            let worker = poisoned - 1;
            panic!(
                "persistent SPMD decode worker {worker} panicked while executing a decode op; \
                 the pool is poisoned and cannot continue. Disable \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL or restart the process"
            );
        }
    }
}

/// A persistent SPMD decode pool: hot worker threads plus the shared barrier
/// state that drives them.
pub struct SpmdDecodePools {
    shared: Arc<SharedState>,
    handles: Vec<JoinHandle<()>>,
    /// Workers assigned to each node, node-major, matching global worker index
    /// order (workers `0..counts[0]` are node 0, and so on).
    node_worker_counts: Vec<usize>,
    total_workers: usize,
}

impl SpmdDecodePools {
    /// Build the pool from per-node worker shards. Global worker indices are
    /// laid out node-major (node 0's workers first) so row segments and weight
    /// placement line up with the node assignment.
    fn build(shards: &[NodeShard]) -> Self {
        let node_count = shards.len();
        let mut worker_node = Vec::new();
        let mut node_worker_counts = Vec::with_capacity(node_count);
        // Global (index, pinned cpu) assignment, node-major.
        let mut assignment: Vec<(usize, Option<usize>)> = Vec::new();
        for (node_position, shard) in shards.iter().enumerate() {
            node_worker_counts.push(shard.workers);
            for worker in 0..shard.workers {
                worker_node.push(node_position);
                let cpu = shard.cpus.get(worker % shard.cpus.len().max(1)).copied();
                assignment.push((node_position, cpu));
            }
        }
        let total_workers = assignment.len();

        let shared = Arc::new(SharedState {
            sequence: Padded(AtomicUsize::new(0)),
            job: UnsafeCell::new(None),
            node_pending: (0..node_count)
                .map(|_| Padded(AtomicUsize::new(0)))
                .collect(),
            parked: (0..total_workers)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
            worker_node,
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(total_workers);
        for (global_index, (node_position, cpu)) in assignment.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("onnx-genai-spmd-n{node_position}-{global_index}"))
                .spawn(move || {
                    if let Some(cpu) = cpu
                        && let Err(message) = crate::decode_affinity::pin_current_thread_to_cpu(cpu)
                    {
                        report_spmd_fallback(&format!(
                            "worker {global_index} could not pin to cpu {cpu}: {message}"
                        ));
                    }
                    worker_loop(shared, global_index);
                })
                .expect("spawn persistent SPMD decode worker");
            handles.push(handle);
        }

        // Block until every worker has entered its loop and is waiting for ops.
        // Without this, a dispatch issued before a worker starts would set the
        // op's pending count for that worker, which would never arrive to
        // decrement it -- hanging the barrier. `total_workers` is bounded and
        // each worker signals readiness immediately, so this is a brief spin.
        while shared.ready.load(Ordering::Acquire) < total_workers {
            std::hint::spin_loop();
        }

        Self {
            shared,
            handles,
            node_worker_counts,
            total_workers,
        }
    }

    /// Total decode workers across all node groups.
    pub fn total_workers(&self) -> usize {
        self.total_workers
    }

    /// Number of node groups in the layout.
    pub fn node_count(&self) -> usize {
        self.node_worker_counts.len()
    }

    /// Broadcast `job` to the workers and block until all have finished.
    ///
    /// `job(global_worker_index)` runs the shard owned by that worker. The
    /// dispatcher (this thread) does not compute; it only publishes and waits,
    /// mirroring an external `pool.install` where the caller blocks.
    fn dispatch<F>(&self, job: &F)
    where
        F: Fn(usize) + Sync,
    {
        self.shared.panic_if_poisoned();
        unsafe fn call<F>(data: *const (), global_index: usize)
        where
            F: Fn(usize) + Sync,
        {
            // SAFETY: `data` came from a live `&F`; synchronous dispatch keeps
            // that borrow alive until every worker acknowledges this op.
            let job = unsafe { &*data.cast::<F>() };
            job(global_index);
        }
        let job = Job {
            data: std::ptr::from_ref(job).cast(),
            call: call::<F>,
        };
        self.shared
            .publish(job, &self.node_worker_counts, &self.handles);
        self.shared.wait();
        self.shared.panic_if_poisoned();
    }

    /// Split `n` output rows across the node groups proportionally to their
    /// worker counts (contiguous, non-overlapping, last node absorbs the
    /// remainder), matching [`crate::decode_numa`] so weight placement and
    /// compute dispatch always line up.
    fn node_row_lengths(&self, n: usize) -> Vec<usize> {
        let node_count = self.node_worker_counts.len();
        let mut lengths = Vec::with_capacity(node_count);
        let mut assigned = 0;
        for (position, &node_workers) in self.node_worker_counts.iter().enumerate() {
            let rows = if position + 1 == node_count {
                n - assigned
            } else {
                n.saturating_mul(node_workers) / self.total_workers
            };
            assigned += rows;
            lengths.push(rows);
        }
        lengths
    }

    /// Contiguous `(start, len)` output-row segment for each global worker index,
    /// node-major: a node's rows are split evenly across that node's workers.
    fn worker_row_segments(&self, n: usize) -> Vec<(usize, usize)> {
        let node_lengths = self.node_row_lengths(n);
        let mut segments = Vec::with_capacity(self.total_workers);
        let mut node_start = 0;
        for (&node_len, &node_workers) in node_lengths.iter().zip(&self.node_worker_counts) {
            let base = node_len / node_workers;
            let remainder = node_len % node_workers;
            let mut offset = node_start;
            for worker in 0..node_workers {
                let len = base + usize::from(worker < remainder);
                segments.push((offset, len));
                offset += len;
            }
            node_start += node_len;
        }
        segments
    }

    /// [`Self::worker_row_segments`] with every interior boundary snapped to a
    /// multiple of `align`. The per-worker split is computed exactly as the
    /// unaligned version (so node-major ordering and weight placement still line
    /// up), then each cumulative boundary except the final `n` is rounded to the
    /// nearest multiple of `align`, kept monotonic and in `[0, n]`. The result
    /// still covers `0..n` exactly once; a boundary collision can leave a worker
    /// with a zero-length segment (it simply runs no work), which the dispatch
    /// and shard-build paths already tolerate.
    ///
    /// `align <= 1` is the identity (returns the unaligned segments): callers
    /// whose per-column arithmetic is partition-independent pass `1`.
    fn worker_row_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        let base = self.worker_row_segments(n);
        if align <= 1 {
            return base;
        }
        let mut segments = Vec::with_capacity(base.len());
        let mut prev_boundary = 0;
        let mut cumulative = 0;
        let last = base.len().saturating_sub(1);
        for (index, &(_, len)) in base.iter().enumerate() {
            cumulative += len;
            let boundary = if index == last {
                // The final boundary is always `n`, even when `n` is not a
                // multiple of `align`: the last shard's start is aligned, so its
                // trailing partial N-tile matches the full-width call's tail.
                n
            } else {
                // Round the ideal cumulative boundary to the nearest multiple of
                // `align`, staying monotonic and within bounds.
                let rounded = ((cumulative + align / 2) / align) * align;
                rounded.clamp(prev_boundary, n)
            };
            segments.push((prev_boundary, boundary - prev_boundary));
            prev_boundary = boundary;
        }
        segments
    }

    /// Shard `result`'s output rows across the workers and run `compute` on each
    /// worker's contiguous slice under one lightweight barrier.
    ///
    /// `compute(output_start, outputs)` fills the rows
    /// `output_start .. output_start + outputs.len()` -- the same closure the
    /// flat path hands to `par_chunks_mut`, so the arithmetic is identical.
    /// Tiny ops (below the flat path's parallelization threshold) run serially
    /// on the dispatcher, so the same set of ops parallelize as before.
    pub fn dispatch_output_rows<F>(&self, result: &mut [f32], k: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        if self.total_workers <= 1 || output_chunk_len(n, k) >= n {
            compute(0, result);
            return;
        }
        self.dispatch_rows_across_workers(result, &compute);
    }

    /// Public view of the contiguous `(start, len)` output-column segment each
    /// global worker owns when a length-`n` GEMV output is sharded across the
    /// pool. Callers that pre-partition a weight along N (e.g. one MLAS SQNBit
    /// packed shard per worker) use this to build shards that line up exactly
    /// with [`Self::dispatch_output_rows_indexed`].
    ///
    /// Every segment boundary is snapped to a multiple of `align` (the last,
    /// `n`-terminated segment excepted). This matters for kernels whose SIMD
    /// column-tiling is *not* bit-stable across an arbitrary N-partition: MLAS's
    /// SQNBit GEMV processes output columns in fixed-width N-tiles, so a shard
    /// boundary that falls *mid-tile* forces MLAS's remainder path to reduce a
    /// block-sum in a different order than the full-width call, shifting that
    /// column by ~1 ULP. Aligning every interior boundary to the N-tile width
    /// keeps every tile whole inside a single shard, so each shard reproduces
    /// the full-width tiling exactly and the concatenated output is
    /// bit-identical to the unsharded call (verified `max_ulp = 0`). Pass
    /// `align = 1` for kernels whose per-column result is already
    /// partition-independent (e.g. the hand int4/int8 GEMV).
    pub fn output_column_segments(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        self.worker_row_segments_aligned(n, align)
    }

    /// Like [`Self::dispatch_output_rows`], but hands each worker its global
    /// index alongside its output slice and always dispatches across the pool
    /// (no serial-threshold short-circuit), so a caller can select the matching
    /// pre-partitioned weight shard (`compute(global_index, output_start,
    /// outputs)`). `result.len()` must equal `n` passed to
    /// [`Self::output_column_segments`], and `align` must match so the dispatch
    /// segments line up byte-for-byte with the caller's pre-built shards; each
    /// worker writes only its own segment, so the concatenated result is
    /// bit-identical to the single-worker path.
    pub fn dispatch_output_rows_indexed<F>(&self, result: &mut [f32], align: usize, compute: &F)
    where
        F: Fn(usize, usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.worker_row_segments_aligned(n, align);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds column
            // ranges covering `0..n` exactly once, so each worker's slice never
            // aliases another's.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(global_index, start, outputs);
        };
        self.dispatch(&job);
    }

    /// Shard `result`'s `num_rows` fixed-width rows (each `row_len` elements)
    /// across the resident workers and run `compute(row_index, row_slice)` on
    /// each whole row under one lightweight barrier.
    ///
    /// Unlike [`Self::dispatch_output_rows`] (which shards a GEMV's scalar output
    /// rows), this keeps every `row_len`-element row intact on a single worker,
    /// so a caller whose per-row closure needs the full contiguous row (e.g. an
    /// attention head's output vector) can run on the persistent decode pool
    /// instead of a second, contending thread pool. Rows are handed out
    /// contiguously, so concatenating the per-worker slices reproduces the
    /// single-threaded result bit-for-bit (each row is independent).
    pub fn dispatch_output_row_blocks<F>(
        &self,
        result: &mut [f32],
        row_len: usize,
        num_rows: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        debug_assert_eq!(result.len(), row_len.saturating_mul(num_rows));
        if self.total_workers <= 1 || num_rows <= 1 || row_len == 0 {
            for row in 0..num_rows {
                compute(row, &mut result[row * row_len..(row + 1) * row_len]);
            }
            return;
        }
        let segments = self.worker_row_segments(num_rows);
        let table = RowBlockTable {
            base: result.as_mut_ptr(),
            row_len,
            segments: &segments,
        };
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            for row in start..start + len {
                // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
                // ranges covering `0..num_rows` exactly once, so each worker's
                // `[row*row_len, (row+1)*row_len)` slice never aliases another's.
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        table.base.add(row * table.row_len),
                        table.row_len,
                    )
                };
                compute(row, slice);
            }
        };
        self.dispatch(&job);
    }

    /// Broadcast the output-row shards to every worker under one barrier,
    /// unconditionally (no serial-threshold check). The public
    /// [`Self::dispatch_output_rows`] applies the threshold before calling this;
    /// tests exercise the multi-worker path directly through it.
    fn dispatch_rows_across_workers<F>(&self, result: &mut [f32], compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.worker_row_segments(n);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        // Bind a reference so the `move` closure captures the whole `RowTable`
        // (which carries the manual `Sync` impl) rather than its raw pointer
        // field individually (disjoint capture does not reach through a
        // reference).
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
            // ranges covering `0..n` exactly once, and each worker touches only
            // its own segment, so these mutable slices never alias.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(start, outputs);
        };
        self.dispatch(&job);
    }

    /// Copy `src` into a fresh buffer whose per-node row shards are first-touched
    /// on their owning node, so each worker later streams node-local memory.
    ///
    /// `src` is a row-major `[n, stride]` weight component; the row split matches
    /// [`Self::worker_row_segments`] exactly so it lines up with dispatch.
    pub fn place_rows<T: Copy + Send + Sync>(&self, src: &[T], n: usize) -> Vec<T> {
        if n == 0 || src.is_empty() || self.total_workers <= 1 {
            return src.to_vec();
        }
        let stride = src.len() / n;
        debug_assert_eq!(stride * n, src.len());
        let mut dst: Vec<T> = Vec::with_capacity(src.len());
        // Leave the buffer uninitialized on purpose: zero-filling here would
        // fault every page onto the dispatcher's node, defeating the node-local
        // first-touch performed by the pinned workers below.
        // SAFETY: `T: Copy` has no `Drop`, capacity is exactly `src.len()`, and
        // every element is overwritten by the per-worker `copy_from_slice`
        // (`worker_row_segments` covers `0..n` exactly once) before the buffer is
        // read.
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        let segments = self.worker_row_segments(n);
        let table = CopyTable {
            dst: dst.as_mut_ptr(),
            src: src.as_ptr(),
            stride,
            segments: &segments,
        };
        // Capture the whole `CopyTable` (manual `Sync`) rather than its raw
        // pointer fields individually.
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: disjoint, in-bounds `[start, start+len)` row ranges (in
            // units of `stride`), covering every row exactly once; the pinned
            // worker's write faults these destination pages onto its own node.
            unsafe {
                let dst = table.dst.add(start * table.stride);
                let src = table.src.add(start * table.stride);
                std::ptr::copy_nonoverlapping(src, dst, len * table.stride);
            }
        };
        self.dispatch(&job);
        dst
    }
}

impl Drop for SpmdDecodePools {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // Bump the sequence and unpark so parked workers observe the shutdown.
        self.shared.sequence.0.fetch_add(1, Ordering::AcqRel);
        for handle in &self.handles {
            handle.thread().unpark();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Row-major output view handed to a dispatched compute job. Disjoint per-worker
/// segments make the raw base pointer safe to share.
struct RowTable<'a> {
    base: *mut f32,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index reads only its own disjoint row segment.
unsafe impl Sync for RowTable<'_> {}

/// Output view for fixed-width row-block dispatch: `base` is a `[num_rows,
/// row_len]` row-major buffer, and each worker owns the disjoint row range
/// `segments[worker]`.
struct RowBlockTable<'a> {
    base: *mut f32,
    row_len: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index writes only its own disjoint row range.
unsafe impl Sync for RowBlockTable<'_> {}

/// Source/destination view for node-local weight placement.
struct CopyTable<'a, T> {
    dst: *mut T,
    src: *const T,
    stride: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each worker copies only its own disjoint row range.
unsafe impl<T: Send + Sync> Sync for CopyTable<'_, T> {}

/// Ensures a worker always acknowledges the current op while making the normal
/// path no more expensive than the existing atomic decrement. `complete`
/// forgets the guard after decrementing; only unwinding executes `Drop`, which
/// poisons the pool before decrementing so the dispatcher cannot miss the panic.
struct WorkerCompletion<'a> {
    shared: &'a SharedState,
    node: usize,
    global_index: usize,
}

impl WorkerCompletion<'_> {
    fn complete(self) {
        self.shared.node_pending[self.node]
            .0
            .fetch_sub(1, Ordering::AcqRel);
        std::mem::forget(self);
    }
}

impl Drop for WorkerCompletion<'_> {
    fn drop(&mut self) {
        self.shared
            .poisoned_worker
            .compare_exchange(
                0,
                self.global_index + 1,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .ok();
        self.shared.node_pending[self.node]
            .0
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// The persistent worker main loop: wait for a published op, run this worker's
/// shard, acknowledge, repeat until shutdown.
fn worker_loop(shared: Arc<SharedState>, global_index: usize) {
    let node = shared.worker_node[global_index];
    // Start from the pre-dispatch baseline (sequence is 0 until the first op),
    // then announce readiness. The dispatcher blocks in `build` until every
    // worker has done this, so no op can be published before this worker is
    // waiting for it.
    let mut local_seq = 0usize;
    shared.ready.fetch_add(1, Ordering::AcqRel);
    loop {
        let mut spins = 0u32;
        let mut yields = 0u32;
        // Wait for a new op (or shutdown).
        let new_seq = loop {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            let seq = shared.sequence.0.load(Ordering::Acquire);
            if seq != local_seq {
                break seq;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins >= SPIN_BEFORE_YIELD {
                yields = yields.wrapping_add(1);
                if yields >= YIELD_BEFORE_PARK {
                    // Deep idle: park and STAY parked, re-parking on each spurious
                    // or timeout wake, until a real op arrives (or shutdown). This
                    // is critical for `Auto` mode: while the flat path is
                    // committed the pool is idle for the whole generation, and an
                    // idle worker that re-ran the spin cycle every timeout would
                    // burn cycles and contend with the flat path -- polluting the
                    // calibration probe and regressing committed-flat decode under
                    // load (the exact bug that made Auto mis-commit to the pool).
                    // `parked` stays true for the whole loop so the dispatcher's
                    // unpark always lands. Publish the park intent, then re-check
                    // the sequence (SeqCst) so a wakeup that raced the flag store
                    // is not lost; the bounded timeout is a final backstop.
                    shared.parked[global_index].0.store(true, Ordering::SeqCst);
                    while shared.sequence.0.load(Ordering::SeqCst) == local_seq
                        && !shared.shutdown.load(Ordering::SeqCst)
                    {
                        thread::park_timeout(PARK_TIMEOUT);
                    }
                    shared.parked[global_index].0.store(false, Ordering::SeqCst);
                    yields = 0;
                }
                spins = 0;
            }
        };
        local_seq = new_seq;
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        // Read and run the published op. The acquire on `sequence` above
        // established visibility of the job pointer and the pending counts.
        // SAFETY: the dispatcher keeps the pointee alive until every node
        // counter reaches zero, i.e. until after this worker acknowledges below.
        let job = unsafe { (*shared.job.get()).expect("published SPMD job") };
        let completion = WorkerCompletion {
            shared: &shared,
            node,
            global_index,
        };
        // SAFETY: `dispatch` keeps the closure alive until this worker
        // acknowledges through `completion`.
        unsafe { (job.call)(job.data, global_index) };
        completion.complete();
    }
}

/// The lazily built persistent SPMD layout, or `None` when the mode is opted out
/// or the safe auto-enable gate declines. Built once and reused for the whole
/// process.
pub fn pools() -> Option<&'static SpmdDecodePools> {
    static POOLS: OnceLock<Option<SpmdDecodePools>> = OnceLock::new();
    POOLS
        .get_or_init(|| build_from_env(default_threads()))
        .as_ref()
}

/// Resolve the persistent pool's worker count. Honors `ONNX_GENAI_CPU_DECODE_THREADS`
/// when set (`0` opts out); when unset it uses the persistent-specific default
/// (about half the logical CPUs), *not* the flat pool's eight-worker ceiling --
/// see [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`].
fn default_threads() -> Option<usize> {
    crate::kernels::matmul_nbits::configured_persistent_decode_threads()
}

/// How the persistent pool was selected, parsed from `PERSISTENT_POOL_ENV`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistenceMode {
    /// `=0`: explicit opt-out; the decode path stays on the flat legacy pool.
    Off,
    /// Unset (or an unrecognized value): the default. The pool is opt-in, so this
    /// leaves decode on the flat path (same effective path as `Off`).
    Auto,
    /// `=1`: opt in to the persistent pool (operator override for dedicated hosts).
    Forced,
}

/// Parse the persistence mode from the raw env value (`None` = unset). Only the
/// exact string `1` opts in to the persistent pool; `0` is the explicit opt-out
/// and unset or any other value maps to `Auto`, which uses the flat path by
/// default (the pool is opt-in).
pub(crate) fn persistence_mode_from_raw(raw: Option<&str>) -> PersistenceMode {
    match raw.map(str::trim) {
        Some("0") => PersistenceMode::Off,
        Some("1") => PersistenceMode::Forced,
        _ => PersistenceMode::Auto,
    }
}

fn persistence_mode() -> PersistenceMode {
    persistence_mode_from_raw(std::env::var(PERSISTENT_POOL_ENV).ok().as_deref())
}

/// Whether a persistence mode **builds** the persistent SPMD pool. Both `Forced`
/// (`=1`) and `Auto` (the unset default) build it: `Forced` always dispatches to
/// it, and `Auto` needs it available so calibration can time the real workload on
/// it and adopt it when it is faster. Only `Off` (`=0`) never builds it. Pure so
/// the gating is unit-tested without env races.
fn pool_mode_builds(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::Forced | PersistenceMode::Auto)
}

/// Whether a persistence mode **unconditionally** dispatches to the pool (no
/// calibration): only `Forced` (`=1`). `Auto` builds the pool but lets the
/// [`Calibrator`] pick per step; `Off` never uses it.
fn pool_mode_forces(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::Forced)
}

/// Whether the persistent pool was **explicitly opted into** (`PERSISTENT_POOL=1`).
/// Used to keep the `numa-split` mutual-exclusion diagnostic scoped to users who
/// actually asked for the persistent pool, to make dense-f32 decode still
/// eligible for the pool when forced, and to skip calibration (always dispatch).
pub(crate) fn is_forced() -> bool {
    pool_mode_forces(persistence_mode())
}

/// Build the persistent SPMD layout when the mode builds it (`=1` Forced or the
/// unset `Auto` default); `=0` (Off) or `THREADS=0` return `None` so decode stays
/// on the flat path. Under `Auto` the pool is built but only *used* when
/// calibration adopts it (see [`Calibrator`]); under `Forced` it is always used.
///
/// Two or more usable NUMA nodes yield the two-level node-pinned layout; a
/// single-node host, a non-NUMA machine, or a platform without pinning yields a
/// single unpinned worker group (still the lightweight barrier, still correct).
pub fn build_from_env(threads: Option<usize>) -> Option<SpmdDecodePools> {
    // Build for `Forced` (`=1`) and the `Auto` default (unset). `Auto` needs the
    // pool available so the calibrator can time the live decode step on it and
    // adopt it only when it is measured faster than the flat path (it stays on
    // the flat path under load); `Forced` always dispatches to it. `Off` (`=0`)
    // and `THREADS=0` leave decode on the flat Rayon path. See `PERSISTENT_POOL_ENV`.
    let mode = persistence_mode();
    if !pool_mode_builds(mode) {
        return None;
    }
    // Auto defers to an explicit decode-affinity request: if the user set
    // `ONNX_GENAI_CPU_DECODE_AFFINITY` (numa-split, compact, node:N, off, ...),
    // they picked a specific strategy, so Auto does not build/calibrate the
    // persistent pool and lets that request drive decode (numa-split via
    // `numa_pools`, everything else via the flat path + `plan_decode_affinity`).
    // `Forced` (`=1`) still builds the pool and keeps its documented precedence.
    if matches!(mode, PersistenceMode::Auto) && explicit_decode_affinity_set() {
        return None;
    }
    let Some(total) = threads else {
        report_spmd_fallback(
            "ONNX_GENAI_CPU_DECODE_THREADS=0 opts out of the bounded pool; the persistent \
             SPMD pool needs a bounded worker count -- leaving the decode path unchanged",
        );
        return None;
    };
    if total == 0 {
        return None;
    }
    report_pool_built(mode);
    let shards = node_shards(total);
    Some(SpmdDecodePools::build(&shards))
}

/// Whether `ONNX_GENAI_CPU_DECODE_AFFINITY` is set to a non-empty value. Auto
/// calibration only engages when it is unset, so an explicit affinity request is
/// honored on the flat/numa path exactly as before.
fn explicit_decode_affinity_set() -> bool {
    std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Resolve the node shards for `total` workers: the multi-node split when the
/// (cpuset-restricted) topology exposes >=2 nodes, otherwise a single group.
fn node_shards(total: usize) -> Vec<NodeShard> {
    let allowed = crate::decode_affinity::allowed_cpus();
    if let Some(topology) = NumaTopology::detect() {
        let topology = topology.restrict_to_allowed(allowed.as_deref());
        if let Some(shards) = topology.split_workers(total) {
            return shards;
        }
    }
    // Single-node / non-NUMA / no-pinning fallback: one group. Pin to the
    // process's allowed CPUs when known (best-effort), else leave unpinned.
    let cpus = allowed.unwrap_or_default();
    vec![NodeShard {
        index: 0,
        cpus,
        workers: total,
    }]
}

/// Log the first persistent-pool fallback/pinning problem once so a restricted
/// or unsupported host surfaces the reason without spamming every worker.
fn report_spmd_fallback(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: persistent SPMD decode pool: {message}");
    }
}

/// Announce once how the persistent SPMD pool was built so the non-default
/// decode path is inspectable: `Forced` (`=1`) always dispatches to it, while the
/// `Auto` default builds it but only adopts it when calibration measures it
/// faster than the flat path (and stays on the flat path under load).
fn report_pool_built(mode: PersistenceMode) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        match mode {
            PersistenceMode::Forced => eprintln!(
                "onnx-genai: persistent SPMD decode pool forced on via \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1 (always dispatches to the pool)"
            ),
            _ => eprintln!(
                "onnx-genai: persistent SPMD decode pool built for auto-calibration \
                 (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL unset); each decode step is timed \
                 both ways and the faster path is kept -- the flat path stays committed \
                 under load. Set =0 to force flat, =1 to force the pool"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-calibration: pick the pool or the flat path by measuring the live decode
// step both ways, keeping the faster one.
// ---------------------------------------------------------------------------

/// Which decode path a single `Auto`-mode decode step should take. Both paths are
/// token-exact (the pool is N-tile aligned, PR #110), so switching between them
/// never changes the emitted tokens -- that is exactly what lets calibration time
/// the *same real workload* both ways.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoPath {
    /// Dispatch this step's projections to the persistent SPMD pool.
    Pool,
    /// Run this step on the flat Rayon decode pool (the safe default under load).
    Flat,
}

/// Decode steps spent on the pool before the first measurement, so the pool's
/// one-time constant weights are prepacked and node-locally first-touched (see
/// [`SpmdDecodePools::place_rows`]) and caches are warm before it is timed.
/// Without this the pool would be measured reading cross-node memory and unfairly
/// lose; it is a handful of steps, amortized over the whole generation.
const CALIB_WARMUP_STEPS: u64 = 2;
/// Samples collected per path during a probe. The decision uses the median, so an
/// odd count with a small majority rejects a single load-spike outlier while
/// keeping the probe short (a probe costs `2 * CALIB_PROBE_SAMPLES` real steps,
/// half of them possibly-slower pool steps).
const CALIB_PROBE_SAMPLES: usize = 5;
/// Committed steps between re-probes. Long enough that probe overhead is
/// negligible (a probe is `<= 2 * CALIB_PROBE_SAMPLES` steps per period, so
/// worst-case < ~2% of steps ever run a possibly-slower pool probe), short enough
/// that a host which becomes loaded mid-generation falls back within one window.
const CALIB_RECAL_PERIOD: u64 = 600;
/// Hysteresis margin: the pool is adopted only when its median step time is at
/// least this percent faster than the flat path. Biases toward the flat path (the
/// regression-safe default) and prevents flapping when the two paths are close.
const CALIB_SWITCH_MARGIN_PCT: u64 = 8;
/// Samples discarded at the start of each probe block. The persistent pool's
/// worker threads keep spinning for a short while after their last dispatch, so
/// the *first* flat step after a pool step (and vice-versa) is polluted by the
/// other path's threads still winding down. Discarding the transition sample
/// makes each block measure its path in isolation -- critical because measuring
/// flat while pool workers are still hot makes flat look slow and would bias the
/// choice *toward* the pool (the regression). See the block-ordered probe below.
const CALIB_PROBE_DISCARD: usize = 1;

/// The calibration probe measures the two paths in **separate contiguous blocks**
/// (all flat, then all pool) rather than interleaving them, so a just-finished
/// pool step's still-spinning workers never pollute a flat measurement. Flat is
/// measured first, while the pool is quiesced (parked), which is exactly the
/// steady state a committed-flat `Auto` run experiences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CalibPhase {
    /// Warm the pool (prepack + node-local first-touch) before any measurement.
    Warmup,
    /// Collect the flat block, with the pool quiesced (its worker threads parked).
    ProbeFlat,
    /// Collect the pool block.
    ProbePool,
    /// Run the committed path until the recalibration period elapses.
    Committed,
}

/// Runtime calibrate-and-pick state machine for `Auto` mode.
///
/// # Why this cannot regress under load
///
/// * The committed path starts as -- and defaults back to -- [`AutoPath::Flat`],
///   today's flat decode path. A host that never lets the pool win keeps running
///   exactly the flat path.
/// * The pool is adopted only when its *measured* median step time beats the flat
///   path's by [`CALIB_SWITCH_MARGIN_PCT`]. Under co-tenant load the spinning
///   pool is slower, so it never clears the bar and the flat path stays committed.
/// * Flat and pool are measured in **separate blocks** (flat first, pool
///   quiesced), with the transition sample discarded ([`CALIB_PROBE_DISCARD`]),
///   so a pool step's still-spinning workers cannot make the flat measurement
///   look slow. (An interleaved probe *did* mis-commit to the pool under load;
///   see `.squad/decisions.md`, Hudson 2026-07-24.)
/// * The only pool work done while the flat path is committed is a bounded probe
///   (`<= CALIB_PROBE_SAMPLES` pool steps per [`CALIB_RECAL_PERIOD`] steps, plus a
///   one-time warmup), so the worst case is a small, self-correcting number of
///   possibly-slower steps -- never a sustained regression.
/// * Re-probing lets a host that *becomes* loaded mid-generation fall back within
///   one recalibration window, and a host that *becomes* idle adopt the pool.
///
/// The logic is pure (no threads, no clock, no env), so it is unit-tested
/// deterministically by feeding synthetic per-path samples.
struct Calibrator {
    phase: CalibPhase,
    warmup_left: u64,
    discard_left: usize,
    pool_ns: Vec<u64>,
    flat_ns: Vec<u64>,
    committed: AutoPath,
    committed_left: u64,
}

impl Calibrator {
    fn new() -> Self {
        Self {
            phase: CalibPhase::Warmup,
            warmup_left: CALIB_WARMUP_STEPS,
            discard_left: 0,
            pool_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            flat_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            // Default to the flat path: the safe, no-regression baseline that a
            // host which never lets the pool win keeps forever.
            committed: AutoPath::Flat,
            committed_left: 0,
        }
    }

    /// The path the next decode step should take. Warmup uses the pool (to place
    /// weights node-locally); the probe runs the flat block then the pool block;
    /// a committed phase returns the committed path.
    fn choose(&self) -> AutoPath {
        match self.phase {
            CalibPhase::Warmup | CalibPhase::ProbePool => AutoPath::Pool,
            CalibPhase::ProbeFlat => AutoPath::Flat,
            CalibPhase::Committed => self.committed,
        }
    }

    /// Feed back the measured wall time (nanoseconds) of a step that took `path`.
    fn record(&mut self, path: AutoPath, ns: u64) {
        match self.phase {
            CalibPhase::Warmup => {
                self.warmup_left = self.warmup_left.saturating_sub(1);
                if self.warmup_left == 0 {
                    self.enter_flat_probe();
                }
            }
            CalibPhase::ProbeFlat => {
                if path == AutoPath::Flat {
                    self.push_sample_or_discard(ns, true);
                }
                if self.flat_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.enter_pool_probe();
                }
            }
            CalibPhase::ProbePool => {
                if path == AutoPath::Pool {
                    self.push_sample_or_discard(ns, false);
                }
                if self.pool_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.commit_from_samples();
                }
            }
            CalibPhase::Committed => {
                self.committed_left = self.committed_left.saturating_sub(1);
                if self.committed_left == 0 {
                    self.enter_flat_probe();
                }
            }
        }
    }

    /// Record a sample into the current block, discarding the leading transition
    /// sample(s) so the other path's winding-down threads do not pollute it.
    fn push_sample_or_discard(&mut self, ns: u64, flat: bool) {
        if self.discard_left > 0 {
            self.discard_left -= 1;
            return;
        }
        let block = if flat {
            &mut self.flat_ns
        } else {
            &mut self.pool_ns
        };
        if block.len() < CALIB_PROBE_SAMPLES {
            block.push(ns);
        }
    }

    fn enter_flat_probe(&mut self) {
        self.phase = CalibPhase::ProbeFlat;
        self.flat_ns.clear();
        self.pool_ns.clear();
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn enter_pool_probe(&mut self) {
        self.phase = CalibPhase::ProbePool;
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn commit_from_samples(&mut self) {
        let pool = median_ns(&mut self.pool_ns);
        let flat = median_ns(&mut self.flat_ns);
        // Adopt the pool only when it is at least CALIB_SWITCH_MARGIN_PCT faster:
        // pool <= flat * (100 - margin) / 100. Use u128 so the multiply cannot
        // overflow for pathologically large samples.
        let pool_scaled = u128::from(pool) * 100;
        let flat_scaled = u128::from(flat) * u128::from(100 - CALIB_SWITCH_MARGIN_PCT);
        self.committed = if pool_scaled <= flat_scaled {
            AutoPath::Pool
        } else {
            AutoPath::Flat
        };
        self.phase = CalibPhase::Committed;
        self.committed_left = CALIB_RECAL_PERIOD;
        self.pool_ns.clear();
        self.flat_ns.clear();
    }
}

/// Median of the samples (upper-middle for an even count). `u64::MAX` for an empty
/// slice so an unmeasured path never looks like the fast choice.
fn median_ns(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return u64::MAX;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn calibrator() -> &'static Mutex<Calibrator> {
    static CALIBRATOR: OnceLock<Mutex<Calibrator>> = OnceLock::new();
    CALIBRATOR.get_or_init(|| Mutex::new(Calibrator::new()))
}

/// The path the next `Auto`-mode decode step should take (see [`Calibrator`]).
pub(crate) fn auto_choose_path() -> AutoPath {
    calibrator()
        .lock()
        .map(|calib| calib.choose())
        // A poisoned lock (a panic in a prior decode step) should never change
        // tokens or crash decode -- fall back to the safe flat path.
        .unwrap_or(AutoPath::Flat)
}

/// Feed the measured wall time of an `Auto`-mode decode step back to calibration.
pub(crate) fn auto_record_sample(path: AutoPath, elapsed: Duration) {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    if let Ok(mut calib) = calibrator().lock() {
        calib.record(path, ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_group_pool() -> SpmdDecodePools {
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![],
                workers: 2,
            },
            NodeShard {
                index: 1,
                cpus: vec![],
                workers: 2,
            },
        ];
        SpmdDecodePools::build(&shards)
    }

    fn single_group_pool(workers: usize) -> SpmdDecodePools {
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![],
            workers,
        }];
        SpmdDecodePools::build(&shards)
    }

    #[test]
    fn node_row_lengths_split_proportionally_and_cover_all_rows() {
        let pool = two_group_pool();
        assert_eq!(pool.node_row_lengths(100), vec![50, 50]);
        assert_eq!(pool.node_row_lengths(101), vec![50, 51]);
        assert_eq!(pool.node_row_lengths(1), vec![0, 1]);
        assert_eq!(pool.node_row_lengths(0), vec![0, 0]);
    }

    #[test]
    fn worker_row_segments_are_disjoint_and_cover_every_row() {
        let pool = two_group_pool();
        let n = 37usize;
        let segments = pool.worker_row_segments(n);
        assert_eq!(segments.len(), pool.total_workers());
        // Contiguous, non-overlapping, covering exactly 0..n.
        let mut expected_start = 0;
        for (start, len) in &segments {
            assert_eq!(*start, expected_start);
            expected_start += len;
        }
        assert_eq!(expected_start, n);
    }

    #[test]
    fn worker_row_segments_aligned_snaps_interior_boundaries_and_covers_every_row() {
        // Every interior boundary must be a multiple of `align`; the segments
        // must still be contiguous, disjoint, and cover exactly 0..n. This is
        // the invariant the MLAS SQNBit decode shard path relies on to keep each
        // N-tile whole (and thus bit-identical to the full-width call).
        let pool = single_group_pool(3);
        for &align in &[4usize, 16] {
            for &n in &[97usize, 128, 151936, 1, 0, 5, 17] {
                let segments = pool.worker_row_segments_aligned(n, align);
                assert_eq!(segments.len(), pool.total_workers());
                let mut expected_start = 0;
                for (index, &(start, len)) in segments.iter().enumerate() {
                    assert_eq!(start, expected_start, "n={n} align={align} seg {index}");
                    // Interior boundaries (every start past the first) must be
                    // align-aligned; the final segment may end at an unaligned n.
                    assert_eq!(
                        start % align,
                        0,
                        "n={n} align={align}: segment start {start} not aligned"
                    );
                    expected_start += len;
                }
                assert_eq!(expected_start, n, "n={n} align={align}: must cover 0..n");
            }
        }
    }

    #[test]
    fn worker_row_segments_aligned_is_identity_for_align_one() {
        let pool = two_group_pool();
        for &n in &[0usize, 1, 37, 100, 101] {
            assert_eq!(
                pool.worker_row_segments_aligned(n, 1),
                pool.worker_row_segments(n),
                "align=1 must reproduce the unaligned split (n={n})"
            );
        }
    }

    #[test]
    fn dispatch_output_rows_matches_flat_computation() {
        let pool = two_group_pool();
        let n = 101usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32 * 2.5 - 3.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn dispatch_preserves_per_row_reduction_bit_for_bit() {
        // Mirror the real GEMV: each output row is a full-K f32 dot product. Row
        // sharding must not reorder the per-row accumulation, so the SPMD result
        // must be *byte-for-byte* identical to a single-threaded reference (this
        // is the parity invariant the greedy-token equality relies on).
        let pool = two_group_pool();
        let n = 257usize;
        let k = 320usize;
        // Deterministic pseudo-random-ish weights/activation, mixed signs/scales.
        let activation: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.031_25)
            .collect();
        let weight = |row: usize, col: usize| -> f32 {
            (((row * 131 + col * 17) % 251) as f32 - 125.0) * 0.007_812_5
        };
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                let mut acc = 0.0f32;
                for (col, &a) in activation.iter().enumerate() {
                    acc += a * weight(row, col);
                }
                *out = acc;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut reference = vec![0.0f32; n];
        compute(0, &mut reference);
        assert_eq!(
            sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row-sharded dispatch must be bit-identical to the serial reference"
        );
    }

    #[test]
    fn dispatch_output_row_blocks_matches_flat_computation() {
        // Fixed-width row blocks (mirrors GroupQueryAttention's per-head output
        // rows): every `row_len`-element row is computed whole on one worker, so
        // the sharded result must equal the single-threaded reference row-for-row
        // and bit-for-bit (rows are independent).
        for (num_rows, row_len) in [
            (28usize, 128usize),
            (3, 128),
            (1, 64),
            (5, 3),
            (37, 1),
            (0, 8),
        ] {
            let pool = two_group_pool();
            let compute = |row_index: usize, row: &mut [f32]| {
                for (offset, out) in row.iter_mut().enumerate() {
                    // Order-sensitive accumulation to catch any row reordering.
                    let mut acc = 0.0f32;
                    for step in 0..=offset {
                        acc += (row_index * 7 + step) as f32 * 0.015_625 - 1.0;
                    }
                    *out = acc;
                }
            };
            let mut sharded = vec![0.0f32; num_rows * row_len];
            pool.dispatch_output_row_blocks(&mut sharded, row_len, num_rows, &compute);
            let mut reference = vec![0.0f32; num_rows * row_len];
            for row in 0..num_rows {
                compute(row, &mut reference[row * row_len..(row + 1) * row_len]);
            }
            assert_eq!(
                sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row-block dispatch must be bit-identical to the serial reference \
                 (num_rows={num_rows}, row_len={row_len})"
            );
        }
    }

    #[test]
    fn dispatch_is_reusable_across_many_ops() {
        // Exercises the barrier repeatedly: every worker must re-arm and the
        // dispatcher must observe completion each time (regression guard for
        // the sequence/pending protocol).
        let pool = single_group_pool(4);
        for round in 0..200usize {
            let n = 53usize;
            let compute = move |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (round * 1000 + output_start + offset) as f32;
                }
            };
            let mut got = vec![0.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want, "round {round}");
        }
    }

    #[test]
    fn build_then_immediate_dispatch_never_hangs() {
        // Regression guard: a dispatch issued right after `build` must not race
        // a not-yet-started worker (which would hang the barrier). Rebuild a
        // fresh pool and dispatch across all workers immediately, many times.
        for _ in 0..40usize {
            let pool = single_group_pool(6);
            let n = 61usize;
            let compute = |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (output_start + offset) as f32;
                }
            };
            let mut got = vec![-1.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn place_rows_preserves_bytes() {
        let pool = two_group_pool();
        let n = 7usize;
        let stride = 4usize;
        let src: Vec<u8> = (0..(n * stride) as u8).collect();
        assert_eq!(pool.place_rows(&src, n), src);

        let scales: Vec<f32> = (0..n).map(|row| row as f32 * 0.5).collect();
        assert_eq!(pool.place_rows(&scales, n), scales);
    }

    #[test]
    fn tiny_ops_run_serially_but_correctly() {
        // Below the parallelization threshold the op runs on the dispatcher;
        // the result must still be correct.
        let pool = single_group_pool(8);
        let n = 3usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };
        let mut got = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut got, 4096, &compute);
        assert_eq!(got, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn panicking_worker_poison_is_reported_without_hanging() {
        let pool = single_group_pool(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch(&|worker| {
                assert_ne!(worker, 2, "intentional SPMD worker panic");
            });
        }));
        let panic = result.expect_err("dispatcher must report a worker panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            message.contains("persistent SPMD decode worker 2 panicked")
                && message.contains("pool is poisoned"),
            "unexpected dispatcher diagnostic: {message}"
        );
    }

    #[test]
    fn persistence_mode_parses_env_values() {
        // Mode parsing (the pool is opt-in): unset -> Auto (flat path), `0` -> Off
        // (flat path), `1` -> Forced (pool). Whitespace is trimmed, and any
        // unrecognized value maps to Auto rather than surprising the user.
        assert_eq!(persistence_mode_from_raw(None), PersistenceMode::Auto);
        assert_eq!(persistence_mode_from_raw(Some("")), PersistenceMode::Auto);
        assert_eq!(
            persistence_mode_from_raw(Some("   ")),
            PersistenceMode::Auto
        );
        assert_eq!(persistence_mode_from_raw(Some("0")), PersistenceMode::Off);
        assert_eq!(persistence_mode_from_raw(Some(" 0 ")), PersistenceMode::Off);
        assert_eq!(
            persistence_mode_from_raw(Some("1")),
            PersistenceMode::Forced
        );
        assert_eq!(
            persistence_mode_from_raw(Some(" 1 ")),
            PersistenceMode::Forced
        );
        // Unknown values map to Auto (flat path), never a surprise pool activation.
        assert_eq!(
            persistence_mode_from_raw(Some("true")),
            PersistenceMode::Auto
        );
        assert_eq!(persistence_mode_from_raw(Some("2")), PersistenceMode::Auto);

        // The pool is BUILT for `Forced` (`=1`) and the `Auto` default (unset);
        // only `Off` (`=0`) skips building. Auto is not *forced* (calibrated).
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("1"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
        assert!(!pool_mode_forces(persistence_mode_from_raw(None)));
    }

    #[test]
    fn auto_and_forced_build_the_pool_but_only_forced_dispatches_unconditionally() {
        // Auto-enable design (Hudson, 2026-07-24): the pool is built for `Auto`
        // (the unset default) so calibration can time the live decode step on it
        // and adopt it only when it is measured faster than the flat path -- while
        // `Off` (`=0`) never builds it. `Forced` (`=1`) both builds it and forces
        // dispatch (no calibration). This preserves the no-regression guarantee:
        // Auto's committed default is the flat path (see `Calibrator`).
        assert!(pool_mode_builds(PersistenceMode::Auto));
        assert!(pool_mode_builds(PersistenceMode::Forced));
        assert!(!pool_mode_builds(PersistenceMode::Off));

        assert!(pool_mode_forces(PersistenceMode::Forced));
        assert!(!pool_mode_forces(PersistenceMode::Auto));
        assert!(!pool_mode_forces(PersistenceMode::Off));

        // The default env value (unset) maps to Auto: built, calibrated, not forced.
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(!pool_mode_forces(persistence_mode_from_raw(None)));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("2"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
    }

    /// Drive the calibrator from its current phase until it commits, feeding each
    /// step the per-path time it chose. Returns once the committed phase is reached.
    fn drive_to_commit(calib: &mut Calibrator, pool_ns: u64, flat_ns: u64) {
        for _ in 0..100_000 {
            if calib.phase == CalibPhase::Committed {
                return;
            }
            let path = calib.choose();
            let ns = match path {
                AutoPath::Pool => pool_ns,
                AutoPath::Flat => flat_ns,
            };
            calib.record(path, ns);
        }
        panic!("calibrator never reached the committed phase");
    }

    /// Fresh calibrator driven through warmup + one probe with the given per-path
    /// step times; returns the committed decision.
    fn run_one_probe(pool_ns: u64, flat_ns: u64) -> Calibrator {
        let mut calib = Calibrator::new();
        drive_to_commit(&mut calib, pool_ns, flat_ns);
        calib
    }

    #[test]
    fn calibrator_defaults_to_flat_before_any_measurement() {
        // The no-regression baseline: a fresh calibrator's committed path is the
        // flat path, so a host that never lets the pool win runs exactly today's
        // flat decode path.
        let calib = Calibrator::new();
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_probe_measures_flat_block_before_pool_block() {
        // Warmup runs on the pool (node-local placement), then the flat block is
        // measured first (pool quiesced), then the pool block -- never interleaved,
        // so a hot pool step cannot pollute a flat measurement.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            assert_eq!(calib.choose(), AutoPath::Pool);
            assert_eq!(calib.phase, CalibPhase::Warmup);
            calib.record(AutoPath::Pool, 1_000);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        // The whole flat block chooses Flat.
        while calib.phase == CalibPhase::ProbeFlat {
            assert_eq!(calib.choose(), AutoPath::Flat);
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.phase, CalibPhase::ProbePool);
        // The whole pool block chooses Pool.
        while calib.phase == CalibPhase::ProbePool {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 100);
        }
        assert_eq!(calib.phase, CalibPhase::Committed);
    }

    #[test]
    fn calibrator_probe_discards_the_transition_sample() {
        // The first sample of each block is discarded so the other path's
        // winding-down threads do not pollute it: only CALIB_PROBE_SAMPLES land.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        assert_eq!(calib.discard_left, CALIB_PROBE_DISCARD);
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.flat_ns.len(), CALIB_PROBE_SAMPLES);
    }

    #[test]
    fn calibrator_commits_pool_only_when_clearly_faster() {
        // Pool 20% faster than flat clears the 8% hysteresis margin -> adopt pool.
        let calib = run_one_probe(80, 100);
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(calib.committed, AutoPath::Pool);
        assert_eq!(calib.choose(), AutoPath::Pool);
    }

    #[test]
    fn calibrator_stays_flat_when_pool_slower_simulating_contention() {
        // Regression guard: under (simulated) co-tenant load the spinning pool is
        // slower, so its median probe time loses and the flat path stays
        // committed -- decode behaves exactly like today's flat path. This is the
        // property that makes "never regress under load" a measured guarantee.
        let calib = run_one_probe(200, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
        assert_eq!(calib.choose(), AutoPath::Flat);
    }

    #[test]
    fn calibrator_stays_flat_within_the_hysteresis_margin() {
        // Pool only ~5% faster (< 8% margin): not worth switching, avoids flapping
        // and keeps the safe flat default.
        let calib = run_one_probe(95, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_reprobes_after_the_recal_period_and_can_fall_back() {
        // Adopt the pool on a quiet probe, then simulate the host becoming loaded:
        // after CALIB_RECAL_PERIOD committed steps the machine re-probes, measures
        // the pool as slower, and falls back to the flat path within one window.
        let mut calib = run_one_probe(80, 100);
        assert_eq!(calib.committed, AutoPath::Pool);
        // Burn the committed window; choose() keeps returning Pool until re-probe.
        for _ in 0..CALIB_RECAL_PERIOD {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 80);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        // Now the host is loaded: pool probes slow, flat probes fast.
        drive_to_commit(&mut calib, 300, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_probe_median_rejects_a_single_load_spike() {
        // The pool is genuinely faster (median 80) but one probe sample spikes
        // under a transient stall; the median ignores the outlier and still adopts
        // the pool, so a single blip does not cost the win.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 80);
        }
        // Flat block: all 100.
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        // Pool block: mostly fast with one spike; the (discarded) transition
        // sample plus four 80s and one huge spike keep the median at 80.
        let mut pool_samples = [80u64, 80, 80, 80, 80, 100_000].into_iter();
        while calib.phase == CalibPhase::ProbePool {
            calib.record(AutoPath::Pool, pool_samples.next().unwrap_or(80));
        }
        assert_eq!(calib.committed, AutoPath::Pool);
    }

    #[test]
    fn median_ns_picks_the_middle_and_guards_empty() {
        assert_eq!(median_ns(&mut []), u64::MAX);
        assert_eq!(median_ns(&mut [5]), 5);
        assert_eq!(median_ns(&mut [30, 10, 20]), 20);
        assert_eq!(median_ns(&mut [10, 40, 20, 30]), 30);
    }
}
