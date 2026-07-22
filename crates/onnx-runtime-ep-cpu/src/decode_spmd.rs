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
//! of atomics per projection.
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
//! lightweight one, and stays correct. The whole layout is opt-in behind
//! `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1` (rule 5); when unset the decode path
//! is byte-for-byte unchanged.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use crate::decode_affinity::{NodeShard, NumaTopology};
use crate::kernels::matmul_nbits::output_chunk_len;

/// Environment switch that opts a run into the persistent SPMD decode pool.
/// Default OFF: only the exact value `1` enables it (rule 5).
pub const PERSISTENT_POOL_ENV: &str = "ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL";

/// Spin iterations a worker or the dispatcher busy-waits before yielding /
/// parking. Sized so back-to-back decode projections (microseconds apart) are
/// always caught while spinning; only genuinely idle gaps fall through to park.
const SPIN_BEFORE_YIELD: u32 = 1 << 12;
const YIELD_BEFORE_PARK: u32 = 1 << 6;
/// Bounded park so a (rare, off-hot-path) lost wakeup self-heals within 1 ms
/// rather than hanging.
const PARK_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1);

/// Cache-line pad so per-node completion counters and per-worker park flags do
/// not false-share (which would reintroduce cross-socket coherency traffic).
#[repr(align(128))]
struct Padded<T>(T);

/// A type-erased decode job: run the shard for the given global worker index.
/// The pointer is only dereferenced by workers between a [`SharedState::publish`]
/// and the matching dispatcher wait, both of which live entirely inside one
/// blocking `dispatch` call, so the borrow it erases always outlives its use.
type JobPtr = *const (dyn Fn(usize) + Sync);

/// State shared between the dispatcher (the engine thread running the forward)
/// and the persistent worker threads.
struct SharedState {
    /// Bumped once per dispatched op; workers wait for it to change.
    sequence: Padded<AtomicUsize>,
    /// The current op, published before `sequence` bumps and read after the
    /// bump is observed (release/acquire pairing on `sequence`).
    job: UnsafeCell<Option<JobPtr>>,
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
    fn publish(&self, job: JobPtr, counts: &[usize], handles: &[JoinHandle<()>]) {
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
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(total_workers);
        for (global_index, (node_position, cpu)) in assignment.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("onnx-genai-spmd-n{node_position}-{global_index}"))
                .spawn(move || {
                    if let Some(cpu) = cpu
                        && let Err(message) =
                            crate::decode_affinity::pin_current_thread_to_cpu(cpu)
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
    fn dispatch(&self, job: &(dyn Fn(usize) + Sync)) {
        // SAFETY: erase the borrow's lifetime to store the trait-object pointer.
        // `dispatch` blocks in `wait()` until every worker has finished reading
        // and running `job`, so the pointee outlives all worker accesses; the
        // `Sync` bound makes the shared &-access sound.
        let job_ptr = job as *const (dyn Fn(usize) + Sync);
        let job_ptr: JobPtr = unsafe { std::mem::transmute::<_, JobPtr>(job_ptr) };
        self.shared
            .publish(job_ptr, &self.node_worker_counts, &self.handles);
        self.shared.wait();
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

/// Source/destination view for node-local weight placement.
struct CopyTable<'a, T> {
    dst: *mut T,
    src: *const T,
    stride: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each worker copies only its own disjoint row range.
unsafe impl<T: Send + Sync> Sync for CopyTable<'_, T> {}

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
                    // Park to stop burning a core while idle. Publish the park
                    // intent, then re-check the sequence (SeqCst) so a wakeup
                    // that raced the flag store is not lost; the bounded timeout
                    // is a final backstop.
                    shared.parked[global_index].0.store(true, Ordering::SeqCst);
                    if shared.sequence.0.load(Ordering::SeqCst) == local_seq
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
        let job: &(dyn Fn(usize) + Sync) = unsafe { &*job };
        job(global_index);
        shared.node_pending[node].0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The lazily built persistent SPMD layout, or `None` when the mode is not
/// requested (its default). Built once and reused for the whole process.
pub fn pools() -> Option<&'static SpmdDecodePools> {
    static POOLS: OnceLock<Option<SpmdDecodePools>> = OnceLock::new();
    POOLS.get_or_init(|| build_from_env(default_threads())).as_ref()
}

/// Resolve the configured decode thread count the same way the flat pool does,
/// so the persistent pool honors `ONNX_GENAI_CPU_DECODE_THREADS`.
fn default_threads() -> Option<usize> {
    crate::kernels::matmul_nbits::configured_decode_threads_public()
}

/// Build the persistent SPMD layout when `PERSISTENT_POOL_ENV=1` and a worker
/// count is available; otherwise `None` (the decode path stays unchanged).
///
/// Two or more usable NUMA nodes yield the two-level node-pinned layout; a
/// single-node host, a non-NUMA machine, or a platform without pinning yields a
/// single unpinned worker group (still the lightweight barrier, still correct).
pub fn build_from_env(threads: Option<usize>) -> Option<SpmdDecodePools> {
    if !enabled() {
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
    let shards = node_shards(total);
    Some(SpmdDecodePools::build(&shards))
}

/// True only when the opt-in switch is exactly `1`.
fn enabled() -> bool {
    std::env::var(PERSISTENT_POOL_ENV)
        .map(|value| value.trim() == "1")
        .unwrap_or(false)
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
}
