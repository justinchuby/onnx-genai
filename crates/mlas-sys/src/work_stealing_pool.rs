//! Persistent, low-overhead work-stealing pool for decode-sized parallel-for.
//!
//! This is the Rust-side prototype of the ORT/Eigen design: workers are created
//! once, stay in a spin-then-park loop between dispatches, and each parallel-for
//! publishes ORT-style `LoopCounter` shards that workers dynamically claim from.
//! The API accepts stack-borrowing closures because `parallel_for` does not
//! return until every worker has left the closure.

use std::cell::UnsafeCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

type JobFn = unsafe fn(*const (), usize, usize);

const SPIN_LOOP_BUDGET: usize = 1 << 12;
const YIELD_ROUNDS: usize = 64;
const PARK_TIMEOUT: Duration = Duration::from_micros(50);
const MAX_LOOP_COUNTER_SHARDS: usize = 8;

#[repr(align(128))]
struct PaddedAtomicUsize(AtomicUsize);

impl PaddedAtomicUsize {
    fn new(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }

    fn load(&self, ordering: Ordering) -> usize {
        self.0.load(ordering)
    }

    fn store(&self, value: usize, ordering: Ordering) {
        self.0.store(value, ordering);
    }

    fn fetch_add(&self, value: usize, ordering: Ordering) -> usize {
        self.0.fetch_add(value, ordering)
    }
}

struct LoopCounterShard {
    next: PaddedAtomicUsize,
    end: PaddedAtomicUsize,
}

impl LoopCounterShard {
    fn new() -> Self {
        Self {
            next: PaddedAtomicUsize::new(0),
            end: PaddedAtomicUsize::new(0),
        }
    }
}

#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: Option<JobFn>,
    begin: usize,
    end: usize,
    grain: usize,
    num_shards: usize,
    work_items: usize,
    claims: usize,
}

impl Job {
    const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            call: None,
            begin: 0,
            end: 0,
            grain: 1,
            num_shards: 1,
            work_items: 0,
            claims: 0,
        }
    }
}

struct Shared {
    epoch: AtomicUsize,
    ready: AtomicUsize,
    remaining: AtomicUsize,
    active: AtomicUsize,
    shutdown: AtomicBool,
    panicked: AtomicBool,
    shards: Vec<LoopCounterShard>,
    thread_count: usize,
    job: UnsafeCell<Job>,
}

unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// Persistent, spin-then-park, per-worker-queue parallel-for pool.
///
/// The pool is intentionally small and synchronous: at most one `parallel_for`
/// may run at a time, and the caller blocks until all worker threads complete.
/// Work is published exactly like ORT's `ParallelForFixedBlockSizeScheduling`:
/// at most one work item per pool lane, up to eight `LoopCounter` shards, and a
/// fixed block size (`grain`). Each work item starts at its home shard and
/// atomically claims the next block, then scans the remaining shards. A delayed
/// worker therefore cannot strand a static range behind a barrier.
pub struct WorkStealingThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    worker_threads: Vec<Thread>,
    dispatch_lock: Mutex<()>,
}

impl WorkStealingThreadPool {
    /// Create a pool with `thread_count` total threads, including the caller.
    pub fn new(thread_count: usize) -> std::io::Result<Self> {
        assert!(thread_count > 0, "thread_count must be non-zero");
        let worker_count = thread_count.saturating_sub(1);

        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            shards: (0..MAX_LOOP_COUNTER_SHARDS)
                .map(|_| LoopCounterShard::new())
                .collect(),
            thread_count,
            job: UnsafeCell::new(Job::empty()),
        });

        let mut workers = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let shared = Arc::clone(&shared);
            let queue_id = worker_id + 1;
            workers.push(
                thread::Builder::new()
                    .name(format!("mlas-sys-ws-{worker_id}"))
                    .spawn(move || worker_loop(shared, queue_id))?,
            );
        }

        let worker_threads = workers
            .iter()
            .map(|worker| worker.thread().clone())
            .collect();
        while shared.ready.load(Ordering::Acquire) != worker_count {
            std::hint::spin_loop();
        }
        Ok(Self {
            shared,
            workers,
            worker_threads,
            dispatch_lock: Mutex::new(()),
        })
    }

    /// Degree of parallelism, including the caller thread.
    pub fn thread_count(&self) -> usize {
        self.shared.thread_count
    }

    /// Run `body(start, end)` over `[begin, end)` using chunks of at least `grain`.
    ///
    /// A single-chunk range runs inline, matching ORT's trivial-loop fast path.
    pub fn parallel_for<F>(&self, begin: usize, end: usize, grain: usize, body: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        assert!(begin <= end, "parallel_for begin must be <= end");
        assert!(grain > 0, "parallel_for grain must be non-zero");
        let len = end - begin;
        if len == 0 {
            return;
        }
        if len <= grain {
            body(begin, end);
            return;
        }

        let num_blocks = len / grain;
        let claims = len.div_ceil(grain);
        let num_shards = loop_counter_shards(len, self.thread_count(), grain);
        let work_items = self.thread_count().min(num_blocks.max(1));
        let _dispatch_guard = self.dispatch_lock.lock().unwrap();
        self.shared.panicked.store(false, Ordering::Relaxed);

        let blocks_per_shard = num_blocks / num_shards;
        let iterations_per_shard = blocks_per_shard * grain;
        for (shard_idx, shard) in self.shared.shards.iter().enumerate() {
            if shard_idx < num_shards {
                let start = shard_idx * iterations_per_shard;
                let end = if shard_idx + 1 == num_shards {
                    len
                } else {
                    (shard_idx + 1) * iterations_per_shard
                };
                shard.next.store(start, Ordering::Relaxed);
                shard.end.store(end, Ordering::Relaxed);
            } else {
                shard.next.store(0, Ordering::Relaxed);
                shard.end.store(0, Ordering::Relaxed);
            }
        }

        unsafe fn call<F>(data: *const (), begin: usize, end: usize)
        where
            F: Fn(usize, usize) + Sync,
        {
            let body = unsafe { &*(data as *const F) };
            body(begin, end);
        }

        unsafe {
            *self.shared.job.get() = Job {
                data: &body as *const F as *const (),
                call: Some(call::<F>),
                begin,
                end,
                grain,
                num_shards,
                work_items,
                claims,
            };
        }

        self.shared.remaining.store(claims, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in self
            .worker_threads
            .iter()
            .take(work_items.saturating_sub(1))
        {
            worker.unpark();
        }

        let caller_result = panic::catch_unwind(AssertUnwindSafe(|| run_job(&self.shared, 0)));
        if caller_result.is_err() {
            self.shared.panicked.store(true, Ordering::Release);
        }
        wait_for_completion(&self.shared);
        if self.shared.panicked.load(Ordering::Acquire) {
            wait_for_active(&self.shared);
        }

        unsafe {
            *self.shared.job.get() = Job::empty();
        }

        if caller_result.is_err() || self.shared.panicked.load(Ordering::Acquire) {
            panic!("WorkStealingThreadPool worker panicked");
        }
    }
}

impl Drop for WorkStealingThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in &self.worker_threads {
            worker.unpark();
        }
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize) {
    let mut seen_epoch = shared.epoch.load(Ordering::Acquire);
    shared.ready.fetch_add(1, Ordering::Release);
    loop {
        let epoch = wait_for_epoch(&shared, seen_epoch);
        seen_epoch = epoch;
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        let result = panic::catch_unwind(AssertUnwindSafe(|| run_job(&shared, worker_id)));
        if result.is_err() {
            shared.panicked.store(true, Ordering::Release);
        }
    }
}

fn wait_for_epoch(shared: &Shared, seen_epoch: usize) -> usize {
    let mut rounds = 0;
    loop {
        for _ in 0..SPIN_LOOP_BUDGET {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen_epoch {
                return epoch;
            }
            std::hint::spin_loop();
        }
        if rounds < YIELD_ROUNDS {
            rounds += 1;
            thread::yield_now();
        } else {
            thread::park_timeout(PARK_TIMEOUT);
        }
    }
}

fn wait_for_completion(shared: &Shared) {
    let mut spins = 0usize;
    while shared.remaining.load(Ordering::Acquire) != 0 && !shared.panicked.load(Ordering::Acquire)
    {
        if spins < SPIN_LOOP_BUDGET {
            spins += 1;
            std::hint::spin_loop();
        } else {
            thread::yield_now();
        }
    }
}

fn wait_for_active(shared: &Shared) {
    while shared.active.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }
}

fn run_job(shared: &Shared, worker_id: usize) {
    let job = unsafe { *shared.job.get() };
    let Some(call) = job.call else {
        return;
    };
    if worker_id >= job.work_items {
        return;
    }

    let home_shard = worker_id % job.num_shards;
    let mut shard = home_shard;
    while !shared.panicked.load(Ordering::Acquire) {
        let Some((iter_begin, iter_end)) =
            claim_iterations(shared, home_shard, &mut shard, job.grain, job.num_shards)
        else {
            break;
        };
        debug_assert!(iter_begin < job.end - job.begin);
        debug_assert!(iter_begin / job.grain <= job.claims);
        let begin = job.begin + iter_begin;
        let end = job.begin + iter_end.min(job.end - job.begin);
        shared.active.fetch_add(1, Ordering::AcqRel);
        let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            call(job.data, begin, end);
        }));
        shared.active.fetch_sub(1, Ordering::AcqRel);
        shared.remaining.fetch_sub(1, Ordering::Release);
        if result.is_err() {
            shared.panicked.store(true, Ordering::Release);
            break;
        }
    }
}

fn claim_iterations(
    shared: &Shared,
    home_shard: usize,
    shard: &mut usize,
    block_size: usize,
    num_shards: usize,
) -> Option<(usize, usize)> {
    loop {
        let counter = &shared.shards[*shard];
        let end = counter.end.load(Ordering::Relaxed);
        if counter.next.load(Ordering::Relaxed) < end {
            let start = counter.next.fetch_add(block_size, Ordering::AcqRel);
            if start < end {
                return Some((start, (start + block_size).min(end)));
            }
        }

        *shard = (*shard + 1) % num_shards;
        if *shard == home_shard {
            return None;
        }
    }
}

fn loop_counter_shards(
    num_iterations: usize,
    degree_of_parallelism: usize,
    block_size: usize,
) -> usize {
    let num_blocks = num_iterations / block_size;
    let mut num_shards = if num_blocks == 0 {
        1
    } else {
        num_blocks.min(MAX_LOOP_COUNTER_SHARDS)
    };
    num_shards = num_shards.min(degree_of_parallelism.max(1));
    num_shards.max(1)
}

#[cfg(test)]
mod tests {
    use super::WorkStealingThreadPool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_visits_every_item_once() {
        let pool = WorkStealingThreadPool::new(4).unwrap();
        let hits = (0..257).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
        pool.parallel_for(0, hits.len(), 7, |begin, end| {
            for hit in &hits[begin..end] {
                hit.fetch_add(1, Ordering::Relaxed);
            }
        });

        for hit in hits {
            assert_eq!(hit.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn parallel_for_accepts_stack_borrows() {
        let pool = WorkStealingThreadPool::new(3).unwrap();
        let input = (0..128).collect::<Vec<usize>>();
        let sum = AtomicUsize::new(0);

        pool.parallel_for(0, input.len(), 5, |begin, end| {
            let local = input[begin..end].iter().sum::<usize>();
            sum.fetch_add(local, Ordering::Relaxed);
        });

        assert_eq!(sum.load(Ordering::Relaxed), input.iter().sum::<usize>());
    }

    #[test]
    fn parallel_for_repeated_dispatches_do_not_miss_epochs() {
        let pool = WorkStealingThreadPool::new(4).unwrap();
        let calls = AtomicUsize::new(0);
        for _ in 0..1000 {
            pool.parallel_for(0, 64, 8, |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
        }

        assert_eq!(calls.load(Ordering::Relaxed), 1000 * 8);
    }
}
