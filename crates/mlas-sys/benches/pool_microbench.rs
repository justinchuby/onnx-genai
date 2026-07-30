use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use mlas_sys::WorkStealingThreadPool;
use rayon::prelude::*;

type JobFn = unsafe fn(*const (), usize, usize);

const DISPATCH_ITERS: usize = 10_000;
const THROUGHPUT_ITERS: usize = 1_000;
const ITEMS: usize = 8_192;
const GRAIN: usize = 32;

#[derive(Default)]
struct Stats {
    mean_us: f64,
    p50_us: f64,
    p90_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn main() {
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .clamp(2, 8);
    println!("pool_microbench: threads={threads}, items={ITEMS}, grain={GRAIN}");

    let eigen = WorkStealingThreadPool::new(threads).expect("create work-stealing pool");
    let fixed = FixedSpmdPool::new(threads).expect("create fixed-spmd pool");
    let rayon = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("mlas-sys-rayon-bench-{i}"))
        .build()
        .expect("create rayon pool");

    let dispatch_items = threads * GRAIN;
    report(
        "dispatch/eigen-style",
        measure(DISPATCH_ITERS, |iter| {
            eigen.parallel_for(0, dispatch_items, GRAIN, |begin, end| {
                std::hint::black_box(((begin as u64) << 32) ^ end as u64 ^ iter as u64);
            });
        }),
    );
    report(
        "dispatch/fixed-spmd",
        measure(DISPATCH_ITERS, |iter| {
            fixed.parallel_for(0, dispatch_items, |begin, end| {
                std::hint::black_box(((begin as u64) << 32) ^ end as u64 ^ iter as u64);
            });
        }),
    );
    report(
        "dispatch/rayon",
        measure(DISPATCH_ITERS, |iter| {
            rayon.install(|| {
                (0..threads).into_par_iter().for_each(|chunk| {
                    let begin = chunk * GRAIN;
                    let end = begin + GRAIN;
                    std::hint::black_box(((begin as u64) << 32) ^ end as u64 ^ iter as u64);
                });
            });
        }),
    );

    let checksum = AtomicU64::new(0);
    report(
        "contention/eigen-style",
        measure(THROUGHPUT_ITERS, |iter| {
            eigen.parallel_for(0, ITEMS, GRAIN, |begin, end| {
                simulated_contention(iter, begin);
                burn(&checksum, begin, end);
            });
        }),
    );
    report(
        "contention/fixed-spmd",
        measure(THROUGHPUT_ITERS, |iter| {
            fixed.parallel_for(0, ITEMS, |begin, end| {
                simulated_contention(iter, begin);
                burn(&checksum, begin, end);
            });
        }),
    );
    report(
        "contention/rayon",
        measure(THROUGHPUT_ITERS, |iter| {
            let chunks = ITEMS.div_ceil(GRAIN);
            rayon.install(|| {
                (0..chunks).into_par_iter().for_each(|chunk| {
                    let begin = chunk * GRAIN;
                    let end = ITEMS.min(begin + GRAIN);
                    simulated_contention(iter, begin);
                    burn(&checksum, begin, end);
                });
            });
        }),
    );

    std::hint::black_box(checksum.load(Ordering::Relaxed));
}

fn simulated_contention(iter: usize, begin: usize) {
    if begin == 0 && iter % 4 == 0 {
        thread::sleep(Duration::from_micros(100));
    }
}

fn burn(checksum: &AtomicU64, begin: usize, end: usize) {
    let mut local = 0u64;
    for i in begin..end {
        local = local.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }
    checksum.fetch_add(local, Ordering::Relaxed);
}

fn measure(mut iters: usize, mut f: impl FnMut(usize)) -> Stats {
    if let Ok(raw) = std::env::var("ONNX_GENAI_FAST_BENCH") {
        let cap = raw.parse::<usize>().unwrap_or(500);
        iters = iters.min(cap);
    }

    let mut samples = Vec::with_capacity(iters);
    for iter in 0..iters {
        let start = Instant::now();
        f(iter);
        samples.push(start.elapsed().as_nanos() as u64);
    }
    summarize(&mut samples)
}

fn summarize(samples: &mut [u64]) -> Stats {
    samples.sort_unstable();
    let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
    Stats {
        mean_us: mean / 1_000.0,
        p50_us: percentile(samples, 50) as f64 / 1_000.0,
        p90_us: percentile(samples, 90) as f64 / 1_000.0,
        p99_us: percentile(samples, 99) as f64 / 1_000.0,
        max_us: samples[samples.len() - 1] as f64 / 1_000.0,
    }
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    let idx = ((sorted.len() - 1) * pct) / 100;
    sorted[idx]
}

fn report(name: &str, stats: Stats) {
    println!(
        "{name:25} mean={:8.3}us p50={:8.3}us p90={:8.3}us p99={:8.3}us max={:8.3}us",
        stats.mean_us, stats.p50_us, stats.p90_us, stats.p99_us, stats.max_us
    );
}

#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: Option<JobFn>,
    begin: usize,
    end: usize,
}

impl Job {
    const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            call: None,
            begin: 0,
            end: 0,
        }
    }
}

struct FixedShared {
    epoch: AtomicUsize,
    ready: AtomicUsize,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    job: UnsafeCell<Job>,
}

unsafe impl Send for FixedShared {}
unsafe impl Sync for FixedShared {}

struct FixedSpmdPool {
    shared: Arc<FixedShared>,
    workers: Vec<JoinHandle<()>>,
    worker_threads: Vec<Thread>,
    dispatch_lock: Mutex<()>,
}

impl FixedSpmdPool {
    fn new(thread_count: usize) -> std::io::Result<Self> {
        let shared = Arc::new(FixedShared {
            epoch: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            job: UnsafeCell::new(Job::empty()),
        });
        let mut workers = Vec::with_capacity(thread_count);
        for worker_id in 0..thread_count {
            let shared = Arc::clone(&shared);
            workers.push(
                thread::Builder::new()
                    .name(format!("mlas-sys-fixed-bench-{worker_id}"))
                    .spawn(move || fixed_worker_loop(shared, worker_id, thread_count))?,
            );
        }
        let worker_threads = workers
            .iter()
            .map(|worker| worker.thread().clone())
            .collect();
        while shared.ready.load(Ordering::Acquire) != thread_count {
            std::hint::spin_loop();
        }
        Ok(Self {
            shared,
            workers,
            worker_threads,
            dispatch_lock: Mutex::new(()),
        })
    }

    fn parallel_for<F>(&self, begin: usize, end: usize, body: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let _dispatch_guard = self.dispatch_lock.lock().unwrap();
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
            };
        }

        self.shared
            .pending
            .store(self.workers.len(), Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for worker in &self.worker_threads {
            worker.unpark();
        }
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        unsafe {
            *self.shared.job.get() = Job::empty();
        }
    }
}

impl Drop for FixedSpmdPool {
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

fn fixed_worker_loop(shared: Arc<FixedShared>, worker_id: usize, workers: usize) {
    let mut seen_epoch = shared.epoch.load(Ordering::Acquire);
    shared.ready.fetch_add(1, Ordering::Release);
    loop {
        loop {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen_epoch {
                seen_epoch = epoch;
                break;
            }
            for _ in 0..4096 {
                std::hint::spin_loop();
            }
            thread::park_timeout(Duration::from_micros(50));
        }
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        let job = unsafe { *shared.job.get() };
        if let Some(call) = job.call {
            let len = job.end - job.begin;
            let begin = job.begin + worker_id * len / workers;
            let end = job.begin + (worker_id + 1) * len / workers;
            unsafe {
                call(job.data, begin, end);
            }
        }
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}
