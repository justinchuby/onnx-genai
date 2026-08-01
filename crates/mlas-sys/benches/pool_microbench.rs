use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use mlas_sys::{
    mlas_threading_degree, mlas_threading_stats, reset_mlas_threading_stats,
    sqnbit_gemm_into_with_workspace, sqnbit_gemm_with_workspace, sqnbit_mlas_partitioning,
    SQNBitComputeType, SQNBitGemmWorkspace, SQNBitPackedB, WorkStealingThreadPool,
};
use rayon::prelude::*;

type JobFn = unsafe fn(*const (), usize, usize);

const DISPATCH_ITERS: usize = 10_000;
const THROUGHPUT_ITERS: usize = 1_000;
const QNBIT_ITERS: usize = 200;
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

    qnbit_internal_pool_bench();

    std::hint::black_box(checksum.load(Ordering::Relaxed));
}

fn qnbit_internal_pool_bench() {
    let (m, n, k, block_size) = (1usize, 3072usize, 1024usize, 128usize);
    let weights: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 * 0.017 + 0.11).sin()) * 1.3)
        .collect();
    let (packed_b, scales, zps) = quantize_int4(&weights, n, k, block_size, true);
    let Some((comp, packed)) = [SQNBitComputeType::Int8, SQNBitComputeType::Fp32]
        .into_iter()
        .find_map(|comp| {
            SQNBitPackedB::new(
                n,
                k,
                4,
                block_size,
                comp,
                &packed_b,
                &scales,
                zps.as_deref(),
            )
            .map(|packed| (comp, packed))
        })
    else {
        println!("mlas-qnbit/internal     skipped: SQNBit unavailable on this host");
        return;
    };
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.013 + 0.29).cos()) * 0.9)
        .collect();
    let mut c = vec![0.0f32; m * n];
    let mut workspace = SQNBitGemmWorkspace::new();
    let partition = sqnbit_mlas_partitioning(m, n, k, 1, mlas_threading_degree());
    println!(
        "mlas-qnbit/partition   M={m} N={n} K={k} blk={block_size} comp={comp:?} \
         stride_n={} tiles={} claimants={} shards={}",
        partition.stride_n,
        partition.work_items,
        partition.ort_claimants,
        partition.ort_loop_counter_shards
    );

    for _ in 0..5 {
        sqnbit_gemm_with_workspace(&packed, m, &a, None, &mut c, &mut workspace, true);
    }
    reset_mlas_threading_stats();
    report(
        "mlas-qnbit/internal",
        measure(QNBIT_ITERS, |_| {
            sqnbit_gemm_with_workspace(&packed, m, &a, None, &mut c, &mut workspace, true);
        }),
    );
    let stats = mlas_threading_stats();
    println!(
        "mlas-qnbit/stats        comp={comp:?} calls={} iterations={} dynamic_blocks={} pool_threads={} fallbacks={}",
        stats.parallel_for_calls,
        stats.scheduled_iterations,
        stats.dynamic_blocks_claimed,
        stats.pool_threads,
        stats.serial_fallback_calls
    );

    let static_bench = StaticQnbitBench::new(StaticQnbitInputs {
        threads: threads_for_static(),
        n,
        k,
        block_size,
        packed_b: &packed_b,
        scales: &scales,
        zps: zps.as_deref(),
        comp,
    });
    if let Some(static_bench) = static_bench {
        let mut c_static = vec![0.0f32; m * n];
        for _ in 0..5 {
            static_bench.run(m, &a, &mut c_static);
        }
        report(
            "mlas-qnbit/static-split",
            measure(QNBIT_ITERS, |_| {
                static_bench.run(m, &a, &mut c_static);
            }),
        );
        let balance = static_bench.measure_balance(m, &a, &mut c_static);
        println!("mlas-qnbit/static-balance ns/thread={balance:?}");
    }
}

fn threads_for_static() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .clamp(2, 8)
}

struct StaticQnbitShard {
    start_n: usize,
    packed: SQNBitPackedB,
    workspace: UnsafeCell<SQNBitGemmWorkspace>,
}

unsafe impl Sync for StaticQnbitShard {}

struct StaticQnbitBench {
    pool: FixedSpmdPool,
    shards: Vec<StaticQnbitShard>,
    ldc: usize,
}

struct StaticQnbitInputs<'a> {
    threads: usize,
    n: usize,
    k: usize,
    block_size: usize,
    packed_b: &'a [u8],
    scales: &'a [f32],
    zps: Option<&'a [u8]>,
    comp: SQNBitComputeType,
}

impl StaticQnbitBench {
    fn new(inputs: StaticQnbitInputs<'_>) -> Option<Self> {
        let StaticQnbitInputs {
            threads,
            n,
            k,
            block_size,
            packed_b,
            scales,
            zps,
            comp,
        } = inputs;
        let blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let zp_row = blocks.div_ceil(2);
        let ranges = aligned_ranges(n, threads, 16);
        let mut shards = Vec::with_capacity(ranges.len());
        for (start, len) in ranges {
            let pb = &packed_b[start * blocks * blob..(start + len) * blocks * blob];
            let sc = &scales[start * blocks..(start + len) * blocks];
            let zp = zps.map(|z| &z[start * zp_row..(start + len) * zp_row]);
            let packed = SQNBitPackedB::new(len, k, 4, block_size, comp, pb, sc, zp)?;
            shards.push(StaticQnbitShard {
                start_n: start,
                packed,
                workspace: UnsafeCell::new(SQNBitGemmWorkspace::new()),
            });
        }
        Some(Self {
            pool: FixedSpmdPool::new(shards.len()).ok()?,
            shards,
            ldc: n,
        })
    }

    fn run(&self, m: usize, a: &[f32], c: &mut [f32]) {
        let c_addr = c.as_mut_ptr() as usize;
        self.pool.parallel_for(0, self.shards.len(), |begin, end| {
            for shard_id in begin..end {
                let shard = &self.shards[shard_id];
                unsafe {
                    sqnbit_gemm_into_with_workspace(
                        &shard.packed,
                        m,
                        a,
                        None,
                        (c_addr as *mut f32).add(shard.start_n),
                        self.ldc,
                        &mut *shard.workspace.get(),
                        false,
                    );
                }
            }
        });
    }

    fn measure_balance(&self, m: usize, a: &[f32], c: &mut [f32]) -> Vec<u64> {
        let c_addr = c.as_mut_ptr() as usize;
        let timings = (0..self.shards.len())
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>();
        self.pool.parallel_for(0, self.shards.len(), |begin, end| {
            for (shard_id, timing) in timings.iter().enumerate().take(end).skip(begin) {
                let shard = &self.shards[shard_id];
                let start = Instant::now();
                unsafe {
                    sqnbit_gemm_into_with_workspace(
                        &shard.packed,
                        m,
                        a,
                        None,
                        (c_addr as *mut f32).add(shard.start_n),
                        self.ldc,
                        &mut *shard.workspace.get(),
                        false,
                    );
                }
                timing.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        });
        timings
            .iter()
            .map(|timing| timing.load(Ordering::Relaxed))
            .collect()
    }
}

fn aligned_ranges(n: usize, parts: usize, align: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for part in 0..parts {
        let mut end = if part + 1 == parts {
            n
        } else {
            ((part + 1) * n / parts).div_ceil(align) * align
        };
        end = end.clamp(start, n);
        if end > start {
            ranges.push((start, end - start));
        }
        start = end;
    }
    ranges
}

fn simulated_contention(iter: usize, begin: usize) {
    if begin == 0 && iter.is_multiple_of(4) {
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

fn quantize_int4(
    weights_nk: &[f32],
    n: usize,
    k: usize,
    block_size: usize,
    asymmetric: bool,
) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>) {
    let blocks = k.div_ceil(block_size);
    let blob = block_size / 2;
    let zp_row = blocks.div_ceil(2);
    let mut packed = vec![0u8; n * blocks * blob];
    let mut scales = vec![0.0f32; n * blocks];
    let mut zps = vec![0u8; n * zp_row];
    for row in 0..n {
        for block in 0..blocks {
            let start = block * block_size;
            let end = (start + block_size).min(k);
            let values = &weights_nk[row * k + start..row * k + end];
            let (scale, zp) = if asymmetric {
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = ((max - min) / 15.0).max(1e-6);
                (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
            } else {
                let max_abs = values.iter().map(|v| v.abs()).fold(0.0, f32::max);
                ((max_abs / 7.0).max(1e-6), 8u8)
            };
            scales[row * blocks + block] = scale;
            if asymmetric {
                zps[row * zp_row + block / 2] |= zp << (4 * (block % 2));
            }
            for (offset, &value) in values.iter().enumerate() {
                let q = (value / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                packed[(row * blocks + block) * blob + offset / 2] |= q << (4 * (offset % 2));
            }
        }
    }
    (packed, scales, asymmetric.then_some(zps))
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
