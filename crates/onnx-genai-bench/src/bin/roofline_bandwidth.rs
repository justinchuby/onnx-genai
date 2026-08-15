//! STREAM-like read bandwidth probe for decode roofline estimates.

use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Measure sustained DRAM read bandwidth with a large read-sum sweep")]
struct Args {
    /// Comma-separated thread counts to test.
    #[arg(long, default_value = "1,2,4,6,8,12")]
    threads: String,
    /// Total buffer size in MiB. Keep this far above LLC size.
    #[arg(long, default_value_t = 1024)]
    mib: usize,
    /// Minimum measurement duration per thread count.
    #[arg(long, default_value_t = 2.0)]
    seconds: f64,
    /// Warmup sweeps before timing.
    #[arg(long, default_value_t = 1)]
    warmups: usize,
}

#[derive(Debug)]
struct Measurement {
    threads: usize,
    bytes: u128,
    elapsed: Duration,
    checksum: u64,
}

fn parse_threads(value: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let threads = part
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse thread count {part:?}"))?;
        if threads == 0 {
            bail!("thread counts must be positive");
        }
        out.push(threads);
    }
    Ok(out)
}

fn sweep(data: Arc<[u64]>, threads: usize, min_duration: Duration, warmups: usize) -> Measurement {
    let chunk_len = data.len().div_ceil(threads);
    let run_once = |timed: bool| {
        let start = Instant::now();
        let mut bytes = 0u128;
        let mut checksum = 0u64;
        thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for thread_index in 0..threads {
                let data = Arc::clone(&data);
                handles.push(scope.spawn(move || {
                    let start_index = thread_index * chunk_len;
                    let end_index = ((thread_index + 1) * chunk_len).min(data.len());
                    let slice = &data[start_index..end_index];
                    let mut local = 0u64;
                    let mut local_bytes = 0u128;
                    loop {
                        for &value in slice {
                            local = local.wrapping_add(value);
                        }
                        local_bytes += std::mem::size_of_val(slice) as u128;
                        if !timed || start.elapsed() >= min_duration {
                            break;
                        }
                    }
                    (local_bytes, local)
                }));
            }
            for handle in handles {
                let (local_bytes, local) = handle.join().expect("bandwidth worker panicked");
                bytes += local_bytes;
                checksum ^= local;
            }
        });
        (bytes, checksum, start.elapsed())
    };

    for _ in 0..warmups {
        black_box(run_once(false));
    }
    let (bytes, checksum, elapsed) = run_once(true);
    Measurement {
        threads,
        bytes,
        elapsed,
        checksum,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let thread_counts = parse_threads(&args.threads)?;
    let elements = args
        .mib
        .checked_mul(1024 * 1024 / std::mem::size_of::<u64>())
        .context("buffer size overflow")?;
    let data: Vec<u64> = (0..elements)
        .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let data: Arc<[u64]> = data.into();
    let min_duration = Duration::from_secs_f64(args.seconds);

    println!(
        "roofline_bandwidth: buffer={} MiB elements={} seconds={:.3} warmups={}",
        args.mib,
        data.len(),
        args.seconds,
        args.warmups
    );
    println!("threads,gb_s,elapsed_s,bytes_read,checksum");
    for threads in thread_counts {
        let measurement = sweep(Arc::clone(&data), threads, min_duration, args.warmups);
        let gb_s = measurement.bytes as f64 / measurement.elapsed.as_secs_f64() / 1.0e9;
        println!(
            "{},{:.3},{:.6},{},{}",
            measurement.threads,
            gb_s,
            measurement.elapsed.as_secs_f64(),
            measurement.bytes,
            measurement.checksum
        );
    }
    Ok(())
}
