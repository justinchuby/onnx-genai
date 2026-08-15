//! Host<->device transfer-bandwidth probe for the placement cost model.
//!
//! This is the device-link companion to `roofline_bandwidth` (which measures
//! host DRAM). It measures the *machine-specific* rate that
//! `docs/architecture/ORT2.md` §6.2 `TransferProfile` needs — `latency_base`
//! and `bandwidth` — for the host<->device PCIe/NVLink link, so the cost model
//! never has to hardcode a link constant (issue #995).
//!
//! It reports the four regimes that actually differ on real hardware and that
//! the cost model must be able to tell apart:
//!   * **H2D vs D2H** — the two directions are asymmetric on most links.
//!   * **pageable vs pinned** host memory — the driver must bounce pageable
//!     memory through an internal pinned staging buffer, so pinned is usually
//!     much faster; the cost model has to know which one the runtime will use.
//!
//! Both regimes of the roofline matter, so it **sweeps transfer size**: small
//! transfers are latency-bound and large ones bandwidth-bound. It then fits
//! `time = latency_base + bytes / bandwidth` by least squares over the sweep,
//! which is exactly the two `TransferProfile` fields. Reporting a single size
//! would mislead — a small size understates bandwidth, a large one hides
//! latency.
//!
//! Timing uses CUDA events around a batch of async copies on a dedicated
//! stream, so the measured interval brackets the copy *completing* on the
//! device timeline, not merely an async enqueue returning (the
//! `h2d_enqueue_copy_ms` mistake called out in the measurement-discipline
//! skill). All measurements are repeated so a distribution can be reported
//! rather than a single sample.

use std::ffi::c_void;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Direction {
    /// Host to device (upload).
    H2d,
    /// Device to host (download).
    D2h,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Direction::H2d => "h2d",
            Direction::D2h => "d2h",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum HostKind {
    /// Ordinary pageable `Vec<u8>` host memory.
    Pageable,
    /// Page-locked (pinned) host memory via `cuMemHostAlloc`.
    Pinned,
}

impl HostKind {
    fn label(self) -> &'static str {
        match self {
            HostKind::Pageable => "pageable",
            HostKind::Pinned => "pinned",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    about = "Measure sustained host<->device bandwidth (H2D/D2H, pageable/pinned) \
                   and fit latency_base + bandwidth for the cost model's TransferProfile"
)]
struct Args {
    /// GPU ordinal to probe.
    #[arg(long, default_value_t = 0)]
    device: u32,
    /// Comma-separated transfer sizes in KiB to sweep. Small sizes expose
    /// launch latency; large sizes expose sustained bandwidth.
    #[arg(long, default_value = "4,16,64,256,1024,4096,16384,65536,262144")]
    sizes_kib: String,
    /// Copies timed per measurement (amortizes event overhead).
    #[arg(long, default_value_t = 30)]
    iters: usize,
    /// Warmup copies before timing (pays first-touch / JIT staging costs).
    #[arg(long, default_value_t = 5)]
    warmups: usize,
    /// Repeated measurements per point, so a distribution can be reported.
    #[arg(long, default_value_t = 7)]
    repeats: usize,
    /// Directions to measure.
    #[arg(long, value_delimiter = ',', default_values = ["h2d", "d2h"])]
    directions: Vec<Direction>,
    /// Host memory kinds to measure.
    #[arg(long, value_delimiter = ',', default_values = ["pageable", "pinned"])]
    host_kinds: Vec<HostKind>,
}

fn parse_sizes_kib(value: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        let kib = part
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse size {part:?}"))?;
        if kib == 0 {
            bail!("transfer sizes must be positive");
        }
        out.push(
            kib.checked_mul(1024)
                .with_context(|| format!("size {kib} KiB overflows"))?,
        );
    }
    if out.is_empty() {
        bail!("at least one size is required");
    }
    Ok(out)
}

/// An owned host buffer of the requested kind. Pinned buffers are freed on drop.
enum HostBuffer {
    Pageable(Vec<u8>),
    Pinned { ptr: *mut c_void, len: usize },
}

impl HostBuffer {
    fn alloc(kind: HostKind, bytes: usize) -> Result<Self> {
        match kind {
            HostKind::Pageable => Ok(HostBuffer::Pageable(vec![0u8; bytes])),
            HostKind::Pinned => {
                // SAFETY: `malloc_host` returns a fresh page-locked host
                // allocation on the bound context; freed once in `Drop`.
                let ptr =
                    unsafe { result::malloc_host(bytes.max(1), 0) }.context("cuMemHostAlloc")?;
                // Touch every page so the measured copy is not also paying a
                // first-touch fault.
                // SAFETY: `ptr` covers `bytes` writable bytes.
                unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, bytes) };
                Ok(HostBuffer::Pinned { ptr, len: bytes })
            }
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            HostBuffer::Pageable(v) => v,
            // SAFETY: `ptr` covers `len` initialized bytes for this buffer's life.
            HostBuffer::Pinned { ptr, len } => unsafe {
                std::slice::from_raw_parts(ptr.cast::<u8>(), *len)
            },
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            HostBuffer::Pageable(v) => v,
            // SAFETY: `ptr` covers `len` initialized bytes, uniquely borrowed here.
            HostBuffer::Pinned { ptr, len } => unsafe {
                std::slice::from_raw_parts_mut(ptr.cast::<u8>(), *len)
            },
        }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        if let HostBuffer::Pinned { ptr, .. } = self {
            // SAFETY: `ptr` came from `malloc_host` and is freed exactly once.
            let _ = unsafe { result::free_host(*ptr) };
        }
    }
}

/// One timed measurement: milliseconds for `iters` copies of `bytes` each.
fn time_copies(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    direction: Direction,
    host: &mut HostBuffer,
    device: cudarc::driver::sys::CUdeviceptr,
    bytes: usize,
    iters: usize,
) -> Result<f64> {
    let start = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .context("cuEventCreate(start)")?;
    let end = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .context("cuEventCreate(end)")?;
    let cu_stream = stream.cu_stream();
    start.record(stream).context("record start")?;
    for _ in 0..iters {
        match direction {
            Direction::H2d => {
                let src = &host.as_slice()[..bytes];
                // SAFETY: `device` covers `bytes`; `src` is `bytes` long.
                unsafe { result::memcpy_htod_async(device, src, cu_stream) }
                    .context("cuMemcpyHtoDAsync")?;
            }
            Direction::D2h => {
                let dst = &mut host.as_mut_slice()[..bytes];
                // SAFETY: `device` covers `bytes`; `dst` is `bytes` long.
                unsafe { result::memcpy_dtoh_async(dst, device, cu_stream) }
                    .context("cuMemcpyDtoHAsync")?;
            }
        }
    }
    end.record(stream).context("record end")?;
    end.synchronize().context("event synchronize")?;
    let ms = start.elapsed_ms(&end).context("cuEventElapsedTime")?;
    Ok(ms as f64)
}

#[derive(Debug, Clone)]
struct Point {
    bytes: usize,
    /// Per-copy time in seconds (median across repeats).
    time_s: f64,
    /// Sustained bandwidth (GB/s) at this size (median).
    gb_s: f64,
    /// Min/max bandwidth across repeats, for the distribution.
    gb_s_min: f64,
    gb_s_max: f64,
}

/// Fit `time = latency_base + bytes / bandwidth` over the sweep, respecting the
/// two regimes the roofline actually has.
///
/// A plain OLS over all sizes is dominated by the largest transfers (their
/// per-copy times are ~4 orders of magnitude larger than the small ones), which
/// drives the fitted intercept to ~0 and *hides the latency floor entirely* —
/// the exact "single number that misleads" failure the probe exists to avoid.
/// Instead we separate the regimes:
///   * **bandwidth** — OLS slope over the upper (bandwidth-bound) half of the
///     sweep, where per-byte cost dominates and the plateau is flat.
///   * **latency_base** — the median of `t - bytes / bandwidth` over the lower
///     (latency-bound) third, where fixed overhead dominates.
///
/// Returns `(latency_base_seconds, bandwidth_bytes_per_sec)`.
fn fit_latency_bandwidth(points: &[Point]) -> Option<(f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let mut sorted = points.to_vec();
    sorted.sort_by_key(|p| p.bytes);

    // Bandwidth from the upper half (bandwidth-bound regime).
    let upper_start = sorted.len() / 2;
    let upper = &sorted[upper_start..];
    let n = upper.len() as f64;
    let sx: f64 = upper.iter().map(|p| p.bytes as f64).sum();
    let sy: f64 = upper.iter().map(|p| p.time_s).sum();
    let sxx: f64 = upper.iter().map(|p| (p.bytes as f64).powi(2)).sum();
    let sxy: f64 = upper.iter().map(|p| p.bytes as f64 * p.time_s).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let slope = (n * sxy - sx * sy) / denom;
    if slope <= 0.0 {
        return None;
    }
    let bandwidth = 1.0 / slope;

    // Latency floor from the lower third (latency-bound regime): subtract the
    // now-known transfer term from each small-size time and take the median.
    let lower_end = (sorted.len() / 3).max(1);
    let mut residuals: Vec<f64> = sorted[..lower_end]
        .iter()
        .map(|p| p.time_s - p.bytes as f64 / bandwidth)
        .collect();
    let latency = median(&mut residuals).max(0.0);
    Some((latency, bandwidth))
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Query the GPU's current performance state (pstate) via `nvidia-smi`.
///
/// Returns the raw pstate string (e.g. `"P0"`, `"P8"`) if it could be read.
/// A **parked** laptop GPU sits in a deep pstate (`P8`) that downclocks the
/// PCIe link (Gen4→Gen1), collapsing measured host<->device bandwidth by ~7×
/// on the #995 box. The number measured in that state is *not* the rate that
/// applies during decode (when the GPU is active), so the operator must record
/// it as `MeasuredLinkState::Parked` in the cost model rather than trusting it.
fn query_pstate(device: u32) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=pstate",
            "--format=csv,noheader",
            "-i",
            &device.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Whether a pstate string denotes a parked / low-power state (`P8` or deeper),
/// where the PCIe link is downclocked and bandwidth is unrepresentative of
/// decode. `P0`–`P7` are treated as active-enough; anything at or past `P8` is
/// parked. An unparseable string is treated as not-parked (we do not know).
fn pstate_is_parked(pstate: &str) -> bool {
    pstate
        .strip_prefix('P')
        .or_else(|| pstate.strip_prefix('p'))
        .and_then(|n| n.trim().parse::<u32>().ok())
        .is_some_and(|n| n >= 8)
}

fn measure_regime(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    direction: Direction,
    kind: HostKind,
    sizes: &[usize],
    args: &Args,
) -> Result<Vec<Point>> {
    let max_bytes = *sizes.iter().max().unwrap();
    let mut host = HostBuffer::alloc(kind, max_bytes)?;
    // Fill the source with non-trivial data so nothing is optimized away.
    for (i, b) in host.as_mut_slice().iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }
    let device = unsafe { result::malloc_sync(max_bytes) }.context("cuMemAlloc")?;
    // Prime the device buffer so D2H has real bytes to read.
    // SAFETY: `device` covers `max_bytes`; `host` is `max_bytes` long.
    unsafe { result::memcpy_htod_sync(device, host.as_slice()) }.context("prime device")?;

    let mut points = Vec::with_capacity(sizes.len());
    for &bytes in sizes {
        // Warmup.
        for _ in 0..args.warmups {
            let _ = black_box(time_copies(
                ctx, stream, direction, &mut host, device, bytes, 1,
            )?);
        }
        let mut gbs_samples = Vec::with_capacity(args.repeats);
        let mut time_samples = Vec::with_capacity(args.repeats);
        for _ in 0..args.repeats {
            let ms = time_copies(ctx, stream, direction, &mut host, device, bytes, args.iters)?;
            let per_copy_s = (ms / 1000.0) / args.iters as f64;
            let gb_s = bytes as f64 / per_copy_s / 1.0e9;
            gbs_samples.push(gb_s);
            time_samples.push(per_copy_s);
        }
        let gb_s_min = gbs_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let gb_s_max = gbs_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let gb_s = median(&mut gbs_samples);
        let time_s = median(&mut time_samples);
        points.push(Point {
            bytes,
            time_s,
            gb_s,
            gb_s_min,
            gb_s_max,
        });
    }

    // SAFETY: `device` came from `malloc_sync` and is freed exactly once here.
    unsafe { result::free_sync(device) }.context("cuMemFree")?;
    Ok(points)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sizes = parse_sizes_kib(&args.sizes_kib)?;

    let wall = Instant::now();
    let ctx = CudaContext::new(args.device as usize)
        .with_context(|| format!("open CUDA device {}", args.device))?;
    let stream = ctx.default_stream();

    // Sample the pstate before the sweep so the operator knows whether these
    // rates are decode-representative or a parked-GPU under-estimate (§6.2
    // MeasuredLinkState). See `query_pstate` for why this matters.
    let pstate_before = query_pstate(args.device);
    let link_state = match pstate_before.as_deref() {
        Some(p) if pstate_is_parked(p) => "parked",
        Some(_) => "active",
        None => "unknown",
    };

    println!(
        "roofline_transfer: device={} iters={} warmups={} repeats={} sizes_kib=[{}]",
        args.device,
        args.iters,
        args.warmups,
        args.repeats,
        sizes
            .iter()
            .map(|b| (b / 1024).to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "roofline_transfer: pstate_at_start={} measured_link_state={} \
         (pass this to the cost model as MeasuredLinkState::{})",
        pstate_before.as_deref().unwrap_or("unknown"),
        link_state,
        match link_state {
            "parked" => "Parked",
            "active" => "Active",
            _ => "Unknown",
        }
    );
    if link_state == "parked" {
        eprintln!(
            "roofline_transfer: WARNING — GPU is parked ({}); the PCIe link is \
             downclocked and these bandwidths UNDER-STATE the decode-time link \
             (up to ~7x low on RTX 4060 Laptop). Do NOT record them as \
             decode-representative: run a GPU workload to wake the device first, \
             or record MeasuredLinkState::Parked.",
            pstate_before.as_deref().unwrap_or("P8")
        );
    }
    println!("direction,host,bytes,gb_s_median,gb_s_min,gb_s_max,per_copy_us");

    let mut fits: Vec<(Direction, HostKind, f64, f64, f64)> = Vec::new();
    for &direction in &args.directions {
        for &kind in &args.host_kinds {
            let points = measure_regime(&ctx, &stream, direction, kind, &sizes, &args)?;
            for p in &points {
                println!(
                    "{},{},{},{:.3},{:.3},{:.3},{:.3}",
                    direction.label(),
                    kind.label(),
                    p.bytes,
                    p.gb_s,
                    p.gb_s_min,
                    p.gb_s_max,
                    p.time_s * 1.0e6
                );
            }
            // Peak sustained is the largest-size (bandwidth-bound) median.
            let peak = points.last().map(|p| p.gb_s).unwrap_or(0.0);
            if let Some((latency_s, bandwidth_bps)) = fit_latency_bandwidth(&points) {
                fits.push((
                    direction,
                    kind,
                    latency_s * 1.0e6,
                    bandwidth_bps / 1.0e9,
                    peak,
                ));
            }
        }
    }

    println!();
    println!("# Fitted TransferProfile (time = latency_base + bytes / bandwidth)");
    println!("direction,host,latency_base_us,fit_bandwidth_gb_s,peak_sustained_gb_s");
    for (direction, kind, latency_us, bw_gb_s, peak) in &fits {
        println!(
            "{},{},{:.3},{:.3},{:.3}",
            direction.label(),
            kind.label(),
            latency_us,
            bw_gb_s,
            peak
        );
    }
    // Re-sample the pstate: if the GPU parked partway through the sweep, the
    // later (larger-size) points are contaminated and the operator should know.
    if let Some(after) = query_pstate(args.device)
        && pstate_is_parked(&after)
        && link_state != "parked"
    {
        eprintln!(
            "roofline_transfer: WARNING — GPU parked during the run \
             (pstate now {after}); later points may under-state bandwidth. \
             Re-run with the GPU kept active for a decode-representative rate."
        );
    }
    eprintln!(
        "roofline_transfer: done in {:.1}s",
        wall.elapsed().as_secs_f64()
    );
    Ok(())
}
