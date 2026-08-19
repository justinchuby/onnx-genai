//! CPU int4 GEMV roofline probe for the placement cost model (issue #994).
//!
//! This is the compute-side companion to `roofline_bandwidth` (host DRAM
//! read-sum ceiling) and `roofline_transfer` (host<->device link). It measures
//! the one term #994's placement criterion still assumes rather than measures:
//! the **effective memory bandwidth the real CPU `MatMulNBits` decode kernel
//! achieves** at `lm_head` shape — specifically on the #979 borrowed zero-copy
//! int4 path, which is the path that streams weights from DRAM on every token
//! and so the only one whose bandwidth the criterion actually turns on. A build
//! that instead keeps an f32 `weight_nk` cache resident, or that hands fp32
//! compute to MLAS SQNBit, is answering a different question; see below.
//!
//! # Why this exists
//!
//! #994's criterion prefers computing an op where its weights already live over
//! moving the weights, when `F/C_slow < W/B_link + F/C_fast`. For the 14B
//! `lm_head` (`K=5120, N=152064, block_size=32, bits=4`, m=1) the arithmetic
//! intensity is `1.557 GFLOP / 389,283,840 B ≈ 4 FLOP/byte`, far below any CPU
//! balance point, so the GEMV *should* be memory-bound and its per-token cost
//! *should* be `weight_bytes / B_dram`. But "should be memory-bound" and
//! "achieves the 49.28 GB/s STREAM ceiling" are different claims: the int4 GEMV
//! also unpacks nibbles, applies per-block scales, and accumulates — real work
//! per byte — and it additionally streams ~97 MB of f32 scales on top of the
//! 389 MB of packed weights. Either can keep it off the roofline. This probe
//! measures where it actually lands.
//!
//! # What it drives
//!
//! It builds the real `com.microsoft::MatMulNBits` kernel through the CPU EP's
//! own `get_kernel` dispatch and calls `execute` in a loop, so it runs the exact
//! code the runtime decodes with — specifically the **#979 symmetric-int4
//! borrowed zero-copy path** (`accuracy_level=0`, no zero points), which streams
//! the packed int4 weights in place instead of materializing the ~8x f32
//! `weight_nk` cache. That is the case #994 cares about: weights that are *not*
//! resident as f32. `mlas` is intentionally not enabled — symmetric int4
//! `accuracy_level=0` returns on the borrowed path before the MLAS SQNBit
//! interception, so the measured path is identical either way.
//!
//! # Threads
//!
//! Decode fans the N output rows across the CPU-EP decode pool, whose worker
//! count is a process-global `OnceLock` built on first use. To sweep thread
//! counts faithfully — as `roofline_bandwidth` does, because the DRAM ceiling
//! is itself a strong function of thread count (10.07 GB/s at 1 thread to
//! 49.28 GB/s at 20 on the box #994 was written against; 33.7 to 75.8 GB/s on a
//! 32-core box) — run this probe once per `--threads N`; each fresh process
//! sizes its pool to `N` via `set_decode_thread_budget`. A single-threaded GEMV
//! number would answer a different question than the one #994 asks.
//!
//! # Reading the output
//!
//! Every number this prints is a property of *one host and one build*, so a
//! result quoted from an issue thread is evidence about that run, not a
//! constant. Pair it with `roofline_bandwidth` **on the same box, in the same
//! session** before turning a ratio into a placement decision — the point of
//! having the probe is that re-measuring is cheap. It has already paid for
//! itself once: between this PR being opened and being merged the kernel it
//! drives got ~4x faster on the same host, which no amount of re-reading the
//! original number would have revealed.

use std::hint::black_box;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, set_decode_thread_budget, with_decode_pool_scope};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};

#[derive(Debug, Parser)]
#[command(
    about = "Measure the effective memory bandwidth of the real CPU int4 MatMulNBits \
             decode GEMV (the #979 borrowed zero-copy path) at lm_head shape, for \
             #994's placement criterion"
)]
struct Args {
    /// Decode-pool worker count for this run. Sweep by invoking once per value
    /// (the pool is a process-global OnceLock, so one process = one thread
    /// count). Mirrors `roofline_bandwidth`'s thread sweep.
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Reduction dim K (activation length). Default is the 14B lm_head.
    #[arg(long, default_value_t = 5120)]
    k: usize,
    /// Output dim N (vocab). Default is the 14B lm_head.
    #[arg(long, default_value_t = 152064)]
    n: usize,
    /// Quantization block size.
    #[arg(long, default_value_t = 32)]
    block_size: usize,
    /// Timed `execute` calls per measurement.
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Warmup `execute` calls before timing (first-touch faults, page-ins).
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    /// Repeated measurements, so a distribution is reported rather than a single
    /// sample (the box is shared and bandwidth is contention-sensitive).
    #[arg(long, default_value_t = 12)]
    repeats: usize,
}

/// Owned, contiguous host buffer plus the shape/stride metadata a view needs.
struct HostTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    strides: Vec<i64>,
    dtype: DataType,
}

impl HostTensor {
    fn from_f32(shape: &[usize], data: Vec<f32>) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            bytes,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype: DataType::Float32,
        }
    }

    fn from_u8(shape: &[usize], bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype: DataType::Uint8,
        }
    }

    fn zeros_f32(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_f32(shape, vec![0.0; n])
    }

    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.bytes.as_ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    fn view_mut(&mut self) -> TensorMut<'_> {
        TensorMut::new(
            DevicePtrMut(self.bytes.as_mut_ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }
}

/// Build the `com.microsoft::MatMulNBits` node for the given decode shape, with
/// `accuracy_level` left at the default 0 so symmetric int4 takes the #979
/// borrowed zero-copy path.
fn build_matmulnbits_graph(
    k: usize,
    n: usize,
    block_size: usize,
    k_blocks: usize,
    blob: usize,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let mut inputs = Vec::new();
    for (name, dtype, shape) in [
        ("A", DataType::Float32, vec![1, k]),
        ("B", DataType::Uint8, vec![n, k_blocks, blob]),
        ("scales", DataType::Float32, vec![n, k_blocks]),
    ] {
        let value = graph.create_named_value(name, dtype, static_shape(shape));
        graph.add_input(value);
        inputs.push(Some(value));
    }
    let output = graph.create_named_value("Y", DataType::Float32, static_shape(vec![1, n]));
    let mut node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![output]);
    node.domain = "com.microsoft".into();
    node.attributes.insert("K".into(), Attribute::Int(k as i64));
    node.attributes.insert("N".into(), Attribute::Int(n as i64));
    node.attributes.insert("bits".into(), Attribute::Int(4));
    node.attributes
        .insert("block_size".into(), Attribute::Int(block_size as i64));
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.threads == 0 {
        bail!("--threads must be positive");
    }
    if args.repeats == 0 {
        bail!("--repeats must be positive (the summary reports a distribution)");
    }
    if args.block_size == 0 || !args.block_size.is_multiple_of(2) {
        bail!("--block-size must be a positive even number (int4 packs two nibbles per byte)");
    }

    // Size the decode pool BEFORE the first kernel execution: the pool is a
    // process-global OnceLock, so this must win the race against any lazy init.
    set_decode_thread_budget(Some(args.threads))
        .map_err(|e| anyhow::anyhow!("set decode thread budget: {e}"))?;

    let k = args.k;
    let n = args.n;
    let block_size = args.block_size;
    let k_blocks = k.div_ceil(block_size);
    let blob = block_size / 2; // int4: two weights per byte.

    let weight_bytes = n * k_blocks * blob;
    let scales_len = n * k_blocks;
    let scales_bytes = scales_len * 4;
    let act_bytes = k * 4;
    let out_bytes = n * 4;
    // The GEMV reads the packed weight and the scales, reads the activation, and
    // writes the logits. The weight dominates; the scales are a real +25% that
    // the criterion's `W` term omits, so we report both a weight-only rate (the
    // criterion input) and a total-traffic rate (the honest roofline efficiency).
    let total_traffic_bytes = weight_bytes + scales_bytes + act_bytes + out_bytes;
    let flops = 2u64 * (n as u64) * (k as u64); // m=1.

    // Build the buffers once. The weight is filled with a cheap pseudo-random
    // pattern: every nibble 0..=15 is a valid packed int4 code, so this is a
    // well-formed symmetric weight — the byte traffic is identical to a real one
    // and only the arithmetic result differs (irrelevant to a bandwidth probe).
    let b_bytes: Vec<u8> = (0..weight_bytes)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();
    let scales_vec: Vec<f32> = (0..scales_len)
        .map(|i| 0.01 + ((i % 17) as f32) * 1.0e-4)
        .collect();
    let act_vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();

    let a = HostTensor::from_f32(&[1, k], act_vec);
    let b = HostTensor::from_u8(&[n, k_blocks, blob], b_bytes);
    let scales = HostTensor::from_f32(&[n, k_blocks], scales_vec);
    let mut y = HostTensor::zeros_f32(&[1, n]);

    let input_shapes = vec![vec![1, k], vec![n, k_blocks, blob], vec![n, k_blocks]];
    let (graph, node_id) = build_matmulnbits_graph(k, n, block_size, k_blocks, blob);
    let provider = CpuExecutionProvider::new();
    let kernel = provider
        .get_kernel(graph.node(node_id), &input_shapes, 1)
        .context("create MatMulNBits kernel")?;

    println!(
        "roofline_gemv: threads={} K={} N={} block_size={} bits=4 accuracy_level=0 (borrowed \
         symmetric int4) iters={} warmups={} repeats={}",
        args.threads, k, n, block_size, args.iters, args.warmups, args.repeats
    );
    println!(
        "roofline_gemv: weight_bytes={} scales_bytes={} total_traffic_bytes={} flops={} \
         arithmetic_intensity={:.3} FLOP/byte",
        weight_bytes,
        scales_bytes,
        total_traffic_bytes,
        flops,
        flops as f64 / weight_bytes as f64
    );

    println!("repeat,per_call_ms,weight_gb_s,total_gb_s");
    // Run warmup + timing INSIDE one persistent decode scope, exactly as the
    // runtime installs the decode pool once per forward pass. This is the
    // difference between measuring the kernel and measuring per-call fork/join:
    // `with_decode_pool_scope(true, ..)` selects the default persistent SPMD
    // pool (mode `On`), so each `execute`'s `parallel_output_rows` broadcasts its
    // N output-row shards to hot, already-spun workers under one lightweight
    // barrier — instead of re-forking the flat pool on every call, which would
    // understate the achieved bandwidth by an order of magnitude at these sizes.
    // The scope wraps the whole measurement so the workers stay resident across
    // all iterations, matching steady-state decode.
    let (per_call_ms_samples, weight_gbs_samples, total_gbs_samples) =
        with_decode_pool_scope(true, move || -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
            // Warmup: pay first-touch faults and let the pool spin up.
            for _ in 0..args.warmups {
                kernel
                    .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
                    .context("warmup execute")?;
                black_box(&y.bytes);
            }

            let mut per_call_ms_samples = Vec::with_capacity(args.repeats);
            let mut weight_gbs_samples = Vec::with_capacity(args.repeats);
            let mut total_gbs_samples = Vec::with_capacity(args.repeats);
            for repeat in 0..args.repeats {
                let start = Instant::now();
                for _ in 0..args.iters {
                    kernel
                        .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
                        .context("timed execute")?;
                    black_box(&y.bytes);
                }
                let per_call_s = start.elapsed().as_secs_f64() / args.iters as f64;
                let per_call_ms = per_call_s * 1.0e3;
                let weight_gb_s = weight_bytes as f64 / per_call_s / 1.0e9;
                let total_gb_s = total_traffic_bytes as f64 / per_call_s / 1.0e9;
                println!("{repeat},{per_call_ms:.3},{weight_gb_s:.3},{total_gb_s:.3}");
                per_call_ms_samples.push(per_call_ms);
                weight_gbs_samples.push(weight_gb_s);
                total_gbs_samples.push(total_gb_s);
            }
            Ok((per_call_ms_samples, weight_gbs_samples, total_gbs_samples))
        })?;

    let ms_min = per_call_ms_samples
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let ms_max = per_call_ms_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let ms_med = median(&mut per_call_ms_samples.clone());
    let weight_gbs_med = median(&mut weight_gbs_samples.clone());
    let weight_gbs_max = weight_gbs_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let total_gbs_med = median(&mut total_gbs_samples.clone());
    let total_gbs_max = total_gbs_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("# Summary (headline = best/quietest sample)");
    println!(
        "threads,per_call_ms_median,per_call_ms_min,per_call_ms_max,weight_gb_s_median,\
         weight_gb_s_best,total_gb_s_median,total_gb_s_best"
    );
    println!(
        "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        args.threads,
        ms_med,
        ms_min,
        ms_max,
        weight_gbs_med,
        weight_gbs_max,
        total_gbs_med,
        total_gbs_max
    );

    Ok(())
}
