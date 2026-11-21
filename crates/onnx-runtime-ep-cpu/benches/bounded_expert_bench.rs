//! Isolated-process MoE memory/CPU-time comparison. Set the existing cache budget
//! before process startup: zero for `streamed`, sufficient bytes for `cold` or
//! `warm`. Arguments: mode rows hidden intermediate iterations output-file. The unchanged
//! Relu control helps reject measurements distorted by shared-host contention.
//! For RSS comparisons, set ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES=1048576
//! identically in every arm: the default startup calibration otherwise touches
//! about 512 MiB and obscures the weight-working-set difference.
mod common;
use common::{FloatDType, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, TensorView};
use onnx_runtime_ep_cpu::{
    CpuExecutionProvider, kernels::block_quantized_moe::BLOCK_QUANT_MOE_DENSE_EXPANSIONS,
};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId};
use std::{hint::black_box, sync::atomic::Ordering, time::Instant};

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn usage() -> (f64, u64) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the entire output structure on success.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    let usage = unsafe { usage.assume_init() };
    let cpu = usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6;
    let rss = usage.ru_maxrss as u64;
    (
        cpu,
        if cfg!(target_os = "linux") {
            rss * 1024
        } else {
            rss
        },
    )
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn usage() -> (f64, u64) {
    (f64::NAN, 0)
}

fn packed(experts: usize, output: usize, input: usize) -> Tensor {
    let shape = [experts, output, input / 32, 17];
    let mut bytes = vec![0u8; shape.iter().product()];
    for (block, data) in bytes.chunks_exact_mut(17).enumerate() {
        data[0] = 125;
        for (i, byte) in data[1..].iter_mut().enumerate() {
            *byte = ((block + i) % 16) as u8 | (((block * 7 + i * 3) % 16) as u8) << 4;
        }
    }
    Tensor::u8(&shape, &bytes)
}

fn main() {
    let startup_peak = usage().1;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        6,
        "usage: bounded_expert_bench <streamed|cold|warm> <rows> <hidden> <intermediate> <iterations> <output-file>; set ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES and RAYON_NUM_THREADS before launching"
    );
    let mode = &args[0];
    assert!(matches!(mode.as_str(), "streamed" | "cold" | "warm"));
    let number = |i: usize| {
        args[i]
            .parse::<usize>()
            .expect("dimensions and iteration count must be unsigned integers")
    };
    let (rows, hidden, inter, iterations) = (number(1), number(2), number(3), number(4));
    assert!(
        hidden > 0
            && inter > 0
            && hidden.is_multiple_of(32)
            && inter.is_multiple_of(32)
            && iterations > 0
    );
    let budget = std::env::var("ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES")
        .expect("set the explicit cache budget")
        .parse::<usize>()
        .unwrap();
    let dense_bytes = 2 * hidden * inter * 4;
    assert!(
        if mode == "streamed" {
            budget < dense_bytes
        } else {
            budget >= dense_bytes
        },
        "mode must agree with the real kernel's budget decision"
    );
    common::init_decode_topology();
    let initialized_peak = usage().1;
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![
            Some(ValueId(0)),
            Some(ValueId(1)),
            Some(ValueId(2)),
            None,
            Some(ValueId(4)),
        ],
        vec![ValueId(5)],
    );
    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (name, value) in [
        ("fc1_format", Attribute::String(b"mxfp4".to_vec())),
        ("fc2_format", Attribute::String(b"mxfp4".to_vec())),
        ("block_layout_version", Attribute::Int(1)),
        ("activation_type", Attribute::String(b"silu".to_vec())),
        ("k", Attribute::Int(1)),
        ("normalize_routing_weights", Attribute::Int(1)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    let mut kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &[], 1)
        .unwrap();
    let compiled_peak = usage().1;
    if mode == "warm" {
        kernel.set_constant_inputs(&[false, false, true, false, true]);
    }
    let input = Tensor::floats(
        FloatDType::F32,
        &[rows, hidden],
        &(0..rows * hidden)
            .map(|i| (i % 31) as f32 / 1000. - 0.015)
            .collect::<Vec<_>>(),
    );
    let logits = Tensor::floats(
        FloatDType::F32,
        &[rows, 2],
        &(0..rows).flat_map(|_| [2., 0.]).collect::<Vec<_>>(),
    );
    let fc1 = packed(2, inter, hidden);
    let fc2 = packed(2, hidden, inter);
    let inputs = [
        input.view(),
        logits.view(),
        fc1.view(),
        TensorView::absent(DataType::Float32),
        fc2.view(),
    ];
    let mut output = Tensor::zeros(FloatDType::F32, &[rows, hidden]);
    let inputs_peak = usage().1;
    kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
    let before = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
    let cpu = usage().0;
    let wall = Instant::now();
    for _ in 0..iterations {
        kernel
            .execute(black_box(&inputs), black_box(&mut [output.view_mut()]))
            .unwrap();
    }
    let wall_s = wall.elapsed().as_secs_f64();
    let (end_cpu, peak_rss) = usage();
    let expansions = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        expansions,
        if mode == "cold" && rows != 0 {
            2 * iterations
        } else {
            0
        },
        "counter must prove the selected compute route"
    );
    let checksum = output
        .f32s()
        .iter()
        .map(|&value| f64::from(value))
        .sum::<f64>();
    std::fs::write(
        &args[5],
        output
            .f32s()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("write exact benchmark output for numerical comparison");
    let control = common::make_kernel("Relu", [], &[vec![16384]], 24);
    let values = Tensor::floats(FloatDType::F32, &[16384], &common::float_values(16384));
    let mut result = Tensor::zeros(FloatDType::F32, &[16384]);
    let control_cpu = usage().0;
    for _ in 0..1000 {
        control
            .execute(&[values.view()], &mut [result.view_mut()])
            .unwrap();
        black_box(result.f32s());
    }
    println!(
        "mode={mode} rows={rows} hidden={hidden} inter={inter} iterations={iterations} threads={} cpu_ms_per_call={:.6} wall_ms_per_call={:.6} startup_peak_rss_bytes={startup_peak} initialized_peak_rss_bytes={initialized_peak} compiled_peak_rss_bytes={compiled_peak} inputs_peak_rss_bytes={inputs_peak} peak_rss_bytes={peak_rss} dense_expansions={expansions} checksum={checksum:.9} control_cpu_ms={:.6}",
        rayon::current_num_threads(),
        (end_cpu - cpu) * 1000. / iterations as f64,
        wall_s * 1000. / iterations as f64,
        (usage().0 - control_cpu) * 1000.
    );
}
