#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::{CudaRuntime, cuptr};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

#[derive(Clone)]
struct HostTensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl HostTensor {
    fn f32(shape: &[usize], values: &[f32]) -> Self {
        Self {
            dtype: DataType::Float32,
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
        }
    }

    fn u8(shape: &[usize], values: Vec<u8>) -> Self {
        Self {
            dtype: DataType::Uint8,
            shape: shape.to_vec(),
            bytes: values,
        }
    }

    fn activation(dtype: DataType, shape: &[usize], values: &[f32]) -> Self {
        let bytes = match dtype {
            DataType::Float32 => values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect(),
            DataType::Float16 => values
                .iter()
                .flat_map(|value| f16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            DataType::BFloat16 => values
                .iter()
                .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
                .collect(),
            other => panic!("unsupported activation dtype {other:?}"),
        };
        Self {
            dtype,
            shape: shape.to_vec(),
            bytes,
        }
    }
}

#[derive(Clone)]
struct Quantized {
    packed: HostTensor,
    scales: HostTensor,
    zero_points: Option<HostTensor>,
}

#[allow(clippy::too_many_arguments)]
fn quantize(
    experts: usize,
    out_features: usize,
    in_features: usize,
    bits: usize,
    block_size: usize,
    affine: bool,
    seed: usize,
) -> Quantized {
    let pack_size = 8 / bits;
    let blocks = in_features / block_size;
    let packed_in = in_features / pack_size;
    let zero_point_bytes = blocks.div_ceil(pack_size);
    let mask = if bits == 8 {
        u8::MAX
    } else {
        (1u8 << bits) - 1
    };
    let default_zero = 1u8 << (bits - 1);
    let mut packed = vec![0u8; experts * out_features * packed_in];
    let mut scales = vec![0.0f32; experts * out_features * blocks];
    let mut zero_points = affine.then(|| vec![0u8; experts * out_features * zero_point_bytes]);
    for expert in 0..experts {
        for output in 0..out_features {
            let expert_row = expert * out_features + output;
            for block in 0..blocks {
                let scale =
                    0.025 + 0.0125 * ((seed + expert * 3 + output * 5 + block * 7) % 5) as f32;
                scales[expert_row * blocks + block] = scale;
                let zero = if affine {
                    default_zero.saturating_sub(
                        ((seed + expert + output + block) % 3).min(default_zero as usize) as u8,
                    )
                } else {
                    default_zero
                };
                if let Some(points) = &mut zero_points {
                    points[expert_row * zero_point_bytes + block / pack_size] |=
                        zero << ((block % pack_size) * bits);
                }
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let span = if bits == 8 { 31 } else { 7 };
                    let centered = ((seed + expert * 11 + output * 13 + depth * 17) % span) as i16
                        - (span / 2) as i16;
                    let quantized = (centered + i16::from(zero)).clamp(0, i16::from(mask)) as u8;
                    packed[expert_row * packed_in + depth / pack_size] |=
                        quantized << ((depth % pack_size) * bits);
                }
            }
        }
    }
    Quantized {
        packed: HostTensor::u8(&[experts, out_features, packed_in], packed),
        scales: HostTensor::f32(&[experts, out_features, blocks], &scales),
        zero_points: zero_points
            .map(|points| HostTensor::u8(&[experts, out_features, zero_point_bytes], points)),
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    experts: usize,
    rows: usize,
    hidden: usize,
    inter: usize,
    bits: usize,
    top_k: usize,
    activation: &'static str,
    swiglu_fusion: usize,
    affine: bool,
    fc3: bool,
    biases: bool,
    normalize: bool,
    router_weights: bool,
}

fn case_inputs(case: Case, dtype: DataType) -> Vec<Option<HostTensor>> {
    let fc1_size = if case.activation == "swiglu" && case.swiglu_fusion != 0 {
        case.inter * 2
    } else {
        case.inter
    };
    let x: Vec<f32> = (0..case.rows * case.hidden)
        .map(|index| ((index * 19 + 3) % 29) as f32 / 13.0 - 1.0)
        .collect();
    let router: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| ((index * 7 + 5) % 17) as f32 / 4.0 - 2.0)
        .collect();
    let aggregation: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| 0.1 + ((index * 5 + 2) % 11) as f32 / 10.0)
        .collect();
    let fc1 = quantize(
        case.experts,
        fc1_size,
        case.hidden,
        case.bits,
        16,
        case.affine,
        1,
    );
    let fc2 = quantize(
        case.experts,
        case.hidden,
        case.inter,
        case.bits,
        16,
        case.affine,
        2,
    );
    match (case.activation, case.swiglu_fusion) {
        ("swiglu", 0) => assert!(case.fc3, "unfused SwiGLU requires FC3"),
        ("swiglu", _) => assert!(!case.fc3, "fused SwiGLU must not provide FC3"),
        ("silu", 0) => {}
        _ => assert!(!case.fc3, "FC3 is only valid for SwiGLU or gated SiLU"),
    }
    let fc3 = case.fc3.then(|| {
        quantize(
            case.experts,
            case.inter,
            case.hidden,
            case.bits,
            16,
            case.affine,
            3,
        )
    });
    let bias = |width: usize, seed: usize| {
        let values: Vec<f32> = (0..case.experts * width)
            .map(|index| ((index * 3 + seed) % 7) as f32 * 0.01 - 0.03)
            .collect();
        HostTensor::f32(&[case.experts, width], &values)
    };
    vec![
        Some(HostTensor::activation(dtype, &[case.rows, case.hidden], &x)),
        Some(HostTensor::f32(&[case.rows, case.experts], &router)),
        Some(fc1.packed),
        Some(fc1.scales),
        case.biases.then(|| bias(fc1_size, 1)),
        Some(fc2.packed),
        Some(fc2.scales),
        case.biases.then(|| bias(case.hidden, 2)),
        fc3.as_ref().map(|weights| weights.packed.clone()),
        fc3.as_ref().map(|weights| weights.scales.clone()),
        (case.biases && case.fc3).then(|| bias(case.inter, 3)),
        fc1.zero_points,
        fc2.zero_points,
        fc3.and_then(|weights| weights.zero_points),
        case.router_weights
            .then(|| HostTensor::f32(&[case.rows, case.experts], &aggregation)),
    ]
}

fn router_with_top_experts(case: Case, first_expert: usize) -> HostTensor {
    assert!(first_expert + case.top_k <= case.experts);
    let values: Vec<f32> = (0..case.rows)
        .flat_map(|_| {
            (0..case.experts).map(move |expert| {
                if (first_expert..first_expert + case.top_k).contains(&expert) {
                    10.0 - (expert - first_expert) as f32
                } else {
                    -10.0
                }
            })
        })
        .collect();
    HostTensor::f32(&[case.rows, case.experts], &values)
}

fn router_with_hot_expert(case: Case) -> HostTensor {
    assert!(case.top_k >= 2);
    let values: Vec<f32> = (0..case.rows)
        .flat_map(|_| {
            (0..case.experts).map(move |expert| match expert {
                0 => 20.0,
                expert if expert < case.top_k => 10.0 - expert as f32,
                _ => -10.0,
            })
        })
        .collect();
    HostTensor::f32(&[case.rows, case.experts], &values)
}

fn model_node(
    inputs: &[Option<HostTensor>],
    output_dtype: DataType,
    output_shape: &[usize],
    case: Case,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let values = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input.as_ref().map(|input| {
                let value = graph.create_named_value(
                    format!("input_{index}"),
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                );
                graph.add_input(value);
                value
            })
        })
        .collect();
    let output = graph.create_named_value(
        "output",
        output_dtype,
        static_shape(output_shape.iter().copied()),
    );
    let mut node = Node::new(NodeId(0), "QMoE", values, vec![output]);
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("expert_weight_bits", Attribute::Int(case.bits as i64)),
        ("block_size", Attribute::Int(16)),
        ("k", Attribute::Int(case.top_k as i64)),
        (
            "activation_type",
            Attribute::String(case.activation.as_bytes().to_vec()),
        ),
        (
            "normalize_routing_weights",
            Attribute::Int(i64::from(case.normalize)),
        ),
        ("swiglu_fusion", Attribute::Int(case.swiglu_fusion as i64)),
    ] {
        node.attributes.insert(name.into(), value);
    }
    node.attributes
        .insert("activation_alpha".into(), Attribute::Float(1.125));
    node.attributes
        .insert("activation_beta".into(), Attribute::Float(-0.0625));
    node.attributes
        .insert("swiglu_limit".into(), Attribute::Float(4.0));
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn absent_dtype(index: usize, activation_dtype: DataType) -> DataType {
    match index {
        0 => activation_dtype,
        2 | 5 | 8 | 11 | 12 | 13 => DataType::Uint8,
        _ => DataType::Float32,
    }
}

fn run_cpu(case: Case, inputs: &[Option<HostTensor>]) -> Vec<f32> {
    let output_shape = [case.rows, case.hidden];
    let (graph, node) = model_node(inputs, DataType::Float32, &output_shape, case);
    let model = Model::new(&graph);
    let kernel = CpuExecutionProvider::new()
        .get_kernel(model.graph.node(node), &[], 1)
        .unwrap();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&strides)
        .enumerate()
        .map(|(index, (input, strides))| match (input, strides) {
            (Some(input), Some(strides)) => TensorView::new(
                DevicePtr(input.bytes.as_ptr().cast()),
                input.dtype,
                &input.shape,
                strides,
                DeviceId::cpu(),
            ),
            _ => TensorView::absent(absent_dtype(index, DataType::Float32)),
        })
        .collect();
    let mut output = vec![0u8; case.rows * case.hidden * 4];
    let output_strides = compute_contiguous_strides(&output_shape);
    kernel
        .execute(
            &views,
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr().cast()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                DeviceId::cpu(),
            )],
        )
        .unwrap();
    output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Serializes GPU test bodies within this binary. The capture/replay tests use
/// `CU_STREAM_CAPTURE_MODE_GLOBAL`, under which any concurrent CUDA alloc/launch
/// from another test thread in the same process/context errors out. Holding this
/// lock for the whole test body (via [`GpuGuard`]) keeps capture from overlapping
/// other CUDA work. Separate test binaries run in separate processes/contexts, so
/// no cross-binary serialization is needed.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A live CUDA EP plus the held [`GPU_SERIAL`] guard. Derefs to the EP so every
/// existing `require_cuda()` call site is unchanged.
struct GpuGuard {
    ep: CudaExecutionProvider,
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl std::ops::Deref for GpuGuard {
    type Target = CudaExecutionProvider;
    fn deref(&self) -> &CudaExecutionProvider {
        &self.ep
    }
}

fn require_cuda() -> GpuGuard {
    // Ignore poisoning: a panicking test still leaves the device usable, and we
    // must not cascade one failure into spurious lock failures elsewhere.
    let serial = GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => GpuGuard {
            ep,
            _serial: serial,
        },
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    case: Case,
    inputs: &[Option<HostTensor>],
    dtype: DataType,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    run_gpu_impl(ep, case, inputs, dtype, None, None, true)
}

fn run_gpu_with_prefill_min_tokens(
    ep: &CudaExecutionProvider,
    case: Case,
    inputs: &[Option<HostTensor>],
    dtype: DataType,
    prefill_min_tokens: Option<usize>,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    run_gpu_impl(ep, case, inputs, dtype, prefill_min_tokens, None, true)
}

fn run_gpu_impl(
    ep: &CudaExecutionProvider,
    case: Case,
    inputs: &[Option<HostTensor>],
    dtype: DataType,
    prefill_min_tokens: Option<usize>,
    replay_router: Option<&HostTensor>,
    capture: bool,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let output_shape = [case.rows, case.hidden];
    let (mut graph, node) = model_node(inputs, dtype, &output_shape, case);
    if let Some(prefill_min_tokens) = prefill_min_tokens {
        graph.node_mut(node).attributes.insert(
            "prefill_min_tokens".into(),
            Attribute::Int(prefill_min_tokens as i64),
        );
    }
    let model = Model::new(&graph);
    let concrete_shapes: Vec<_> = inputs
        .iter()
        .filter_map(|input| input.as_ref().map(|input| input.shape.clone()))
        .collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::<Option<DeviceBuffer>>::new();
    for input in inputs {
        if let Some(input) = input {
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            // SAFETY: allocation size equals the source tensor byte length.
            unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            buffers.push(Some(buffer));
        } else {
            buffers.push(None);
        }
    }
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views: Vec<_> = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .enumerate()
        .map(
            |(index, ((input, buffer), strides))| match (input, buffer, strides) {
                (Some(input), Some(buffer), Some(strides)) => TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    input.dtype,
                    &input.shape,
                    strides,
                    ep.device_id(),
                ),
                _ => TensorView::absent(absent_dtype(index, dtype)),
            },
        )
        .collect();
    let output_bytes = case.rows * case.hidden * dtype.byte_size();
    let mut output_buffer = ep.allocate(output_bytes, 256)?;
    let output_strides = compute_contiguous_strides(&output_shape);
    kernel.execute(
        &views,
        &mut [TensorMut::new(
            DevicePtrMut(output_buffer.as_mut_ptr()),
            dtype,
            &output_shape,
            &output_strides,
            ep.device_id(),
        )],
    )?;
    if capture {
        assert!(
            kernel.capture_support().is_supported(),
            "successful eager QMoE execution must warm its capture workspace"
        );
        runtime.begin_graph_capture(&[kernel.as_ref()])?;
        if let Err(error) = kernel.execute(
            &views,
            &mut [TensorMut::new(
                DevicePtrMut(output_buffer.as_mut_ptr()),
                dtype,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
        ) {
            let _ = runtime.abort_graph_capture();
            return Err(error);
        }
        runtime.end_graph_capture()?;
        if let Some(router) = replay_router {
            let router_buffer = buffers[1].as_ref().expect("router_probs must be present");
            // SAFETY: router shape is unchanged and its byte length matches the allocation.
            unsafe { runtime.htod(&router.bytes, cuptr(router_buffer.as_ptr()))? };
        }
        // SAFETY: the output allocation is exactly `output_bytes` bytes.
        unsafe { runtime.htod(&vec![0u8; output_bytes], cuptr(output_buffer.as_ptr()))? };
        runtime.replay_graph()?;
    }
    runtime.synchronize()?;
    if capture {
        runtime.reset_graph()?;
    }
    let mut bytes = vec![0u8; output_bytes];
    // SAFETY: output allocation contains exactly the requested output tensor.
    unsafe { runtime.dtoh(&mut bytes, cuptr(output_buffer.as_ptr()))? };
    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    Ok(match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported output dtype {other:?}"),
    })
}

fn rounded_cpu_inputs(inputs: &[Option<HostTensor>], dtype: DataType) -> Vec<Option<HostTensor>> {
    let mut rounded = inputs.to_vec();
    let activation = rounded[0].as_ref().unwrap();
    let values: Vec<f32> = match dtype {
        DataType::Float32 => activation
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => activation
            .bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => activation
            .bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    };
    rounded[0] = Some(HostTensor::f32(&activation.shape, &values));
    rounded
}

fn assert_conforms(actual: &[f32], expected: &[f32], case: Case, dtype: DataType) {
    assert_eq!(actual.len(), expected.len());
    let (absolute, relative) = match dtype {
        DataType::Float32 => (2e-5, 1e-4),
        DataType::Float16 => (2e-5, 6e-4),
        DataType::BFloat16 => (2e-5, 4.1e-3),
        other => panic!("unsupported dtype {other:?}"),
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute + relative * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}, \
             absolute={absolute}, relative={relative}, dtype={dtype:?}, case={case:?}"
        );
    }
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> (f32, u32) {
    actual.iter().zip(expected).fold(
        (0.0f32, 0u32),
        |(max_abs, max_ulp), (&actual, &expected)| {
            let actual_key = if actual.is_sign_negative() {
                !actual.to_bits()
            } else {
                actual.to_bits() | 0x8000_0000
            };
            let expected_key = if expected.is_sign_negative() {
                !expected.to_bits()
            } else {
                expected.to_bits() | 0x8000_0000
            };
            (
                max_abs.max((actual - expected).abs()),
                max_ulp.max(actual_key.abs_diff(expected_key)),
            )
        },
    )
}

fn host_f32(tensor: &HostTensor) -> Vec<f32> {
    tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// f64 truth for an identity-activation, symmetric int4 QMoE forward.
///
/// This mirrors the dense `int8_f64_reference` philosophy (evaluate every
/// dequant+FMA in f64) but walks the *expert path*: top-k softmax routing,
/// two chained int4 GEMVs (fc1 -> identity -> fc2), and the routing-weighted
/// expert combination. `identity` activation keeps the expert body a pure pair
/// of int4 dot-product reductions, so the only source of CPU-vs-CUDA drift is
/// the reduction order of those GEMVs -- exactly the reassociation that tips
/// borderline argmaxes in DeepSeek-V2-Lite. Returns per-output
/// `(reference, fc2_sum_abs)` so the caller can form a roundoff envelope.
///
/// Restricted to symmetric int4, identity activation, no fc3/bias/aggregation
/// so the reference is exact-by-construction and cannot silently mismodel the
/// kernel. `case_inputs` quantizes expert weights with `block_size = 16`.
fn qmoe_identity_f64_reference(case: Case, inputs: &[Option<HostTensor>]) -> Vec<(f64, f64)> {
    assert_eq!(case.activation, "identity");
    assert!(
        !case.affine,
        "reference assumes symmetric int4 (zero-point = 8)"
    );
    assert!(
        !case.fc3 && !case.biases && !case.router_weights,
        "reference assumes a plain fc1->identity->fc2 expert body"
    );
    let (hidden, inter, experts, bits) = (case.hidden, case.inter, case.experts, case.bits);
    let block = 16usize; // case_inputs quantizes expert weights with block_size = 16
    let pack = 8 / bits;
    let zero = 1i32 << (bits - 1); // symmetric zero-point (8 for int4)
    let mask = (1u16 << bits) - 1;
    let x = host_f32(inputs[0].as_ref().unwrap());
    let router = host_f32(inputs[1].as_ref().unwrap());
    let fc1_packed = &inputs[2].as_ref().unwrap().bytes;
    let fc1_scales = host_f32(inputs[3].as_ref().unwrap());
    let fc2_packed = &inputs[5].as_ref().unwrap().bytes;
    let fc2_scales = host_f32(inputs[6].as_ref().unwrap());

    let dequant = |packed: &[u8],
                   scales: &[f32],
                   expert: usize,
                   out_features: usize,
                   in_features: usize,
                   output: usize,
                   depth: usize|
     -> f64 {
        let row = expert * out_features + output;
        let packed_in = in_features / pack;
        let byte = packed[row * packed_in + depth / pack];
        let nibble = i32::from((u16::from(byte) >> ((depth % pack) * bits)) & mask);
        let scale = f64::from(scales[row * (in_features / block) + depth / block]);
        f64::from(nibble - zero) * scale
    };

    let mut out = vec![(0.0f64, 0.0f64); case.rows * hidden];
    for row in 0..case.rows {
        let logits = &router[row * experts..(row + 1) * experts];
        // Identical top-k rule to routing_weights(): total_cmp desc, index asc.
        let mut selected: Vec<usize> = (0..experts).collect();
        selected.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        selected.truncate(case.top_k);
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f64> = logits
            .iter()
            .map(|&value| (f64::from(value) - f64::from(max_logit)).exp())
            .collect();
        let denominator: f64 = if case.normalize {
            selected.iter().map(|&i| exponentials[i]).sum()
        } else {
            exponentials.iter().sum()
        };
        for &expert in &selected {
            let weight = exponentials[expert] / denominator;
            // fc1: y[j] = sum_d x[d] * dequant(W1[e, j, d]); identity activation.
            let mut activated = vec![0.0f64; inter];
            for j in 0..inter {
                let mut acc = 0.0f64;
                for d in 0..hidden {
                    acc += f64::from(x[row * hidden + d])
                        * dequant(fc1_packed, &fc1_scales, expert, inter, hidden, j, d);
                }
                activated[j] = acc;
            }
            // fc2: o[h] = sum_j activated[j] * dequant(W2[e, h, j]).
            for h in 0..hidden {
                let mut acc = 0.0f64;
                let mut sum_abs = 0.0f64;
                for (j, &value) in activated.iter().enumerate() {
                    let term =
                        value * dequant(fc2_packed, &fc2_scales, expert, hidden, inter, h, j);
                    acc += term;
                    sum_abs += term.abs();
                }
                let slot = &mut out[row * hidden + h];
                slot.0 += weight * acc;
                slot.1 += weight.abs() * sum_abs;
            }
        }
    }
    out
}

fn compare(case: Case, dtype: DataType) -> (f32, u32) {
    let ep = require_cuda();
    let gpu_inputs = case_inputs(case, dtype);
    let cpu_inputs = rounded_cpu_inputs(&gpu_inputs, dtype);
    let expected = run_cpu(case, &cpu_inputs);
    let actual = run_gpu(&ep, case, &gpu_inputs, dtype).unwrap();
    assert_conforms(&actual, &expected, case, dtype);
    error_metrics(&actual, &expected)
}

/// Read the f32 values back out of an f32-encoded [`HostTensor`].
fn f32_values(tensor: &HostTensor) -> Vec<f32> {
    tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

/// Re-encode an f32 [`HostTensor`] as a half (fp16/bf16) tensor, matching how a
/// real all-`T` fp16/bf16 QMoE graph stores its router probs, scales and biases.
fn to_half_tensor(tensor: &HostTensor, dtype: DataType) -> HostTensor {
    HostTensor::activation(dtype, &tensor.shape, &f32_values(tensor))
}

/// Round an f32 [`HostTensor`] through fp16/bf16 back to f32, so the CPU
/// reference dequantizes with the same values the GPU sees after widening.
fn round_through_half(tensor: &HostTensor, dtype: DataType) -> HostTensor {
    let rounded: Vec<f32> = f32_values(tensor)
        .into_iter()
        .map(|value| match dtype {
            DataType::Float16 => f16::from_f32(value).to_f32(),
            DataType::BFloat16 => bf16::from_f32(value).to_f32(),
            _ => value,
        })
        .collect();
    HostTensor::f32(&tensor.shape, &rounded)
}

/// Build an all-half QMoE input set (input, router_probs, all scales, all
/// biases and the aggregation weights are fp16/bf16) plus a matching CPU
/// reference input set whose float operands are the same values rounded through
/// the half type. The uint8 packed weights and zero points stay untouched.
fn all_half_inputs(
    case: Case,
    dtype: DataType,
) -> (Vec<Option<HostTensor>>, Vec<Option<HostTensor>>) {
    let base = case_inputs(case, dtype);
    let mut gpu = base.clone();
    let mut cpu = rounded_cpu_inputs(&base, dtype);
    // Indices of the T-typed float operands: router_probs, fc1/fc2/fc3 scales,
    // fc1/fc2/fc3 biases and the aggregation weights.
    for index in [1usize, 3, 4, 6, 7, 9, 10, 14] {
        if let Some(tensor) = base[index].clone() {
            gpu[index] = Some(to_half_tensor(&tensor, dtype));
            cpu[index] = Some(round_through_half(&tensor, dtype));
        }
    }
    (gpu, cpu)
}

/// The all-`T` fp16/bf16 QMoE path: router_probs, scales, biases and the
/// aggregation weights all carry the input's half type (matching ORT's single
/// type parameter `T`, e.g. the fused Mobius fp16 QMoE export). The GPU widens
/// each to f32 and must match a CPU reference dequantized with the same
/// half-rounded values.
fn compare_all_half(case: Case, dtype: DataType) {
    let ep = require_cuda();
    let (gpu_inputs, cpu_inputs) = all_half_inputs(case, dtype);
    let expected = run_cpu(case, &cpu_inputs);
    let actual = run_gpu(&ep, case, &gpu_inputs, dtype).unwrap();
    assert_conforms(&actual, &expected, case, dtype);
}

fn all_half_rich_case(rows: usize) -> Case {
    Case {
        experts: 4,
        rows,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: true,
        biases: true,
        normalize: true,
        router_weights: true,
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_all_half_router_scales_bias_match_rounded_cpu() {
    // rows=1 exercises the decode/fused path; rows=6 exercises prefill grouping.
    for rows in [1, 6] {
        compare_all_half(all_half_rich_case(rows), DataType::Float16);
        compare_all_half(all_half_rich_case(rows), DataType::BFloat16);
    }
}

fn qmoe_64expert_case(rows: usize) -> Case {
    Case {
        experts: 64,
        rows,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 6,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: true,
    }
}

fn compare_64expert_decode_and_prefill(dtype: DataType) {
    for rows in [1, 8] {
        compare(qmoe_64expert_case(rows), dtype);
    }
}

fn compare_gemv_gemm_and_cpu(case: Case) {
    assert_eq!(case.rows, 6);
    assert_eq!(case.experts, 4);
    assert_eq!(case.top_k, 2);
    let ep = require_cuda();
    let mut inputs = case_inputs(case, DataType::Float32);
    inputs[1] = Some(HostTensor::f32(
        &[case.rows, case.experts],
        &[
            9.0, 8.0, 0.0, -1.0, 8.0, 9.0, -1.0, 0.0, 9.0, 7.0, 1.0, 0.0, 0.0, -1.0, 9.0, 8.0,
            -1.0, 0.0, 8.0, 9.0, 1.0, 0.0, 9.0, 7.0,
        ],
    ));
    let expected = run_cpu(case, &inputs);
    let gemm =
        run_gpu_with_prefill_min_tokens(&ep, case, &inputs, DataType::Float32, Some(2)).unwrap();
    let gemv =
        run_gpu_with_prefill_min_tokens(&ep, case, &inputs, DataType::Float32, Some(1024)).unwrap();
    assert_conforms(&gemm, &gemv, case, DataType::Float32);
    assert_conforms(&gemm, &expected, case, DataType::Float32);
    assert_conforms(&gemv, &expected, case, DataType::Float32);
}

fn activation_case(activation: &'static str, swiglu_fusion: usize, fc3: bool) -> Case {
    Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation,
        swiglu_fusion,
        affine: true,
        fc3,
        biases: false,
        normalize: true,
        router_weights: false,
    }
}

macro_rules! activation_path_test {
    ($name:ident, $activation:literal, $fusion:expr, $separate_gate:expr) => {
        #[cfg_attr(
            not(feature = "gpu-tests"),
            ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
        )]
        #[test]
        fn $name() {
            compare_gemv_gemm_and_cpu(activation_case($activation, $fusion, $separate_gate));
        }
    };
}

activation_path_test!(qmoe_relu_gemv_gemm_matches_cpu, "relu", 0, false);
activation_path_test!(qmoe_gelu_gemv_gemm_matches_cpu, "gelu", 0, false);
activation_path_test!(qmoe_silu_gemv_gemm_matches_cpu, "silu", 0, false);
activation_path_test!(qmoe_silu_gated_gemv_gemm_matches_cpu, "silu", 0, true);
activation_path_test!(qmoe_swiglu_unfused_gemv_gemm_matches_cpu, "swiglu", 0, true);
activation_path_test!(
    qmoe_swiglu_interleaved_gemv_gemm_matches_cpu,
    "swiglu",
    1,
    false
);
activation_path_test!(qmoe_swiglu_split_gemv_gemm_matches_cpu, "swiglu", 2, false);
activation_path_test!(qmoe_identity_gemv_gemm_matches_cpu, "identity", 0, false);

fn assert_fused_gate_up_decode_case(case: Case) {
    assert_eq!(case.rows, 1);
    assert!(case.rows * case.top_k <= 16);
    assert!(
        (case.fc3 && matches!(case.activation, "silu" | "swiglu"))
            || (!case.fc3 && case.activation == "swiglu" && case.swiglu_fusion != 0),
        "case must satisfy the qmoe_gate_up_activate launch gate"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_decode_fused_swiglu_interleaved_matches_cpu() {
    let case = Case {
        experts: 4,
        rows: 1,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "swiglu",
        swiglu_fusion: 1,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: false,
    };
    assert_fused_gate_up_decode_case(case);
    compare(case, DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_decode_fused_silu_fc3_matches_cpu() {
    let case = Case {
        experts: 4,
        rows: 1,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: true,
        biases: true,
        normalize: true,
        router_weights: false,
    };
    assert_fused_gate_up_decode_case(case);
    compare(case, DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_biases_gemv_gemm_match_cpu() {
    compare_gemv_gemm_and_cpu(Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: false,
    });
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_separate_router_weights_gemv_gemm_match_cpu() {
    compare_gemv_gemm_and_cpu(Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: false,
        normalize: true,
        router_weights: true,
    });
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_glm_silu_fc3_biases_separate_router_matches_cpu_all_dtypes() {
    let case = Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: true,
        biases: true,
        normalize: true,
        router_weights: true,
    };
    compare_gemv_gemm_and_cpu(case);
    compare(case, DataType::Float16);
    compare(case, DataType::BFloat16);
}

macro_rules! sub_byte_path_test {
    ($name:ident, $bits:expr, $affine:expr) => {
        #[cfg_attr(
            not(feature = "gpu-tests"),
            ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
        )]
        #[test]
        fn $name() {
            compare_gemv_gemm_and_cpu(Case {
                experts: 4,
                rows: 6,
                hidden: 16,
                inter: 16,
                bits: $bits,
                top_k: 2,
                activation: "identity",
                swiglu_fusion: 0,
                affine: $affine,
                fc3: false,
                biases: true,
                normalize: true,
                router_weights: false,
            });
        }
    };
}

sub_byte_path_test!(qmoe_int1_symmetric_gemv_gemm_matches_cpu, 1, false);
sub_byte_path_test!(qmoe_int1_affine_gemv_gemm_matches_cpu, 1, true);
sub_byte_path_test!(qmoe_int2_symmetric_gemv_gemm_matches_cpu, 2, false);
sub_byte_path_test!(qmoe_int2_affine_gemv_gemm_matches_cpu, 2, true);

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_int4_top2_symmetric_matches_cpu() {
    let (max_abs, max_ulp) = compare(
        Case {
            experts: 4,
            rows: 3,
            hidden: 16,
            inter: 16,
            bits: 4,
            top_k: 2,
            activation: "identity",
            swiglu_fusion: 0,
            affine: false,
            fc3: false,
            biases: false,
            normalize: true,
            router_weights: false,
        },
        DataType::Float32,
    );
    eprintln!("QMoE int4 top-2 CPU/CUDA max_abs_diff={max_abs:e} max_ulp_diff={max_ulp}");
}

/// f64-reference parity for the int4 QMoE expert GEMV -- the QMoE analogue of the
/// dense `run_int8_f64_reference_parity`. Proves the native-CUDA expert reduction
/// is within an f64 tree-reduction roundoff bound (not "whatever CUDA produced"),
/// and reports the CPU-vs-CUDA-vs-f64 gap: the sequential CPU fold drifts further
/// from f64 truth than the CUDA tree reduction, so where a borderline top-k argmax
/// flips (as in DeepSeek-V2-Lite token 5) native CUDA is the more-accurate stream.
///
/// `top_k == 1` isolates the pure single-expert two-GEMV body (routing weight is
/// exactly 1.0 under normalize); `top_k == 2` additionally exercises the
/// softmax-weighted expert combination. Symmetric int4, identity activation.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_int4_identity_expert_gemv_within_f64_roundoff() {
    let ep = require_cuda();
    for top_k in [1usize, 2] {
        let case = Case {
            experts: 8,
            rows: 1,
            hidden: 512,
            inter: 512,
            bits: 4,
            top_k,
            activation: "identity",
            swiglu_fusion: 0,
            affine: false,
            fc3: false,
            biases: false,
            normalize: true,
            router_weights: false,
        };
        let inputs = case_inputs(case, DataType::Float32);
        let cpu = run_cpu(case, &inputs);
        let gpu = run_gpu(&ep, case, &inputs, DataType::Float32).unwrap();
        let reference = qmoe_identity_f64_reference(case, &inputs);
        assert_eq!(gpu.len(), reference.len());
        assert_eq!(cpu.len(), reference.len());

        let mut max_gpu_f64 = 0.0f64;
        let mut max_cpu_f64 = 0.0f64;
        for (index, ((&g, &c), &(reference, sum_abs))) in
            gpu.iter().zip(&cpu).zip(&reference).enumerate()
        {
            let gpu_error = (f64::from(g) - reference).abs();
            let cpu_error = (f64::from(c) - reference).abs();
            max_gpu_f64 = max_gpu_f64.max(gpu_error);
            max_cpu_f64 = max_cpu_f64.max(cpu_error);

            // Two chained int4 GEMVs (fc1 -> identity -> fc2), each dequant a
            // product (2 roundings) and each K-reduction a log-depth tree over the
            // hidden/inter lanes. ~(log2(hidden) + log2(inter) + products +
            // scale/route) roundings envelope the fp32 path; keep generous margin
            // for contraction/codegen differences across SMs.
            let tolerance = (sum_abs * f64::from(f32::EPSILON) * 32.0).max(1e-6);
            assert!(
                gpu_error <= tolerance,
                "top_k={top_k} index {index}: CUDA={g}, f64={reference}, error={gpu_error:e}, \
                 roundoff_bound={tolerance:e}, CPU={c}, CPU_error={cpu_error:e}"
            );

            // Sequential CPU fold: up to (hidden + inter) reduction roundings plus
            // the same products; a deliberately loose two-sided envelope.
            let cpu_tolerance = (sum_abs * f64::from(f32::EPSILON) * 200.0).max(1e-6);
            assert!(
                cpu_error <= cpu_tolerance,
                "top_k={top_k} index {index}: CPU={c}, f64={reference}, error={cpu_error:e}, \
                 sequential_bound={cpu_tolerance:e}, CUDA={g}"
            );
        }
        // Absolute regression tripwire complementing the conditioning-scaled bound.
        assert!(
            max_gpu_f64 < 1e-4,
            "top_k={top_k}: CUDA/f64 max_abs_diff={max_gpu_f64:e} exceeds the 1e-4 regression guard"
        );
        eprintln!(
            "QMoE int4 identity expert-GEMV top_k={top_k}: \
             CUDA/f64 max_abs_diff={max_gpu_f64:e}; CPU/f64 max_abs_diff={max_cpu_f64:e}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_64experts_top6_fp16_decode_and_prefill_match_cpu() {
    compare_64expert_decode_and_prefill(DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_64experts_top6_bf16_decode_and_prefill_match_cpu() {
    compare_64expert_decode_and_prefill(DataType::BFloat16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_64experts_top6_handles_empty_and_hot_experts() {
    let ep = require_cuda();
    let case = qmoe_64expert_case(16);
    let mut inputs = case_inputs(case, DataType::Float16);
    inputs[1] = Some(router_with_hot_expert(case));

    let cpu_inputs = rounded_cpu_inputs(&inputs, DataType::Float16);
    let expected = run_cpu(case, &cpu_inputs);
    let actual = run_gpu(&ep, case, &inputs, DataType::Float16).unwrap();
    assert_conforms(&actual, &expected, case, DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_capture_replay_reresolves_changed_router_probs() {
    let ep = require_cuda();
    let case = Case {
        experts: 4,
        rows: 3,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "identity",
        swiglu_fusion: 0,
        affine: false,
        fc3: false,
        biases: false,
        normalize: true,
        router_weights: false,
    };
    let mut capture_inputs = case_inputs(case, DataType::Float32);
    capture_inputs[1] = Some(HostTensor::f32(
        &[case.rows, case.experts],
        &[
            9.0, 8.0, 0.0, -1.0, 9.0, 8.0, 0.0, -1.0, 9.0, 8.0, 0.0, -1.0,
        ],
    ));
    let replay_router = HostTensor::f32(
        &[case.rows, case.experts],
        &[
            -1.0, 0.0, 9.0, 8.0, -1.0, 0.0, 9.0, 8.0, -1.0, 0.0, 9.0, 8.0,
        ],
    );

    let replay = run_gpu_impl(
        &ep,
        case,
        &capture_inputs,
        DataType::Float32,
        None,
        Some(&replay_router),
        true,
    )
    .unwrap();
    let mut eager_inputs = capture_inputs.clone();
    eager_inputs[1] = Some(replay_router);
    let eager = run_gpu_impl(
        &ep,
        case,
        &eager_inputs,
        DataType::Float32,
        None,
        None,
        false,
    )
    .unwrap();

    assert_conforms(&replay, &eager, case, DataType::Float32);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_64experts_top6_capture_replay_reresolves_changed_router_probs() {
    let ep = require_cuda();
    let case = qmoe_64expert_case(8);
    let mut capture_inputs = case_inputs(case, DataType::Float16);
    capture_inputs[1] = Some(router_with_top_experts(case, 0));
    let replay_router = router_with_top_experts(case, 32);

    let replay = run_gpu_impl(
        &ep,
        case,
        &capture_inputs,
        DataType::Float16,
        None,
        Some(&replay_router),
        true,
    )
    .unwrap();
    let mut eager_inputs = capture_inputs;
    eager_inputs[1] = Some(replay_router);
    let eager = run_gpu_impl(
        &ep,
        case,
        &eager_inputs,
        DataType::Float16,
        None,
        None,
        false,
    )
    .unwrap();
    let expected = run_cpu(case, &rounded_cpu_inputs(&eager_inputs, DataType::Float16));

    assert_conforms(&replay, &eager, case, DataType::Float16);
    assert_conforms(&replay, &expected, case, DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_64experts_top6_worst_case_scratch_sizing_matches_cpu() {
    compare(qmoe_64expert_case(64), DataType::Float16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_int8_top1_affine_bias_matches_cpu() {
    compare(
        Case {
            experts: 4,
            rows: 2,
            hidden: 16,
            inter: 16,
            bits: 8,
            top_k: 1,
            activation: "relu",
            swiglu_fusion: 0,
            affine: true,
            fc3: false,
            biases: true,
            normalize: false,
            router_weights: true,
        },
        DataType::Float32,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_single_expert_top1_matches_cpu() {
    compare(
        Case {
            experts: 1,
            rows: 2,
            hidden: 16,
            inter: 16,
            bits: 4,
            top_k: 1,
            activation: "gelu",
            swiglu_fusion: 0,
            affine: true,
            fc3: false,
            biases: true,
            normalize: true,
            router_weights: false,
        },
        DataType::Float32,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_fp16_and_bf16_storage_match_rounded_cpu_reference() {
    let case = Case {
        experts: 4,
        rows: 2,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: false,
        router_weights: false,
    };
    compare(case, DataType::Float16);
    compare(case, DataType::BFloat16);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_prefill_gemm_matches_gemv_and_cpu_oracle() {
    let case = Case {
        experts: 4,
        rows: 6,
        hidden: 16,
        inter: 16,
        bits: 4,
        top_k: 2,
        activation: "silu",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: true,
        router_weights: false,
    };
    compare_gemv_gemm_and_cpu(case);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_prefill_handles_empty_experts_and_all_routes_to_one_expert() {
    let ep = require_cuda();
    let case = Case {
        experts: 4,
        rows: 5,
        hidden: 16,
        inter: 16,
        bits: 8,
        top_k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        affine: true,
        fc3: false,
        biases: true,
        normalize: false,
        router_weights: true,
    };
    let mut inputs = case_inputs(case, DataType::Float32);
    inputs[1] = Some(HostTensor::f32(
        &[case.rows, case.experts],
        &[
            0.0, 1.0, 9.0, -1.0, 1.0, 0.0, 8.0, -1.0, -1.0, 0.0, 7.0, 1.0, 0.0, -1.0, 9.0, 1.0,
            1.0, 0.0, 8.0, -1.0,
        ],
    ));

    let expected = run_cpu(case, &inputs);
    let actual = run_gpu(&ep, case, &inputs, DataType::Float32).unwrap();
    assert_conforms(&actual, &expected, case, DataType::Float32);
}

// ---------------------------------------------------------------------------
// QMoE expert-GEMV microbench + byte-identity A/B (Luv, squad/qmoe-vec).
//
// Realistic DeepSeek-V2-Lite-shaped decode case for profiling `qmoe_linear`
// (hidden=2048, moe_intermediate=1408, 64 experts, top-6). The bench is opt-in
// (set QMOE_BENCH=1) so ordinary `cargo test` runs skip the heavy loop; ncu
// filters the launches with `-k qmoe_linear`.
//
// NOTE (issue #82 baseline cycle): `median_e2e` below is per-iteration cost of
// `run_gpu`, which rebuilds the `Graph`/`Model`, fetches a fresh `Kernel`, and
// re-allocates + re-uploads (H2D) every input tensor -- including the full
// fc1/fc2/fc3 expert weight banks -- on EVERY timed iteration (see
// `run_gpu_impl`). That makes this number dominated by model reconstruction
// and full weight re-upload, not by the expert-GEMV kernel's own achieved
// bandwidth. It is left as-is (still a valid "cost of calling this op from
// scratch every time" measurement) but is NOT a kernel bandwidth probe. See
// `qmoe_expert_gemv_bandwidth_probe` below for one that allocates/uploads
// once and times only repeated `kernel.execute()` calls.
// ---------------------------------------------------------------------------

/// `hidden_size=2048`, `moe_intermediate_size=1408`, `n_routed_experts=64`,
/// `num_experts_per_tok=6` -- `huggingface.co/deepseek-ai/DeepSeek-V2-Lite`
/// `config.json`, confirmed against the upstream file 2026-08-22 (this case
/// predates that citation; the numbers were already correct). `n_shared_experts=2`
/// is a separate always-on dense MLP added to the routed-expert output in the
/// real model and is not part of the QMoE node modeled here.
fn deepseek_v2_lite_decode_case() -> Case {
    Case {
        experts: 64,
        rows: 1,
        hidden: 2048,
        inter: 1408,
        bits: 4,
        top_k: 6,
        activation: "swiglu",
        swiglu_fusion: 0,
        affine: false,
        fc3: true,
        biases: false,
        normalize: true,
        router_weights: true,
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_deepseek_decode_bench() {
    let ep = require_cuda();
    let case = deepseek_v2_lite_decode_case();
    let dtype = DataType::Float16;
    let inputs = case_inputs(case, dtype);
    // Warm the NVRTC cache + capture workspace.
    let _ = run_gpu(&ep, case, &inputs, dtype).unwrap();
    if std::env::var("QMOE_BENCH").is_err() {
        // Under ncu we only need the launches; skip the wall-clock loop.
        let _ = run_gpu(&ep, case, &inputs, dtype).unwrap();
        return;
    }
    let iters = 60usize;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = std::time::Instant::now();
        let _ = run_gpu(&ep, case, &inputs, dtype).unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let flag = std::env::var("ONNX_GENAI_QMOE_OCC").unwrap_or_else(|_| "unset".into());
    eprintln!(
        "QMOE_BENCH deepseek-v2-lite decode: median_e2e={median:.4} ms (iters={iters}, \
         ONNX_GENAI_QMOE_OCC={flag}, dtype={dtype:?})"
    );
}

// ---------------------------------------------------------------------------
// Sound QMoE expert-GEMV/GEMM bandwidth probe (issue #82 baseline).
//
// Unlike `qmoe_deepseek_decode_bench` above, every device allocation and H2D
// upload happens exactly once per (model shape, M) configuration, before any
// timing loop starts (measurement-discipline: "allocate outside the timed
// region"). Only repeated `kernel.execute()` calls are timed, so the number
// this reports is the expert-GEMV/GEMM kernel's own achieved bandwidth, not
// the cost of rebuilding the graph/model or re-uploading expert weights.
//
// It follows the same *mechanical* host-vs-device split used by
// `decode_gemv_achieved_bandwidth_by_projection_shape` in `matmul_nbits.rs`
// (host wall-clock around a batch of raw launches, separately from a batch of
// the same launches bracketed by two CUDA events with one
// `event::synchronize` at the end), but it deliberately does NOT attempt
// `begin_graph_capture`/`replay_graph`. QMoE's capture eligibility is
// conditional on a stateful `warmed` flag set by a prior eager pass
// (`QMoEKernel::capture_support`, `src/kernels/qmoe.rs`) and
// `BlockQuantizedMoEKernel::capture_support` is unconditionally `Unsupported`
// -- relying on capture here would make the very thing this probe reports on
// (fast-path/capture eligibility) also decide whether the timing methodology
// itself is valid. Every sample below is instead `BATCH` raw `execute()`
// launches between two events (the "repeat-launch + CUDA-event/single-sync"
// pattern, not graph replay), which is also the honest cost of the *current*
// uncaptured decode step in production.
//
// IMPORTANT, verified-against-source caveat on what "host_us" actually means
// here: unlike `MatMulNBitsKernel::run`, `QMoEKernel::execute` ends with
// `let result = if capturing { Ok(()) } else { self.runtime.synchronize() };`
// (`src/kernels/qmoe.rs`, end of `execute`) -- i.e. it performs a *full,
// unconditional, blocking device synchronize inside every single call* when
// not capturing, for both the M=1 fused-decode and the M>1 grouped path
// alike. That means the host loop below can never race ahead of the device:
// every `execute_once()` call is already a synchronous host<->device round
// trip by construction, so `median_host_us` and `median_gpu_us` measure
// nearly the same serialized per-call cost rather than "host dispatch"
// isolated from "device execution" the way they do for `MatMulNBitsKernel`.
// Measured on an idle A100: at M=1 the two numbers diverge a lot (host_us
// ~25us vs. gpu_us 145-430us) because the fused-decode kernel itself is fast
// enough that fixed per-call CPU-side bookkeeping (view construction, kernel
// dispatch) is comparatively small; at M>1 they converge to within 7-38% of
// each other and the gap shrinks further as M grows (e.g. GLM-5.2 M=8:
// host_us=6652 vs gpu_us=7154, 93%). This is *not* evidence of a grouped-path-
// specific extra sync -- both paths hit the identical `execute()` sync -- it
// is evidence that, absent capture, QMoE pays a full host round trip on every
// decode step, and that round trip's relative weight only becomes visible
// once the per-call GPU work is large enough (grouped path) that CPU-side
// bookkeeping is no longer the dominant term to notice separately. Any future
// attempt to hide this cost behind pipelining (without graph capture) would
// first need to remove this unconditional `synchronize()`, which is there
// (per the surrounding code) to make the pooled scratch buffers
// (`self.scratch`) safe to reuse across calls without capture.
// ---------------------------------------------------------------------------

/// A100-SXM4-80GB HBM2e datasheet peak. Percent-of-peak below is only
/// meaningful on this device; see the identical constant/comment in
/// `decode_gemv_achieved_bandwidth_by_projection_shape` (`matmul_nbits.rs`).
const A100_SXM4_80GB_PEAK_GBPS: f64 = 2039.0;

/// A real target model's routed-expert MoE shape. Every numeric field is read
/// verbatim from the model's published `config.json` -- none are invented
/// (`measurement-discipline`: shapes must come from a cited real config).
#[derive(Clone, Copy, Debug)]
struct MoeModelShape {
    name: &'static str,
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
}

/// `hidden_size=2048`, `moe_intermediate_size=1408`, `n_routed_experts=64`,
/// `num_experts_per_tok=6`, `hidden_act=silu` --
/// `huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json`, fetched
/// 2026-08-22. `n_shared_experts=2` is a separate always-on dense MLP added to
/// the routed-expert output and is out of scope for the QMoE node measured
/// here (matches `deepseek_v2_lite_decode_case` above).
const DEEPSEEK_V2_LITE_MOE: MoeModelShape = MoeModelShape {
    name: "deepseek-v2-lite",
    hidden: 2048,
    inter: 1408,
    experts: 64,
    top_k: 6,
};

/// `hidden_size=6144`, `moe_intermediate_size=2048`, `n_routed_experts=256`,
/// `num_experts_per_tok=8`, `hidden_act=silu`, `model_type=glm_moe_dsa` --
/// `huggingface.co/zai-org/GLM-5.2/raw/main/config.json`, fetched 2026-08-22.
/// `n_shared_experts=1` is likewise a separate dense path, out of scope here.
/// `glm_moe_dsa` matches the architecture family already named in this repo's
/// `tests/fixtures/tiny-glm52-qmoe-indexshare/manifest.json`.
const GLM_5_2_MOE: MoeModelShape = MoeModelShape {
    name: "glm-5.2",
    hidden: 6144,
    inter: 2048,
    experts: 256,
    top_k: 8,
};

/// Builds the `Case` for one (real model shape, decode batch `rows`) point on
/// the probe's matrix. `bits: 4`, `block_size: 16` (hardcoded by `model_node`)
/// and `affine: false` are this repo's existing QMoE test/deployment
/// convention for these two models (matches `deepseek_v2_lite_decode_case`),
/// not a field either upstream `config.json` specifies -- neither config names
/// a serving quantization scheme, so this is a stated methodology choice, not
/// a config citation.
fn moe_bench_case(shape: MoeModelShape, rows: usize) -> Case {
    Case {
        experts: shape.experts,
        rows,
        hidden: shape.hidden,
        inter: shape.inter,
        bits: 4,
        top_k: shape.top_k,
        activation: "swiglu",
        swiglu_fusion: 0,
        affine: false,
        fc3: true,
        biases: false,
        normalize: true,
        router_weights: true,
    }
}

/// A reduced-expert-count proxy of `shape`'s bench case, used ONLY to prove
/// numerical correctness against the CPU oracle at a tractable `quantize()`
/// cost: `quantize()`'s per-element dequant construction at GLM-5.2's real
/// 256-expert scale takes upward of ten CPU-minutes per projection (measured
/// on this box), which is fine to pay once but not four times (one per M) on
/// top of the bandwidth sweep itself. Expert COUNT does not change any
/// per-expert GEMV/GEMM dequantization or matmul numerics -- each expert's
/// weights are independent -- so a correctness pass at a smaller expert count
/// is evidence for the same kernel code path the bandwidth run below
/// exercises, while still being large enough (`top_k + 2`) that the M > 1
/// grouped/gather-scatter path has at least one non-selected expert to skip.
/// hidden, inter, top_k, bits, and activation are identical to the real case.
fn correctness_proxy_case(shape: MoeModelShape, rows: usize) -> Case {
    let mut case = moe_bench_case(shape, rows);
    case.experts = (shape.top_k + 2).max(8).min(shape.experts);
    case
}

/// Cheap LCG fill -- same generator as
/// `decode_gemv_achieved_bandwidth_by_projection_shape` in `matmul_nbits.rs`
/// uses for the same reason: bandwidth does not depend on weight VALUES, only
/// on byte traffic and access pattern.
fn fast_fill_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        })
        .collect()
}

/// Byte-shape-identical to `quantize()`'s output (same packed/scale array
/// sizes for the given `experts`/`out_features`/`in_features`/`bits`/
/// `block_size`) but filled with [`fast_fill_bytes`] instead of run through
/// `quantize()`'s per-element dequant construction -- seconds instead of
/// minutes at GLM-5.2's full 256-expert scale. Scales are a fixed small
/// constant rather than LCG-filled bits, so no NaN/Inf/denormal value can
/// reach the GEMV and skew its timing; the packed weight nibbles are safe to
/// fill arbitrarily because they are just integer codes dequantized by that
/// scale, never interpreted as floats directly. No zero points: every case
/// this probe builds has `affine: false`.
fn fast_fill_quantized(
    experts: usize,
    out_features: usize,
    in_features: usize,
    bits: usize,
    block_size: usize,
    seed: u64,
) -> Quantized {
    let pack_size = 8 / bits;
    let packed_in = in_features / pack_size;
    let blocks = in_features / block_size;
    let packed = fast_fill_bytes(experts * out_features * packed_in, seed);
    let scales = vec![0.02f32; experts * out_features * blocks];
    Quantized {
        packed: HostTensor::u8(&[experts, out_features, packed_in], packed),
        scales: HostTensor::f32(&[experts, out_features, blocks], &scales),
        zero_points: None,
    }
}

/// Same 15-input layout as `case_inputs` (matching `model_node`'s attribute
/// contract exactly) but expert weight banks come from
/// [`fast_fill_quantized`] instead of `quantize()`. Activation/router/
/// aggregation are cheap regardless of expert count, so they keep
/// `case_inputs`'s real deterministic formulas -- only the weight banks,
/// which are what makes GLM-5.2's real scale intractable, are fast-filled.
/// Requires `!case.affine && !case.biases`, which every case
/// [`moe_bench_case`] builds satisfies.
fn fast_case_inputs(case: Case, dtype: DataType) -> Vec<Option<HostTensor>> {
    assert!(!case.affine, "fast_case_inputs has no zero-point fill path");
    assert!(!case.biases, "fast_case_inputs has no bias fill path");
    const BLOCK_SIZE: usize = 16;
    let fc1_size = if case.activation == "swiglu" && case.swiglu_fusion != 0 {
        case.inter * 2
    } else {
        case.inter
    };
    let x: Vec<f32> = (0..case.rows * case.hidden)
        .map(|index| ((index * 19 + 3) % 29) as f32 / 13.0 - 1.0)
        .collect();
    let router: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| ((index * 7 + 5) % 17) as f32 / 4.0 - 2.0)
        .collect();
    let aggregation: Vec<f32> = (0..case.rows * case.experts)
        .map(|index| 0.1 + ((index * 5 + 2) % 11) as f32 / 10.0)
        .collect();
    let fc1 = fast_fill_quantized(
        case.experts,
        fc1_size,
        case.hidden,
        case.bits,
        BLOCK_SIZE,
        1,
    );
    let fc2 = fast_fill_quantized(
        case.experts,
        case.hidden,
        case.inter,
        case.bits,
        BLOCK_SIZE,
        2,
    );
    let fc3 = case.fc3.then(|| {
        fast_fill_quantized(
            case.experts,
            case.inter,
            case.hidden,
            case.bits,
            BLOCK_SIZE,
            3,
        )
    });
    vec![
        Some(HostTensor::activation(dtype, &[case.rows, case.hidden], &x)),
        Some(HostTensor::f32(&[case.rows, case.experts], &router)),
        Some(fc1.packed),
        Some(fc1.scales),
        None,
        Some(fc2.packed),
        Some(fc2.scales),
        None,
        fc3.as_ref().map(|weights| weights.packed.clone()),
        fc3.as_ref().map(|weights| weights.scales.clone()),
        None,
        None,
        None,
        None,
        case.router_weights
            .then(|| HostTensor::f32(&[case.rows, case.experts], &aggregation)),
    ]
}

/// Packed weight + fp32-scale bytes for one expert's `[out_features,
/// in_features]` projection, no zero points (`affine: false` here, matching
/// `quantize()`'s symmetric path). Scales are fp32 -- not fp16 -- because
/// `qmoe.rs`'s CUDA kernels declare `const float* fc1_scales` /
/// `fc2_scales` / `fc3_scales` and `case_inputs` uploads `HostTensor::f32`
/// scale tensors accordingly; this is why the byte accounting here is
/// re-derived from this file's own `quantize()` layout instead of reusing the
/// int4/fp16-scale formula in `cuda-perf-measurement/SKILL.md` verbatim (that
/// formula was written for `MatMulNBits`, whose scales really are fp16).
fn expert_projection_bytes(
    out_features: usize,
    in_features: usize,
    bits: usize,
    block_size: usize,
) -> u64 {
    let pack_size = 8 / bits;
    let packed_in = in_features / pack_size;
    let blocks = in_features / block_size;
    (out_features * packed_in) as u64 + (out_features * blocks * 4) as u64
}

/// Total fc1 (gate) + fc3 (up) + fc2 (down) bytes for ONE expert under this
/// probe's cases (`fc3: true`, unfused SwiGLU, so fc1/fc3 share the
/// `[inter, hidden]` shape and fc2 is `[hidden, inter]`).
fn expert_bytes(case: Case) -> u64 {
    const BLOCK_SIZE: usize = 16; // hardcoded by `model_node`'s `block_size` attribute
    let gate = expert_projection_bytes(case.inter, case.hidden, case.bits, BLOCK_SIZE);
    let up = expert_projection_bytes(case.inter, case.hidden, case.bits, BLOCK_SIZE);
    let down = expert_projection_bytes(case.hidden, case.inter, case.bits, BLOCK_SIZE);
    gate + up + down
}

/// Replicates the kernel's top-k tie-break (`qmoe_route`'s
/// `total_order_key`: descending value, ascending index on ties) on the host,
/// against the actual `router_probs` this call uploads, so the "how many
/// distinct experts does this call actually touch" count used for the
/// bandwidth roofline is derived from the exact routing decision the kernel
/// makes, rather than assumed.
fn top_k_distinct_experts(case: Case, router: &[f32]) -> std::collections::BTreeSet<usize> {
    let mut touched = std::collections::BTreeSet::new();
    for row in 0..case.rows {
        let row_router = &router[row * case.experts..(row + 1) * case.experts];
        let mut order: Vec<usize> = (0..case.experts).collect();
        order.sort_by(|&a, &b| {
            row_router[b]
                .partial_cmp(&row_router[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        touched.extend(order.into_iter().take(case.top_k));
    }
    touched
}

/// Boolean mirror of `assert_fused_gate_up_decode_case` above (same three
/// conditions: `rows == 1`, `rows * top_k <= 16`, and an fc3/activation/fusion
/// combination the `qmoe_gate_up_activate` launch gate accepts), so this probe
/// can report which dispatch path (fused decode vs. grouped/gather-scatter)
/// is structurally expected at each M without panicking for M > 1 -- this
/// reads the same public `Case` fields `qmoe.rs`'s `fused_gate_up_decode`
/// eligibility check reads; it does not add or change any kernel-internal
/// instrumentation (no paging/residency/kernel changes in this PR).
fn is_fused_decode_eligible(case: Case) -> bool {
    case.rows == 1
        && case.rows * case.top_k <= 16
        && ((case.fc3 && matches!(case.activation, "silu" | "swiglu"))
            || (!case.fc3 && case.activation == "swiglu" && case.swiglu_fusion != 0))
}

/// Persistent GPU state for one (shape, M) point: built and uploaded exactly
/// once by [`setup_gemv_bench`], then reused for every timed `execute()`
/// call. No *device* allocation, H2D copy, or `Graph`/`Kernel` construction
/// happens after `setup_gemv_bench` returns -- `execute_once` still builds a
/// small host-side `Vec<TensorView>` per call (via `views`), which is
/// intentional: that bookkeeping cost is part of what `median_us`'s
/// `host_us` leg is measuring, not something this harness hides.
struct GemvBenchSetup {
    kernel: Box<dyn onnx_runtime_ep_api::Kernel>,
    buffers: Vec<Option<DeviceBuffer>>,
    input_shapes: Vec<Vec<usize>>,
    input_strides: Vec<Vec<i64>>,
    input_dtypes: Vec<DataType>,
    input_ptrs: Vec<Option<DevicePtr>>,
    output_buffer: DeviceBuffer,
    output_ptr: DevicePtrMut,
    output_shape: [usize; 2],
    output_strides: Vec<i64>,
    output_bytes: usize,
    dtype: DataType,
    device_id: DeviceId,
}

impl GemvBenchSetup {
    fn views(&self) -> Vec<TensorView<'_>> {
        self.input_shapes
            .iter()
            .zip(&self.input_strides)
            .zip(&self.input_ptrs)
            .enumerate()
            .map(|(index, ((shape, strides), ptr))| match ptr {
                Some(ptr) => TensorView::new(
                    *ptr,
                    self.input_dtypes[index],
                    shape,
                    strides,
                    self.device_id,
                ),
                None => TensorView::absent(self.input_dtypes[index]),
            })
            .collect()
    }

    fn output_view(&self) -> TensorMut<'_> {
        TensorMut::new(
            self.output_ptr,
            self.dtype,
            &self.output_shape,
            &self.output_strides,
            self.device_id,
        )
    }

    fn execute_once(&self) -> onnx_runtime_ep_api::Result<()> {
        self.kernel
            .execute(&self.views(), &mut [self.output_view()])
    }

    /// Reads the output back and frees every buffer. Consumes `self` so a
    /// torn-down setup cannot be executed again.
    fn teardown(self, ep: &CudaExecutionProvider) -> onnx_runtime_ep_api::Result<Vec<u8>> {
        let runtime = ep.runtime();
        runtime.synchronize()?;
        let mut bytes = vec![0u8; self.output_bytes];
        // SAFETY: `output_buffer` was sized `output_bytes` when allocated.
        unsafe { runtime.dtoh(&mut bytes, cuptr(self.output_buffer.as_ptr()))? };
        for buffer in self.buffers.into_iter().flatten() {
            ep.deallocate(buffer)?;
        }
        ep.deallocate(self.output_buffer)?;
        Ok(bytes)
    }
}

fn decode_output_bytes(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported output dtype {other:?}"),
    }
}

fn setup_gemv_bench(
    ep: &CudaExecutionProvider,
    case: Case,
    dtype: DataType,
    inputs: &[Option<HostTensor>],
) -> onnx_runtime_ep_api::Result<GemvBenchSetup> {
    let output_shape = [case.rows, case.hidden];
    let (graph, node) = model_node(inputs, dtype, &output_shape, case);
    let model = Model::new(&graph);
    let concrete_shapes: Vec<_> = inputs
        .iter()
        .filter_map(|input| input.as_ref().map(|input| input.shape.clone()))
        .collect();
    let kernel = ep.get_kernel(model.graph.node(node), &concrete_shapes, 1)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::<Option<DeviceBuffer>>::new();
    for input in inputs {
        if let Some(input) = input {
            let buffer = ep.allocate(input.bytes.len(), 256)?;
            // SAFETY: allocation size equals the source tensor byte length.
            unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
            buffers.push(Some(buffer));
        } else {
            buffers.push(None);
        }
    }
    let input_shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| input.shape.clone())
                .unwrap_or_default()
        })
        .collect();
    let input_strides: Vec<Vec<i64>> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
                .unwrap_or_default()
        })
        .collect();
    let input_dtypes: Vec<DataType> = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input
                .as_ref()
                .map(|input| input.dtype)
                .unwrap_or_else(|| absent_dtype(index, dtype))
        })
        .collect();
    let input_ptrs: Vec<Option<DevicePtr>> = buffers
        .iter()
        .map(|buffer| buffer.as_ref().map(|buffer| DevicePtr(buffer.as_ptr())))
        .collect();

    let output_bytes = case.rows * case.hidden * dtype.byte_size();
    let mut output_buffer = ep.allocate(output_bytes, 256)?;
    let output_ptr = DevicePtrMut(output_buffer.as_mut_ptr());
    let output_strides = compute_contiguous_strides(&output_shape);

    let setup = GemvBenchSetup {
        kernel,
        buffers,
        input_shapes,
        input_strides,
        input_dtypes,
        input_ptrs,
        output_buffer,
        output_ptr,
        output_shape,
        output_strides,
        output_bytes,
        dtype,
        device_id: ep.device_id(),
    };
    // First call also compiles the NVRTC module and (for the fused decode
    // path) sets QMoE's `warmed` flag; never time it.
    setup.execute_once()?;
    runtime.synchronize()?;
    Ok(setup)
}

/// Times one sample: `batch` raw `execute()` calls bracketed by two CUDA
/// events (single `event::synchronize` + `elapsed()`, divided by `batch`),
/// and a separate wall-clock enqueue loop (synchronized only at its end).
/// No graph capture is attempted (see the module doc above). Because
/// `QMoEKernel::execute` itself blocks on `runtime.synchronize()` internally
/// whenever it is not capturing (verified in `src/kernels/qmoe.rs`), the two
/// numbers below are NOT a clean host-dispatch-vs-device-execution split the
/// way they are for `MatMulNBitsKernel` -- both legs measure a serialized
/// per-call round trip, and their ratio mainly reflects how large a fraction
/// of that round trip is fixed CPU-side bookkeeping vs. actual device work.
/// Returns `(median_gpu_us, median_host_us, sorted_gpu_samples_us)`.
fn median_us(
    setup: &GemvBenchSetup,
    runtime: &CudaRuntime,
    reps: usize,
    batch: usize,
) -> (f64, f64, Vec<f64>) {
    use cudarc::driver::result::event;
    use cudarc::driver::sys::CUevent_flags;

    // Host dispatch cost first, on the only path this probe has (uncaptured).
    let mut host = Vec::with_capacity(reps);
    for _ in 0..reps {
        let enqueue_begin = std::time::Instant::now();
        for _ in 0..batch {
            setup.execute_once().unwrap();
        }
        host.push(enqueue_begin.elapsed().as_secs_f64() * 1e6 / batch as f64);
        runtime.synchronize().unwrap();
    }
    host.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let host_us = host[host.len() / 2];

    let mut gpu = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
        let end = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
        // SAFETY: both events belong to this context and bracket `batch`
        // back-to-back `execute()` launches on the runtime's own stream.
        unsafe {
            event::record(start, runtime.stream_ptr()).unwrap();
            for _ in 0..batch {
                setup.execute_once().unwrap();
            }
            event::record(end, runtime.stream_ptr()).unwrap();
            event::synchronize(end).unwrap();
            gpu.push(event::elapsed(start, end).unwrap() as f64 / batch as f64 * 1000.0);
            event::destroy(start).ok();
            event::destroy(end).ok();
        }
    }
    gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = gpu[gpu.len() / 2];
    (median, host_us, gpu)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_expert_gemv_bandwidth_probe() {
    let ep = require_cuda();
    let runtime = ep.runtime().clone();
    let dtype = DataType::Float16;
    // 25 reps, not 9: a follow-up local reproducibility sweep (3x reps=9 runs,
    // then one reps=25 run, all on the same idle A100) showed
    // `deepseek-v2-lite M=2` and `glm-5.2 M=1` occasionally landing their
    // reps=9 median in a slower tail (e.g. 822us vs. a typical 673us; 520us
    // vs. a typical 430us) purely from small-sample luck -- at reps=25 both
    // medians settled tightly onto the typical/fast value every time, with
    // only a narrow, rare excursion visible in `range_us`'s upper bound. This
    // is small-sample median instability, not genuine 50/50 bimodal kernel
    // behavior; raising the default trades ~15% more wall-clock time for a
    // materially more trustworthy median on every configuration.
    let reps = env_usize("QMOE_GEMV_PROBE_REPS", 25).max(5);
    let batch = env_usize("QMOE_GEMV_PROBE_BATCH", 16).max(1);

    // Bring the SM clock up BEFORE measuring anything (cuda-perf-measurement
    // Trap 5): an idle A100 in this environment can sit far below its rated
    // clock with persistence mode off, and a probe that starts timing
    // immediately reports whatever partial ramp it happened to catch. Ramp on
    // the heaviest configuration (GLM-5.2, M=8) so the device is loaded the
    // way the largest probed decode step loads it, until the timing itself
    // stops improving AND at least 8s of continuous work has elapsed.
    let ramp_case = moe_bench_case(GLM_5_2_MOE, 8);
    let ramp_inputs = fast_case_inputs(ramp_case, dtype);
    let ramp_setup = setup_gemv_bench(&ep, ramp_case, dtype, &ramp_inputs).unwrap();
    let ramp_start = std::time::Instant::now();
    let ramp_deadline = ramp_start + std::time::Duration::from_secs(30);
    let mut previous = f64::INFINITY;
    let mut ramp_trace = Vec::new();
    loop {
        let (now, _, _) = median_us(&ramp_setup, &runtime, 5, batch);
        ramp_trace.push(now);
        let settled = now > previous * 0.985;
        previous = now;
        if settled && ramp_start.elapsed() >= std::time::Duration::from_secs(8) {
            break;
        }
        if std::time::Instant::now() >= ramp_deadline {
            eprintln!(
                "WARNING: clock had not settled after 30s of ramping; absolute numbers \
                 below are not comparable across runs"
            );
            break;
        }
    }
    let ramp_best = ramp_trace.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "clock ramp on glm-5.2 M=8: {:.0} -> {:.0} us over {} readings in {:.1}s (best {:.0} us, \
         {:.0}% off the first)",
        ramp_trace[0],
        ramp_trace[ramp_trace.len() - 1],
        ramp_trace.len(),
        ramp_start.elapsed().as_secs_f64(),
        ramp_best,
        100.0 * (ramp_trace[0] - ramp_best) / ramp_trace[0]
    );
    let (first_before, _, _) = median_us(&ramp_setup, &runtime, reps, batch);
    ramp_setup.teardown(&ep).unwrap();

    println!(
        "{:<16} {:>2} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "shape",
        "M",
        "dispatch",
        "correct",
        "median_us",
        "host_us",
        "distinct",
        "routes",
        "GB/s",
        "%peak(dedup)"
    );

    for shape in [DEEPSEEK_V2_LITE_MOE, GLM_5_2_MOE] {
        for &rows in &[1usize, 2, 4, 8] {
            // --- Correctness gate, at a reduced expert count (see
            // `correctness_proxy_case`), through the REAL `quantize()` +
            // CPU-oracle path. A fast wrong kernel must not pass, and this
            // must happen before any timing is trusted. ---
            let proxy_case = correctness_proxy_case(shape, rows);
            let proxy_inputs = case_inputs(proxy_case, dtype);
            let proxy_cpu_inputs = rounded_cpu_inputs(&proxy_inputs, dtype);
            let expected = run_cpu(proxy_case, &proxy_cpu_inputs);
            let proxy_setup = setup_gemv_bench(&ep, proxy_case, dtype, &proxy_inputs).unwrap();
            let proxy_bytes = {
                let runtime = ep.runtime();
                runtime.synchronize().unwrap();
                let mut bytes = vec![0u8; proxy_setup.output_bytes];
                // SAFETY: `output_buffer` was sized `output_bytes`.
                unsafe {
                    runtime
                        .dtoh(&mut bytes, cuptr(proxy_setup.output_buffer.as_ptr()))
                        .unwrap()
                };
                bytes
            };
            let actual = decode_output_bytes(&proxy_bytes, dtype);
            assert_conforms(&actual, &expected, proxy_case, dtype);
            proxy_setup.teardown(&ep).unwrap();

            // --- Bandwidth measurement, at the REAL model expert count, with
            // fast-filled weights (see `fast_case_inputs` for why: values do
            // not matter for a bandwidth number, and `quantize()` at GLM-5.2's
            // real 256-expert scale is not tractable to pay 4x per shape). ---
            let case = moe_bench_case(shape, rows);
            let inputs = fast_case_inputs(case, dtype);
            let setup = setup_gemv_bench(&ep, case, dtype, &inputs).unwrap();

            let router = host_f32(inputs[1].as_ref().unwrap());
            let distinct = top_k_distinct_experts(case, &router);
            let total_routes = case.rows * case.top_k;
            let dispatch = if is_fused_decode_eligible(case) {
                "fused_decode"
            } else {
                "grouped"
            };

            let (median_gpu_us, host_us, gpu_samples) = median_us(&setup, &runtime, reps, batch);
            let per_expert_bytes = expert_bytes(case) as f64;
            let dedup_bytes = per_expert_bytes * distinct.len() as f64;
            let no_dedup_bytes = per_expert_bytes * total_routes as f64;
            let gbps_dedup = dedup_bytes / (median_gpu_us * 1e-6) / 1e9;
            let gbps_no_dedup = no_dedup_bytes / (median_gpu_us * 1e-6) / 1e9;
            let pct_dedup = 100.0 * gbps_dedup / A100_SXM4_80GB_PEAK_GBPS;
            let pct_no_dedup = 100.0 * gbps_no_dedup / A100_SXM4_80GB_PEAK_GBPS;
            let range_us = (
                gpu_samples.first().copied().unwrap_or(median_gpu_us),
                gpu_samples.last().copied().unwrap_or(median_gpu_us),
            );

            println!(
                "{:<16} {:>2} {:>10} {:>10} {:>10.2} {:>9.2} {:>9} {:>9} {:>9.0} {:>11.1}%",
                shape.name,
                rows,
                dispatch,
                "pass",
                median_gpu_us,
                host_us,
                distinct.len(),
                total_routes,
                gbps_dedup,
                pct_dedup
            );
            // Stable machine-readable line (issue #82 baseline):
            // {model_shape, M, iterations, median_us, achieved_GBps, pct_of_theoretical_memory_bw}.
            // Two bandwidth hypotheses are reported because the grouped path's
            // actual dedup behaviour is exactly what this probe is trying to
            // establish evidence for, not assume; if `pct_of_theoretical_memory_bw`
            // under the dedup hypothesis exceeds 100%, that falsifies the dedup
            // assumption for this configuration (a fast KERNEL cannot exceed the
            // hardware's bandwidth; an inflated PERCENTAGE means the byte count
            // fed into it was too small).
            println!(
                "QMOE_GEMV_BW model_shape={} M={} iterations={} median_us={:.3} \
                 achieved_GBps_dedup={:.1} pct_of_theoretical_memory_bw_dedup={:.2} \
                 achieved_GBps_no_dedup={:.1} pct_of_theoretical_memory_bw_no_dedup={:.2} \
                 host_us={:.3} range_us=[{:.3},{:.3}] distinct_experts={} total_routes={} \
                 dispatch={} correctness=pass",
                shape.name,
                rows,
                reps * batch,
                median_gpu_us,
                gbps_dedup,
                pct_dedup,
                gbps_no_dedup,
                pct_no_dedup,
                host_us,
                range_us.0,
                range_us.1,
                distinct.len(),
                total_routes,
                dispatch,
            );
            if pct_dedup > 100.0 {
                eprintln!(
                    "WARNING: {} M={} achieved {:.1}% of peak under the dedup byte hypothesis; \
                     an impossible result means the grouped path is reading fewer bytes than \
                     `distinct_experts * per_expert_bytes` assumes (e.g. re-using a cached read),\
                     not that the kernel exceeds hardware bandwidth",
                    shape.name, rows, pct_dedup
                );
            }

            setup.teardown(&ep).unwrap();
        }
    }

    // Re-measure the ramp configuration last. If the device drifted (clock
    // throttle, a neighbour starting work) the sweep above compared
    // configurations measured under different conditions.
    let ramp_inputs_after = fast_case_inputs(ramp_case, dtype);
    let ramp_setup_after = setup_gemv_bench(&ep, ramp_case, dtype, &ramp_inputs_after).unwrap();
    let (first_after, _, _) = median_us(&ramp_setup_after, &runtime, reps, batch);
    ramp_setup_after.teardown(&ep).unwrap();
    let drift = (first_after - first_before).abs() / first_before;
    if drift > 0.03 {
        eprintln!(
            "WARNING: glm-5.2 M=8 drifted {:.1}% across the sweep ({:.1} -> {:.1} us); the \
             device was not stable and these rows are not comparable",
            100.0 * drift,
            first_before,
            first_after
        );
    } else {
        println!("\ndevice stable across sweep: {:.1}% drift", 100.0 * drift);
    }

    assert!(
        first_before.is_finite() && first_before > 0.0,
        "QMoE GEMV bandwidth probe produced no usable timing"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn qmoe_occ_is_bit_identical() {
    // Byte-identity gate for the `ONNX_GENAI_QMOE_OCC` expert-GEMV variant:
    // the `_occ` (`__launch_bounds__`) path must reproduce the default path
    // bit-for-bit (0 differing bits) across dtypes and shapes.
    //
    // Coverage note: `_occ` only rebinds the fused gate/up SwiGLU expert GEMV,
    // which is a decode-only (rows==1, routes<=16) launch. The rows>1 arm here
    // routes through the grouped prefill path where the flag is a no-op, so it
    // is an unchanged-output guard rather than genuine `_occ` coverage; rows==1
    // is what actually exercises the `_occ` kernel.
    let ep = require_cuda();
    for &rows in &[1usize, 4, 6, 8] {
        let mut case = deepseek_v2_lite_decode_case();
        case.rows = rows;
        for &dtype in &[DataType::Float16, DataType::BFloat16, DataType::Float32] {
            let inputs = case_inputs(case, dtype);
            // SAFETY: single-threaded test; the flag only selects a kernel entry.
            unsafe { std::env::set_var("ONNX_GENAI_QMOE_OCC", "0") };
            let base = run_gpu(&ep, case, &inputs, dtype).unwrap();
            unsafe { std::env::set_var("ONNX_GENAI_QMOE_OCC", "1") };
            let vec_path = run_gpu(&ep, case, &inputs, dtype).unwrap();
            unsafe { std::env::remove_var("ONNX_GENAI_QMOE_OCC") };
            assert_eq!(base.len(), vec_path.len());
            let mismatches = base
                .iter()
                .zip(&vec_path)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "ONNX_GENAI_QMOE_OCC diverged from default: {mismatches} of {} elements \
                 differ (rows={rows}, dtype={dtype:?})",
                base.len()
            );
        }
    }
}
