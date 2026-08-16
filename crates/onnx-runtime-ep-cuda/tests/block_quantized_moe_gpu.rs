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
//! GPU parity regressions for the frozen `pkg.nxrt::BlockQuantizedMoE` v1 op.
//!
//! Every case builds small synthetic block-quantized MoE tensors (mxfp4-packed
//! expert weights), runs the CPU reference kernel as the parity oracle, and
//! asserts the CUDA `BlockQuantizedMoE` kernel reproduces it within tolerance.
//! The suite covers multiple experts, top-k routing (`k=1` and `k>1`), a single
//! expert, optional biases, router-weight aggregation, and the relu/gelu/silu/
//! identity/swiglu activation paths (fused and unfused). CPU-only CI reports
//! these as ignored unless `gpu-tests` is enabled.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMetadata,
    TensorMut, TensorView, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "pkg.nxrt";
const FORMAT: &str = "mxfp4";
const QK: usize = 32;
const BLOCK_BYTES: usize = 17;

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
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

fn random_u32(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

fn uniform(state: &mut u64, low: f32, high: f32) -> f32 {
    low + (random_u32(state) as f32 / u32::MAX as f32) * (high - low)
}

/// mxfp4-packs one `[experts, out_features, ceil(in_features/32), 17]` projection.
///
/// Each 17-byte block stores an E8M0 scale exponent in byte 0 followed by 16
/// bytes of packed fp4 codes. The exponent is kept in a tame range so the
/// two-layer expert math stays well within f32 dynamic range.
fn pack_projection(
    state: &mut u64,
    experts: usize,
    out_features: usize,
    in_features: usize,
) -> HostTensor {
    let blocks = in_features.div_ceil(QK);
    let mut packed = vec![0u8; experts * out_features * blocks * BLOCK_BYTES];
    for block in packed.chunks_exact_mut(BLOCK_BYTES) {
        block[0] = 122 + (random_u32(state) % 4) as u8;
        for byte in &mut block[1..] {
            *byte = random_u32(state) as u8;
        }
    }
    HostTensor::u8(&[experts, out_features, blocks, BLOCK_BYTES], packed)
}

struct Config {
    rows: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    k: usize,
    activation: &'static str,
    swiglu_fusion: usize,
    with_bias: bool,
    with_router_weights: bool,
    normalize: bool,
}

impl Config {
    fn needs_fc3(&self) -> bool {
        self.activation == "swiglu" && self.swiglu_fusion == 0
    }

    fn fc1_size(&self) -> usize {
        if self.activation == "swiglu" && self.swiglu_fusion != 0 {
            self.inter * 2
        } else {
            self.inter
        }
    }
}

/// Builds the positional input list (`Vec<Option<HostTensor>>`, length 6..=9)
/// for a MoE case. Omitted optional inputs are `None`.
fn build_inputs(config: &Config, seed: u64) -> Vec<Option<HostTensor>> {
    let mut state = seed;
    let Config {
        rows,
        hidden,
        inter,
        experts,
        ..
    } = *config;
    let fc1_size = config.fc1_size();

    let input: Vec<f32> = (0..rows * hidden)
        .map(|_| uniform(&mut state, -1.0, 1.0))
        .collect();
    let router_logits: Vec<f32> = (0..rows * experts)
        .map(|_| uniform(&mut state, -2.5, 2.5))
        .collect();

    let fc1 = pack_projection(&mut state, experts, fc1_size, hidden);
    let fc2 = pack_projection(&mut state, experts, hidden, inter);

    let mut inputs: Vec<Option<HostTensor>> = vec![
        Some(HostTensor::f32(&[rows, hidden], &input)),
        Some(HostTensor::f32(&[rows, experts], &router_logits)),
        Some(fc1),
        None,
        Some(fc2),
        None,
    ];

    if config.with_bias {
        let fc1_bias: Vec<f32> = (0..experts * fc1_size)
            .map(|_| uniform(&mut state, -0.2, 0.2))
            .collect();
        let fc2_bias: Vec<f32> = (0..experts * hidden)
            .map(|_| uniform(&mut state, -0.2, 0.2))
            .collect();
        inputs[3] = Some(HostTensor::f32(&[experts, fc1_size], &fc1_bias));
        inputs[5] = Some(HostTensor::f32(&[experts, hidden], &fc2_bias));
    }

    if config.needs_fc3() {
        let fc3 = pack_projection(&mut state, experts, inter, hidden);
        inputs.push(Some(fc3));
        if config.with_bias {
            let fc3_bias: Vec<f32> = (0..experts * inter)
                .map(|_| uniform(&mut state, -0.2, 0.2))
                .collect();
            inputs.push(Some(HostTensor::f32(&[experts, inter], &fc3_bias)));
        } else {
            inputs.push(None);
        }
    }

    if config.with_router_weights {
        while inputs.len() < 8 {
            inputs.push(None);
        }
        let router_weights: Vec<f32> = (0..rows * experts)
            .map(|_| uniform(&mut state, 0.0, 1.0))
            .collect();
        inputs.push(Some(HostTensor::f32(&[rows, experts], &router_weights)));
    }

    inputs
}

fn absent_dtype(index: usize) -> DataType {
    match index {
        2 | 4 | 6 => DataType::Uint8,
        _ => DataType::Float32,
    }
}

fn model_node(config: &Config, inputs: &[Option<HostTensor>]) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(DOMAIN.into(), 1);
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
        DataType::Float32,
        static_shape([config.rows, config.hidden]),
    );
    let mut node = Node::new(NodeId(0), "BlockQuantizedMoE", values, vec![output]);
    node.domain = DOMAIN.into();
    node.attributes
        .insert("k".into(), Attribute::Int(config.k as i64));
    node.attributes.insert(
        "activation_type".into(),
        Attribute::String(config.activation.as_bytes().to_vec()),
    );
    node.attributes.insert(
        "normalize_routing_weights".into(),
        Attribute::Int(i64::from(config.normalize)),
    );
    node.attributes.insert(
        "swiglu_fusion".into(),
        Attribute::Int(config.swiglu_fusion as i64),
    );
    node.attributes.insert(
        "format".into(),
        Attribute::String(FORMAT.as_bytes().to_vec()),
    );
    node.attributes
        .insert("block_layout_version".into(), Attribute::Int(1));
    if config.activation == "swiglu" {
        node.attributes
            .insert("activation_alpha".into(), Attribute::Float(1.125));
        node.attributes
            .insert("activation_beta".into(), Attribute::Float(-0.0625));
        node.attributes
            .insert("swiglu_limit".into(), Attribute::Float(4.0));
    }
    let node = graph.insert_node(node);
    graph.add_output(output);
    (graph, node)
}

fn build_views<'a>(
    inputs: &'a [Option<HostTensor>],
    strides: &'a [Option<Vec<i64>>],
    buffers: Option<&'a [Option<DeviceBuffer>]>,
    device: DeviceId,
) -> Vec<TensorView<'a>> {
    inputs
        .iter()
        .zip(strides)
        .enumerate()
        .map(|(index, (input, strides))| match (input, strides) {
            (Some(input), Some(strides)) => {
                let ptr: *const std::ffi::c_void = match buffers {
                    Some(buffers) => buffers[index]
                        .as_ref()
                        .expect("present input must have a device buffer")
                        .as_ptr(),
                    None => input.bytes.as_ptr().cast(),
                };
                TensorView::new(DevicePtr(ptr), input.dtype, &input.shape, strides, device)
            }
            _ => TensorView::absent(absent_dtype(index)),
        })
        .collect()
}

fn run_cpu(config: &Config, inputs: &[Option<HostTensor>]) -> Vec<f32> {
    let (graph, node) = model_node(config, inputs);
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
    let views = build_views(inputs, &strides, None, DeviceId::cpu());
    let output_shape = [config.rows, config.hidden];
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut output = vec![0u8; config.rows * config.hidden * 4];
    let output_view = TensorMut::new(
        DevicePtrMut(output.as_mut_ptr().cast()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        DeviceId::cpu(),
    );
    kernel.execute(&views, &mut [output_view]).unwrap();
    output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn run_gpu(
    ep: &CudaExecutionProvider,
    config: &Config,
    inputs: &[Option<HostTensor>],
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let (graph, node) = model_node(config, inputs);
    let model = Model::new(&graph);
    let concrete_shapes: Vec<Vec<usize>> = inputs
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
    let views = build_views(inputs, &strides, Some(&buffers), ep.device_id());
    let output_shape = [config.rows, config.hidden];
    let output_len = config.rows * config.hidden;
    let mut output_buffer = ep.allocate(output_len * 4, 256)?;
    let output_strides = compute_contiguous_strides(&output_shape);
    let output_view = TensorMut::new(
        DevicePtrMut(output_buffer.as_mut_ptr()),
        DataType::Float32,
        &output_shape,
        &output_strides,
        ep.device_id(),
    );
    let metadata = views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata)?;
    let workspace_bytes = usize::try_from(requirement.bytes)
        .map_err(|_| onnx_runtime_ep_api::EpError::KernelFailed("workspace too large".into()))?;
    let mut workspace = ep.allocate(workspace_bytes, requirement.alignment)?;
    kernel.execute_with_workspace(
        &views,
        &mut [output_view],
        Some(WorkspaceView::new(
            DevicePtrMut(workspace.as_mut_ptr()),
            workspace_bytes,
        )),
    )?;
    let mut output = vec![0u8; output_len * 4];
    // SAFETY: the destination exactly covers the f32 output allocation.
    unsafe { runtime.dtoh(&mut output, cuptr(output_buffer.as_ptr()))? };
    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer)?;
    }
    ep.deallocate(output_buffer)?;
    ep.deallocate(workspace)?;
    Ok(output
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 3e-3_f32.max(expected.abs() * 3e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label} index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn check_case(ep: &CudaExecutionProvider, config: &Config, seed: u64, label: &str) {
    let inputs = build_inputs(config, seed);
    let expected = run_cpu(config, &inputs);
    let actual = run_gpu(ep, config, &inputs).unwrap();
    assert_close(&actual, &expected, label);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_matches_cpu_across_activations() {
    let ep = require_cuda();
    let base = Config {
        rows: 5,
        hidden: 64,
        inter: 32,
        experts: 4,
        k: 2,
        activation: "relu",
        swiglu_fusion: 0,
        with_bias: true,
        with_router_weights: false,
        normalize: false,
    };
    for activation in ["relu", "gelu", "silu", "identity"] {
        let config = Config {
            activation,
            ..base_like(&base)
        };
        check_case(
            &ep,
            &config,
            0x1111_2222_3333_4444 ^ label_seed(activation),
            activation,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_matches_cpu_for_swiglu_variants() {
    let ep = require_cuda();
    // Unfused SwiGLU (fc1 gate + fc3 linear) and both fused layouts.
    for (fusion, label) in [
        (0usize, "swiglu-unfused"),
        (1, "swiglu-fused1"),
        (2, "swiglu-fused2"),
    ] {
        let config = Config {
            rows: 4,
            hidden: 64,
            inter: 32,
            experts: 3,
            k: 2,
            activation: "swiglu",
            swiglu_fusion: fusion,
            with_bias: true,
            with_router_weights: false,
            normalize: true,
        };
        check_case(&ep, &config, 0xabcd_0000 ^ fusion as u64, label);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_matches_cpu_for_routing_edge_cases() {
    let ep = require_cuda();

    // k = 1 top-1 routing across several experts.
    check_case(
        &ep,
        &Config {
            rows: 6,
            hidden: 64,
            inter: 32,
            experts: 4,
            k: 1,
            activation: "gelu",
            swiglu_fusion: 0,
            with_bias: false,
            with_router_weights: false,
            normalize: true,
        },
        0x5150_5150,
        "k1-topk",
    );

    // Single expert, k = 1 (every token routed to the same expert).
    check_case(
        &ep,
        &Config {
            rows: 3,
            hidden: 32,
            inter: 32,
            experts: 1,
            k: 1,
            activation: "relu",
            swiglu_fusion: 0,
            with_bias: true,
            with_router_weights: false,
            normalize: false,
        },
        0x1234_5678,
        "single-expert",
    );

    // k equal to the expert count (every token touches every expert).
    check_case(
        &ep,
        &Config {
            rows: 4,
            hidden: 64,
            inter: 32,
            experts: 3,
            k: 3,
            activation: "silu",
            swiglu_fusion: 0,
            with_bias: true,
            with_router_weights: false,
            normalize: true,
        },
        0x0f0f_0f0f,
        "k-equals-experts",
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_matches_cpu_with_router_weight_aggregation() {
    let ep = require_cuda();
    check_case(
        &ep,
        &Config {
            rows: 5,
            hidden: 64,
            inter: 32,
            experts: 4,
            k: 2,
            activation: "relu",
            swiglu_fusion: 0,
            with_bias: false,
            with_router_weights: true,
            normalize: false,
        },
        0x7777_9999,
        "router-weights",
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_claim_gate_matches_implemented_config() {
    let ep = require_cuda();
    let config = Config {
        rows: 2,
        hidden: 64,
        inter: 32,
        experts: 3,
        k: 2,
        activation: "relu",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: false,
    };
    let inputs = build_inputs(&config, 0x2020_2020);
    let (graph, node) = model_node(&config, &inputs);
    let model = Model::new(&graph);
    let shapes: Vec<_> = inputs
        .iter()
        .map(|input| match input {
            Some(input) => static_shape(input.shape.iter().copied()),
            None => static_shape([0usize; 0]),
        })
        .collect();
    let dtypes: Vec<_> = inputs
        .iter()
        .map(|input| match input {
            Some(input) => input.dtype,
            None => DataType::Undefined,
        })
        .collect();

    // Supported: mxfp4 all-f32 activations.
    assert!(matches!(
        ep.supports_op(model.graph.node(node), 1, &shapes, &dtypes, &[]),
        KernelMatch::Supported { .. }
    ));

    // Unsupported format falls back to CPU.
    let mut bad_format = graph.clone();
    bad_format
        .node_mut(node)
        .attributes
        .insert("format".into(), Attribute::String(b"q4_0".to_vec()));
    let bad_model = Model::new(&bad_format);
    assert!(matches!(
        ep.supports_op(bad_model.graph.node(node), 1, &shapes, &dtypes, &[]),
        KernelMatch::Unsupported { .. }
    ));

    // Non-f32 activation dtype falls back to CPU.
    let mut bad_dtypes = dtypes.clone();
    bad_dtypes[0] = DataType::Float16;
    assert!(matches!(
        ep.supports_op(model.graph.node(node), 1, &shapes, &bad_dtypes, &[]),
        KernelMatch::Unsupported { .. }
    ));
}

fn base_like(config: &Config) -> Config {
    Config {
        rows: config.rows,
        hidden: config.hidden,
        inter: config.inter,
        experts: config.experts,
        k: config.k,
        activation: config.activation,
        swiglu_fusion: config.swiglu_fusion,
        with_bias: config.with_bias,
        with_router_weights: config.with_router_weights,
        normalize: config.normalize,
    }
}

fn label_seed(label: &str) -> u64 {
    label.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
