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
//! GPU parity regressions for the mixed-projection `pkg.nxrt::BlockQuantizedMoE` op.
//!
//! Synthetic cases cover every supported mixed GLM projection pair and use the
//! CPU kernel as the independent oracle. Opt-in tests also read official
//! checkpoint blocks in place, exercise capture/replay, and measure exact-shape
//! selected-expert execution. CPU-only CI reports GPU cases as ignored unless
//! `gpu-tests` is enabled.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, KernelConstantInput, KernelMatch,
    TensorMetadata, TensorMut, TensorView, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, TensorData, WeightRef,
    compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{Read, Seek, SeekFrom};

struct CountingAllocator;

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static HOST_ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                HOST_ALLOCATIONS.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_host_allocations<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    HOST_ALLOCATIONS.with(|count| count.set(0));
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(true));
    let result = operation();
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
    let allocations = HOST_ALLOCATIONS.with(Cell::get);
    (result, allocations)
}

const DOMAIN: &str = "pkg.nxrt";
const FORMAT: &str = "mxfp4";

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

fn format_info(format: &str) -> (usize, usize) {
    match format {
        "mxfp4" => (32, 17),
        "iq1_s" => (256, 50),
        "iq2_xxs" => (256, 66),
        "iq3_xxs" => (256, 98),
        "iq4_xs" => (256, 136),
        "q2_k" => (256, 84),
        "q3_k" => (256, 110),
        "q5_k" => (256, 176),
        "q6_k" => (256, 210),
        "q8_0" => (32, 34),
        other => panic!("unknown test format {other}"),
    }
}

fn pack_projection_format(
    state: &mut u64,
    format: &str,
    experts: usize,
    out_features: usize,
    in_features: usize,
) -> HostTensor {
    let (qk, block_bytes) = format_info(format);
    assert!(in_features.is_multiple_of(qk));
    let blocks = in_features / qk;
    let mut packed = vec![0u8; experts * out_features * blocks * block_bytes];
    for block in packed.chunks_exact_mut(block_bytes) {
        match format {
            "mxfp4" => {
                block[0] = 122 + (random_u32(state) % 4) as u8;
                for byte in &mut block[1..] {
                    *byte = random_u32(state) as u8;
                }
            }
            "q2_k" => {
                for byte in &mut block[..80] {
                    *byte = random_u32(state) as u8;
                }
                block[80..82].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
                block[82..84].copy_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
            }
            "q3_k" => {
                for byte in &mut block[..108] {
                    *byte = random_u32(state) as u8;
                }
                block[108..110].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
            }
            "q5_k" => {
                block[..2].copy_from_slice(&half::f16::from_f32(0.0005).to_le_bytes());
                block[2..4].copy_from_slice(&half::f16::from_f32(0.00025).to_le_bytes());
                for byte in &mut block[4..] {
                    *byte = random_u32(state) as u8;
                }
            }
            "q6_k" => {
                for byte in &mut block[..208] {
                    *byte = random_u32(state) as u8;
                }
                block[208..210].copy_from_slice(&half::f16::from_f32(0.0005).to_le_bytes());
            }
            "q8_0" => {
                block[..2].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
                for byte in &mut block[2..] {
                    *byte = random_u32(state) as u8;
                }
            }
            _ => {
                block[..2].copy_from_slice(&half::f16::from_f32(0.002).to_le_bytes());
                for byte in &mut block[2..] {
                    *byte = random_u32(state) as u8;
                }
            }
        }
    }
    HostTensor::u8(&[experts, out_features, blocks, block_bytes], packed)
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
fn build_inputs_formats(
    config: &Config,
    seed: u64,
    fc1_format: &str,
    fc2_format: &str,
    fc3_format: Option<&str>,
) -> Vec<Option<HostTensor>> {
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

    let fc1 = pack_projection_format(&mut state, fc1_format, experts, fc1_size, hidden);
    let fc2 = pack_projection_format(&mut state, fc2_format, experts, hidden, inter);

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
        let fc3 = pack_projection_format(
            &mut state,
            fc3_format.expect("separate gate requires a test format"),
            experts,
            inter,
            hidden,
        );
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

fn build_inputs(config: &Config, seed: u64) -> Vec<Option<HostTensor>> {
    build_inputs_formats(
        config,
        seed,
        FORMAT,
        FORMAT,
        config.needs_fc3().then_some(FORMAT),
    )
}

fn absent_dtype(index: usize) -> DataType {
    match index {
        2 | 4 | 6 => DataType::Uint8,
        _ => DataType::Float32,
    }
}

fn model_node_formats(
    config: &Config,
    inputs: &[Option<HostTensor>],
    fc1_format: &str,
    fc2_format: &str,
    fc3_format: Option<&str>,
) -> (Graph, NodeId) {
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
                if matches!(index, 2 | 4 | 6) {
                    graph.set_initializer(
                        value,
                        WeightRef::Inline(TensorData::from_raw(
                            input.dtype,
                            input.shape.clone(),
                            input.bytes.clone(),
                        )),
                    );
                } else {
                    graph.add_input(value);
                }
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
        "fc1_format".into(),
        Attribute::String(fc1_format.as_bytes().to_vec()),
    );
    node.attributes.insert(
        "fc2_format".into(),
        Attribute::String(fc2_format.as_bytes().to_vec()),
    );
    if config.needs_fc3() {
        node.attributes.insert(
            "fc3_format".into(),
            Attribute::String(
                fc3_format
                    .expect("separate gate requires a test format")
                    .as_bytes()
                    .to_vec(),
            ),
        );
    }
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

fn model_node(config: &Config, inputs: &[Option<HostTensor>]) -> (Graph, NodeId) {
    model_node_formats(
        config,
        inputs,
        FORMAT,
        FORMAT,
        config.needs_fc3().then_some(FORMAT),
    )
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

fn run_cpu_node(
    config: &Config,
    inputs: &[Option<HostTensor>],
    graph: &Graph,
    node: NodeId,
) -> Vec<f32> {
    let model = Model::new(graph);
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

fn run_cpu(config: &Config, inputs: &[Option<HostTensor>]) -> Vec<f32> {
    let (graph, node) = model_node(config, inputs);
    run_cpu_node(config, inputs, &graph, node)
}

fn prepare_gpu_kernel(
    ep: &CudaExecutionProvider,
    graph: &Graph,
    node: NodeId,
    inputs: &[Option<HostTensor>],
) -> onnx_runtime_ep_api::Result<(
    Box<dyn onnx_runtime_ep_api::Kernel>,
    Vec<Option<DeviceBuffer>>,
)> {
    let concrete_shapes = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map_or_else(Vec::new, |input| input.shape.clone())
        })
        .collect::<Vec<_>>();
    let mut kernel = ep.get_kernel(graph.node(node), &concrete_shapes, 1)?;
    let node_ref = graph.node(node);
    let constants = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let value = node_ref.inputs.get(index).copied().flatten()?;
            graph.initializers.contains_key(&value).then(|| {
                let input = input.as_ref().expect("initializer input must be present");
                KernelConstantInput {
                    dtype: input.dtype,
                    shape: &input.shape,
                    bytes: &input.bytes,
                }
            })
        })
        .collect::<Vec<_>>();
    kernel.prepare_constant_inputs(&constants, ep)?;
    let runtime = ep.runtime();
    let mut buffers = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        if let Some(input) = input {
            if kernel.constant_input_override(index).is_some() {
                buffers.push(None);
            } else {
                let buffer = ep.allocate(input.bytes.len(), 256)?;
                // SAFETY: allocation size equals the source tensor byte length.
                unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
                buffers.push(Some(buffer));
            }
        } else {
            buffers.push(None);
        }
    }
    Ok((kernel, buffers))
}

fn build_gpu_views<'a>(
    kernel: &'a dyn onnx_runtime_ep_api::Kernel,
    inputs: &'a [Option<HostTensor>],
    strides: &'a [Option<Vec<i64>>],
    buffers: &'a [Option<DeviceBuffer>],
    device: DeviceId,
) -> Vec<TensorView<'a>> {
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            if let Some(sealed) = kernel.constant_input_override(index) {
                return sealed;
            }
            match input {
                Some(input) => TensorView::new(
                    DevicePtr(
                        buffers[index]
                            .as_ref()
                            .expect("non-sealed input must have a device buffer")
                            .as_ptr(),
                    ),
                    input.dtype,
                    &input.shape,
                    strides[index]
                        .as_ref()
                        .expect("present input must have strides"),
                    device,
                ),
                None => TensorView::absent(absent_dtype(index)),
            }
        })
        .collect()
}

fn run_gpu_node(
    ep: &CudaExecutionProvider,
    config: &Config,
    inputs: &[Option<HostTensor>],
    graph: &Graph,
    node: NodeId,
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let model = Model::new(graph);
    let (kernel, buffers) = prepare_gpu_kernel(ep, model.graph, node, inputs)?;
    let runtime = ep.runtime();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views = build_gpu_views(kernel.as_ref(), inputs, &strides, &buffers, ep.device_id());
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

fn run_gpu(
    ep: &CudaExecutionProvider,
    config: &Config,
    inputs: &[Option<HostTensor>],
) -> onnx_runtime_ep_api::Result<Vec<f32>> {
    let (graph, node) = model_node(config, inputs);
    run_gpu_node(ep, config, inputs, &graph, node)
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
fn block_quantized_moe_glm52_mixed_projection_formats_match_cpu() {
    let ep = require_cuda();
    let config = Config {
        rows: 2,
        hidden: 256,
        inter: 256,
        experts: 2,
        k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: false,
    };
    for (index, (fc1_format, fc2_format)) in [
        ("iq1_s", "iq3_xxs"),
        ("iq2_xxs", "iq3_xxs"),
        ("iq2_xxs", "iq4_xs"),
        ("q2_k", "q3_k"),
        ("q5_k", "q6_k"),
        ("q6_k", "q8_0"),
    ]
    .into_iter()
    .enumerate()
    {
        let inputs = build_inputs_formats(
            &config,
            0x5200_0000 ^ index as u64,
            fc1_format,
            fc2_format,
            None,
        );
        let (graph, node) = model_node_formats(&config, &inputs, fc1_format, fc2_format, None);
        let expected = run_cpu_node(&config, &inputs, &graph, node);
        let actual = run_gpu_node(&ep, &config, &inputs, &graph, node).unwrap();
        assert_close(&actual, &expected, &format!("{fc1_format}/{fc2_format}"));
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_sealed_bank_admission_is_device_bound_and_whole_bank() {
    let provider = std::sync::Arc::new(require_cuda());
    let mut state = 0x5ea1_ed00;
    let fc1 = pack_projection_format(&mut state, "mxfp4", 2, 32, 32);
    let fc2 = pack_projection_format(&mut state, "q8_0", 2, 32, 32);
    let admitted = onnx_runtime_ep_cuda::admit_block_quantized_moe_banks(
        &provider,
        onnx_runtime_ep_cuda::BlockQuantizedMoeBank {
            format: "mxfp4",
            packed: &fc1.bytes,
            experts: 2,
            out_features: 32,
            in_features: 32,
        },
        onnx_runtime_ep_cuda::BlockQuantizedMoeBank {
            format: "q8_0",
            packed: &fc2.bytes,
            experts: 2,
            out_features: 32,
            in_features: 32,
        },
        None,
    )
    .unwrap();
    assert_eq!(admitted.device(), provider.device_id());
    assert_eq!(admitted.projection_count(), 2);
    assert_eq!(
        admitted.residency(),
        onnx_runtime_ep_cuda::BlockQuantizedMoeResidency::WholeProjectionBank
    );
    assert!(
        admitted
            .diagnostic_identities()
            .into_iter()
            .flatten()
            .count()
            == 2
    );
    let traffic = admitted.no_residency_traffic(&[1]).unwrap();
    assert_eq!(
        traffic.uploaded_whole_bank_bytes,
        (2 * 32 * (17 + 34)) as u64
    );
    assert_eq!(traffic.logical_route_demand_bytes, (32 * (17 + 34)) as u64);
    assert_eq!(
        traffic.unique_selected_expert_bytes,
        (32 * (17 + 34)) as u64
    );
    assert_eq!(traffic.physical_dram_bytes, None);
    assert_eq!(traffic.page_ins, 0);
    assert_eq!(traffic.byte_hit_rate, None);
    let repeated = admitted.no_residency_traffic(&[1, 1]).unwrap();
    let broad = admitted.no_residency_traffic(&[0, 1]).unwrap();
    assert_eq!(
        repeated.logical_route_demand_bytes,
        broad.logical_route_demand_bytes
    );
    assert_eq!(
        repeated.unique_selected_expert_bytes,
        traffic.unique_selected_expert_bytes
    );
    assert_eq!(
        broad.unique_selected_expert_bytes,
        2 * traffic.unique_selected_expert_bytes
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_rejects_raw_substituted_and_mutated_projection_views() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let config = Config {
        rows: 1,
        hidden: 32,
        inter: 32,
        experts: 2,
        k: 1,
        activation: "silu",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: true,
    };
    let mut inputs = build_inputs_formats(&config, 0x5ea1_0001, "q8_0", "q8_0", Some("q8_0"));
    let (graph, node) = model_node_formats(&config, &inputs, "q8_0", "q8_0", Some("q8_0"));
    let (kernel, buffers) = prepare_gpu_kernel(&ep, &graph, node, &inputs).unwrap();
    let strides = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect::<Vec<_>>();
    let views = build_gpu_views(kernel.as_ref(), &inputs, &strides, &buffers, ep.device_id());
    let metadata = views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata).unwrap();
    let workspace_bytes = requirement.bytes as usize;
    let mut workspace = ep.allocate(workspace_bytes, requirement.alignment).unwrap();
    let mut output = ep.allocate(32 * 4, 256).unwrap();
    let workspace_ptr = workspace.as_mut_ptr();
    let output_ptr_mut = output.as_mut_ptr();
    let output_ptr = cuptr(output.as_ptr());
    let output_shape = [1, 32];
    let output_strides = compute_contiguous_strides(&output_shape);
    let run = |candidate: &[TensorView]| {
        kernel.execute_with_workspace(
            candidate,
            &mut [TensorMut::new(
                DevicePtrMut(output_ptr_mut),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
            Some(WorkspaceView::new(
                DevicePtrMut(workspace_ptr),
                workspace_bytes,
            )),
        )
    };

    let mut substituted = views.clone();
    substituted[2] = views[4];
    assert!(
        run(&substituted)
            .unwrap_err()
            .to_string()
            .contains("exact immutable admitted projection")
    );

    let mut raw = views.clone();
    raw[2] = TensorView::new(
        views[2].data,
        views[2].dtype,
        views[2].shape,
        views[2].strides,
        views[2].device,
    );
    assert!(
        run(&raw)
            .unwrap_err()
            .to_string()
            .contains("exact immutable admitted projection")
    );

    let mut wrong_device = views.clone();
    wrong_device[2] = TensorView::new(
        views[2].data,
        views[2].dtype,
        views[2].shape,
        views[2].strides,
        DeviceId::cpu(),
    )
    .with_backing(views[2].backing);
    assert!(
        run(&wrong_device)
            .unwrap_err()
            .to_string()
            .contains("wrong CUDA device")
    );

    let mut mutated = views.clone();
    mutated[2] = views[2].with_byte_offset(1);
    assert!(
        run(&mutated)
            .unwrap_err()
            .to_string()
            .contains("exact immutable admitted projection")
    );

    let foreign_ep = require_cuda();
    let (foreign_kernel, foreign_buffers) =
        prepare_gpu_kernel(&foreign_ep, &graph, node, &inputs).unwrap();
    let foreign_views = build_gpu_views(
        foreign_kernel.as_ref(),
        &inputs,
        &strides,
        &foreign_buffers,
        foreign_ep.device_id(),
    );
    let mut wrong_context = views.clone();
    wrong_context[2] = foreign_views[2];
    assert!(
        run(&wrong_context)
            .unwrap_err()
            .to_string()
            .contains("exact immutable admitted projection")
    );

    run(&views).unwrap();
    runtime.synchronize().unwrap();
    let mut before_mutation = vec![0u8; 32 * 4];
    unsafe { runtime.dtoh(&mut before_mutation, output_ptr).unwrap() };
    drop(views);
    drop(substituted);
    drop(raw);
    drop(wrong_device);
    drop(mutated);
    drop(wrong_context);
    drop(foreign_views);
    drop(foreign_kernel);
    for buffer in foreign_buffers.into_iter().flatten() {
        foreign_ep.deallocate(buffer).unwrap();
    }
    inputs[2].as_mut().unwrap().bytes.fill(0xff);
    let views_after_source_mutation =
        build_gpu_views(kernel.as_ref(), &inputs, &strides, &buffers, ep.device_id());
    run(&views_after_source_mutation).unwrap();
    runtime.synchronize().unwrap();
    let mut after_mutation = vec![0u8; 32 * 4];
    unsafe { runtime.dtoh(&mut after_mutation, output_ptr).unwrap() };
    assert_eq!(
        after_mutation, before_mutation,
        "mutating host source bytes after admission must not change execution"
    );
    drop(views_after_source_mutation);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer).unwrap();
    }

    ep.deallocate(output).unwrap();
    ep.deallocate(workspace).unwrap();
    runtime.synchronize().unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_rejects_bank_admission_during_capture() {
    let provider = std::sync::Arc::new(require_cuda());
    let runtime = provider.runtime();
    let mut state = 0xca97_ad01;
    let fc1 = pack_projection_format(&mut state, "q8_0", 2, 32, 32);
    let fc2 = pack_projection_format(&mut state, "q8_0", 2, 32, 32);
    runtime.test_begin_unregistered_graph_capture().unwrap();
    let error = match onnx_runtime_ep_cuda::admit_block_quantized_moe_banks(
        &provider,
        onnx_runtime_ep_cuda::BlockQuantizedMoeBank {
            format: "q8_0",
            packed: &fc1.bytes,
            experts: 2,
            out_features: 32,
            in_features: 32,
        },
        onnx_runtime_ep_cuda::BlockQuantizedMoeBank {
            format: "q8_0",
            packed: &fc2.bytes,
            experts: 2,
            out_features: 32,
            in_features: 32,
        },
        None,
    ) {
        Ok(_) => panic!("admission during capture must reject"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("during CUDA graph capture"));
    runtime.test_end_unregistered_graph_capture().unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_mixed_decode_capture_replays_without_fallback() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let config = Config {
        rows: 1,
        hidden: 256,
        inter: 256,
        experts: 2,
        k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: false,
    };
    let inputs = build_inputs_formats(&config, 0xca97_0001, "iq1_s", "iq3_xxs", None);
    let (graph, node) = model_node_formats(&config, &inputs, "iq1_s", "iq3_xxs", None);
    let model = Model::new(&graph);
    let (kernel, buffers) = prepare_gpu_kernel(&ep, model.graph, node, &inputs).unwrap();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views = build_gpu_views(kernel.as_ref(), &inputs, &strides, &buffers, ep.device_id());
    let output_shape = [config.rows, config.hidden];
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut output_buffer = ep.allocate(config.hidden * 4, 256).unwrap();
    let output_ptr_mut = output_buffer.as_mut_ptr();
    let output_ptr = cuptr(output_buffer.as_ptr());
    let output_view = || {
        TensorMut::new(
            DevicePtrMut(output_ptr_mut),
            DataType::Float32,
            &output_shape,
            &output_strides,
            ep.device_id(),
        )
    };
    let metadata = views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata).unwrap();
    let workspace_bytes = usize::try_from(requirement.bytes).unwrap();
    let mut workspace = ep.allocate(workspace_bytes, requirement.alignment).unwrap();
    let workspace_ptr = workspace.as_mut_ptr();
    let workspace_view = || WorkspaceView::new(DevicePtrMut(workspace_ptr), workspace_bytes);

    kernel
        .execute_with_workspace(&views, &mut [output_view()], Some(workspace_view()))
        .unwrap();
    runtime.synchronize().unwrap();
    let mut eager = vec![0u8; config.hidden * 4];
    // SAFETY: destination exactly covers the output allocation.
    unsafe { runtime.dtoh(&mut eager, output_ptr).unwrap() };

    let allocations = runtime.allocation_counts();
    let transfers = runtime.transfer_counts();
    let synchronizations = runtime.forced_synchronization_count();
    let preparation = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
    let (warmed_result, host_allocations) = count_host_allocations(|| {
        kernel.execute_with_workspace(&views, &mut [output_view()], Some(workspace_view()))
    });
    warmed_result.unwrap();
    assert_eq!(host_allocations, 0, "warmed eager host allocations");
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(
        onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
        preparation,
        "warmed eager format parsing/workspace layout"
    );
    assert_eq!(
        runtime.forced_synchronization_count(),
        synchronizations,
        "warmed eager operator synchronizations"
    );
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    kernel
        .execute_with_workspace(&views, &mut [output_view()], Some(workspace_view()))
        .unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(runtime.graph_segment_count().unwrap(), 1, "captures");
    assert_eq!(runtime.allocation_counts(), allocations);
    assert_eq!(runtime.transfer_counts(), transfers);
    assert_eq!(
        onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
        preparation,
        "captured replay format parsing/workspace layout"
    );
    // Prime any one-time CUDA driver graph-launch state before measuring the
    // warmed replay path. The assertions below cover every subsequent replay.
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    drop(views);
    drop(kernel);

    let mut replayed = vec![0u8; eager.len()];
    for _ in 0..3 {
        let replay_transfers = runtime.transfer_counts();
        let replay_synchronizations = runtime.forced_synchronization_count();
        let replay_preparation = onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts();
        let (replay_result, replay_host_allocations) =
            count_host_allocations(|| runtime.replay_graph());
        replay_result.unwrap();
        assert_eq!(
            replay_host_allocations, 0,
            "captured replay host allocations"
        );
        assert_eq!(runtime.allocation_counts(), allocations);
        assert_eq!(runtime.transfer_counts(), replay_transfers);
        assert_eq!(
            runtime.forced_synchronization_count(),
            replay_synchronizations,
            "captured replay operator synchronizations"
        );
        assert_eq!(
            onnx_runtime_ep_cuda::block_quantized_moe_preparation_counts(),
            replay_preparation,
            "captured replay format parsing/workspace layout"
        );
        runtime.synchronize().unwrap();
        // SAFETY: destination exactly covers the output allocation.
        unsafe { runtime.dtoh(&mut replayed, output_ptr).unwrap() };
        assert_eq!(replayed, eager);
    }
    assert!(runtime.reset_graph().unwrap());
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();
    ep.deallocate(workspace).unwrap();
    eprintln!("captures=1 fallbacks=0 warmup_replays=1 measured_replays=3");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn block_quantized_moe_reads_only_selected_expert_projection_bytes() {
    let ep = require_cuda();
    let config = Config {
        rows: 1,
        hidden: 256,
        inter: 256,
        experts: 2,
        k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: false,
    };
    let mut baseline = build_inputs_formats(&config, 0x5e1e_c7ed, "iq1_s", "iq3_xxs", None);
    baseline[1] = Some(HostTensor::f32(&[1, 2], &[10.0, -10.0]));
    let alternate = build_inputs_formats(&config, 0xa17e_12ed, "iq1_s", "iq3_xxs", None);
    let mut altered = baseline.clone();
    let fc1_stride = 256 * 50;
    altered[2].as_mut().unwrap().bytes[fc1_stride..]
        .copy_from_slice(&alternate[2].as_ref().unwrap().bytes[fc1_stride..]);
    let fc2_stride = 256 * 98;
    altered[4].as_mut().unwrap().bytes[fc2_stride..]
        .copy_from_slice(&alternate[4].as_ref().unwrap().bytes[fc2_stride..]);

    let (baseline_graph, baseline_node) =
        model_node_formats(&config, &baseline, "iq1_s", "iq3_xxs", None);
    let (altered_graph, altered_node) =
        model_node_formats(&config, &altered, "iq1_s", "iq3_xxs", None);
    let expected = run_gpu_node(&ep, &config, &baseline, &baseline_graph, baseline_node).unwrap();
    let actual = run_gpu_node(&ep, &config, &altered, &altered_graph, altered_node).unwrap();
    assert_eq!(actual, expected);
    eprintln!(
        "uploaded_whole_bank_bytes={} logical_route_demand_bytes={} \
         unique_selected_expert_bytes={} physical_dram_bytes=None page_ins=0 byte_hit_rate=None",
        2 * (fc1_stride + fc2_stride),
        fc1_stride + fc2_stride,
        fc1_stride + fc2_stride
    );
}

fn read_real_projection_blocks(
    root: &std::path::Path,
    shard: &str,
    tensor_offset: u64,
    format: &str,
    source_in_features: usize,
    out_features: usize,
) -> HostTensor {
    let (qk, block_bytes) = format_info(format);
    assert!(source_in_features.is_multiple_of(qk));
    let source_row_bytes = (source_in_features / qk) * block_bytes;
    let mut file = std::fs::File::open(root.join(shard)).unwrap();
    let mut packed = vec![0u8; out_features * block_bytes];
    for output in 0..out_features {
        file.seek(SeekFrom::Start(
            tensor_offset + (output * source_row_bytes) as u64,
        ))
        .unwrap();
        file.read_exact(&mut packed[output * block_bytes..][..block_bytes])
            .unwrap();
    }
    HostTensor::u8(&[1, out_features, 1, block_bytes], packed)
}

fn read_real_projection_expert(
    root: &std::path::Path,
    shard: &str,
    tensor_offset: u64,
    format: &str,
    in_features: usize,
    out_features: usize,
) -> HostTensor {
    let (qk, block_bytes) = format_info(format);
    assert!(in_features.is_multiple_of(qk));
    let blocks = in_features / qk;
    let mut file = std::fs::File::open(root.join(shard)).unwrap();
    file.seek(SeekFrom::Start(tensor_offset)).unwrap();
    let mut packed = vec![0u8; out_features * blocks * block_bytes];
    file.read_exact(&mut packed).unwrap();
    HostTensor::u8(&[1, out_features, blocks, block_bytes], packed)
}

fn measure_captured_moe(
    ep: &CudaExecutionProvider,
    config: &Config,
    inputs: &[Option<HostTensor>],
    graph: &Graph,
    node: NodeId,
    ramp_seconds: f64,
    batch_replays: usize,
) -> Vec<f64> {
    let model = Model::new(graph);
    let (kernel, buffers) = prepare_gpu_kernel(ep, model.graph, node, inputs).unwrap();
    let runtime = ep.runtime();
    let strides: Vec<_> = inputs
        .iter()
        .map(|input| {
            input
                .as_ref()
                .map(|input| compute_contiguous_strides(&input.shape))
        })
        .collect();
    let views = build_gpu_views(kernel.as_ref(), inputs, &strides, &buffers, ep.device_id());
    let output_shape = [config.rows, config.hidden];
    let output_strides = compute_contiguous_strides(&output_shape);
    let mut output_buffer = ep.allocate(config.rows * config.hidden * 4, 256).unwrap();
    let output_ptr_mut = output_buffer.as_mut_ptr();
    let output_view = || {
        TensorMut::new(
            DevicePtrMut(output_ptr_mut),
            DataType::Float32,
            &output_shape,
            &output_strides,
            ep.device_id(),
        )
    };
    let metadata = views
        .iter()
        .map(|view| TensorMetadata::new(view.dtype, view.shape, !view.is_absent()))
        .collect::<Vec<_>>();
    let requirement = kernel.workspace_requirement(&metadata).unwrap();
    let workspace_bytes = usize::try_from(requirement.bytes).unwrap();
    let mut workspace = ep.allocate(workspace_bytes, requirement.alignment).unwrap();
    let workspace_ptr = workspace.as_mut_ptr();
    let workspace_view = || WorkspaceView::new(DevicePtrMut(workspace_ptr), workspace_bytes);

    kernel
        .execute_with_workspace(&views, &mut [output_view()], Some(workspace_view()))
        .unwrap();
    runtime.synchronize().unwrap();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    kernel
        .execute_with_workspace(&views, &mut [output_view()], Some(workspace_view()))
        .unwrap();
    runtime.end_graph_capture().unwrap();
    assert_eq!(runtime.graph_segment_count().unwrap(), 1);

    let allocation_counts = runtime.allocation_counts();
    let transfer_counts = runtime.transfer_counts();
    let ramp_start = std::time::Instant::now();
    while ramp_start.elapsed().as_secs_f64() < ramp_seconds {
        for _ in 0..batch_replays {
            runtime.replay_graph().unwrap();
        }
        runtime.synchronize().unwrap();
    }

    let mut samples_us = Vec::with_capacity(3);
    for _ in 0..3 {
        runtime.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..batch_replays {
            runtime.replay_graph().unwrap();
        }
        runtime.synchronize().unwrap();
        samples_us.push(start.elapsed().as_secs_f64() * 1.0e6 / batch_replays as f64);
    }
    assert_eq!(runtime.allocation_counts(), allocation_counts);
    assert_eq!(runtime.transfer_counts(), transfer_counts);
    assert!(runtime.reset_graph().unwrap());

    drop(views);
    for buffer in buffers.into_iter().flatten() {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output_buffer).unwrap();
    ep.deallocate(workspace).unwrap();
    samples_us
}

#[test]
#[ignore = "opt-in real-checkpoint test; set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT"]
fn glm52_ud_iq1s_real_mixed_projection_blocks_execute_block_quantized_moe() {
    let root = std::env::var("ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT")
        .expect("set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT to the official checkpoint directory");
    let root = std::path::Path::new(&root);
    let ep = require_cuda();
    let config = Config {
        rows: 1,
        hidden: 256,
        inter: 256,
        experts: 1,
        k: 1,
        activation: "identity",
        swiglu_fusion: 0,
        with_bias: false,
        with_router_weights: false,
        normalize: false,
    };
    for (label, shard, fc1_format, fc1_offset, fc2_format, fc2_offset) in [
        (
            "layer56-iq1s-iq3xxs",
            "GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            "iq1_s",
            2_668_820_832u64,
            "iq3_xxs",
            1_425_373_536u64,
        ),
        (
            "layer74-iq2xxs-iq3xxs",
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            "iq2_xxs",
            3_071_455_584,
            "iq3_xxs",
            1_828_008_288,
        ),
        (
            "layer8-iq2xxs-iq4xs",
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            "iq2_xxs",
            17_617_400_576,
            "iq4_xs",
            15_892_755_200,
        ),
        (
            "layer78-q2k-q3k",
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            "q2_k",
            16_942_690_656,
            "q3_k",
            15_548_248_416,
        ),
    ] {
        let input: Vec<f32> = (0..256)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let inputs = vec![
            Some(HostTensor::f32(&[1, 256], &input)),
            Some(HostTensor::f32(&[1, 1], &[1.0])),
            Some(read_real_projection_blocks(
                root, shard, fc1_offset, fc1_format, 6144, 256,
            )),
            None,
            Some(read_real_projection_blocks(
                root, shard, fc2_offset, fc2_format, 2048, 256,
            )),
            None,
        ];
        let (graph, node) = model_node_formats(&config, &inputs, fc1_format, fc2_format, None);
        let expected = run_cpu_node(&config, &inputs, &graph, node);
        let actual = run_gpu_node(&ep, &config, &inputs, &graph, node).unwrap();
        assert_close(&actual, &expected, label);
    }
}

#[test]
#[ignore = "dedicated idle A100 perf probe; set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT"]
fn glm52_ud_iq1s_real_selected_expert_captured_perf() {
    let root = std::env::var("ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT")
        .expect("set ONNX_GENAI_GLM52_UD_IQ1S_CHECKPOINT to the official checkpoint directory");
    let root = std::path::Path::new(&root);
    let ramp_seconds = std::env::var("ONNX_GENAI_CUDA_PERF_RAMP_SECONDS")
        .map(|value| value.parse::<f64>().expect("valid perf ramp seconds"))
        .unwrap_or(8.0);
    assert!(ramp_seconds.is_finite() && ramp_seconds > 0.0);
    let ep = require_cuda();
    let cases = [
        (
            "layer56-iq1s-iq3xxs",
            "decode",
            1usize,
            "GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            "iq1_s",
            2_668_820_832u64,
            "iq3_xxs",
            1_425_373_536u64,
        ),
        (
            "layer56-iq1s-iq3xxs",
            "prefill",
            8,
            "GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            "iq1_s",
            2_668_820_832,
            "iq3_xxs",
            1_425_373_536,
        ),
        (
            "layer74-iq2xxs-iq3xxs",
            "decode",
            1,
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            "iq2_xxs",
            3_071_455_584,
            "iq3_xxs",
            1_828_008_288,
        ),
        (
            "layer8-iq2xxs-iq4xs",
            "decode",
            1,
            "GLM-5.2-UD-IQ1_S-00002-of-00006.gguf",
            "iq2_xxs",
            17_617_400_576,
            "iq4_xs",
            15_892_755_200,
        ),
        (
            "layer78-q2k-q3k",
            "decode",
            1,
            "GLM-5.2-UD-IQ1_S-00006-of-00006.gguf",
            "q2_k",
            16_942_690_656,
            "q3_k",
            15_548_248_416,
        ),
        (
            "layer56-iq1s-iq3xxs-repeat",
            "decode",
            1,
            "GLM-5.2-UD-IQ1_S-00005-of-00006.gguf",
            "iq1_s",
            2_668_820_832,
            "iq3_xxs",
            1_425_373_536,
        ),
    ];
    for (label, stage, rows, shard, fc1_format, fc1_offset, fc2_format, fc2_offset) in cases {
        let config = Config {
            rows,
            hidden: 6144,
            inter: 2048,
            experts: 1,
            k: 1,
            activation: "identity",
            swiglu_fusion: 0,
            with_bias: false,
            with_router_weights: false,
            normalize: false,
        };
        let input: Vec<f32> = (0..config.rows * config.hidden)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 32.0)
            .collect();
        let router_logits = vec![1.0; config.rows];
        let inputs = vec![
            Some(HostTensor::f32(&[config.rows, config.hidden], &input)),
            Some(HostTensor::f32(&[config.rows, 1], &router_logits)),
            Some(read_real_projection_expert(
                root,
                shard,
                fc1_offset,
                fc1_format,
                config.hidden,
                config.inter,
            )),
            None,
            Some(read_real_projection_expert(
                root,
                shard,
                fc2_offset,
                fc2_format,
                config.inter,
                config.hidden,
            )),
            None,
        ];
        let bytes_per_expert =
            inputs[2].as_ref().unwrap().bytes.len() + inputs[4].as_ref().unwrap().bytes.len();
        let (graph, node) = model_node_formats(&config, &inputs, fc1_format, fc2_format, None);
        let mut samples =
            measure_captured_moe(&ep, &config, &inputs, &graph, node, ramp_seconds, 128);
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: native_cuda=true stage={stage} rows={} H=6144 I=2048 selected_experts=1 \
             uploaded_whole_bank_bytes={bytes_per_expert} \
             logical_route_demand_bytes={} unique_selected_expert_bytes={bytes_per_expert} \
             physical_dram_bytes=None page_ins=0 byte_hit_rate=None captures=1 fallbacks=0 n=3 \
             median_us={:.3} range_us={:.3}..{:.3}",
            config.rows,
            bytes_per_expert * config.rows,
            samples[1],
            samples[0],
            samples[2]
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
        .insert("fc1_format".into(), Attribute::String(b"q4_0".to_vec()));
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

// ---------------------------------------------------------------------------
// Fused, inert route-telemetry coverage (issue #1810 Slice 7A).
//
// BlockQuantizedMoE capture/replay is covered above; this module covers the
// producer fused into `bqmoe_route`: default disabled,
// byte-identical outputs off vs on, a device bitmap that matches a CPU oracle
// for decode and M>1, fresh epochs across consecutive eager calls, typed
// fail-closed rejection on a device mismatch, inert behavior on a capacity
// mismatch (never fails inference), and multi-instance request/device isolation
// with teardown accounting.
mod route_telemetry {
    use super::*;
    use onnx_runtime_ep_api::Kernel;
    use onnx_runtime_ep_cuda::kernels::block_quantized_moe::{
        BlockQuantizedMoEFactory, BlockQuantizedMoEKernel,
    };
    use onnx_runtime_ep_cuda::kernels::expert_route_telemetry::{
        H_DEVICE, H_REQUEST, RouteTelemetryConfig, TelemetryUnsupported, cpu_bitmap,
    };

    fn telemetry_config(rows: usize) -> Config {
        Config {
            rows,
            hidden: 32,
            inter: 32,
            experts: 8,
            k: 2,
            activation: "relu",
            swiglu_fusion: 0,
            with_bias: false,
            with_router_weights: false,
            normalize: false,
        }
    }

    /// Row `r` gives experts `(r + shift + i) % E` strictly descending values, so
    /// its top-`k` selection is deterministic and tie-break independent.
    fn shifted_router(config: &Config, shift: usize) -> HostTensor {
        let mut values = vec![0f32; config.rows * config.experts];
        for row in 0..config.rows {
            for rank in 0..config.experts {
                let expert = (row + shift + rank) % config.experts;
                values[row * config.experts + expert] = (config.experts - rank) as f32;
            }
        }
        HostTensor::f32(&[config.rows, config.experts], &values)
    }

    fn oracle_bitmap(router: &HostTensor, config: &Config) -> Vec<u32> {
        let values: Vec<f32> = router
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        let mut routes = Vec::new();
        for row in 0..config.rows {
            let logits = &values[row * config.experts..(row + 1) * config.experts];
            let mut order: Vec<usize> = (0..config.experts).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
            for &expert in order.iter().take(config.k) {
                routes.push(expert as i32);
            }
        }
        cpu_bitmap(&routes, config.experts).0
    }

    /// CPU oracle for a coarse-boundary *window*: the union of every call's route
    /// bitmap accumulated with no reset in between.
    fn oracle_union(routers: &[HostTensor], config: &Config) -> Vec<u32> {
        let mut acc = vec![0u32; config.experts.div_ceil(32)];
        for router in routers {
            for (word, bit) in acc.iter_mut().zip(oracle_bitmap(router, config)) {
                *word |= bit;
            }
        }
        acc
    }

    fn build_kernel(
        ep: &CudaExecutionProvider,
        config: &Config,
        inputs: &[Option<HostTensor>],
    ) -> BlockQuantizedMoEKernel {
        let (graph, node) = model_node(config, inputs);
        let model = Model::new(&graph);
        let concrete_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|input| {
                input
                    .as_ref()
                    .map_or_else(Vec::new, |input| input.shape.clone())
            })
            .collect();
        let factory = BlockQuantizedMoEFactory {
            runtime: ep.runtime().clone(),
        };
        let mut kernel = factory
            .create_kernel(model.graph.node(node), &concrete_shapes)
            .expect("concrete BlockQuantizedMoE kernel");
        let node_ref = model.graph.node(node);
        let constants = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let value = node_ref.inputs.get(index).copied().flatten()?;
                model.graph.initializers.contains_key(&value).then(|| {
                    let input = input.as_ref().expect("initializer input must be present");
                    KernelConstantInput {
                        dtype: input.dtype,
                        shape: &input.shape,
                        bytes: &input.bytes,
                    }
                })
            })
            .collect::<Vec<_>>();
        kernel
            .prepare_constant_inputs(&constants, ep)
            .expect("admit immutable projection banks");
        kernel
    }

    /// Upload inputs, run the concrete kernel once through its workspace, return
    /// the raw output bytes.
    fn exec_eager(
        ep: &CudaExecutionProvider,
        kernel: &BlockQuantizedMoEKernel,
        config: &Config,
        inputs: &[Option<HostTensor>],
    ) -> onnx_runtime_ep_api::Result<Vec<u8>> {
        let runtime = ep.runtime();
        let mut buffers = Vec::<Option<DeviceBuffer>>::new();
        for (index, input) in inputs.iter().enumerate() {
            if let Some(input) = input {
                if kernel.constant_input_override(index).is_some() {
                    buffers.push(None);
                } else {
                    let buffer = ep.allocate(input.bytes.len(), 256)?;
                    // SAFETY: allocation size equals the source tensor byte length.
                    unsafe { runtime.htod(&input.bytes, cuptr(buffer.as_ptr()))? };
                    buffers.push(Some(buffer));
                }
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
        let views = build_gpu_views(kernel, inputs, &strides, &buffers, ep.device_id());
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
        let workspace_bytes = usize::try_from(requirement.bytes).unwrap();
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
        // SAFETY: destination exactly covers the f32 output allocation.
        unsafe { runtime.dtoh(&mut output, cuptr(output_buffer.as_ptr()))? };
        drop(views);
        for buffer in buffers.into_iter().flatten() {
            ep.deallocate(buffer)?;
        }
        ep.deallocate(output_buffer)?;
        ep.deallocate(workspace)?;
        Ok(output)
    }

    fn ordinal(ep: &CudaExecutionProvider) -> u32 {
        ep.runtime().ordinal()
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_off_on_output_is_byte_identical() {
        let ep = require_cuda();
        for rows in [1usize, 5] {
            let config = telemetry_config(rows);
            let mut inputs = build_inputs(&config, 0xA11CE ^ rows as u64);
            inputs[1] = Some(shifted_router(&config, 0));

            let mut kernel = build_kernel(&ep, &config, &inputs);
            let off = exec_eager(&ep, &kernel, &config, &inputs).unwrap();
            assert!(kernel.route_telemetry_snapshot().unwrap().is_none());

            kernel
                .arm_route_telemetry(RouteTelemetryConfig {
                    request_id: 42,
                    device_id: ordinal(&ep),
                    num_experts: config.experts,
                })
                .expect("arm telemetry");
            let on = exec_eager(&ep, &kernel, &config, &inputs).unwrap();
            assert_eq!(
                off, on,
                "route outputs must be byte-identical with telemetry off vs on (rows={rows})"
            );
        }
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_bitmap_matches_cpu_oracle_decode_and_prefill() {
        let ep = require_cuda();
        for rows in [1usize, 6] {
            let config = telemetry_config(rows);
            let mut inputs = build_inputs(&config, 0xB0B ^ rows as u64);
            let router = shifted_router(&config, 0);
            inputs[1] = Some(router.clone());

            let mut kernel = build_kernel(&ep, &config, &inputs);
            kernel
                .arm_route_telemetry(RouteTelemetryConfig {
                    request_id: 7,
                    device_id: ordinal(&ep),
                    num_experts: config.experts,
                })
                .expect("arm telemetry");
            exec_eager(&ep, &kernel, &config, &inputs).unwrap();

            let snapshot = kernel.route_telemetry_snapshot().unwrap().unwrap();
            assert_eq!(snapshot.header[H_DEVICE], ordinal(&ep));
            assert_eq!(snapshot.header[H_REQUEST], 7);
            assert_eq!(snapshot.epoch(), 1, "first armed launch stamps epoch 1");
            assert!(!snapshot.poison());
            assert!(!snapshot.overflow());
            assert_eq!(snapshot.count() as usize, config.rows * config.k);
            assert_eq!(
                snapshot.bitmap,
                oracle_bitmap(&router, &config),
                "device bitmap must match CPU oracle (rows={rows})"
            );
        }
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_eager_calls_accumulate_union_within_window() {
        let ep = require_cuda();
        let config = telemetry_config(4);
        let mut inputs = build_inputs(&config, 0xC0FFEE);
        let mut kernel = build_kernel(&ep, &config, &inputs);
        kernel
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 3,
                device_id: ordinal(&ep),
                num_experts: config.experts,
            })
            .expect("arm telemetry");

        // Multiple eager calls with different deterministic routes and NO reset
        // in between accumulate the route union and count over the window; the
        // epoch stays fixed (advances only at an explicit coarse boundary).
        let routers: Vec<HostTensor> = (0..3usize)
            .map(|shift| shifted_router(&config, shift))
            .collect();
        let mut expected_count = 0u32;
        for (index, router) in routers.iter().enumerate() {
            inputs[1] = Some(router.clone());
            exec_eager(&ep, &kernel, &config, &inputs).unwrap();
            expected_count += (config.rows * config.k) as u32;
            let snapshot = kernel.route_telemetry_snapshot().unwrap().unwrap();
            assert_eq!(
                snapshot.epoch(),
                1,
                "epoch is fixed within a window (call {index})"
            );
            assert_eq!(
                snapshot.count(),
                expected_count,
                "count accumulates across calls with no per-call reset (call {index})"
            );
            assert_eq!(
                snapshot.bitmap,
                oracle_union(&routers[..=index], &config),
                "bitmap accumulates the routed union across calls (call {index})"
            );
            assert!(!snapshot.poison());
            assert!(!snapshot.overflow());
        }

        // An explicit coarse boundary opens a fresh, empty window at a new epoch.
        kernel.reset_route_telemetry_boundary().unwrap();
        let after = kernel.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(after.epoch(), 2, "boundary reset increments the epoch");
        assert_eq!(after.count(), 0, "next window starts empty");
        assert!(
            after.bitmap.iter().all(|&word| word == 0),
            "no stale carryover"
        );

        // The next window accumulates only its own routes.
        let router = shifted_router(&config, 5);
        inputs[1] = Some(router.clone());
        exec_eager(&ep, &kernel, &config, &inputs).unwrap();
        let window2 = kernel.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(window2.epoch(), 2, "still window 2");
        assert_eq!(window2.count(), (config.rows * config.k) as u32);
        assert_eq!(
            window2.bitmap,
            oracle_bitmap(&router, &config),
            "window 2 records only its own routes, with no window-1 carryover"
        );
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_boundary_reset_increments_epoch_and_starts_empty_window() {
        let ep = require_cuda();
        let config = telemetry_config(4);
        let mut inputs = build_inputs(&config, 0xB0);
        inputs[1] = Some(shifted_router(&config, 0));
        let mut kernel = build_kernel(&ep, &config, &inputs);
        kernel
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 9,
                device_id: ordinal(&ep),
                num_experts: config.experts,
            })
            .expect("arm telemetry");

        exec_eager(&ep, &kernel, &config, &inputs).unwrap();
        let before = kernel.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(before.epoch(), 1);
        assert!(before.count() > 0);
        assert!(before.bitmap.iter().any(|&word| word != 0));

        kernel.reset_route_telemetry_boundary().unwrap();
        let after = kernel.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(
            after.epoch(),
            before.epoch() + 1,
            "reset bumps epoch by one"
        );
        assert_eq!(after.count(), 0, "next window starts with a zero count");
        assert!(
            after.bitmap.iter().all(|&word| word == 0),
            "next window starts empty (no stale carryover)"
        );
        assert_eq!(after.header[H_REQUEST], 9, "request identity is preserved");
        assert_eq!(
            after.header[H_DEVICE],
            ordinal(&ep),
            "device identity preserved"
        );
        assert!(!after.poison());
        assert!(!after.overflow());
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_bqmoe_route_matches_oracle_for_m_1_2_4_8() {
        // BQMoE token counts M in {1,2,4,8} (issue #1810 §5): the fused route
        // bitmap and count must match the CPU oracle at every M.
        let ep = require_cuda();
        for rows in [1usize, 2, 4, 8] {
            let config = telemetry_config(rows);
            let mut inputs = build_inputs(&config, 0x3A5 ^ rows as u64);
            let router = shifted_router(&config, 1);
            inputs[1] = Some(router.clone());

            let mut kernel = build_kernel(&ep, &config, &inputs);
            kernel
                .arm_route_telemetry(RouteTelemetryConfig {
                    request_id: 300 + rows as u32,
                    device_id: ordinal(&ep),
                    num_experts: config.experts,
                })
                .expect("arm telemetry");
            exec_eager(&ep, &kernel, &config, &inputs).unwrap();

            let snapshot = kernel.route_telemetry_snapshot().unwrap().unwrap();
            assert_eq!(snapshot.epoch(), 1, "M={rows}: arm opened window 1");
            assert_eq!(
                snapshot.count() as usize,
                config.rows * config.k,
                "M={rows}: count must equal rows*k"
            );
            assert_eq!(
                snapshot.bitmap,
                oracle_bitmap(&router, &config),
                "M={rows}: bitmap must match the CPU oracle"
            );
            assert!(!snapshot.poison(), "M={rows}: in-range routes never poison");
            assert!(!snapshot.overflow(), "M={rows}: count never overflows");
        }
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_device_mismatch_fails_closed_without_failing_inference() {
        let ep = require_cuda();
        let config = telemetry_config(2);
        let inputs = build_inputs(&config, 0xD00D);
        let mut kernel = build_kernel(&ep, &config, &inputs);

        let wrong_device = ordinal(&ep).wrapping_add(1);
        let error = kernel
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 1,
                device_id: wrong_device,
                num_experts: config.experts,
            })
            .expect_err("device mismatch must be rejected");
        assert!(matches!(error, TelemetryUnsupported::DeviceMismatch { .. }));

        assert!(kernel.route_telemetry_snapshot().unwrap().is_none());
        let out = exec_eager(&ep, &kernel, &config, &inputs).unwrap();
        assert_eq!(out.len(), config.rows * config.hidden * 4);
        assert!(kernel.route_telemetry_snapshot().unwrap().is_none());
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_capacity_mismatch_is_inert_and_never_fails_inference() {
        let ep = require_cuda();
        let config = telemetry_config(2);
        let mut inputs = build_inputs(&config, 0xFEED);
        inputs[1] = Some(shifted_router(&config, 0));
        let mut kernel = build_kernel(&ep, &config, &inputs);

        kernel
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 5,
                device_id: ordinal(&ep),
                num_experts: config.experts + 4,
            })
            .expect("arming with any positive capacity succeeds");

        let out = exec_eager(&ep, &kernel, &config, &inputs).unwrap();
        assert_eq!(out.len(), config.rows * config.hidden * 4);
        let snapshot = kernel.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(snapshot.epoch(), 1, "arm opened window 1; execute is inert");
        assert_eq!(snapshot.count(), 0);
        assert!(snapshot.bitmap.iter().all(|&word| word == 0));
    }

    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
    )]
    #[test]
    fn telemetry_multi_instance_request_isolation_and_accounting() {
        let ep = require_cuda();
        let config = telemetry_config(3);
        let mut inputs = build_inputs(&config, 0x5EED);
        inputs[1] = Some(shifted_router(&config, 0));

        let mut kernel_a = build_kernel(&ep, &config, &inputs);
        let mut kernel_b = build_kernel(&ep, &config, &inputs);
        kernel_a
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 100,
                device_id: ordinal(&ep),
                num_experts: config.experts,
            })
            .unwrap();
        kernel_b
            .arm_route_telemetry(RouteTelemetryConfig {
                request_id: 200,
                device_id: ordinal(&ep),
                num_experts: config.experts,
            })
            .unwrap();

        assert_ne!(
            kernel_a.route_telemetry_bitmap_addr(),
            kernel_b.route_telemetry_bitmap_addr()
        );
        let footprint = 4 * config.experts.div_ceil(32) + 6 * 4;
        assert_eq!(kernel_a.route_telemetry_footprint_bytes(), footprint);

        exec_eager(&ep, &kernel_a, &config, &inputs).unwrap();
        exec_eager(&ep, &kernel_b, &config, &inputs).unwrap();

        let snap_a = kernel_a.route_telemetry_snapshot().unwrap().unwrap();
        let snap_b = kernel_b.route_telemetry_snapshot().unwrap().unwrap();
        assert_eq!(snap_a.header[H_REQUEST], 100);
        assert_eq!(snap_b.header[H_REQUEST], 200);
        assert_eq!(snap_a.header[H_DEVICE], ordinal(&ep));

        kernel_a.disarm_route_telemetry().unwrap();
        assert_eq!(kernel_a.route_telemetry_footprint_bytes(), 0);
        assert!(kernel_a.route_telemetry_snapshot().unwrap().is_none());
        let out = exec_eager(&ep, &kernel_a, &config, &inputs).unwrap();
        assert_eq!(out.len(), config.rows * config.hidden * 4);
    }
}
