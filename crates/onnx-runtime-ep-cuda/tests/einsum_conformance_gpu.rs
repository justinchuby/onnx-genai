mod common;

use std::fmt;
use std::sync::{Mutex, OnceLock};

use common::{Tensor, build_graph};
use onnx_runtime_einsum_conformance::{
    BackendAdapter, BackendKind, BackendObservation, CanonicalTensor, CaptureExpectation,
    CaseLimits, CaseRecord, ComparisonMode, ConformanceDType, ExecutionRequest, ForcedRoute,
    PlannerQuality, RouteProbe, ValueProfile, ValueSpec, WorkspaceClass, default_corpus, evaluate,
    infer_output_shape, materialize_inputs, verify_observation,
};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaEinsumRoute, CudaExecutionProvider, EinsumRouteOverride, einsum_execution_stats,
    execute_einsum_with_route, reset_einsum_execution_stats,
};
use onnx_runtime_ir::{Attribute, DataType, compute_contiguous_strides, static_shape};
use onnx_runtime_loader::Model;

fn suite_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn dtype(dtype: ConformanceDType) -> DataType {
    match dtype {
        ConformanceDType::Uint8 => DataType::Uint8,
        ConformanceDType::Uint16 => DataType::Uint16,
        ConformanceDType::Uint32 => DataType::Uint32,
        ConformanceDType::Uint64 => DataType::Uint64,
        ConformanceDType::Int8 => DataType::Int8,
        ConformanceDType::Int16 => DataType::Int16,
        ConformanceDType::Int32 => DataType::Int32,
        ConformanceDType::Int64 => DataType::Int64,
        ConformanceDType::Float16 => DataType::Float16,
        ConformanceDType::Float32 => DataType::Float32,
        ConformanceDType::Float64 => DataType::Float64,
        ConformanceDType::BFloat16 => DataType::BFloat16,
    }
}

fn tensor_bytes(tensor: &CanonicalTensor) -> Vec<u8> {
    let width = tensor.dtype().byte_size();
    let mut bytes = Vec::with_capacity(tensor.raw_bits().len() * width);
    for &bits in tensor.raw_bits() {
        bytes.extend_from_slice(&bits.to_ne_bytes()[..width]);
    }
    bytes
}

fn tensor_bits(dtype: ConformanceDType, bytes: &[u8]) -> Vec<u64> {
    let width = dtype.byte_size();
    bytes
        .chunks_exact(width)
        .map(|chunk| {
            let mut value = [0u8; 8];
            value[..width].copy_from_slice(chunk);
            u64::from_ne_bytes(value)
        })
        .collect()
}

#[derive(Debug)]
struct CudaAdapterError(String);

impl fmt::Display for CudaAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CudaAdapterError {}

fn adapter_error(error: impl fmt::Display) -> CudaAdapterError {
    CudaAdapterError(error.to_string())
}

struct CudaAdapter<'a> {
    ep: &'a CudaExecutionProvider,
}

impl BackendAdapter for CudaAdapter<'_> {
    type Error = CudaAdapterError;

    fn execute(&self, request: ExecutionRequest<'_>) -> Result<BackendObservation, Self::Error> {
        let case = request.case;
        let dtype = dtype(case.dtype);
        let output_shape =
            infer_output_shape(&case.equation, &case.input_shapes).map_err(adapter_error)?;
        let inputs = request
            .inputs
            .iter()
            .map(|tensor| Tensor {
                dtype,
                shape: tensor.shape().to_vec(),
                bytes: tensor_bytes(tensor),
                absent: false,
            })
            .collect::<Vec<_>>();
        let (graph, node) = build_graph(
            "Einsum",
            "",
            case.opset,
            &inputs,
            &[(dtype, output_shape.clone())],
            &[(
                "equation",
                Attribute::String(case.equation.as_bytes().to_vec()),
            )],
        );
        let model = Model::new(&graph);
        let claim_shapes = case
            .input_shapes
            .iter()
            .map(|shape| static_shape(shape.iter().copied()))
            .collect::<Vec<_>>();
        let claim_dtypes = vec![dtype; inputs.len()];
        let claim = self.ep.supports_op(
            model.graph.node(node),
            case.opset,
            &claim_shapes,
            &claim_dtypes,
            &[],
        );
        if !claim.is_supported() {
            return Err(CudaAdapterError(format!(
                "case {} was not claimed by CUDA: {:?}",
                case.id,
                claim.reason()
            )));
        }
        let kernel = self
            .ep
            .get_kernel(model.graph.node(node), &case.input_shapes, case.opset)
            .map_err(adapter_error)?;
        let input_buffers = inputs
            .iter()
            .map(|input| {
                let buffer = self
                    .ep
                    .allocate(input.bytes.len().max(1), 256)
                    .map_err(adapter_error)?;
                if !input.bytes.is_empty() {
                    // SAFETY: the fresh allocation covers the complete input payload.
                    unsafe {
                        self.ep
                            .runtime()
                            .htod(&input.bytes, cuptr(buffer.as_ptr()))
                            .map_err(adapter_error)?;
                    }
                }
                Ok(buffer)
            })
            .collect::<Result<Vec<_>, CudaAdapterError>>()?;
        let mut output = self
            .ep
            .allocate(
                dtype.storage_bytes(output_shape.iter().product()).max(1),
                256,
            )
            .map_err(adapter_error)?;
        let output_bytes_len = dtype.storage_bytes(output_shape.iter().product());
        if output_bytes_len != 0 {
            let zeros = vec![0u8; output_bytes_len];
            // SAFETY: the output allocation is at least `output_bytes_len`.
            unsafe {
                self.ep
                    .runtime()
                    .htod(&zeros, cuptr(output.as_ptr()))
                    .map_err(adapter_error)?;
            }
        }
        let input_strides = case
            .input_shapes
            .iter()
            .map(|shape| compute_contiguous_strides(shape))
            .collect::<Vec<_>>();
        let output_strides = compute_contiguous_strides(&output_shape);
        let views = input_buffers
            .iter()
            .zip(&case.input_shapes)
            .zip(&input_strides)
            .map(|((buffer, shape), strides)| {
                TensorView::new(
                    DevicePtr(buffer.as_ptr()),
                    dtype,
                    shape,
                    strides,
                    self.ep.device_id(),
                )
            })
            .collect::<Vec<_>>();
        let override_route = match request.probe.route {
            ForcedRoute::GenericNative => EinsumRouteOverride::GenericNative,
            ForcedRoute::OptimizedDp | ForcedRoute::OptimizedHeuristic => {
                EinsumRouteOverride::Optimized
            }
            ForcedRoute::CudaCublas => EinsumRouteOverride::CudaCublas,
            ForcedRoute::MatMul => {
                return Err(CudaAdapterError(
                    "CPU MatMul route was supplied to the CUDA adapter".into(),
                ));
            }
        };
        let execute = |output: &mut onnx_runtime_ep_api::DeviceBuffer| {
            execute_einsum_with_route(
                kernel.as_ref(),
                &views,
                &mut [TensorMut::new(
                    DevicePtrMut(output.as_mut_ptr()),
                    dtype,
                    &output_shape,
                    &output_strides,
                    self.ep.device_id(),
                )],
                override_route,
            )
        };

        reset_einsum_execution_stats();
        execute(&mut output).map_err(adapter_error)?;
        if !kernel.capture_support().is_supported() {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} did not become capture-ready: {:?}",
                case.id,
                request.probe.route,
                kernel.capture_support().reason()
            )));
        }
        let allocations = self.ep.runtime().allocation_counts();
        let transfers = self.ep.runtime().transfer_counts();
        let synchronizations = self.ep.runtime().forced_synchronization_count();
        let graph_before = self.ep.runtime().graph_execution_counts();
        self.ep
            .runtime()
            .begin_graph_capture(&[kernel.as_ref()])
            .map_err(adapter_error)?;
        execute(&mut output).map_err(adapter_error)?;
        self.ep
            .runtime()
            .end_graph_capture()
            .map_err(adapter_error)?;
        if self.ep.runtime().allocation_counts() != allocations {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} allocated or freed device memory during capture",
                case.id, request.probe.route
            )));
        }
        if self.ep.runtime().transfer_counts() != transfers {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} transferred host/device data during capture",
                case.id, request.probe.route
            )));
        }
        if self.ep.runtime().forced_synchronization_count() != synchronizations {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} forced a host synchronization during capture",
                case.id, request.probe.route
            )));
        }
        for _ in 0..2 {
            self.ep.runtime().replay_graph().map_err(adapter_error)?;
        }
        if self.ep.runtime().allocation_counts() != allocations {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} allocated or freed device memory during replay",
                case.id, request.probe.route
            )));
        }
        if self.ep.runtime().transfer_counts() != transfers {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} transferred host/device data during replay",
                case.id, request.probe.route
            )));
        }
        if self.ep.runtime().forced_synchronization_count() != synchronizations {
            return Err(CudaAdapterError(format!(
                "case {} route {:?} forced a host synchronization during replay",
                case.id, request.probe.route
            )));
        }
        self.ep.runtime().synchronize().map_err(adapter_error)?;
        let graph_after = self.ep.runtime().graph_execution_counts();
        let mut output_bytes = vec![0u8; output_bytes_len];
        if !output_bytes.is_empty() {
            // SAFETY: the host vector exactly covers the contiguous output.
            unsafe {
                self.ep
                    .runtime()
                    .dtoh(&mut output_bytes, cuptr(output.as_ptr()))
                    .map_err(adapter_error)?;
            }
        }
        self.ep.runtime().reset_graph().map_err(adapter_error)?;
        let stats = einsum_execution_stats();
        let actual_route = match stats.last_route {
            Some(CudaEinsumRoute::GenericNative) => ForcedRoute::GenericNative,
            Some(CudaEinsumRoute::OptimizedDp) => ForcedRoute::OptimizedDp,
            Some(CudaEinsumRoute::OptimizedHeuristic) => ForcedRoute::OptimizedHeuristic,
            Some(CudaEinsumRoute::CudaCublas) => ForcedRoute::CudaCublas,
            other => {
                return Err(CudaAdapterError(format!(
                    "case {} route {:?} reported non-arithmetic CUDA route {other:?}",
                    case.id, request.probe.route
                )));
            }
        };
        let planner_quality = match stats.last_route {
            Some(CudaEinsumRoute::OptimizedDp) => Some(PlannerQuality::ExactSubsetDp),
            Some(CudaEinsumRoute::OptimizedHeuristic) => {
                Some(PlannerQuality::DeterministicHeuristic)
            }
            _ => None,
        };
        let output_tensor = CanonicalTensor::new(
            case.dtype,
            output_shape,
            tensor_bits(case.dtype, &output_bytes),
        )
        .map_err(adapter_error)?;
        for buffer in input_buffers {
            self.ep.deallocate(buffer).map_err(adapter_error)?;
        }
        self.ep.deallocate(output).map_err(adapter_error)?;
        Ok(BackendObservation {
            backend: BackendKind::Cuda,
            route: actual_route,
            planner_quality,
            workspace_bytes: usize::try_from(
                stats
                    .workspace_bytes
                    .saturating_add(stats.persistent_metadata_bytes),
            )
            .map_err(adapter_error)?,
            captures: (graph_after.captures - graph_before.captures) as usize,
            replays: (graph_after.replays - graph_before.replays) as usize,
            capture_fallbacks: 0,
            output: output_tensor,
        })
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_legal_corpus_executes_all_forced_cuda_routes() {
    let _lock = suite_lock();
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let adapter = CudaAdapter { ep: &ep };
    let cases = default_corpus();
    let mut generic = 0usize;
    let mut optimized_dp = 0usize;
    let mut optimized_heuristic = 0usize;
    let mut cublas = 0usize;
    for case in &cases {
        let inputs = materialize_inputs(case)
            .unwrap_or_else(|error| panic!("{} input materialization: {error}", case.id));
        let expected =
            evaluate(case, &inputs).unwrap_or_else(|error| panic!("{} oracle: {error}", case.id));
        for probe in case
            .route_probes
            .iter()
            .filter(|probe| probe.backend == BackendKind::Cuda)
        {
            match probe.route {
                ForcedRoute::GenericNative => generic += 1,
                ForcedRoute::OptimizedDp => optimized_dp += 1,
                ForcedRoute::OptimizedHeuristic => optimized_heuristic += 1,
                ForcedRoute::CudaCublas => cublas += 1,
                ForcedRoute::MatMul => panic!("CUDA corpus contained a CPU MatMul route"),
            }

            let observed = adapter
                .execute(ExecutionRequest {
                    case,
                    inputs: &inputs,
                    probe,
                })
                .unwrap_or_else(|error| panic!("{} forced {:?}: {error}", case.id, probe.route));
            verify_observation(case, &expected, probe, &observed)
                .unwrap_or_else(|error| panic!("{} forced {:?}: {error}", case.id, probe.route));
        }
    }
    assert_eq!(generic, cases.len(), "every legal case needs GenericNative");
    assert!(optimized_dp > 0, "exact-DP route did not fire");
    assert!(optimized_heuristic > 0, "heuristic route did not fire");
    assert!(cublas > 0, "cuBLASLt route did not fire");
    eprintln!(
        "CUDA_EINSUM_CORPUS cases={} generic={} optimized_dp={} optimized_heuristic={} cublas={}",
        cases.len(),
        generic,
        optimized_dp,
        optimized_heuristic,
        cublas
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn every_schema_numeric_dtype_executes_generic_cuda() {
    let _lock = suite_lock();
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let adapter = CudaAdapter { ep: &ep };
    let dtypes = [
        ConformanceDType::Uint8,
        ConformanceDType::Uint16,
        ConformanceDType::Uint32,
        ConformanceDType::Uint64,
        ConformanceDType::Int8,
        ConformanceDType::Int16,
        ConformanceDType::Int32,
        ConformanceDType::Int64,
        ConformanceDType::Float16,
        ConformanceDType::Float32,
        ConformanceDType::Float64,
        ConformanceDType::BFloat16,
    ];
    for dtype in dtypes {
        let case = CaseRecord {
            id: format!("cuda-dtype-{dtype:?}"),
            equation: "i,i->".into(),
            opset: if dtype == ConformanceDType::BFloat16 {
                28
            } else {
                12
            },
            dtype,
            input_shapes: vec![vec![7], vec![7]],
            values: ValueSpec {
                seed: 0x0C0D_AE15,
                profile: if dtype.integer_bits().is_some() {
                    ValueProfile::IntegerEdges
                } else {
                    ValueProfile::Finite
                },
            },
            limits: CaseLimits::default(),
            route_probes: vec![RouteProbe {
                backend: BackendKind::Cuda,
                route: ForcedRoute::GenericNative,
                planner_quality: None,
                comparison: ComparisonMode::ConditionAware,
                workspace: WorkspaceClass::Gpu64MiB,
                capture: CaptureExpectation::MustCapture,
            }],
        };
        let inputs = materialize_inputs(&case).unwrap();
        let expected = evaluate(&case, &inputs).unwrap();
        let observed = adapter
            .execute(ExecutionRequest {
                case: &case,
                inputs: &inputs,
                probe: &case.route_probes[0],
            })
            .unwrap_or_else(|error| panic!("{dtype:?}: {error}"));
        verify_observation(&case, &expected, &case.route_probes[0], &observed)
            .unwrap_or_else(|error| panic!("{dtype:?}: {error}"));
    }
}
