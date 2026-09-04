use std::collections::HashSet;
use std::fmt;

use onnx_runtime_einsum_conformance::{
    BackendAdapter, BackendKind, BackendObservation, CanonicalTensor, CaseLimits, CaseRecord,
    ComparisonMode, ConformanceDType, ExecutionRequest, ForcedRoute, PlannerQuality, RouteProbe,
    ValueProfile, ValueSpec, WorkspaceClass, default_corpus, evaluate, infer_output_shape,
    materialize_inputs, verify_observation,
};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, KernelFactory, TensorMut, TensorView};
use onnx_runtime_ep_cpu::kernels::einsum::{
    EinsumExecutionMode, EinsumFactory, EinsumScratchRetention, benchmark_execute_route,
    benchmark_last_workspace_bytes,
};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

#[derive(Debug)]
struct AdapterError(String);

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AdapterError {}

struct AlignedTensor {
    storage: Vec<u64>,
    shape: Vec<usize>,
    strides: Vec<i64>,
    dtype: DataType,
}

impl AlignedTensor {
    fn from_canonical(tensor: &CanonicalTensor) -> Self {
        let dtype = data_type(tensor.dtype());
        let byte_len = tensor.raw_bits().len() * tensor.dtype().byte_size();
        let mut storage = vec![0u64; byte_len.div_ceil(8).max(1)];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), storage.len() * 8)
        };
        for (index, &bits) in tensor.raw_bits().iter().enumerate() {
            write_bits(
                tensor.dtype(),
                bits,
                &mut bytes[index * tensor.dtype().byte_size()..][..tensor.dtype().byte_size()],
            );
        }
        Self {
            storage,
            shape: tensor.shape().to_vec(),
            strides: compute_contiguous_strides(tensor.shape()),
            dtype,
        }
    }

    fn zeros(dtype: ConformanceDType, shape: Vec<usize>) -> Self {
        let byte_len = shape.iter().product::<usize>() * dtype.byte_size();
        Self {
            storage: vec![0u64; byte_len.div_ceil(8).max(1)],
            strides: compute_contiguous_strides(&shape),
            shape,
            dtype: data_type(dtype),
        }
    }

    fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.storage.as_ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    fn view_mut(&mut self) -> TensorMut<'_> {
        TensorMut::new(
            DevicePtrMut(self.storage.as_mut_ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    fn to_canonical(&self, dtype: ConformanceDType) -> CanonicalTensor {
        let count = self.shape.iter().product::<usize>();
        let bytes = unsafe {
            std::slice::from_raw_parts(self.storage.as_ptr().cast::<u8>(), self.storage.len() * 8)
        };
        let bits = (0..count)
            .map(|index| {
                read_bits(
                    dtype,
                    &bytes[index * dtype.byte_size()..][..dtype.byte_size()],
                )
            })
            .collect();
        CanonicalTensor::new(dtype, self.shape.clone(), bits).unwrap()
    }
}

fn data_type(dtype: ConformanceDType) -> DataType {
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

fn write_bits(dtype: ConformanceDType, bits: u64, destination: &mut [u8]) {
    match dtype.byte_size() {
        1 => destination.copy_from_slice(&(bits as u8).to_ne_bytes()),
        2 => destination.copy_from_slice(&(bits as u16).to_ne_bytes()),
        4 => destination.copy_from_slice(&(bits as u32).to_ne_bytes()),
        8 => destination.copy_from_slice(&bits.to_ne_bytes()),
        width => unreachable!("unsupported conformance element width {width}"),
    }
}

fn read_bits(dtype: ConformanceDType, source: &[u8]) -> u64 {
    match dtype.byte_size() {
        1 => u64::from(u8::from_ne_bytes([source[0]])),
        2 => u64::from(u16::from_ne_bytes(source.try_into().unwrap())),
        4 => u64::from(u32::from_ne_bytes(source.try_into().unwrap())),
        8 => u64::from_ne_bytes(source.try_into().unwrap()),
        width => unreachable!("unsupported conformance element width {width}"),
    }
}

struct CpuAdapter;

impl BackendAdapter for CpuAdapter {
    type Error = AdapterError;

    fn execute(&self, request: ExecutionRequest<'_>) -> Result<BackendObservation, Self::Error> {
        let mode = match request.probe.route {
            ForcedRoute::GenericNative => EinsumExecutionMode::GenericNative,
            ForcedRoute::OptimizedDp | ForcedRoute::OptimizedHeuristic | ForcedRoute::MatMul => {
                EinsumExecutionMode::Optimized
            }
            ForcedRoute::CudaCublas => {
                return Err(AdapterError(
                    "CPU adapter cannot execute a CUDA cuBLAS route".into(),
                ));
            }
        };
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.version = Some(request.case.opset as i64);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(request.case.equation.as_bytes().to_vec()),
        );
        let kernel = EinsumFactory::with_execution_mode(EinsumScratchRetention::default(), mode)
            .create(&node, &request.case.input_shapes)
            .map_err(|error| AdapterError(error.to_string()))?;
        let inputs = request
            .inputs
            .iter()
            .map(AlignedTensor::from_canonical)
            .collect::<Vec<_>>();
        let input_views = inputs.iter().map(AlignedTensor::view).collect::<Vec<_>>();
        let output_shape = infer_output_shape(&request.case.equation, &request.case.input_shapes)
            .map_err(|error| AdapterError(error.to_string()))?;
        let mut output = AlignedTensor::zeros(request.case.dtype, output_shape);
        let route = benchmark_execute_route(&*kernel, &input_views, &mut [output.view_mut()])
            .map_err(|error| AdapterError(error.to_string()))?
            .ok_or_else(|| {
                AdapterError("Einsum route probe did not recognize its kernel".into())
            })?;
        let (route, planner_quality) = match route {
            "generic-native" => (ForcedRoute::GenericNative, None),
            "optimized-dp" => (
                ForcedRoute::OptimizedDp,
                Some(PlannerQuality::ExactSubsetDp),
            ),
            "optimized-heuristic" => (
                ForcedRoute::OptimizedHeuristic,
                Some(PlannerQuality::DeterministicHeuristic),
            ),
            "matmul-direct" | "matmul-materialized" | "matmul-scalar" => {
                (ForcedRoute::MatMul, None)
            }
            other => {
                return Err(AdapterError(format!(
                    "forced {:?} unexpectedly executed CPU route {other}",
                    request.probe.route
                )));
            }
        };
        Ok(BackendObservation {
            backend: BackendKind::Cpu,
            route,
            planner_quality,
            workspace_bytes: benchmark_last_workspace_bytes(&*kernel).expect("known Einsum kernel"),
            captures: 0,
            replays: 0,
            capture_fallbacks: 0,
            output: output.to_canonical(request.case.dtype),
        })
    }
}

#[test]
fn shared_legal_corpus_executes_every_forced_cpu_route() {
    let adapter = CpuAdapter;
    let mut cases = default_corpus();
    let shared_case_count = cases.len();
    for dtype in ConformanceDType::V12_TYPES {
        if cases.iter().any(|case| case.dtype == dtype) {
            continue;
        }
        cases.push(CaseRecord {
            id: format!("cpu-dtype-{dtype:?}").to_ascii_lowercase(),
            equation: "i,i->i".into(),
            opset: 12,
            dtype,
            input_shapes: vec![vec![8], vec![8]],
            values: ValueSpec {
                seed: 0xD7_0000 + dtype.byte_size() as u64,
                profile: if dtype.integer_bits().is_some() {
                    ValueProfile::IntegerEdges
                } else {
                    ValueProfile::Finite
                },
            },
            limits: CaseLimits::default(),
            route_probes: vec![RouteProbe {
                backend: BackendKind::Cpu,
                route: ForcedRoute::GenericNative,
                planner_quality: None,
                comparison: ComparisonMode::ConditionAware,
                workspace: WorkspaceClass::Cpu32MiB,
                capture: onnx_runtime_einsum_conformance::CaptureExpectation::NotApplicable,
            }],
        });
    }
    let mut generic = 0usize;
    let mut exact = 0usize;
    let mut heuristic = 0usize;
    let mut matmul = 0usize;
    let mut dtypes = HashSet::new();
    let mut arities = HashSet::new();

    for case in &cases {
        dtypes.insert(case.dtype);
        arities.insert(case.input_shapes.len());
        let inputs = materialize_inputs(case)
            .unwrap_or_else(|error| panic!("{} input materialization failed: {error}", case.id));
        let expected = evaluate(case, &inputs)
            .unwrap_or_else(|error| panic!("{} oracle failed: {error}", case.id));
        for probe in case
            .route_probes
            .iter()
            .filter(|probe| probe.backend == BackendKind::Cpu)
        {
            let observed = adapter
                .execute(ExecutionRequest {
                    case,
                    inputs: &inputs,
                    probe,
                })
                .unwrap_or_else(|error| {
                    panic!("{} {:?} execution failed: {error}", case.id, probe.route)
                });
            verify_observation(case, &expected, probe, &observed).unwrap_or_else(|error| {
                panic!("{} {:?} conformance failed: {error}", case.id, probe.route)
            });
            match probe.route {
                ForcedRoute::GenericNative => generic += 1,
                ForcedRoute::OptimizedDp => exact += 1,
                ForcedRoute::OptimizedHeuristic => heuristic += 1,
                ForcedRoute::MatMul => matmul += 1,
                ForcedRoute::CudaCublas => unreachable!(),
            }
        }
    }

    assert_eq!(
        generic,
        cases.len(),
        "every legal case must force GenericNative"
    );
    assert!(exact > 0, "the exact-DP route must fire");
    assert!(heuristic > 0, "the deterministic heuristic route must fire");
    assert!(matmul > 0, "the MatMul route must fire");
    for dtype in ConformanceDType::V12_TYPES {
        assert!(dtypes.contains(&dtype), "corpus omitted {dtype:?}");
    }
    assert!(dtypes.contains(&ConformanceDType::BFloat16));
    for arity in [1usize, 2, 3, 4, 8, 16, 64] {
        assert!(arities.contains(&arity), "corpus omitted arity {arity}");
    }
    eprintln!(
        "CPU Einsum corpus: cases={}, generic={generic}, optimized_dp={exact}, \
         optimized_heuristic={heuristic}, matmul={matmul}, added_dtype_cases={}",
        cases.len(),
        cases.len() - shared_case_count
    );
}

#[test]
fn one_hundred_twenty_eight_operand_low_work_uses_generic_fallback() {
    let arity = 128;
    let case = CaseRecord {
        id: "scalar-product-128-operands".into(),
        equation: format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        ),
        opset: 12,
        dtype: ConformanceDType::Int8,
        input_shapes: vec![vec![1]; arity],
        values: ValueSpec {
            seed: 0x128,
            profile: ValueProfile::IntegerEdges,
        },
        limits: CaseLimits::default(),
        route_probes: vec![RouteProbe {
            backend: BackendKind::Cpu,
            route: ForcedRoute::GenericNative,
            planner_quality: None,
            comparison: ComparisonMode::CanonicalBits,
            workspace: WorkspaceClass::Cpu32MiB,
            capture: onnx_runtime_einsum_conformance::CaptureExpectation::NotApplicable,
        }],
    };
    let inputs = materialize_inputs(&case).unwrap();
    let expected = evaluate(&case, &inputs).unwrap();
    let observed = CpuAdapter
        .execute(ExecutionRequest {
            case: &case,
            inputs: &inputs,
            probe: &case.route_probes[0],
        })
        .unwrap();
    verify_observation(&case, &expected, &case.route_probes[0], &observed).unwrap();
}

#[test]
fn generic_parallel_output_tiles_match_the_scalar_fallback_bit_for_bit() {
    let case = CaseRecord {
        id: "parallel-output-tiles".into(),
        equation: "ij,ij->ij".into(),
        opset: 12,
        dtype: ConformanceDType::Float32,
        input_shapes: vec![vec![512, 512], vec![512, 512]],
        values: ValueSpec {
            seed: 0x5151,
            profile: ValueProfile::Finite,
        },
        limits: CaseLimits::default(),
        route_probes: vec![RouteProbe {
            backend: BackendKind::Cpu,
            route: ForcedRoute::GenericNative,
            planner_quality: None,
            comparison: ComparisonMode::CanonicalBits,
            workspace: WorkspaceClass::Cpu32MiB,
            capture: onnx_runtime_einsum_conformance::CaptureExpectation::NotApplicable,
        }],
    };
    let inputs = materialize_inputs(&case).unwrap();
    let before = onnx_runtime_ep_cpu::task_runtime::testing::counters();
    let parallel = CpuAdapter
        .execute(ExecutionRequest {
            case: &case,
            inputs: &inputs,
            probe: &case.route_probes[0],
        })
        .unwrap();
    let after = onnx_runtime_ep_cpu::task_runtime::testing::counters();
    if onnx_runtime_ep_cpu::task_runtime::testing::pool_width() > 1 {
        assert!(
            after.dispatches > before.dispatches,
            "the large GenericNative case must exercise the native output-tile scheduler"
        );
    }
    let serial = {
        let _serial = onnx_runtime_ep_cpu::task_runtime::testing::force_serial();
        CpuAdapter
            .execute(ExecutionRequest {
                case: &case,
                inputs: &inputs,
                probe: &case.route_probes[0],
            })
            .unwrap()
    };
    assert_eq!(
        parallel.output.raw_bits(),
        serial.output.raw_bits(),
        "parallel output tiling must not change per-output reduction order"
    );
}
