//! Exercise the exact adapter used by the packed-expert benchmark executables.
#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::{
    EpError, ExecutionProvider, Kernel, OutputMetadata, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceExecutionPreparation, WorkspacePreparation, WorkspaceRequirement,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};
use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

#[derive(Default)]
struct Observations {
    metadata: AtomicUsize,
    refinements: AtomicUsize,
    launches: AtomicUsize,
    last_origin: AtomicUsize,
    bytes: AtomicU64,
    alignment: AtomicUsize,
    fail_prepare: AtomicBool,
    fail_launch: AtomicBool,
    bind: AtomicBool,
}
struct ConsumedPlan {
    observations: Arc<Observations>,
    prepared: Cell<bool>,
}
fn request(bytes: u64, alignment: usize) -> WorkspaceRequirement {
    WorkspaceRequirement {
        bytes,
        alignment,
        ..WorkspaceRequirement::NONE
    }
}
fn fixture() -> (Box<dyn Kernel>, Arc<Observations>) {
    let observations = Arc::new(Observations::default());
    observations.bytes.store(16, Ordering::Relaxed);
    observations.alignment.store(4, Ordering::Relaxed);
    (
        Box::new(ConsumedPlan {
            observations: observations.clone(),
            prepared: Cell::new(false),
        }),
        observations,
    )
}
impl Kernel for ConsumedPlan {
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata],
        outputs: &[OutputMetadata],
    ) -> Result<WorkspacePreparation> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);
        self.observations.metadata.fetch_add(1, Ordering::Relaxed);
        Ok(WorkspacePreparation::Ready(request(16, 4)))
    }
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata],
        output_metadata: &[OutputMetadata],
        cached: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        self.prepared.set(false);
        let o = &self.observations;
        o.refinements.fetch_add(1, Ordering::Relaxed);
        assert_eq!(cached, Some(WorkspacePreparation::Ready(request(16, 4))));
        assert_eq!(inputs.len(), metadata.len());
        assert_eq!(outputs.len(), output_metadata.len());
        assert_eq!(
            outputs.len(),
            1,
            "initial refinement must receive actual outputs too"
        );
        o.last_origin
            .store(inputs[0].data.0 as usize, Ordering::Relaxed);
        if o.fail_prepare.load(Ordering::Relaxed) {
            return Err(EpError::KernelFailed(
                "injected live-binding preparation failure".into(),
            ));
        }
        let req = request(
            o.bytes.load(Ordering::Relaxed),
            o.alignment.load(Ordering::Relaxed),
        );
        if o.bind.load(Ordering::Relaxed) {
            Ok(WorkspaceExecutionPreparation::Bind(req))
        } else {
            self.prepared.set(true);
            Ok(WorkspaceExecutionPreparation::Ready(req))
        }
    }
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
        assert!(
            self.prepared.replace(false),
            "every launch consumes one fresh plan"
        );
        self.observations.launches.fetch_add(1, Ordering::Relaxed);
        if self.observations.fail_launch.load(Ordering::Relaxed) {
            Err(EpError::KernelFailed(
                "injected launch failure after consuming plan".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn benchmark_adapter_caches_metadata_but_refines_every_live_launch() {
    let cpu = CpuExecutionProvider::new();
    let (kernel, counts) = fixture();
    let a = Tensor::floats(FloatDType::F32, &[1], &[1.]);
    let b = Tensor::floats(FloatDType::F32, &[1], &[2.]);
    let mut y = Tensor::zeros(FloatDType::F32, &[1]);
    let mut prepared = PreparedKernel::new(&cpu, kernel, &[a.view()], &[y.view_mut()]);
    for input in [&a, &b, &a, &b] {
        prepared
            .execute(&[input.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(
            counts.last_origin.load(Ordering::Relaxed),
            input.view().data.0 as usize
        );
    }
    assert_eq!(counts.metadata.load(Ordering::Relaxed), 1);
    assert_eq!(counts.refinements.load(Ordering::Relaxed), 5);
    assert_eq!(counts.launches.load(Ordering::Relaxed), 4);
}

#[test]
fn benchmark_adapter_rejects_changed_geometry_before_reusing_cached_facts() {
    let cpu = CpuExecutionProvider::new();
    let (kernel, counts) = fixture();
    let a = Tensor::floats(FloatDType::F32, &[1], &[1.]);
    let wrong_shape = Tensor::floats(FloatDType::F32, &[1, 1], &[1.]);
    let wrong_dtype = Tensor::u8(&[1], &[1]);
    let mut y = Tensor::zeros(FloatDType::F32, &[1]);
    let mut wrong_output = Tensor::zeros(FloatDType::F32, &[1, 1]);
    let mut prepared = PreparedKernel::new(&cpu, kernel, &[a.view()], &[y.view_mut()]);
    for input in [
        wrong_shape.view(),
        wrong_dtype.view(),
        TensorView::absent(DataType::Float32),
    ] {
        let error = prepared.execute(&[input], &mut [y.view_mut()]).unwrap_err();
        assert!(error.to_string().contains("new geometry"), "{error}");
    }
    assert!(prepared.execute(&[], &mut [y.view_mut()]).is_err());
    assert!(
        prepared
            .execute(&[a.view()], &mut [wrong_output.view_mut()])
            .is_err()
    );
    assert!(prepared.execute(&[a.view()], &mut []).is_err());
    assert_eq!(counts.refinements.load(Ordering::Relaxed), 1);
    assert_eq!(counts.launches.load(Ordering::Relaxed), 0);
    prepared.execute(&[a.view()], &mut [y.view_mut()]).unwrap();
}

#[test]
fn benchmark_adapter_refuses_growth_alignment_and_unresolved_binding_without_launch() {
    let cpu = CpuExecutionProvider::new();
    let (kernel, counts) = fixture();
    let a = Tensor::floats(FloatDType::F32, &[1], &[1.]);
    let mut y = Tensor::zeros(FloatDType::F32, &[1]);
    let mut prepared = PreparedKernel::new(&cpu, kernel, &[a.view()], &[y.view_mut()]);
    for (bytes, alignment) in [(17, 4), (16, 32), (16, 0), (16, 3)] {
        counts.bytes.store(bytes, Ordering::Relaxed);
        counts.alignment.store(alignment, Ordering::Relaxed);
        let error = prepared
            .execute(&[a.view()], &mut [y.view_mut()])
            .unwrap_err();
        assert!(error.to_string().contains("fixed scratch owner"), "{error}");
    }
    counts.bytes.store(16, Ordering::Relaxed);
    counts.alignment.store(4, Ordering::Relaxed);
    counts.bind.store(true, Ordering::Relaxed);
    let error = prepared
        .execute(&[a.view()], &mut [y.view_mut()])
        .unwrap_err();
    assert!(error.to_string().contains("real prefix binding"), "{error}");
    assert_eq!(counts.launches.load(Ordering::Relaxed), 0);
    counts.bind.store(false, Ordering::Relaxed);
    prepared.execute(&[a.view()], &mut [y.view_mut()]).unwrap();
}

#[test]
fn benchmark_adapter_reprepares_after_preparation_and_consuming_launch_failures() {
    let cpu = CpuExecutionProvider::new();
    let (kernel, counts) = fixture();
    let a = Tensor::floats(FloatDType::F32, &[1], &[1.]);
    let mut y = Tensor::zeros(FloatDType::F32, &[1]);
    let mut prepared = PreparedKernel::new(&cpu, kernel, &[a.view()], &[y.view_mut()]);
    counts.fail_prepare.store(true, Ordering::Relaxed);
    assert!(prepared.execute(&[a.view()], &mut [y.view_mut()]).is_err());
    assert_eq!(counts.launches.load(Ordering::Relaxed), 0);
    counts.fail_prepare.store(false, Ordering::Relaxed);
    counts.fail_launch.store(true, Ordering::Relaxed);
    assert!(prepared.execute(&[a.view()], &mut [y.view_mut()]).is_err());
    counts.fail_launch.store(false, Ordering::Relaxed);
    prepared.execute(&[a.view()], &mut [y.view_mut()]).unwrap();
    assert_eq!(counts.refinements.load(Ordering::Relaxed), 4);
    assert_eq!(counts.launches.load(Ordering::Relaxed), 2);
}

#[test]
fn benchmark_adapter_repeats_real_float32_and_q8_packed_kernels_with_complete_outputs() {
    for rows in [1, 3] {
        for q4 in [false, true] {
            let width = if q4 { 18 } else { 34 };
            let mut packed = vec![if q4 { 0x88 } else { 0 }; 2 * width];
            for feature in 0..2 {
                let block = &mut packed[feature * width..][..width];
                block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
                if q4 {
                    block[2 + feature] += 1;
                } else {
                    block[2 + feature] = 1;
                }
            }
            let x = Tensor::floats(FloatDType::F32, &[rows, 32], &vec![127.; rows * 32]);
            let w = Tensor::u8(&[2, 1, width], &packed);
            for arithmetic in ["float32", "q8_0"] {
                let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
                node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
                for (name, value) in [
                    ("K", Attribute::Int(32)),
                    ("N", Attribute::Int(2)),
                    (
                        "format",
                        Attribute::String(if q4 {
                            b"gguf:q4_0".to_vec()
                        } else {
                            b"q8_0".to_vec()
                        }),
                    ),
                    (
                        "activation_arithmetic",
                        Attribute::String(arithmetic.as_bytes().to_vec()),
                    ),
                    (
                        "block_layout_version",
                        Attribute::Int(if arithmetic == "float32" { 1 } else { 2 }),
                    ),
                    (
                        onnx_runtime_optimizer::quantized_matmul::STREAM_DECODED_ROWS,
                        Attribute::Int(1),
                    ),
                ] {
                    node.attributes.insert(name.into(), value);
                }
                let cpu = CpuExecutionProvider::new();
                let kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                let views = [x.view(), w.view()];
                let mut y = Tensor::zeros(FloatDType::F32, &[rows, 2]);
                let mut prepared = PreparedKernel::new(&cpu, kernel, &views, &[y.view_mut()]);
                for _ in 0..6 {
                    prepared.execute(&views, &mut [y.view_mut()]).unwrap();
                    assert_eq!(y.f32s(), vec![127.; rows * 2]);
                }
            }
        }
    }
}

#[test]
fn benchmark_adapter_scopes_cached_preparation_to_the_provider_pool_context() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let same_width = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let cpu = CpuExecutionProvider::new();
    let x = Tensor::floats(FloatDType::F32, &[3, 32], &[127.; 96]);
    let mut packed = vec![0u8; 68];
    for feature in 0..2 {
        packed[feature * 34..feature * 34 + 2].copy_from_slice(&half::f16::ONE.to_le_bytes());
        packed[feature * 34 + 2 + feature] = 1;
    }
    let w = Tensor::u8(&[2, 1, 34], &packed);
    for arithmetic in ["float32", "q8_0"] {
        let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
        node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
        for (name, value) in [
            ("K", Attribute::Int(32)),
            ("N", Attribute::Int(2)),
            ("format", Attribute::String(b"q8_0".to_vec())),
            (
                "activation_arithmetic",
                Attribute::String(arithmetic.as_bytes().to_vec()),
            ),
            (
                "block_layout_version",
                Attribute::Int(if arithmetic == "float32" { 1 } else { 2 }),
            ),
            (
                onnx_runtime_optimizer::quantized_matmul::STREAM_DECODED_ROWS,
                Attribute::Int(1),
            ),
        ] {
            node.attributes.insert(name.into(), value);
        }
        let kernel = cpu.get_kernel(&node, &[], 1).unwrap();
        let mut output = Tensor::zeros(FloatDType::F32, &[3, 2]);
        let mut prepared = one.install(|| {
            PreparedKernel::new(&cpu, kernel, &[x.view(), w.view()], &[output.view_mut()])
        });
        let error = four
            .install(|| prepared.execute(&[x.view(), w.view()], &mut [output.view_mut()]))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("provider workspace context changed"),
            "{error}"
        );
        assert_eq!(output.f32s(), [0.; 6]);
        // Context equality means an equivalent bound, not identical pool IDs.
        same_width
            .install(|| prepared.execute(&[x.view(), w.view()], &mut [output.view_mut()]))
            .unwrap();
        assert_eq!(output.f32s(), [127.; 6]);
        one.install(|| prepared.execute(&[x.view(), w.view()], &mut [output.view_mut()]))
            .unwrap();
        assert_eq!(output.f32s(), [127.; 6]);
    }
}
