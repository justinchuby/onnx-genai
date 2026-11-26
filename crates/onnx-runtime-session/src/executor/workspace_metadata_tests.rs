//! Metadata memo regressions; registered as a child of kernel_cache.
use super::*;
use onnx_runtime_ep_api::{
    EpError, Result as EpResult, WorkspaceContextKey, WorkspaceExecutionPreparation,
    WorkspacePreparation, WorkspaceRequirement,
};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Default)]
struct Observations {
    queries: AtomicUsize,
    fail_next: AtomicBool,
    live_bytes: AtomicUsize,
    live_calls: AtomicUsize,
    last_origin: AtomicUsize,
    drops: AtomicUsize,
}
struct Probe {
    answer: WorkspacePreparation,
    observed: Arc<Observations>,
    execution: Cell<Option<usize>>,
}
fn requirement(bytes: usize) -> WorkspaceRequirement {
    WorkspaceRequirement {
        bytes: bytes as u64,
        alignment: 4,
        ..WorkspaceRequirement::NONE
    }
}
fn entry(answer: WorkspacePreparation, observed: &Arc<Observations>) -> CachedKernel {
    CachedKernel::new(Box::new(Probe {
        answer,
        observed: observed.clone(),
        execution: Cell::new(None),
    }))
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.observed.drops.fetch_add(1, Ordering::Relaxed);
    }
}
impl Kernel for Probe {
    fn workspace_requirement(
        &self,
        _: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
    ) -> EpResult<WorkspacePreparation> {
        self.observed.queries.fetch_add(1, Ordering::Relaxed);
        if self.observed.fail_next.swap(false, Ordering::Relaxed) {
            return Err(EpError::KernelFailed(
                "metadata probe query rejected; retry preparation".into(),
            ));
        }
        Ok(self.answer)
    }
    fn set_constant_inputs(&mut self, inputs: &[bool]) {
        self.answer = WorkspacePreparation::Ready(requirement(if inputs.first() == Some(&true) {
            128
        } else {
            64
        }));
        self.execution.set(None);
    }
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        _: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
        cached: Option<WorkspacePreparation>,
    ) -> EpResult<WorkspaceExecutionPreparation> {
        self.execution.set(None);
        if inputs.len() != 1 || metadata.len() != 1 || inputs[0].shape != metadata[0].shape {
            return Err(EpError::KernelFailed(
                "live probe requires one input matching canonical metadata".into(),
            ));
        }
        assert_eq!(cached, Some(self.answer));
        self.observed.live_calls.fetch_add(1, Ordering::Relaxed);
        self.observed
            .last_origin
            .store(inputs[0].data_ptr::<u8>() as usize, Ordering::Relaxed);
        let bytes = self.observed.live_bytes.load(Ordering::Relaxed);
        self.execution.set(Some(bytes));
        Ok(WorkspaceExecutionPreparation::Ready(requirement(bytes)))
    }
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> EpResult<()> {
        self.execution.take().ok_or_else(|| {
            EpError::KernelFailed(
                "live probe plan absent or consumed; refine before execution".into(),
            )
        })?;
        Ok(())
    }
}
fn inputs() -> [TensorMetadata<'static>; 1] {
    [TensorMetadata::new(DataType::Float32, &[2], true)]
}
fn outputs() -> [OutputMetadata<'static>; 1] {
    [OutputMetadata::tensor(DataType::Float32, &[2])]
}

#[test]
fn ready_and_deferred_metadata_each_query_only_once_on_exact_hits() {
    for answer in [
        WorkspacePreparation::Ready(requirement(64)),
        WorkspacePreparation::Deferred,
    ] {
        let observed = Arc::new(Observations::default());
        let cached = entry(answer, &observed);
        for _ in 0..8 {
            assert_eq!(
                cached
                    .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
                    .unwrap(),
                answer
            );
        }
        assert_eq!(observed.queries.load(Ordering::Relaxed), 1);
        assert_eq!(observed.live_calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn input_dtype_shape_presence_and_arity_are_distinct_memo_keys() {
    let observed = Arc::new(Observations::default());
    let cached = entry(WorkspacePreparation::Deferred, &observed);
    let cases = [
        vec![TensorMetadata::new(DataType::Float32, &[2], true)],
        vec![TensorMetadata::new(DataType::Float64, &[2], true)],
        vec![TensorMetadata::new(DataType::Float64, &[3], true)],
        vec![TensorMetadata::new(DataType::Float64, &[3], false)],
        vec![],
        vec![TensorMetadata::new(DataType::Float64, &[3], false); 2],
    ];
    for (index, facts) in cases.iter().enumerate() {
        for _ in 0..2 {
            assert_eq!(
                cached
                    .workspace_preparation(WorkspaceContextKey::STABLE, facts, &outputs())
                    .unwrap(),
                WorkspacePreparation::Deferred
            );
        }
        assert_eq!(observed.queries.load(Ordering::Relaxed), index + 1);
    }
}

#[test]
fn output_absent_unresolved_tensor_dtype_shape_arity_and_context_invalidate() {
    let observed = Arc::new(Observations::default());
    let cached = entry(WorkspacePreparation::Ready(requirement(64)), &observed);
    let cases = [
        vec![OutputMetadata::Absent],
        vec![OutputMetadata::Unresolved {
            dtype: DataType::Float32,
        }],
        vec![OutputMetadata::tensor(DataType::Float32, &[2])],
        vec![OutputMetadata::tensor(DataType::Float32, &[3])],
        vec![OutputMetadata::tensor(DataType::Float64, &[3])],
        vec![],
        vec![OutputMetadata::Absent; 2],
    ];
    for (index, facts) in cases.iter().enumerate() {
        for _ in 0..2 {
            cached
                .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), facts)
                .unwrap();
        }
        assert_eq!(observed.queries.load(Ordering::Relaxed), index + 1);
    }
    for (index, context) in [1, 4, 1].into_iter().enumerate() {
        for _ in 0..2 {
            cached
                .workspace_preparation(WorkspaceContextKey::new(context), &inputs(), &outputs())
                .unwrap();
        }
        assert_eq!(
            observed.queries.load(Ordering::Relaxed),
            cases.len() + index + 1
        );
    }
}

#[test]
fn failed_query_never_publishes_and_preserves_a_previous_successful_key() {
    let observed = Arc::new(Observations::default());
    let cached = entry(WorkspacePreparation::Deferred, &observed);
    let context = WorkspaceContextKey::STABLE;
    observed.fail_next.store(true, Ordering::Relaxed);
    assert!(
        cached
            .workspace_preparation(context, &inputs(), &outputs())
            .is_err()
    );
    assert!(cached.workspace.borrow().is_none());
    cached
        .workspace_preparation(context, &inputs(), &outputs())
        .unwrap();
    assert_eq!(observed.queries.load(Ordering::Relaxed), 2);
    let changed = [TensorMetadata::new(DataType::Float32, &[3], true)];
    observed.fail_next.store(true, Ordering::Relaxed);
    assert!(
        cached
            .workspace_preparation(context, &changed, &outputs())
            .is_err()
    );
    cached
        .workspace_preparation(context, &inputs(), &outputs())
        .unwrap();
    assert_eq!(
        observed.queries.load(Ordering::Relaxed),
        3,
        "failed candidate must not replace the prior memo"
    );
    cached
        .workspace_preparation(context, &changed, &outputs())
        .unwrap();
    cached
        .workspace_preparation(context, &changed, &outputs())
        .unwrap();
    assert_eq!(observed.queries.load(Ordering::Relaxed), 4);
}

#[test]
fn removing_and_recreating_the_same_kernel_key_never_inherits_a_memo() {
    let observed = Arc::new(Observations::default());
    let mut cache = KernelCache::default();
    let key = KernelKey {
        node: 7,
        shapes: vec![vec![2]],
        shape_facts: KernelShapeFacts::default(),
    };
    cache.entries.insert(
        key.clone(),
        entry(WorkspacePreparation::Ready(requirement(64)), &observed),
    );
    cache.entries[&key]
        .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
        .unwrap();
    drop(cache.entries.remove(&key).unwrap());
    assert_eq!(observed.drops.load(Ordering::Relaxed), 1);
    cache.entries.insert(
        key.clone(),
        entry(WorkspacePreparation::Ready(requirement(128)), &observed),
    );
    assert_eq!(
        cache.entries[&key]
            .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
            .unwrap(),
        WorkspacePreparation::Ready(requirement(128))
    );
    assert_eq!(observed.queries.load(Ordering::Relaxed), 2);
}

#[test]
fn mutable_kernel_reconfiguration_clears_metadata_before_the_next_query() {
    let observed = Arc::new(Observations::default());
    let mut cached = entry(WorkspacePreparation::Ready(requirement(64)), &observed);
    cached
        .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
        .unwrap();
    cached.kernel_mut().set_constant_inputs(&[true]);
    assert!(cached.workspace.borrow().is_none());
    for _ in 0..2 {
        assert_eq!(
            cached
                .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
                .unwrap(),
            WorkspacePreparation::Ready(requirement(128))
        );
    }
    assert_eq!(observed.queries.load(Ordering::Relaxed), 2);
}

#[test]
fn metadata_hit_never_reuses_a_consumed_live_plan_or_storage_origin() {
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::DeviceId;
    let observed = Arc::new(Observations::default());
    let cached = entry(WorkspacePreparation::Deferred, &observed);
    let a = [1f32, 2.];
    let b = [3f32, 4.];
    for (index, bytes) in [0usize, 64, 128, 32, 0].into_iter().enumerate() {
        let storage = if index % 2 == 0 { &a } else { &b };
        let view = TensorView::new(
            DevicePtr(storage.as_ptr().cast()),
            DataType::Float32,
            &[2],
            &[1],
            DeviceId::cpu(),
        );
        observed.live_bytes.store(bytes, Ordering::Relaxed);
        let answer = cached
            .workspace_preparation(WorkspaceContextKey::STABLE, &inputs(), &outputs())
            .unwrap();
        let ready = cached
            .kernel
            .workspace_requirement_for_execution(&[view], &[], &inputs(), &outputs(), Some(answer))
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(ready.bytes, bytes as u64);
        assert_eq!(
            observed.last_origin.load(Ordering::Relaxed),
            storage.as_ptr() as usize
        );
        cached.kernel.execute(&[view], &mut []).unwrap();
        assert!(
            cached
                .kernel
                .execute(&[view], &mut [])
                .unwrap_err()
                .to_string()
                .contains("absent or consumed")
        );
    }
    assert_eq!(observed.queries.load(Ordering::Relaxed), 1);
    assert_eq!(observed.live_calls.load(Ordering::Relaxed), 5);
}
