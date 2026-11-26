use super::*;
use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use std::sync::Arc;

fn resources(
    limit: u64,
) -> (
    Arc<CpuExecutionProvider>,
    ProcessMemoryManager,
    Arc<LedgerGovernor>,
) {
    let manager = ProcessMemoryManager::new().unwrap();
    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        limit,
        0,
    )));
    let memory = CpuBufferMemory::governed(&manager, books.clone()).unwrap();
    let cpu = Arc::new(
        CpuExecutionProvider::initialized_with_resources(
            CpuMemoryResources::default()
                .with_buffer_memory(memory)
                .unwrap(),
        )
        .unwrap(),
    );
    (cpu, manager, books)
}
fn managed_tiny(
    limit: u64,
) -> (
    NativeDecodeSession,
    ProcessMemoryManager,
    Arc<LedgerGovernor>,
) {
    let (cpu, manager, books) = resources(limit);
    let original = tests::tiny_decoder(false);
    let graph = original.graph().clone();
    drop(original);
    let session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        std::path::Path::new("."),
        cpu,
        None,
    )
    .unwrap();
    let native =
        NativeDecodeSession::from_session_with_io(session, &tests::tiny_decoder_io()).unwrap();
    (native, manager, books)
}
#[test]
fn real_native_input_denial_keeps_empty_state_and_retry_executes_numeric_decoder() {
    let (mut native, manager, books) = managed_tiny(4096);
    let context = native.session.host_tensor_allocator().unwrap();
    let used = books.used(Tier::Host);
    let blocker = context
        .array::<u8>(usize::try_from(4096 - used - 8).unwrap())
        .unwrap();
    let error = native.decode(&[2, 3], 0).unwrap_err();
    assert!(format!("{error:#}").contains("ordinary Host"), "{error:#}");
    assert_eq!(native.current_len(), 0);
    assert!(native.past.is_empty());
    drop(blocker);
    let rows = native.decode(&[2, 3], 0).unwrap();
    assert_eq!(
        rows,
        LogitsBatch::from_host_rows(vec![vec![2.0], vec![3.0]])
    );
    assert_eq!(native.current_len(), 2);
    drop(rows);
    drop(context);
    drop(native);
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}
#[test]
fn kv_prefix_candidate_denial_preserves_source_and_retries_with_independent_geometry() {
    let (native, manager, books) = managed_tiny(4096);
    let context = native.session.host_tensor_allocator().unwrap();
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let source = context.from_f32(&[1, 2, 4, 3], &values).unwrap();
    let small = tensor::prefix_slice(&context, &source, 2, 2).unwrap();
    assert_eq!(
        small.to_vec_f32(),
        vec![0., 1., 2., 3., 4., 5., 12., 13., 14., 15., 16., 17.]
    );
    let used = books.used(Tier::Host);
    let blocker = context
        .array::<u8>(usize::try_from(4096 - used - 32).unwrap())
        .unwrap();
    assert!(tensor::prefix_slice(&context, &source, 2, 3).is_err());
    assert_eq!(source.to_vec_f32(), values);
    assert_eq!(small.shape, [1, 2, 2, 3]);
    drop(blocker);
    drop(small);
    let large = tensor::prefix_slice(&context, &source, 2, 3).unwrap();
    assert_eq!(
        large.to_vec_f32(),
        vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 12., 13., 14., 15., 16., 17., 18., 19., 20.
        ]
    );
    drop(large);
    drop(source);
    drop(context);
    drop(native);
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}
#[test]
fn multi_binding_prefix_staging_denial_publishes_no_partial_candidate() {
    let (native, manager, books) = managed_tiny(4096);
    let context = native.session.host_tensor_allocator().unwrap();
    let mut past = HashMap::new();
    past.insert(
        "first".to_string(),
        context.from_f32(&[1, 1, 4, 1], &[1., 2., 3., 4.]).unwrap(),
    );
    past.insert(
        "second".to_string(),
        context
            .from_f32(&[1, 1, 4, 1], &[10., 20., 30., 40.])
            .unwrap(),
    );
    let used = books.used(Tier::Host);
    let blocker = context
        .array::<u8>(usize::try_from(4096 - used - 16).unwrap())
        .unwrap();
    assert!(backend::stage_host_prefix_rewind(&past, &HashSet::new(), 3, &context).is_err());
    assert_eq!(books.used(Tier::Host), used + blocker.len() as u64);
    assert_eq!(past["first"].to_vec_f32(), [1., 2., 3., 4.]);
    assert_eq!(past["second"].to_vec_f32(), [10., 20., 30., 40.]);
    drop(blocker);
    let staged = backend::stage_host_prefix_rewind(&past, &HashSet::new(), 3, &context).unwrap();
    assert_eq!(staged.len(), 2);
    for (name, tensor) in &staged {
        assert_eq!(tensor.to_vec_f32(), past[name].to_vec_f32()[..3]);
    }
    drop(staged);
    drop(past);
    drop(context);
    drop(native);
    assert_eq!(manager.process_used(Tier::Host), 0);
}

#[test]
fn actual_routed_numeric_views_preserve_transposed_and_padded_layouts_under_admission() {
    use onnx_runtime_ep_api::DeviceBuffer;
    use onnx_runtime_ir::{Dim, TensorLayout};
    for (physical, strides, expected) in [
        (
            vec![10_f32, 20., 30., 40.],
            vec![4, 1, 2],
            vec![vec![11., 32.], vec![23., 44.]],
        ),
        (
            vec![10_f32, 20., 999., 30., 40., 888.],
            vec![6, 3, 1],
            vec![vec![11., 22.], vec![33., 44.]],
        ),
    ] {
        let (cpu, manager, books) = resources(4096);
        let original = tests::tiny_embedding_target(true);
        let mut graph = original.graph().clone();
        drop(original);
        let ports = graph
            .values
            .iter()
            .filter_map(|(id, value)| {
                matches!(
                    value.name.as_deref(),
                    Some("embedded_sequence" | "routed_features" | "logits")
                )
                .then_some(id)
            })
            .collect::<Vec<_>>();
        for port in ports {
            *graph.value_mut(port).shape.last_mut().unwrap() = Dim::Static(2);
        }
        let session = InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            std::path::Path::new("."),
            cpu,
            None,
        )
        .unwrap();
        let mut target = NativeDecodeSession::from_session_with_io(
            session,
            &tests::target_io(onnx_genai_metadata::SequenceInputKind::InputsEmbeds),
        )
        .unwrap();
        let context = target.session.host_tensor_allocator().unwrap();
        // SAFETY: Vec's stable live allocation is transferred into the import
        // guard; the full physical span is initialized and outlives every view.
        let buffer = unsafe {
            DeviceBuffer::from_borrowed_parts(
                physical.as_ptr() as *mut std::ffi::c_void,
                onnx_runtime_ir::DeviceId::cpu(),
                physical.len() * 4,
                4,
            )
        };
        let routed = Tensor::from_borrowed_parts_with_guard(
            onnx_runtime_session::cpu_allocator(),
            DataType::Float32,
            vec![1, 2, 2],
            TensorLayout::strided(strides),
            buffer,
            Box::new(physical),
        );
        let embeddings = Tensor::from_f32(&[1, 2, 2], &[1., 2., 3., 4.]).unwrap();
        let inputs = vec![
            ("embedded_sequence".to_string(), embeddings),
            ("routed_features".to_string(), routed),
        ];
        let used = books.used(Tier::Host);
        let blocker = context
            .array::<u8>(usize::try_from(4096 - used - 8).unwrap())
            .unwrap();
        assert!(target.decode_with_step_inputs(&[7, 8], 0, &inputs).is_err());
        assert_eq!(target.current_len(), 0);
        drop(blocker);
        let actual = target.decode_with_step_inputs(&[7, 8], 0, &inputs).unwrap();
        assert_eq!(
            actual,
            LogitsBatch::from_host_rows(expected),
            "logical matrix addition must respect the source layout, not physical byte order"
        );
        drop(actual);
        drop(inputs);
        drop(context);
        drop(target);
        assert_eq!(books.used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
}

#[test]
fn owned_half_logits_admit_before_widen_and_preserve_independent_bit_oracles() {
    let (native, manager, books) = managed_tiny(4096);
    let context = native.session.host_tensor_allocator().unwrap();
    for (dtype, bits) in [
        (
            DataType::Float16,
            [0xc000_u16, 0x3400, 0x3e00, 0x4200, 0xc400, 0x4000],
        ),
        (
            DataType::BFloat16,
            [0xc000_u16, 0x3e80, 0x3fc0, 0x4040, 0xc080, 0x4000],
        ),
    ] {
        let bytes = bits
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let baseline = books.used(Tier::Host);
        let source = context.from_raw(dtype, vec![1, 2, 3], &bytes).unwrap();
        let occupied = books.used(Tier::Host);
        let blocker = context
            .array::<u8>(usize::try_from(4096 - occupied - 23).unwrap())
            .unwrap();
        assert!(tensor::extract_owned_logits(source, &context).is_err());
        assert_eq!(
            books.used(Tier::Host),
            baseline + blocker.len() as u64,
            "consumed source must release and denied wider target must not publish"
        );
        drop(blocker);
        let source = context.from_raw(dtype, vec![1, 2, 3], &bytes).unwrap();
        let rows = tensor::extract_owned_logits(source, &context).unwrap();
        assert_eq!(
            rows,
            LogitsBatch::from_host_rows(vec![vec![-2., 0.25, 1.5], vec![3., -4., 2.]])
        );
        assert_eq!(books.used(Tier::Host), baseline + 24);
        drop(rows);
        assert_eq!(books.used(Tier::Host), baseline);
    }
    drop(context);
    drop(native);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}

#[test]
fn hidden_final_row_conversion_keeps_only_selected_row_geometry_and_values() {
    let (native, manager, books) = managed_tiny(4096);
    let context = native.session.host_tensor_allocator().unwrap();
    let source = context
        .from_f32(
            &[1, 2, 3],
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3., -4., 2.],
        )
        .unwrap();
    assert_eq!(tensor::extract_last_row(&source).unwrap(), [3., -4., 2.]);
    for (dtype, bits) in [
        (
            DataType::Float16,
            [0x7e00_u16, 0x7c00, 0xfc00, 0x4200, 0xc400, 0x4000],
        ),
        (
            DataType::BFloat16,
            [0x7fc0_u16, 0x7f80, 0xff80, 0x4040, 0xc080, 0x4000],
        ),
    ] {
        let bytes = bits
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let half = context.from_raw(dtype, vec![1, 2, 3], &bytes).unwrap();
        assert_eq!(tensor::extract_last_row(&half).unwrap(), [3., -4., 2.]);
    }
    drop(source);
    drop(context);
    drop(native);
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}
