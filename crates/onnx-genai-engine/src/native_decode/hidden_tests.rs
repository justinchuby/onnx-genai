use super::*;
use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use onnx_runtime_session::{HostTensorAllocator, OwnedHostArray};
use std::io::{Cursor, Read};
use std::sync::Mutex;

const LIMIT: u64 = 1024 * 1024;
fn decoder() -> (
    NativeDecodeSession,
    ProcessMemoryManager,
    Arc<LedgerGovernor>,
) {
    let manager = ProcessMemoryManager::new().unwrap();
    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        LIMIT,
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
    let original = tests::tiny_decoder(false);
    let mut graph = original.graph().clone();
    drop(original);
    let logits = *graph
        .outputs
        .iter()
        .find(|&&id| graph.value(id).name.as_deref() == Some("logits"))
        .unwrap();
    let shape = graph.value(logits).shape.clone();
    let hidden = graph.create_named_value("hidden", DataType::Float32, shape);
    graph.insert_node(onnx_runtime_ir::Node::new(
        onnx_runtime_ir::NodeId(0),
        "Identity",
        vec![Some(logits)],
        vec![hidden],
    ));
    graph.add_output(hidden);
    let session = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        cpu,
        None,
    )
    .unwrap();
    let mut io = tests::tiny_decoder_io();
    io.hidden_output = Some("hidden".into());
    let mut native = NativeDecodeSession::from_session_with_io(session, &io).unwrap();
    native.state_identity = Some([7; 32]);
    (native, manager, books)
}

#[test]
fn hidden_final_row_admits_before_copy_and_retries_with_independent_half_oracles() {
    let (native, manager, books) = decoder();
    let context = native.session.host_tensor_allocator().unwrap();
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
        let source = context.from_raw(dtype, vec![1, 2, 3], &bytes).unwrap();
        let blocker = context
            .array::<u8>(usize::try_from(books.available(Tier::Host) - 11).unwrap())
            .unwrap();
        let before = books.used(Tier::Host);
        let error = tensor::extract_owned_last_row(&source, &context).unwrap_err();
        assert!(format!("{error:#}").contains("admit Native hidden final row"));
        assert_eq!(books.used(Tier::Host), before);
        drop(blocker);
        let row = tensor::extract_owned_last_row(&source, &context).unwrap();
        assert_eq!(row.as_slice(), &[3., -4., 2.]);
    }
    drop(context);
    drop(native);
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}

#[test]
fn real_forward_hidden_snapshots_share_funding_and_survive_replacement_and_source_drop() {
    let (mut native, manager, books) = decoder();
    let first_logits = native.decode(&[2, 3], 0).unwrap();
    assert_eq!(first_logits[0], [2.]);
    assert_eq!(first_logits[1], [3.]);
    let expected = vec![3.];
    assert_eq!(native.last_hidden().unwrap(), expected);
    let pointer = native.last_hidden().unwrap().as_ptr();
    let context = native.session.host_tensor_allocator().unwrap();
    let snapshot = native.snapshot_past().unwrap();
    assert_eq!(snapshot.last_hidden.as_ref().unwrap().as_ptr(), pointer);
    let blocker = context
        .array::<u8>(usize::try_from(books.available(Tier::Host)).unwrap())
        .unwrap();
    let before = books.used(Tier::Host);
    let retained = snapshot.last_hidden.clone().unwrap();
    assert_eq!(retained.as_ptr(), pointer);
    assert_eq!(
        books.used(Tier::Host),
        before,
        "immutable clone needs no duplicate physical hidden allocation"
    );
    assert!(
        native.snapshot_past().is_err(),
        "real KV snapshot copies remain subject to admission"
    );
    assert_eq!(native.current_len(), 2);
    assert_eq!(native.last_hidden().unwrap(), expected);
    drop(blocker);
    drop(native.decode(&[5], 2).unwrap());
    assert_eq!(native.last_hidden().unwrap(), &[5.]);
    assert_eq!(
        first_logits[0],
        [2.],
        "retained first logits must survive a real subsequent forward"
    );
    assert_eq!(first_logits[1], [3.]);
    assert_eq!(retained.as_slice(), expected);
    native.restore_past_snapshot(&snapshot).unwrap();
    assert_eq!(native.last_hidden().unwrap().as_ptr(), pointer);
    drop(snapshot);
    drop(first_logits);
    drop(context);
    drop(native);
    assert_eq!(retained.as_slice(), expected);
    assert_eq!(books.used(Tier::Host), (expected.len() * 4) as u64);
    drop(retained);
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}

struct PressureReader<'a> {
    inner: Cursor<&'a [u8]>,
    trigger: u64,
    context: HostTensorAllocator,
    books: Arc<LedgerGovernor>,
    blocker: Arc<Mutex<Option<OwnedHostArray<u8>>>>,
}
impl Read for PressureReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        if self.inner.position() == self.trigger && self.blocker.lock().unwrap().is_none() {
            let blocker = self
                .context
                .array::<u8>(usize::try_from(self.books.available(Tier::Host)).unwrap())
                .map_err(std::io::Error::other)?;
            *self.blocker.lock().unwrap() = Some(blocker);
        }
        Ok(count)
    }
}

#[test]
fn real_hidden_file_payload_denial_truncation_and_retry_keep_owned_values_and_source_state() {
    let (mut native, manager, books) = decoder();
    drop(native.decode(&[2, 3], 0).unwrap());
    let expected = vec![3.];
    assert_eq!(native.last_hidden().unwrap(), expected);
    let snapshot = native.snapshot_past().unwrap();
    let context = native.session.host_tensor_allocator().unwrap();
    let mut file = Vec::new();
    snapshot
        .write_file(
            &[2, 3],
            [7; 32],
            &mut file,
            NativeStateFileLimits::default(),
            books.as_ref(),
            &context,
        )
        .unwrap();
    let hidden_start = file.len() - 32 - expected.len() * 4;
    let blocker = Arc::new(Mutex::new(None));
    let mut pressured = PressureReader {
        inner: Cursor::new(&file),
        trigger: hidden_start as u64,
        context: context.clone(),
        books: books.clone(),
        blocker: blocker.clone(),
    };
    let error = native
        .read_state_file(
            &mut pressured,
            NativeStateFileLimits::default(),
            books.as_ref(),
        )
        .err()
        .expect("hidden payload must be denied before reading");
    assert!(
        format!("{error:#}").contains("admit Native hidden-state file payload"),
        "{error:#}"
    );
    assert!(blocker.lock().unwrap().is_some());
    assert_eq!(native.current_len(), 2);
    assert_eq!(native.last_hidden().unwrap(), expected);
    drop(pressured);
    drop(blocker.lock().unwrap().take());
    assert!(
        native
            .read_state_file(
                &mut &file[..hidden_start + 1],
                NativeStateFileLimits::default(),
                books.as_ref()
            )
            .is_err()
    );
    let decoded = native
        .read_state_file(
            &mut file.as_slice(),
            NativeStateFileLimits::default(),
            books.as_ref(),
        )
        .unwrap();
    let restored = decoded.state.last_hidden.clone().unwrap();
    assert_eq!(restored.as_slice(), expected);
    drop(decoded);
    drop(snapshot);
    drop(context);
    drop(native);
    assert_eq!(restored.as_slice(), expected);
    assert_eq!(books.used(Tier::Host), (expected.len() * 4) as u64);
    drop(restored);
    assert_eq!(books.used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
}
