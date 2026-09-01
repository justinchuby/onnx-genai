#![cfg(feature = "gpu-tests")]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use onnx_runtime_cuda_memory::capability::host_numa_capability;
use onnx_runtime_cuda_memory::release::{DriverFaultPlan, DriverOperation};
use onnx_runtime_ep_api::{ExecutionProvider, ExecutorInstanceId};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::weight_paging::DeviceOffloadPolicy;
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{DeviceKey, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier};
use onnx_runtime_session::{DeviceGraphCaptureResult, DeviceIoBinding, InferenceSession};

const EXPERTS: usize = 4;
const ROWS: usize = 4;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 2048;
const FC1_PACKED_LEN: usize = EXPERTS * INTERMEDIATE * (HIDDEN / 2);
const FC1_SCALES_LEN: usize = EXPERTS * INTERMEDIATE * (HIDDEN / 16) * 4;
const FC2_PACKED_LEN: usize = EXPERTS * HIDDEN * (INTERMEDIATE / 2);
const FC2_SCALES_LEN: usize = EXPERTS * HIDDEN * (INTERMEDIATE / 16) * 4;
const BANK_LEN: usize = FC1_PACKED_LEN + FC1_SCALES_LEN + FC2_PACKED_LEN + FC2_SCALES_LEN;

static FIXTURE_ID: AtomicU64 = AtomicU64::new(1);
static SERIAL: Mutex<()> = Mutex::new(());

struct GateGuard(Option<String>);

impl GateGuard {
    fn enable() -> Self {
        let previous = std::env::var(COARSE_RESIDENCY_ENABLE_ENV).ok();
        unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
        Self(previous)
    }
}

impl Drop for GateGuard {
    fn drop(&mut self) {
        match self.0.take() {
            Some(value) => unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, value) },
            None => unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) },
        }
    }
}

struct Fixture {
    dir: PathBuf,
    model: PathBuf,
}

impl Fixture {
    fn create(symbolic: bool, banks: usize) -> Self {
        assert!(matches!(banks, 1 | 2));
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/deckard-qmoe-route")
            .join(format!("{}-{id}", std::process::id()));
        fs::create_dir_all(&dir).expect("create project-local QMoE fixture");
        let weights = dir.join("weights.bin");
        write_weights(&weights, banks);
        let model = dir.join("model.onnx.textproto");
        fs::write(&model, model_text(symbolic, banks)).expect("write QMoE model");
        Self { dir, model }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn write_pattern(writer: &mut BufWriter<File>, pattern: &[u8], len: usize) {
    assert!(len.is_multiple_of(pattern.len()));
    let chunk = pattern.repeat((1 << 20) / pattern.len());
    let mut remaining = len;
    while remaining != 0 {
        let take = remaining.min(chunk.len());
        writer.write_all(&chunk[..take]).expect("write weight data");
        remaining -= take;
    }
}

fn write_weights(path: &Path, banks: usize) {
    let mut writer = BufWriter::new(File::create(path).expect("create weights"));
    for _ in 0..banks {
        write_pattern(&mut writer, &[0x99], FC1_PACKED_LEN);
        write_pattern(&mut writer, &0.000_125_f32.to_le_bytes(), FC1_SCALES_LEN);
        write_pattern(&mut writer, &[0x99], FC2_PACKED_LEN);
        write_pattern(&mut writer, &0.000_125_f32.to_le_bytes(), FC2_SCALES_LEN);
    }
    writer.flush().expect("flush weights");
    assert_eq!(
        fs::metadata(path).expect("weight metadata").len() as usize,
        banks * BANK_LEN
    );
}

fn external_initializer(
    text: &mut String,
    name: &str,
    dtype: i32,
    dims: &[usize],
    offset: usize,
    len: usize,
) {
    writeln!(text, "  initializer {{").unwrap();
    for dim in dims {
        writeln!(text, "    dims: {dim}").unwrap();
    }
    writeln!(text, "    data_type: {dtype}").unwrap();
    writeln!(text, "    name: \"{name}\"").unwrap();
    writeln!(
        text,
        "    external_data {{ key: \"location\" value: \"weights.bin\" }}"
    )
    .unwrap();
    writeln!(
        text,
        "    external_data {{ key: \"offset\" value: \"{offset}\" }}"
    )
    .unwrap();
    writeln!(
        text,
        "    external_data {{ key: \"length\" value: \"{len}\" }}"
    )
    .unwrap();
    writeln!(text, "    data_location: EXTERNAL").unwrap();
    writeln!(text, "  }}").unwrap();
}

fn value_info(text: &mut String, kind: &str, name: &str, cols: usize, symbolic: bool) {
    writeln!(text, "  {kind} {{").unwrap();
    writeln!(text, "    name: \"{name}\"").unwrap();
    writeln!(text, "    type {{ tensor_type {{ elem_type: 1 shape {{").unwrap();
    if symbolic {
        writeln!(text, "      dim {{ dim_param: \"tokens\" }}").unwrap();
    } else {
        writeln!(text, "      dim {{ dim_value: {ROWS} }}").unwrap();
    }
    writeln!(text, "      dim {{ dim_value: {cols} }}").unwrap();
    writeln!(text, "    }} }} }}").unwrap();
    writeln!(text, "  }}").unwrap();
}

fn qmoe_node(text: &mut String, bank: usize, input: &str, output: &str) {
    let prefix = if bank == 0 { "a" } else { "b" };
    writeln!(text, "  node {{").unwrap();
    for input_name in [
        input,
        &format!("router_{prefix}"),
        &format!("{prefix}_fc1_packed"),
        &format!("{prefix}_fc1_scales"),
        "",
        &format!("{prefix}_fc2_packed"),
        &format!("{prefix}_fc2_scales"),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ] {
        writeln!(text, "    input: \"{input_name}\"").unwrap();
    }
    writeln!(text, "    output: \"{output}\"").unwrap();
    writeln!(text, "    name: \"qmoe_{prefix}\"").unwrap();
    writeln!(text, "    op_type: \"QMoE\"").unwrap();
    writeln!(text, "    domain: \"com.microsoft\"").unwrap();
    text.push_str(
        "    attribute { name: \"expert_weight_bits\" i: 4 type: INT }\n\
         \x20   attribute { name: \"block_size\" i: 16 type: INT }\n\
         \x20   attribute { name: \"k\" i: 1 type: INT }\n\
         \x20   attribute { name: \"activation_type\" s: \"silu\" type: STRING }\n\
         \x20   attribute { name: \"normalize_routing_weights\" i: 0 type: INT }\n\
         \x20   attribute { name: \"swiglu_fusion\" i: 0 type: INT }\n",
    );
    writeln!(text, "  }}").unwrap();
}

fn model_text(symbolic: bool, banks: usize) -> String {
    let mut text = String::from(
        "ir_version: 11\nproducer_name: \"deckard-route-test\"\ngraph {\n\
         \x20 name: \"symbolic_qmoe_route_residency\"\n",
    );
    qmoe_node(
        &mut text,
        0,
        "hidden",
        if banks == 1 { "output" } else { "mid" },
    );
    if banks == 2 {
        qmoe_node(&mut text, 1, "mid", "output");
    }
    for bank in 0..banks {
        let prefix = if bank == 0 { "a" } else { "b" };
        let base = bank * BANK_LEN;
        external_initializer(
            &mut text,
            &format!("{prefix}_fc1_packed"),
            2,
            &[EXPERTS, INTERMEDIATE, HIDDEN / 2],
            base,
            FC1_PACKED_LEN,
        );
        external_initializer(
            &mut text,
            &format!("{prefix}_fc1_scales"),
            1,
            &[EXPERTS, INTERMEDIATE, HIDDEN / 16],
            base + FC1_PACKED_LEN,
            FC1_SCALES_LEN,
        );
        external_initializer(
            &mut text,
            &format!("{prefix}_fc2_packed"),
            2,
            &[EXPERTS, HIDDEN, INTERMEDIATE / 2],
            base + FC1_PACKED_LEN + FC1_SCALES_LEN,
            FC2_PACKED_LEN,
        );
        external_initializer(
            &mut text,
            &format!("{prefix}_fc2_scales"),
            1,
            &[EXPERTS, HIDDEN, INTERMEDIATE / 16],
            base + FC1_PACKED_LEN + FC1_SCALES_LEN + FC2_PACKED_LEN,
            FC2_SCALES_LEN,
        );
    }
    value_info(&mut text, "input", "hidden", HIDDEN, symbolic);
    value_info(&mut text, "input", "router_a", EXPERTS, symbolic);
    if banks == 2 {
        value_info(&mut text, "input", "router_b", EXPERTS, symbolic);
        value_info(&mut text, "value_info", "mid", HIDDEN, symbolic);
    }
    value_info(&mut text, "output", "output", HIDDEN, symbolic);
    text.push_str(
        "}\nopset_import { domain: \"\" version: 13 }\n\
         opset_import { domain: \"com.microsoft\" version: 1 }\n",
    );
    text
}

struct LiveCase {
    provider: Arc<CudaExecutionProvider>,
    session: InferenceSession,
    bindings: Vec<DeviceIoBinding>,
    scopes_before_build: Vec<ExecutorInstanceId>,
    scope: Option<ExecutorInstanceId>,
    ledger: Arc<LeaseLedger>,
    provider_baseline_device: u64,
}

fn provider_or_skip(device: u32) -> Option<(Arc<CudaExecutionProvider>, Arc<LeaseLedger>)> {
    if let Err(error) = host_numa_capability(device as i32) {
        eprintln!("SKIP CUDA:{device}: HOST_NUMA VMM unavailable: {error}");
        return None;
    }
    let ledger = LeaseLedger::new_for_device(DeviceKey::device(device), 1 << 30, 1 << 30, 0);
    let governor: Arc<dyn MemoryGovernor + Send + Sync> =
        Arc::new(LedgerGovernor::new(Arc::clone(&ledger)));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(256 << 20),
        ..DeviceOffloadPolicy::default()
    };
    let provider = CudaExecutionProvider::initialized_with_offload_policy_and_governor(
        device, policy, governor,
    )
    .ok()?;
    Some((Arc::new(provider), ledger))
}

fn build_case(
    fixture: &Fixture,
    provider: Arc<CudaExecutionProvider>,
    ledger: Arc<LeaseLedger>,
    banks: usize,
) -> LiveCase {
    let scopes_before_build = provider.route_residency_scopes();
    let provider_baseline_device = ledger.used(Tier::Device);
    let session = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()
        .expect("build production QMoE session");
    let hidden = session
        .allocate_device_binding(
            "hidden",
            None::<String>,
            DataType::Float32,
            vec![ROWS, HIDDEN],
            vec![ROWS, HIDDEN],
        )
        .expect("allocate hidden");
    let router_a = session
        .allocate_device_binding(
            "router_a",
            None::<String>,
            DataType::Float32,
            vec![ROWS, EXPERTS],
            vec![ROWS, EXPERTS],
        )
        .expect("allocate router a");
    let mut bindings = vec![hidden, router_a];
    if banks == 2 {
        bindings.push(
            session
                .allocate_device_binding(
                    "router_b",
                    None::<String>,
                    DataType::Float32,
                    vec![ROWS, EXPERTS],
                    vec![ROWS, EXPERTS],
                )
                .expect("allocate router b"),
        );
    }
    bindings.push(
        session
            .allocate_device_output_binding(
                "output",
                DataType::Float32,
                vec![ROWS, HIDDEN],
                vec![ROWS, HIDDEN],
            )
            .expect("allocate output"),
    );
    let hidden_bytes = (0..ROWS * HIDDEN)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    bindings[0]
        .write_bytes(0, &hidden_bytes)
        .expect("upload hidden");
    LiveCase {
        provider,
        session,
        bindings,
        scopes_before_build,
        scope: None,
        ledger,
        provider_baseline_device,
    }
}

fn router_bytes(selected: &[usize; ROWS]) -> Vec<u8> {
    selected
        .iter()
        .flat_map(|&hot| {
            (0..EXPERTS).map(move |expert| if expert == hot { 20.0_f32 } else { -20.0_f32 })
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn set_routes(case: &mut LiveCase, banks: usize, selected: [usize; ROWS]) {
    let bytes = router_bytes(&selected);
    case.bindings[1]
        .write_bytes(0, &bytes)
        .expect("upload router a");
    if banks == 2 {
        case.bindings[2]
            .write_bytes(0, &bytes)
            .expect("upload router b");
    }
}

fn consume_latest_output(case: &mut LiveCase) {
    case.bindings
        .last_mut()
        .expect("QMoE output binding")
        .read_bytes()
        .expect("consume device validation receipt");
}

fn run_first_prefill(case: &mut LiveCase, banks: usize) -> ExecutorInstanceId {
    set_routes(case, banks, [0, 1, 2, 3]);
    case.session
        .run_with_device_bindings(&[], &mut case.bindings)
        .expect("real QMoE prefill");
    consume_latest_output(case);
    let new_scopes: Vec<_> = case
        .provider
        .route_residency_scopes()
        .into_iter()
        .filter(|scope| !case.scopes_before_build.contains(scope))
        .collect();
    assert_eq!(new_scopes.len(), 1);
    case.scope = Some(new_scopes[0]);
    new_scopes[0]
}

fn reservation_ranges(
    provider: &CudaExecutionProvider,
    scope: ExecutorInstanceId,
) -> Vec<(u64, u64, DeviceKey)> {
    provider
        .retained_route_residency_artifacts(scope)
        .expect("retained groups")
        .iter()
        .flat_map(|group| group.members.iter())
        .map(|&value| {
            let allocator = provider
                .residency()
                .expect("residency")
                .coarse_route_bank_reservation(scope, value)
                .expect("dedicated reservation");
            let base = allocator.with_reservation_mut(|reservation, _| reservation.base_ptr());
            let reserved = allocator.committed_and_reserved().1 as u64;
            (base, base + reserved, allocator.device_key())
        })
        .collect()
}

fn assert_nonoverlap(ranges: &[(u64, u64, DeviceKey)]) {
    for (index, lhs) in ranges.iter().enumerate() {
        assert!(lhs.0 < lhs.1);
        for rhs in &ranges[index + 1..] {
            assert!(lhs.1 <= rhs.0 || rhs.1 <= lhs.0, "{lhs:?} overlaps {rhs:?}");
        }
    }
}

fn wait_for_accounting(ledger: &LeaseLedger, device: u64, host: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let actual = (ledger.used(Tier::Device), ledger.used(Tier::Host));
        if actual == (device, host) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "accounting {actual:?}, expected {:?}",
            (device, host)
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn drop_case_to_provider_baseline(case: LiveCase) {
    let LiveCase {
        provider,
        session,
        bindings,
        ledger,
        provider_baseline_device,
        ..
    } = case;
    drop(session);
    drop(bindings);
    provider.sync().expect("settle executor teardown");
    wait_for_accounting(&ledger, provider_baseline_device, 0);
}

fn shutdown_provider(
    provider: Arc<CudaExecutionProvider>,
    ledger: Arc<LeaseLedger>,
    baseline: u64,
) {
    wait_for_accounting(&ledger, baseline, 0);
    let mut provider =
        Arc::try_unwrap(provider).unwrap_or_else(|_| panic!("provider still shared"));
    provider.shutdown().expect("shutdown provider");
    drop(provider);
    wait_for_accounting(&ledger, 0, 0);
}

fn run_fault_case(fixture: &Fixture, quarantine: bool) {
    let Some((provider, ledger)) = provider_or_skip(0) else {
        return;
    };
    let mut case = build_case(fixture, Arc::clone(&provider), Arc::clone(&ledger), 2);
    let scope = run_first_prefill(&mut case, 2);
    set_routes(&mut case, 2, [0; ROWS]);
    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
    assert!(matches!(
        case.session
            .try_capture_with_device_bindings(&[], &mut case.bindings)
            .expect("capture fault case"),
        DeviceGraphCaptureResult::Captured(_)
    ));
    consume_latest_output(&mut case);
    for _ in 0..3 {
        assert!(
            case.session
                .replay_device_graph(&mut case.bindings)
                .expect("fault replay")
        );
        consume_latest_output(&mut case);
    }
    provider.sync().expect("finish fault replay");
    let output_index = case.bindings.len() - 1;
    let baseline = case.bindings[output_index]
        .read_bytes()
        .expect("read pre-fault output");
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };

    let groups = provider
        .retained_route_residency_artifacts(scope)
        .expect("fault groups");
    let mut faults = HashMap::new();
    if quarantine {
        for value in groups
            .iter()
            .flat_map(|group| group.members.iter())
            .copied()
        {
            faults.insert(
                value,
                Arc::new(
                    DriverFaultPlan::new()
                        .fail_nth(DriverOperation::Remap, 1)
                        .fail_nth(DriverOperation::Remap, 2),
                ),
            );
        }
    } else {
        let value = groups[0].members[0];
        faults.insert(
            value,
            Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3)),
        );
    }
    let diag = provider.route_residency_diagnostics(scope);
    let fatal_before = diag.fatal_values();
    let quarantine_before = diag.quarantined_blocks();
    provider
        .consume_route_residency_at_boundary_with_phase8_faults_for_executor(scope, faults)
        .expect("fault boundary");
    assert!(diag.fatal_values() > fatal_before);
    if quarantine {
        assert!(diag.quarantined_blocks() > quarantine_before);
    } else {
        assert_eq!(diag.quarantined_blocks(), quarantine_before);
        assert!(
            case.session
                .replay_device_graph(&mut case.bindings)
                .expect("replay after range rollback")
        );
        provider.sync().expect("complete post-rollback replay");
        assert_eq!(
            case.bindings[output_index]
                .read_bytes()
                .expect("read post-rollback output"),
            baseline,
            "mapping failure rollback preserves the real QMoE bank"
        );
    }
    drop(case);
    let mut provider =
        Arc::try_unwrap(provider).unwrap_or_else(|_| panic!("fault provider shared"));
    provider.sync().expect("settle fault case");
    provider.shutdown().expect("shutdown fault provider");
}

#[test]
#[ignore = "requires an idle CUDA device with HOST_NUMA VMM support"]
fn symbolic_real_qmoe_route_residency_lifecycle() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let fixture = Fixture::create(true, 2);
    let Some((provider, ledger)) = provider_or_skip(0) else {
        return;
    };
    let baseline = ledger.used(Tier::Device);
    let mut primary = build_case(&fixture, Arc::clone(&provider), Arc::clone(&ledger), 2);

    assert!(
        provider.route_residency_scopes().is_empty(),
        "symbolic build cannot install before resolved kernel compilation"
    );
    let scope = run_first_prefill(&mut primary, 2);
    let diag = provider.route_residency_diagnostics(scope);
    assert_eq!(diag.installs(), 1);
    assert_eq!(diag.declines(), 0);
    assert_eq!(
        provider.route_residency_executor_status(scope).outcome,
        Some(
            onnx_runtime_ep_cuda::route_residency::RouteResidencyInstallOutcome::Installed {
                banks: 8
            }
        )
    );
    assert_eq!(
        provider
            .route_residency_executor_status(scope)
            .finalization_attempts,
        1
    );
    assert_eq!(diag.values_touched(), 0, "prefill routed every expert");
    assert!(ledger.used(Tier::Device) > baseline);

    let primary_ranges = reservation_ranges(&provider, scope);
    assert_eq!(primary_ranges.len(), 8);
    assert_nonoverlap(&primary_ranges);
    assert!(
        primary_ranges
            .iter()
            .all(|range| range.2 == DeviceKey::device(0))
    );

    set_routes(&mut primary, 2, [0; ROWS]);
    assert!(matches!(
        primary
            .session
            .try_capture_with_device_bindings(&[], &mut primary.bindings)
            .expect("capture symbolic QMoE"),
        DeviceGraphCaptureResult::Captured(_)
    ));
    consume_latest_output(&mut primary);
    assert!(diag.values_touched() >= 8);
    assert!(diag.device_bytes_released() > 0);
    assert!(diag.host_bytes_committed() > 0);
    assert!(ledger.used(Tier::Host) > 0);
    for _ in 0..3 {
        assert!(
            primary
                .session
                .replay_device_graph(&mut primary.bindings)
                .expect("capture replay")
        );
        consume_latest_output(&mut primary);
    }
    provider.sync().expect("complete captured replays");

    let output_index = primary.bindings.len() - 1;
    let expert_zero_output = primary.bindings[output_index]
        .read_bytes()
        .expect("read expert-zero output");
    set_routes(&mut primary, 2, [1; ROWS]);
    for _ in 0..16 {
        assert!(
            primary
                .session
                .replay_device_graph(&mut primary.bindings)
                .expect("decode through remapped stable VA")
        );
        consume_latest_output(&mut primary);
    }
    provider.sync().expect("complete sixteen decodes");
    assert_eq!(
        primary.bindings[output_index]
            .read_bytes()
            .expect("read expert-one output"),
        expert_zero_output
    );
    let groups = provider
        .retained_route_residency_artifacts(scope)
        .expect("retained banks");
    for group in groups.iter() {
        let snapshot = provider
            .route_telemetry_producer(scope, group.node)
            .expect("real QMoE producer")
            .route_telemetry_snapshot()
            .expect("snapshot")
            .expect("producer armed");
        assert!(
            snapshot.routed_experts().is_empty(),
            "each exact validation receipt consumes its request telemetry window"
        );
        assert_eq!(snapshot.count(), 0);
    }
    let touched_before = diag.values_touched();
    provider
        .consume_route_residency_at_boundary_for_executor(scope)
        .expect("consume expert-one window");
    assert_eq!(
        diag.values_touched(),
        touched_before,
        "the finalized coarse placement installs once"
    );
    for group in groups.iter() {
        let next = provider
            .route_telemetry_producer(scope, group.node)
            .expect("producer")
            .route_telemetry_snapshot()
            .expect("next snapshot")
            .expect("producer armed");
        assert_eq!(next.count(), 0, "window reset is exact");
    }

    let mut sibling = build_case(&fixture, Arc::clone(&provider), Arc::clone(&ledger), 2);
    let sibling_scope = run_first_prefill(&mut sibling, 2);
    let mut all_ranges = primary_ranges;
    all_ranges.extend(reservation_ranges(&provider, sibling_scope));
    assert_nonoverlap(&all_ranges);
    assert_eq!(
        provider
            .residency()
            .expect("residency")
            .route_reservation_count(),
        2
    );
    drop(sibling);
    provider.sync().expect("settle sibling cancellation");
    assert!(!provider.route_residency_scopes().contains(&sibling_scope));
    assert!(provider.route_residency_scopes().contains(&scope));
    assert_eq!(
        provider
            .residency()
            .expect("residency")
            .route_reservation_count(),
        1
    );
    primary
        .session
        .run_with_device_bindings(&[], &mut primary.bindings)
        .expect("primary survives sibling teardown");

    let primary_baseline = primary.provider_baseline_device;
    drop(primary);
    provider.sync().expect("settle primary teardown");
    wait_for_accounting(&ledger, primary_baseline, 0);
    assert!(provider.route_residency_scopes().is_empty());
    assert_eq!(
        provider
            .residency()
            .expect("residency")
            .route_reservation_count(),
        0
    );

    if let Some((device_one, device_one_ledger)) = provider_or_skip(1) {
        let device_one_baseline = device_one_ledger.used(Tier::Device);
        let mut isolated = build_case(
            &fixture,
            Arc::clone(&device_one),
            Arc::clone(&device_one_ledger),
            2,
        );
        let isolated_scope = run_first_prefill(&mut isolated, 2);
        assert!(
            reservation_ranges(&device_one, isolated_scope)
                .iter()
                .all(|range| range.2 == DeviceKey::device(1))
        );
        drop(isolated);
        device_one.sync().expect("settle device-one executor");
        shutdown_provider(device_one, device_one_ledger, device_one_baseline);
    }

    run_fault_case(&fixture, false);
    run_fault_case(&fixture, true);
    shutdown_provider(provider, ledger, baseline);
}

#[test]
#[ignore = "requires an idle CUDA device with HOST_NUMA VMM support"]
fn static_qmoe_installs_at_build_and_tears_down_exactly() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let fixture = Fixture::create(false, 1);
    let Some((provider, ledger)) = provider_or_skip(0) else {
        return;
    };
    let baseline = ledger.used(Tier::Device);
    let case = build_case(&fixture, Arc::clone(&provider), Arc::clone(&ledger), 1);
    let scopes: Vec<_> = provider
        .route_residency_scopes()
        .into_iter()
        .filter(|scope| !case.scopes_before_build.contains(scope))
        .collect();
    assert_eq!(scopes.len(), 1, "static compile installs during build");
    assert_eq!(
        provider.route_residency_diagnostics(scopes[0]).installs(),
        1
    );
    drop_case_to_provider_baseline(case);
    shutdown_provider(provider, ledger, baseline);
}
