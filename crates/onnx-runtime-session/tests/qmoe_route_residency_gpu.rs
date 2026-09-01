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
use onnx_runtime_ep_api::{ExecutionProvider, ExecutorInstanceId, RoutedResidencyRequirement};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::coarse_residency::{
    COARSE_RESIDENCY_ENABLE_ENV, RollbackSafePointInterlock,
};
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
            (0..EXPERTS).map(move |expert| {
                if expert == hot {
                    20.0_f32
                } else if expert == (hot + 1) % EXPERTS {
                    10.0_f32
                } else {
                    -20.0_f32
                }
            })
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

fn run_first_prefill(case: &mut LiveCase, banks: usize) -> ExecutorInstanceId {
    set_routes(case, banks, [0, 1, 2, 3]);
    case.session
        .run_with_device_bindings(&[], &mut case.bindings)
        .expect("real QMoE prefill");
    let output_index = case.bindings.len() - 1;
    case.bindings[output_index]
        .read_bytes()
        .expect("consume prefill validation receipt");
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

#[derive(Clone, Copy)]
enum FaultMember {
    First,
    Middle,
    Last,
}

fn selected_member(
    members: &[onnx_runtime_ir::ValueId],
    position: FaultMember,
) -> onnx_runtime_ir::ValueId {
    let mut members = members.to_vec();
    members.sort_unstable_by_key(|value| value.0);
    members[match position {
        FaultMember::First => 0,
        FaultMember::Middle => members.len() / 2,
        FaultMember::Last => members.len() - 1,
    }]
}

fn run_fault_case(fixture: &Fixture, quarantine: bool, position: FaultMember) {
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
    let output_index = case.bindings.len() - 1;
    let baseline = case.bindings[output_index]
        .read_bytes()
        .expect("consume capture validation receipt");
    for _ in 0..3 {
        assert!(
            case.session
                .replay_device_graph(&mut case.bindings)
                .expect("fault replay")
        );
        assert_eq!(
            case.bindings[output_index]
                .read_bytes()
                .expect("consume fault replay validation receipt"),
            baseline
        );
    }
    provider.sync().expect("finish fault replay");
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };

    let groups = provider
        .retained_route_residency_artifacts(scope)
        .expect("fault groups");
    let mut faults = HashMap::new();
    if quarantine {
        let value = selected_member(&groups[0].members, position);
        faults.insert(
            value,
            Arc::new(
                DriverFaultPlan::new()
                    .fail_nth(DriverOperation::Remap, 1)
                    .fail_nth(DriverOperation::Remap, 2),
            ),
        );
    } else {
        let value = groups[0].members[0];
        faults.insert(
            value,
            Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Unmap, 3)),
        );
    }
    let diag = Arc::clone(provider.route_residency_diagnostics());
    let fatal_before = diag.fatal_values();
    let quarantine_before = diag.quarantined_blocks();
    let boundary_result =
        provider.consume_route_residency_at_boundary_with_phase8_faults_for_executor(scope, faults);
    assert!(diag.fatal_values() > fatal_before);
    if quarantine {
        let error = boundary_result.expect_err("quarantine must invalidate the bank reservation");
        assert!(
            error.to_string().contains("invalidated"),
            "unexpected quarantine error: {error}"
        );
        assert_eq!(
            diag.quarantined_blocks(),
            quarantine_before + 1,
            "one injected ambiguous granule must have exactly one quarantine owner"
        );
        unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
        let replay_error = case
            .session
            .replay_device_graph(&mut case.bindings)
            .expect_err("gate-off poisoned reservations must block captured replay");
        assert!(
            replay_error
                .to_string()
                .contains("route-bank reservation is unusable"),
            "unexpected replay rejection: {replay_error}"
        );
        let dispatch_error = case
            .session
            .run_with_device_bindings(&[], &mut case.bindings)
            .expect_err("gate-off poisoned reservations must block ordinary dispatch");
        assert!(
            dispatch_error
                .to_string()
                .contains("route-bank reservation is unusable"),
            "unexpected dispatch rejection: {dispatch_error}"
        );
        unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    } else {
        assert_eq!(diag.quarantined_blocks(), quarantine_before);
        match boundary_result {
            Ok(()) => {
                assert!(
                    case.session
                        .replay_device_graph(&mut case.bindings)
                        .expect("replay after range rollback")
                );
                let post_rollback = case.bindings[output_index]
                    .read_bytes()
                    .expect("consume post-rollback validation receipt");
                provider.sync().expect("complete post-rollback replay");
                assert_eq!(
                    post_rollback, baseline,
                    "mapping failure rollback preserves the real QMoE bank"
                );
            }
            Err(error) => {
                assert!(
                    error.to_string().contains("invalidated"),
                    "an incomplete rollback must fail closed: {error}"
                );
                unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
                let replay_error = case
                    .session
                    .replay_device_graph(&mut case.bindings)
                    .expect_err("incomplete rollback must block replay");
                assert!(
                    replay_error
                        .to_string()
                        .contains("route-bank reservation is unusable")
                );
                unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
            }
        }
    }
    drop(case);
    let mut provider =
        Arc::try_unwrap(provider).unwrap_or_else(|_| panic!("fault provider shared"));
    provider.sync().expect("settle fault case");
    provider.shutdown().expect("shutdown fault provider");
}

fn run_concurrent_safe_point_loss_case(fixture: &Fixture) {
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
            .expect("capture concurrent rollback case"),
        DeviceGraphCaptureResult::Captured(_)
    ));
    provider.sync().expect("finish concurrent rollback capture");
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };

    let groups = provider
        .retained_route_residency_artifacts(scope)
        .expect("concurrent rollback groups");
    let target = selected_member(&groups[0].members, FaultMember::Middle);
    let mut faults = HashMap::new();
    faults.insert(
        target,
        Arc::new(DriverFaultPlan::new().fail_nth(DriverOperation::Remap, 1)),
    );
    let interlock = RollbackSafePointInterlock::new();
    let touched_before = provider.route_residency_diagnostics().values_touched();
    let quarantine_before = provider.route_residency_diagnostics().quarantined_blocks();
    let worker_provider = Arc::clone(&provider);
    let worker_interlock = Arc::clone(&interlock);
    let worker = std::thread::spawn(move || {
        worker_provider.consume_route_residency_with_rollback_interlock_for_executor(
            scope,
            faults,
            worker_interlock,
        )
    });

    interlock.wait_until_forward_failure();
    let authorities = provider
        .residency()
        .expect("residency")
        .route_reservation_authorities(scope)
        .expect("route authorities");
    let blocker_catalog = authorities
        .catalogs
        .get(&groups[0].members[0])
        .expect("blocker catalog");
    let blocker = provider
        .residency()
        .expect("residency")
        .acquire_routed_residency(
            RoutedResidencyRequirement::FusedRoutingUnknown,
            blocker_catalog,
        );
    assert!(
        provider
            .residency()
            .expect("residency")
            .resize_safe_point(1)
            .routed_guards_active
            > 0,
        "the concurrent safe-point blocker must be live before rollback resumes"
    );
    interlock.resume_rollback();
    let error = worker
        .join()
        .expect("rollback worker panicked")
        .expect_err("safe-point loss with surviving transitions must poison");
    assert!(
        error.to_string().contains("invalidated"),
        "unexpected concurrent rollback error: {error}"
    );
    assert!(
        provider.route_residency_diagnostics().values_touched() > touched_before,
        "surviving HOST_NUMA transitions must be journaled before poison"
    );
    assert_eq!(
        provider.route_residency_diagnostics().quarantined_blocks(),
        quarantine_before,
        "safe-point loss itself must not fabricate quarantine"
    );
    drop(blocker);

    unsafe { std::env::remove_var(COARSE_RESIDENCY_ENABLE_ENV) };
    let replay_error = case
        .session
        .replay_device_graph(&mut case.bindings)
        .expect_err("poison after safe-point loss must block replay with gate off");
    assert!(
        replay_error
            .to_string()
            .contains("route-bank reservation is unusable")
    );
    let dispatch_error = case
        .session
        .run_with_device_bindings(&[], &mut case.bindings)
        .expect_err("poison after safe-point loss must block dispatch with gate off");
    assert!(
        dispatch_error
            .to_string()
            .contains("route-bank reservation is unusable")
    );
    unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
    let retry = provider
        .consume_route_residency_at_boundary_with_phase8_faults_for_executor(scope, HashMap::new());
    assert!(
        retry
            .expect_err("poisoned transition retry must remain blocked")
            .to_string()
            .contains("earlier atomic transition failure")
    );

    drop(authorities);
    drop(case);
    let mut provider =
        Arc::try_unwrap(provider).unwrap_or_else(|_| panic!("concurrent provider shared"));
    provider.sync().expect("settle concurrent rollback case");
    provider.shutdown().expect("shutdown concurrent provider");
    wait_for_accounting(&ledger, 0, 0);
}

#[test]
#[ignore = "requires an idle CUDA device with HOST_NUMA VMM support"]
fn failed_parent_build_retires_child_reservations_once_and_clean_retry_isolated() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let fixture = Fixture::create(false, 1);
    let Some((provider, ledger)) = provider_or_skip(0) else {
        return;
    };
    let provider_baseline = ledger.used(Tier::Device);

    let sibling = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()
        .expect("build sibling reservation owner");
    let sibling_scope = *provider
        .route_residency_scopes()
        .last()
        .expect("sibling installs one route-residency scope");
    let sibling_baseline = ledger.used(Tier::Device);
    let claims_before_failure = provider.executor_artifact_generation_claims();
    let census_before = provider.route_residency_retirement_census();

    provider.fail_next_allocation_after_required_artifact_report_for_test();
    let error = match InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()
    {
        Ok(_) => panic!("post-finalization allocation failure must abort the parent transaction"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("device OOM"),
        "the initiating parent-build failure remains actionable: {error}"
    );

    let claims_after_failure = provider.executor_artifact_generation_claims();
    let failed_claims = claims_after_failure
        .iter()
        .filter(|claim| !claims_before_failure.contains(claim))
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        failed_claims.len(),
        1,
        "the failed build must claim exactly one provider/executor/generation"
    );
    let (failed_executor, failed_generation) = failed_claims[0];
    assert_ne!(failed_executor, sibling_scope);
    assert!(
        provider
            .retired_executor_artifact_generations()
            .contains(&(failed_executor, failed_generation)),
        "the exact failed generation remains a sticky tombstone"
    );

    for _ in 0..500 {
        let census = provider.route_residency_retirement_census();
        if census.reservation_registry_entries == 1
            && census.cleanups_executed == census_before.cleanups_executed + 1
            && ledger.used(Tier::Device) == sibling_baseline
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let failed = provider.route_residency_executor_status(failed_executor);
    let census_after = provider.route_residency_retirement_census();
    assert_eq!(failed.producer_nodes, 0);
    assert_eq!(failed.retained_banks, 0);
    assert!(
        !provider.route_residency_scopes().contains(&failed_executor),
        "the failed owner has no live reservation scope"
    );
    assert_eq!(
        census_after.retirements_started,
        census_before.retirements_started + 1,
        "parent rollback retires exactly one committed child reservation"
    );
    assert_eq!(
        census_after.cleanups_executed,
        census_before.cleanups_executed + 1
    );
    assert_eq!(
        census_after.reservation_registry_entries, 1,
        "only the sibling reservation remains live"
    );
    assert_eq!(ledger.used(Tier::Device), sibling_baseline);
    assert!(
        provider.route_residency_scopes().contains(&sibling_scope),
        "failed rollback cannot consume the sibling owner"
    );

    let retry = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()
        .expect("a fresh generation retries cleanly on real CUDA");
    let retry_scope = provider
        .route_residency_scopes()
        .into_iter()
        .find(|scope| *scope != sibling_scope)
        .expect("retry publishes a distinct reservation scope");
    assert_ne!(retry_scope, failed_executor);
    assert!(matches!(
        provider
            .route_residency_executor_status(retry_scope)
            .outcome,
        Some(onnx_runtime_ep_cuda::route_residency::RouteResidencyInstallOutcome::Installed { .. })
    ));

    drop(retry);
    for _ in 0..500 {
        if provider
            .route_residency_retirement_census()
            .reservation_registry_entries
            == 1
            && ledger.used(Tier::Device) == sibling_baseline
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(sibling);
    for _ in 0..500 {
        if provider
            .route_residency_retirement_census()
            .reservation_registry_entries
            == 0
            && ledger.used(Tier::Device) == provider_baseline
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        provider
            .route_residency_retirement_census()
            .reservation_registry_entries,
        0
    );
    assert_eq!(ledger.used(Tier::Device), provider_baseline);

    let mut provider = Arc::try_unwrap(provider)
        .unwrap_or_else(|_| panic!("failed-build rollback provider still shared"));
    provider.sync().expect("sync rollback provider");
    provider.shutdown().expect("shutdown rollback provider");
    wait_for_accounting(&ledger, 0, 0);
}

#[test]
#[ignore = "requires an idle CUDA device with HOST_NUMA VMM support"]
fn executor_drop_is_bounded_while_public_artifact_guard_is_held() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    let fixture = Fixture::create(true, 1);
    let Some((provider, ledger)) = provider_or_skip(0) else {
        return;
    };
    let mut case = build_case(&fixture, Arc::clone(&provider), Arc::clone(&ledger), 1);
    let scope = run_first_prefill(&mut case, 1);
    let generation = provider
        .executor_artifact_generation_claims()
        .into_iter()
        .find_map(|(executor, generation)| (executor == scope).then_some(generation))
        .expect("installed executor generation claim");
    let provider_id = provider
        .executor_artifact_policy()
        .expect("provider artifact policy")
        .provider();
    let requirement = provider
        .executor_artifact_requirement(provider_id, scope, generation)
        .expect("query executor requirement")
        .expect("resolved QMoE executor installed route reservations");
    let holder = requirement
        .acquire_use()
        .expect("hold public artifact guard across executor Drop");
    let (returned_tx, returned_rx) = std::sync::mpsc::channel();
    let drop_thread = std::thread::spawn(move || {
        drop(case);
        returned_tx.send(()).unwrap();
    });
    returned_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("Executor::drop must not wait for an externally held artifact guard");
    drop_thread.join().expect("executor drop thread");
    assert_eq!(
        provider
            .residency()
            .expect("residency")
            .route_reservation_count(),
        1,
        "the mapping remains quarantined while the public guard is live"
    );
    let census = provider.route_residency_retirement_census();
    assert_eq!(census.active_registry_entries, 0);
    assert_eq!(census.retirement_registry_entries, 1);
    assert_eq!(census.deferred_cleanups, 1);
    assert_eq!(census.cleanups_scheduled, 0);

    drop(holder);
    assert!(
        provider
            .release_queue()
            .wait_until_idle(Duration::from_secs(30)),
        "last public guard release must complete cleanup: {:?}",
        provider.deferred_release_stats()
    );
    assert_eq!(provider.residency().unwrap().route_reservation_count(), 0);
    assert!(
        requirement
            .acquire_use()
            .err()
            .expect("retired requirement remains fail-closed")
            .to_string()
            .contains("retired")
    );
    drop(requirement);
    assert_eq!(
        provider
            .route_residency_retirement_census()
            .retirement_registry_entries,
        0
    );

    let mut provider =
        Arc::try_unwrap(provider).unwrap_or_else(|_| panic!("drop test provider shared"));
    provider.shutdown().expect("shutdown drop-test provider");
    wait_for_accounting(&ledger, 0, 0);
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
    let diag = Arc::clone(provider.route_residency_diagnostics());
    let mut primary = build_case(&fixture, Arc::clone(&provider), Arc::clone(&ledger), 2);

    assert!(
        provider.route_residency_scopes().is_empty(),
        "symbolic build cannot install before resolved kernel compilation"
    );
    assert_eq!(diag.installs(), 0);
    let scope = run_first_prefill(&mut primary, 2);
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
    let output_index = primary.bindings.len() - 1;
    let expert_zero_output = primary.bindings[output_index]
        .read_bytes()
        .expect("consume capture validation receipt");
    assert!(
        diag.values_touched() >= 8,
        "capture boundary did not apply route residency: {:?}",
        diag.last_reason()
    );
    assert!(diag.device_bytes_released() > 0);
    assert!(diag.host_bytes_committed() > 0);
    assert!(ledger.used(Tier::Host) > 0);
    let installed_boundary_count = provider
        .retained_route_residency_artifacts(scope)
        .expect("retained route-residency groups")
        .len() as u64;
    let boundaries_before_replays = diag.boundaries();
    for _ in 0..3 {
        assert!(
            primary
                .session
                .replay_device_graph(&mut primary.bindings)
                .expect("capture replay")
        );
        assert_eq!(
            primary.bindings[output_index]
                .read_bytes()
                .expect("consume capture replay validation receipt"),
            expert_zero_output
        );
    }
    assert_eq!(
        diag.boundaries(),
        boundaries_before_replays + 3 * installed_boundary_count,
        "every public fast replay must consume its request-local telemetry window"
    );

    set_routes(&mut primary, 2, [1; ROWS]);
    let touched_before = diag.values_touched();
    let boundaries_before_decode = diag.boundaries();
    for _ in 0..16 {
        assert!(
            primary
                .session
                .replay_device_graph(&mut primary.bindings)
                .expect("decode through remapped stable VA")
        );
        assert_eq!(
            primary.bindings[output_index]
                .read_bytes()
                .expect("consume decode validation receipt"),
            expert_zero_output
        );
    }
    assert_eq!(
        diag.boundaries(),
        boundaries_before_decode + 16 * installed_boundary_count
    );
    assert_eq!(
        diag.values_touched(),
        touched_before,
        "the finalized coarse placement installs once"
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
        assert_eq!(snapshot.count(), 0, "public replay resets each window");
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
    primary.bindings[output_index]
        .read_bytes()
        .expect("consume post-sibling validation receipt");

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

    run_fault_case(&fixture, false, FaultMember::First);
    run_fault_case(&fixture, true, FaultMember::First);
    run_fault_case(&fixture, true, FaultMember::Middle);
    run_fault_case(&fixture, true, FaultMember::Last);
    run_concurrent_safe_point_loss_case(&fixture);
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
    assert_eq!(provider.route_residency_diagnostics().installs(), 1);
    drop_case_to_provider_baseline(case);
    shutdown_provider(provider, ledger, baseline);
}
