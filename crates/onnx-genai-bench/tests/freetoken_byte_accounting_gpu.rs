#![cfg(feature = "gpu-tests")]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use onnx_runtime_ep_api::{ExecutionProvider, ExecutorInstanceId, ExecutorRouteResidencyConfig};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::byte_telemetry::{
    ObservedByteLedger, ObservedCategory, ObservedPhase, ObservedSnapshot, ObservedStatus,
};
use onnx_runtime_ep_cuda::coarse_residency::COARSE_RESIDENCY_ENABLE_ENV;
use onnx_runtime_ep_cuda::weight_paging::DeviceOffloadPolicy;
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{
    DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
};
use onnx_runtime_session::{DeviceGraphCaptureResult, DeviceIoBinding, InferenceSession};
use serde::Serialize;
use sha2::{Digest, Sha256};

const ROWS: usize = 4;
const EXPERTS: usize = 4;
const EVENT_CAPACITY: usize = 65_536;
const REPLAYS: usize = 3;
const DECODE_STEPS: usize = 4;
static FIXTURE_ID: AtomicU64 = AtomicU64::new(1);
static SERIAL: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FixtureKind {
    ManyExpertHybrid,
    GroupedRecurrent,
}

impl FixtureKind {
    fn label(self) -> &'static str {
        match self {
            Self::ManyExpertHybrid => "deepseek-like",
            Self::GroupedRecurrent => "glm52-like",
        }
    }

    fn hidden(self) -> usize {
        match self {
            Self::ManyExpertHybrid => 4096,
            Self::GroupedRecurrent => 8192,
        }
    }

    fn intermediate(self) -> usize {
        match self {
            Self::ManyExpertHybrid => 2048,
            Self::GroupedRecurrent => 1024,
        }
    }

    fn banks(self) -> usize {
        match self {
            Self::ManyExpertHybrid => 2,
            Self::GroupedRecurrent => 3,
        }
    }

    fn top_k(self) -> usize {
        match self {
            Self::ManyExpertHybrid => 1,
            Self::GroupedRecurrent => 2,
        }
    }

    fn device_budget(self) -> u64 {
        256 << 20
    }
}

struct GateGuard(Option<String>);

impl GateGuard {
    fn enable() -> Self {
        let prior = std::env::var(COARSE_RESIDENCY_ENABLE_ENV).ok();
        unsafe { std::env::set_var(COARSE_RESIDENCY_ENABLE_ENV, "1") };
        Self(prior)
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
    kind: FixtureKind,
}

impl Fixture {
    fn create(kind: FixtureKind) -> Result<Self> {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/deckard-freetoken-production-ab")
            .join(format!("{}-{id}", std::process::id()));
        fs::create_dir_all(&dir)
            .with_context(|| format!("create fixture directory {}", dir.display()))?;
        let weights = dir.join("weights.bin");
        write_weights(&weights, kind)?;
        let model = dir.join("model.onnx.textproto");
        fs::write(&model, model_text(kind))
            .with_context(|| format!("write fixture model {}", model.display()))?;
        Ok(Self { dir, model, kind })
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn write_repeated(writer: &mut BufWriter<File>, pattern: &[u8], bytes: usize) -> Result<()> {
    ensure!(!pattern.is_empty() && bytes.is_multiple_of(pattern.len()));
    let chunk = pattern.repeat((1 << 20) / pattern.len());
    let mut remaining = bytes;
    while remaining != 0 {
        let take = remaining.min(chunk.len());
        writer.write_all(&chunk[..take])?;
        remaining -= take;
    }
    Ok(())
}

fn packed_pattern(bank: usize, expert: usize, projection: usize) -> u8 {
    let nibble = ((bank * EXPERTS + expert + projection + 1) % 7 + 1) as u8;
    nibble | (nibble << 4)
}

fn write_expert_tensor(
    writer: &mut BufWriter<File>,
    bank: usize,
    experts: usize,
    bytes_per_expert: usize,
    projection: usize,
) -> Result<()> {
    for expert in 0..experts {
        write_repeated(
            writer,
            &[packed_pattern(bank, expert, projection)],
            bytes_per_expert,
        )?;
    }
    Ok(())
}

fn write_scale_tensor(
    writer: &mut BufWriter<File>,
    bank: usize,
    experts: usize,
    values_per_expert: usize,
    projection: usize,
) -> Result<()> {
    for expert in 0..experts {
        let scale = 0.000_01_f32 * (1 + bank * EXPERTS + expert + projection * 3) as f32;
        write_repeated(writer, &scale.to_le_bytes(), values_per_expert * 4)?;
    }
    Ok(())
}

fn tensor_extents(kind: FixtureKind) -> [usize; 4] {
    let hidden = kind.hidden();
    let intermediate = kind.intermediate();
    [
        EXPERTS * intermediate * (hidden / 2),
        EXPERTS * intermediate * (hidden / 16) * 4,
        EXPERTS * hidden * (intermediate / 2),
        EXPERTS * hidden * (intermediate / 16) * 4,
    ]
}

fn write_weights(path: &Path, kind: FixtureKind) -> Result<()> {
    let mut writer = BufWriter::new(
        File::create(path).with_context(|| format!("create weights {}", path.display()))?,
    );
    let hidden = kind.hidden();
    let intermediate = kind.intermediate();
    for bank in 0..kind.banks() {
        write_expert_tensor(&mut writer, bank, EXPERTS, intermediate * (hidden / 2), 0)?;
        write_scale_tensor(&mut writer, bank, EXPERTS, intermediate * (hidden / 16), 0)?;
        write_expert_tensor(&mut writer, bank, EXPERTS, hidden * (intermediate / 2), 1)?;
        write_scale_tensor(&mut writer, bank, EXPERTS, hidden * (intermediate / 16), 1)?;
    }
    writer.flush()?;
    Ok(())
}

fn value_info(text: &mut String, kind: &str, name: &str, columns: usize) {
    writeln!(text, "  {kind} {{").unwrap();
    writeln!(text, "    name: \"{name}\"").unwrap();
    writeln!(text, "    type {{ tensor_type {{ elem_type: 1 shape {{").unwrap();
    writeln!(text, "      dim {{ dim_value: {ROWS} }}").unwrap();
    writeln!(text, "      dim {{ dim_value: {columns} }}").unwrap();
    writeln!(text, "    }} }} }}").unwrap();
    writeln!(text, "  }}").unwrap();
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

fn qmoe_node(text: &mut String, kind: FixtureKind, bank: usize, input: &str, output: &str) {
    writeln!(text, "  node {{").unwrap();
    for name in [
        input.to_string(),
        format!("router_{bank}"),
        format!("b{bank}_fc1_packed"),
        format!("b{bank}_fc1_scales"),
        String::new(),
        format!("b{bank}_fc2_packed"),
        format!("b{bank}_fc2_scales"),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ] {
        writeln!(text, "    input: \"{name}\"").unwrap();
    }
    writeln!(text, "    output: \"{output}\"").unwrap();
    writeln!(text, "    name: \"qmoe_{bank}\"").unwrap();
    writeln!(text, "    op_type: \"QMoE\"").unwrap();
    writeln!(text, "    domain: \"com.microsoft\"").unwrap();
    writeln!(
        text,
        "    attribute {{ name: \"expert_weight_bits\" i: 4 type: INT }}"
    )
    .unwrap();
    writeln!(
        text,
        "    attribute {{ name: \"block_size\" i: 16 type: INT }}"
    )
    .unwrap();
    writeln!(
        text,
        "    attribute {{ name: \"k\" i: {} type: INT }}",
        kind.top_k()
    )
    .unwrap();
    text.push_str(
        "    attribute { name: \"activation_type\" s: \"silu\" type: STRING }\n\
         \x20   attribute { name: \"normalize_routing_weights\" i: 0 type: INT }\n\
         \x20   attribute { name: \"swiglu_fusion\" i: 0 type: INT }\n",
    );
    writeln!(text, "  }}").unwrap();
}

fn model_text(kind: FixtureKind) -> String {
    let hidden = kind.hidden();
    let intermediate = kind.intermediate();
    let extents = tensor_extents(kind);
    let mut text = format!(
        "ir_version: 11\nproducer_name: \"deckard-freetoken-production-ab\"\ngraph {{\n  \
         name: \"{}\"\n",
        kind.label()
    );
    for bank in 0..kind.banks() {
        let input = if bank == 0 {
            "hidden".to_string()
        } else {
            format!("bank_{}_out", bank - 1)
        };
        let output = format!("bank_{bank}_out");
        qmoe_node(&mut text, kind, bank, &input, &output);
    }
    let last = format!("bank_{}_out", kind.banks() - 1);
    writeln!(
        text,
        "  node {{ input: \"{last}\" input: \"state\" output: \"output\" op_type: \"Add\" }}"
    )
    .unwrap();
    writeln!(
        text,
        "  node {{ input: \"output\" output: \"state_out\" op_type: \"Identity\" }}"
    )
    .unwrap();

    let mut offset = 0usize;
    for bank in 0..kind.banks() {
        for (suffix, dtype, dims, len) in [
            (
                "fc1_packed",
                2,
                vec![EXPERTS, intermediate, hidden / 2],
                extents[0],
            ),
            (
                "fc1_scales",
                1,
                vec![EXPERTS, intermediate, hidden / 16],
                extents[1],
            ),
            (
                "fc2_packed",
                2,
                vec![EXPERTS, hidden, intermediate / 2],
                extents[2],
            ),
            (
                "fc2_scales",
                1,
                vec![EXPERTS, hidden, intermediate / 16],
                extents[3],
            ),
        ] {
            external_initializer(
                &mut text,
                &format!("b{bank}_{suffix}"),
                dtype,
                &dims,
                offset,
                len,
            );
            offset += len;
        }
    }
    value_info(&mut text, "input", "hidden", hidden);
    value_info(&mut text, "input", "state", hidden);
    for bank in 0..kind.banks() {
        value_info(&mut text, "input", &format!("router_{bank}"), EXPERTS);
        value_info(&mut text, "value_info", &format!("bank_{bank}_out"), hidden);
    }
    value_info(&mut text, "output", "output", hidden);
    value_info(&mut text, "output", "state_out", hidden);
    text.push_str(
        "}\nopset_import { domain: \"\" version: 13 }\n\
         opset_import { domain: \"com.microsoft\" version: 1 }\n",
    );
    text
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn router_bytes(kind: FixtureKind, bank: usize, selected: [usize; ROWS]) -> Vec<u8> {
    selected
        .into_iter()
        .enumerate()
        .flat_map(|(row, hot)| {
            (0..EXPERTS).map(move |expert| {
                if expert == hot {
                    30.0_f32 + bank as f32
                } else if kind.top_k() > 1 && expert == (hot + row + 1) % EXPERTS {
                    15.0_f32 + bank as f32
                } else {
                    -30.0_f32
                }
            })
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

struct LiveArm {
    provider: Arc<CudaExecutionProvider>,
    session: InferenceSession,
    bindings: Vec<DeviceIoBinding>,
    router_start: usize,
    state_index: usize,
    output_index: usize,
    state_output_index: usize,
}

fn build_arm(fixture: &Fixture, route_config: ExecutorRouteResidencyConfig) -> Result<LiveArm> {
    let capacity = 2_u64 << 30;
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
        LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
    ));
    let policy = DeviceOffloadPolicy {
        enabled: true,
        device_budget_bytes: Some(fixture.kind.device_budget()),
        ..DeviceOffloadPolicy::default()
    };
    let mut provider =
        CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
            0,
            policy,
            Arc::clone(&governor),
            route_config,
        )
        .context("construct production CUDA provider")?;
    provider
        .adopt_memory_governor(governor.as_ref(), Tier::Device, HolderId::new(0x2343))
        .context("adopt production weight-residency allowance")?;
    provider.configure_observed_byte_capacity(EVENT_CAPACITY)?;
    let provider = Arc::new(provider);
    let mut session = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()
        .context("build production QMoE session")?;
    session.warmup(&[]).context("finalize provider artifacts")?;

    let hidden = fixture.kind.hidden();
    let mut bindings = vec![
        session.allocate_device_binding(
            "hidden",
            None::<String>,
            DataType::Float32,
            vec![ROWS, hidden],
            vec![ROWS, hidden],
        )?,
        session.allocate_device_binding(
            "state",
            None::<String>,
            DataType::Float32,
            vec![ROWS, hidden],
            vec![ROWS, hidden],
        )?,
    ];
    let router_start = bindings.len();
    for bank in 0..fixture.kind.banks() {
        bindings.push(session.allocate_device_binding(
            format!("router_{bank}"),
            None::<String>,
            DataType::Float32,
            vec![ROWS, EXPERTS],
            vec![ROWS, EXPERTS],
        )?);
    }
    let output_index = bindings.len();
    bindings.push(session.allocate_device_output_binding(
        "output",
        DataType::Float32,
        vec![ROWS, hidden],
        vec![ROWS, hidden],
    )?);
    let state_output_index = bindings.len();
    let mut state_output = session.allocate_device_output_binding(
        "state_out",
        DataType::Float32,
        vec![ROWS, hidden],
        vec![ROWS, hidden],
    )?;
    state_output.mark_state_publication()?;
    bindings.push(state_output);

    let hidden_bytes = (0..ROWS * hidden)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.002)
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    bindings[0].write_bytes(0, &hidden_bytes)?;
    bindings[1].write_bytes(0, &vec![0_u8; ROWS * hidden * 4])?;
    Ok(LiveArm {
        provider,
        session,
        bindings,
        router_start,
        state_index: 1,
        output_index,
        state_output_index,
    })
}

fn set_routes(arm: &mut LiveArm, kind: FixtureKind, routes: &[[usize; ROWS]]) -> Result<()> {
    ensure!(routes.len() == kind.banks());
    for (bank, selected) in routes.iter().copied().enumerate() {
        arm.bindings[arm.router_start + bank]
            .write_bytes(0, &router_bytes(kind, bank, selected))?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct StepProof {
    phase: &'static str,
    generated_length: usize,
    route_digest: String,
    output_digest: String,
    state_digest: String,
}

fn execute_step(
    arm: &mut LiveArm,
    kind: FixtureKind,
    phase: &'static str,
    generated_length: usize,
    routes: &[[usize; ROWS]],
    replay: bool,
) -> Result<StepProof> {
    set_routes(arm, kind, routes)?;
    if replay {
        ensure!(arm.session.replay_device_graph(&mut arm.bindings)?);
    } else {
        arm.session
            .run_with_device_bindings(&[], &mut arm.bindings)?;
    }
    let output = arm.bindings[arm.output_index].read_bytes()?;
    let state = arm.bindings[arm.state_output_index].read_bytes()?;
    ensure!(
        output == state,
        "state output must publish the exact produced output"
    );
    arm.bindings[arm.state_index].write_bytes(0, &state)?;
    Ok(StepProof {
        phase,
        generated_length,
        route_digest: digest(&serde_json::to_vec(routes)?),
        output_digest: digest(&output),
        state_digest: digest(&state),
    })
}

fn routes(kind: FixtureKind, step: usize) -> Vec<[usize; ROWS]> {
    (0..kind.banks())
        .map(|bank| {
            [
                (step + bank) % EXPERTS,
                (step * 2 + bank + 1) % EXPERTS,
                (step + bank + 2) % EXPERTS,
                (step * 3 + bank + 3) % EXPERTS,
            ]
        })
        .collect()
}

#[derive(Serialize)]
struct ArmReport {
    scope: onnx_runtime_ep_cuda::byte_telemetry::ObservedScope,
    optimization_mode: &'static str,
    route_residency_outcome: String,
    events: usize,
    phase_events: BTreeMap<ObservedPhase, usize>,
    event_totals: BTreeMap<
        onnx_runtime_ep_cuda::byte_telemetry::ObservedCategory,
        BTreeMap<ObservedStatus, u64>,
    >,
    cold_h2d: u64,
    warm_h2d: u64,
    cold_d2h: u64,
    warm_d2h: u64,
    cold_d2d_submitted: u64,
    warm_d2d_submitted: u64,
    cold_memset_submitted: u64,
    warm_memset_submitted: u64,
    replay_events: usize,
    state_publications: u64,
    output_publications: u64,
    recorder_context_entries: u64,
    recorder_batch_reservations: u64,
    recorder_retained_clones: u64,
    warm_recorder_retained_clones: u64,
    recorder_mutex_acquisitions: u64,
    recorder_thread_id_lookups: u64,
    recorder_vector_growths: u64,
    device_bytes_without_release_receipt: u64,
    mapped_bytes_without_unmap_receipt: u64,
    proofs: Vec<StepProof>,
}

fn phase_bytes(
    snapshot: &ObservedSnapshot,
    phases: &[ObservedPhase],
    category: ObservedCategory,
    status: ObservedStatus,
) -> u64 {
    phases
        .iter()
        .map(|phase| snapshot.phase_bytes(*phase, category, status))
        .sum()
}

fn observation(arm: &LiveArm) -> Result<&ObservedByteLedger> {
    arm.session
        .provider_artifact_observation::<ObservedByteLedger>()
        .context("session omitted its exact observed-byte owner")
}

fn run_arm(fixture: &Fixture, optimized: bool) -> Result<ArmReport> {
    let route_config = if optimized {
        ExecutorRouteResidencyConfig::Enabled
    } else {
        ExecutorRouteResidencyConfig::Disabled
    };
    let mut arm = build_arm(fixture, route_config)?;
    observation(&arm)?.set_phase(ObservedPhase::Prefill)?;
    let mut proofs = vec![execute_step(
        &mut arm,
        fixture.kind,
        "prefill",
        0,
        &routes(fixture.kind, 0),
        false,
    )?];
    let retained_clones_before_warm = observation(&arm)?.hot_path_stats().retained_recorder_clones;

    observation(&arm)?.set_phase(ObservedPhase::DirectWarmup)?;
    proofs.push(execute_step(
        &mut arm,
        fixture.kind,
        "direct_warmup",
        1,
        &routes(fixture.kind, 1),
        false,
    )?);

    observation(&arm)?.set_phase(ObservedPhase::CaptureSetup)?;
    set_routes(&mut arm, fixture.kind, &routes(fixture.kind, 2))?;
    let capture = arm
        .session
        .try_capture_with_device_bindings(&[], &mut arm.bindings)?;
    if let DeviceGraphCaptureResult::NotCapturable(reason) = capture {
        anyhow::bail!("production capture was declined: {reason}");
    }
    ensure!(
        arm.session.captured_graph_segment_count() > 0,
        "first real CUDA graph capture must publish at least one segment"
    );
    let capture_output = arm.bindings[arm.output_index].read_bytes()?;
    let capture_state = arm.bindings[arm.state_output_index].read_bytes()?;
    ensure!(capture_output == capture_state);
    arm.bindings[arm.state_index].write_bytes(0, &capture_state)?;
    proofs.push(StepProof {
        phase: "capture",
        generated_length: 2,
        route_digest: digest(&serde_json::to_vec(&routes(fixture.kind, 2))?),
        output_digest: digest(&capture_output),
        state_digest: digest(&capture_state),
    });

    observation(&arm)?.set_phase(ObservedPhase::Replay)?;
    for replay in 0..REPLAYS {
        proofs.push(execute_step(
            &mut arm,
            fixture.kind,
            "replay",
            3 + replay,
            &routes(fixture.kind, 2),
            true,
        )?);
    }

    observation(&arm)?.set_phase(ObservedPhase::DecodeSteady)?;
    for step in 0..DECODE_STEPS {
        proofs.push(execute_step(
            &mut arm,
            fixture.kind,
            "decode",
            3 + REPLAYS + step,
            &routes(fixture.kind, step + 3),
            false,
        )?);
    }
    let retained_clones_after_warm = observation(&arm)?.hot_path_stats().retained_recorder_clones;

    let distinct_outputs = proofs
        .iter()
        .map(|proof| &proof.output_digest)
        .collect::<BTreeSet<_>>();
    ensure!(
        distinct_outputs.len() > 2,
        "unique expert routes and carried state must produce nontrivial distinct outputs"
    );
    let reader = observation(&arm)?.read_handle();
    let route_residency_outcome = format!(
        "{:?}",
        arm.provider
            .route_residency_executor_status(ExecutorInstanceId::from_raw(
                observation(&arm)?.scope().executor,
            ))
            .outcome
    );
    observation(&arm)?.set_phase(ObservedPhase::Teardown)?;
    let LiveArm {
        provider,
        session,
        bindings,
        ..
    } = arm;
    drop(bindings);
    drop(session);
    let queue = Arc::clone(provider.release_queue());
    let mut provider = Arc::try_unwrap(provider)
        .map_err(|_| anyhow::anyhow!("production A/B provider remained shared after teardown"))?;
    provider.shutdown()?;
    ensure!(
        queue.wait_until_idle(std::time::Duration::from_secs(30)),
        "production A/B deferred releases did not reach terminal outcomes"
    );
    drop(provider);
    let snapshot = reader.snapshot()?;
    let event_totals = onnx_runtime_ep_cuda::byte_telemetry::ObservedCategory::ALL
        .into_iter()
        .map(|category| {
            let statuses = ObservedStatus::ALL
                .into_iter()
                .map(|status| (status, snapshot.bytes(category, status)))
                .collect();
            (category, statuses)
        })
        .collect();
    let replay_events = snapshot
        .events
        .iter()
        .filter(|event| event.phase == ObservedPhase::Replay)
        .count();
    let phase_events = ObservedPhase::ALL
        .into_iter()
        .map(|phase| {
            (
                phase,
                snapshot
                    .events
                    .iter()
                    .filter(|event| event.phase == phase)
                    .count(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let state_publications = snapshot.bytes(
        ObservedCategory::StatePublication,
        ObservedStatus::Published,
    );
    let output_publications = snapshot.bytes(
        ObservedCategory::OutputPublication,
        ObservedStatus::Published,
    );
    let hot_path = reader.hot_path_stats();
    let cold_phases = [ObservedPhase::Setup, ObservedPhase::Prefill];
    let warm_phases = [
        ObservedPhase::DirectWarmup,
        ObservedPhase::CaptureSetup,
        ObservedPhase::Replay,
        ObservedPhase::DecodeSteady,
    ];
    let device_bytes_without_release_receipt = snapshot
        .bytes(
            ObservedCategory::DeviceAllocation,
            ObservedStatus::Committed,
        )
        .checked_sub(snapshot.bytes(ObservedCategory::DeviceRelease, ObservedStatus::Reclaimed))
        .context("device release receipts exceed observed allocations")?;
    let mapped_bytes_without_unmap_receipt = snapshot
        .bytes(ObservedCategory::VmmMap, ObservedStatus::Committed)
        .checked_sub(snapshot.bytes(ObservedCategory::VmmUnmap, ObservedStatus::Reclaimed))
        .context("VMM unmap receipts exceed observed maps")?;
    ensure!(replay_events >= REPLAYS);
    ensure!(state_publications > 0);
    ensure!(output_publications > 0);
    ensure!(retained_clones_after_warm == retained_clones_before_warm);
    ensure!(
        hot_path.mutex_acquisitions == 0
            && hot_path.thread_id_lookups == 0
            && hot_path.vector_growths == 0
    );
    for phase in [
        ObservedPhase::DirectWarmup,
        ObservedPhase::CaptureSetup,
        ObservedPhase::Replay,
        ObservedPhase::DecodeSteady,
    ] {
        ensure!(
            phase_events.get(&phase).copied().unwrap_or(0) > 0,
            "enabled production telemetry recorded no {phase:?} events"
        );
    }
    Ok(ArmReport {
        scope: snapshot.scope,
        optimization_mode: if optimized { "freetoken" } else { "baseline" },
        route_residency_outcome,
        events: snapshot.events.len(),
        phase_events,
        event_totals,
        cold_h2d: phase_bytes(
            &snapshot,
            &cold_phases,
            ObservedCategory::H2d,
            ObservedStatus::Completed,
        ),
        warm_h2d: phase_bytes(
            &snapshot,
            &warm_phases,
            ObservedCategory::H2d,
            ObservedStatus::Completed,
        ),
        cold_d2h: phase_bytes(
            &snapshot,
            &cold_phases,
            ObservedCategory::D2h,
            ObservedStatus::Completed,
        ),
        warm_d2h: phase_bytes(
            &snapshot,
            &warm_phases,
            ObservedCategory::D2h,
            ObservedStatus::Completed,
        ),
        cold_d2d_submitted: phase_bytes(
            &snapshot,
            &cold_phases,
            ObservedCategory::D2d,
            ObservedStatus::Submitted,
        ),
        warm_d2d_submitted: phase_bytes(
            &snapshot,
            &warm_phases,
            ObservedCategory::D2d,
            ObservedStatus::Submitted,
        ),
        cold_memset_submitted: phase_bytes(
            &snapshot,
            &cold_phases,
            ObservedCategory::CudaMemset,
            ObservedStatus::Submitted,
        ),
        warm_memset_submitted: phase_bytes(
            &snapshot,
            &warm_phases,
            ObservedCategory::CudaMemset,
            ObservedStatus::Submitted,
        ),
        replay_events,
        state_publications,
        output_publications,
        recorder_context_entries: hot_path.context_entries,
        recorder_batch_reservations: hot_path.batch_reservations,
        recorder_retained_clones: hot_path.retained_recorder_clones,
        warm_recorder_retained_clones: retained_clones_after_warm
            .checked_sub(retained_clones_before_warm)
            .context("warm recorder-retain counter regressed")?,
        recorder_mutex_acquisitions: hot_path.mutex_acquisitions,
        recorder_thread_id_lookups: hot_path.thread_id_lookups,
        recorder_vector_growths: hot_path.vector_growths,
        device_bytes_without_release_receipt,
        mapped_bytes_without_unmap_receipt,
        proofs,
    })
}

#[derive(Serialize)]
struct ComparisonReport {
    schema: &'static str,
    fixture: FixtureKind,
    dimensions: BTreeMap<&'static str, usize>,
    semantic_equivalent: bool,
    capture_calls_per_arm: usize,
    replay_calls_per_arm: usize,
    baseline: ArmReport,
    optimized: ArmReport,
    cold_h2d_delta: i128,
    warm_h2d_delta: i128,
    cold_d2h_delta: i128,
    warm_d2h_delta: i128,
    cold_d2d_submitted_delta: i128,
    warm_d2d_submitted_delta: i128,
    limitation: &'static str,
}

fn write_report(report: &ComparisonReport) -> Result<()> {
    let Some(directory) = std::env::var_os("ONNX_GENAI_FREETOKEN_OBSERVED_OUTPUT_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{}.json", report.fixture.label()));
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

#[test]
fn production_session_ab_proves_outputs_state_routes_capture_and_replay() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let _gate = GateGuard::enable();
    for kind in [FixtureKind::ManyExpertHybrid, FixtureKind::GroupedRecurrent] {
        let fixture = Fixture::create(kind)?;
        let baseline = run_arm(&fixture, false)?;
        let optimized = run_arm(&fixture, true)?;
        let semantic_equivalent = baseline.proofs == optimized.proofs;
        ensure!(
            semantic_equivalent,
            "{} baseline and optimized production outputs/state/routes diverged",
            kind.label()
        );
        ensure!(baseline.scope.provider != optimized.scope.provider);
        ensure!(baseline.scope.executor != optimized.scope.executor);
        let (minimum_cold_d2h, minimum_warm_h2d, minimum_warm_d2h) = match kind {
            FixtureKind::ManyExpertHybrid => (131_072, 590_976, 1_179_648),
            FixtureKind::GroupedRecurrent => (262_144, 1_181_376, 2_359_296),
        };
        for (name, arm) in [("baseline", &baseline), ("optimized", &optimized)] {
            ensure!(
                arm.warm_h2d >= minimum_warm_h2d,
                "{name} {} warm H2D omitted known route/state binding traffic: {} < {}",
                kind.label(),
                arm.warm_h2d,
                minimum_warm_h2d
            );
            ensure!(
                arm.cold_d2h >= minimum_cold_d2h && arm.warm_d2h >= minimum_warm_d2h,
                "{name} {} D2H omitted known output/state readback traffic: cold={} warm={}",
                kind.label(),
                arm.cold_d2h,
                arm.warm_d2h
            );
        }
        let report = ComparisonReport {
            schema: "onnx-genai.freetoken-production-ab.v3",
            fixture: kind,
            dimensions: BTreeMap::from([
                ("experts", EXPERTS),
                ("banks", kind.banks()),
                ("top_k", kind.top_k()),
                ("batch", ROWS),
                ("hidden", kind.hidden()),
                ("intermediate", kind.intermediate()),
                ("decode_steps", DECODE_STEPS),
            ]),
            semantic_equivalent,
            capture_calls_per_arm: 1,
            replay_calls_per_arm: REPLAYS,
            cold_h2d_delta: optimized.cold_h2d as i128 - baseline.cold_h2d as i128,
            warm_h2d_delta: optimized.warm_h2d as i128 - baseline.warm_h2d as i128,
            cold_d2h_delta: optimized.cold_d2h as i128 - baseline.cold_d2h as i128,
            warm_d2h_delta: optimized.warm_d2h as i128 - baseline.warm_d2h as i128,
            cold_d2d_submitted_delta: optimized.cold_d2d_submitted as i128
                - baseline.cold_d2d_submitted as i128,
            warm_d2d_submitted_delta: optimized.warm_d2d_submitted as i128
                - baseline.warm_d2d_submitted as i128,
            baseline,
            optimized,
            limitation: "synthetic structurally typed QMoE fixture; not a full checkpoint or \
                         throughput claim",
        };
        eprintln!(
            "freetoken_production_ab={}",
            serde_json::to_string(&report)?
        );
        write_report(&report)?;
    }
    Ok(())
}

#[test]
fn capacity_one_rejects_second_real_session_allocation_before_cuda_work() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let fixture = Fixture::create(FixtureKind::ManyExpertHybrid)?;
    let capacity = 2_u64 << 30;
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
        LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
    ));
    let mut provider =
        CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some(fixture.kind.device_budget()),
                ..DeviceOffloadPolicy::default()
            },
            governor,
            ExecutorRouteResidencyConfig::Disabled,
        )?;
    provider.configure_observed_byte_capacity(1)?;
    let provider = Arc::new(provider);
    let mut session = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()?;
    session.warmup(&[])?;
    let before = provider
        .device_allocation_counts()
        .context("allocation counters")?
        .0;
    let first = session.allocate_device_binding(
        "hidden",
        None::<String>,
        DataType::Float32,
        vec![ROWS, fixture.kind.hidden()],
        vec![ROWS, fixture.kind.hidden()],
    )?;
    let after_first = provider
        .device_allocation_counts()
        .context("allocation counters")?
        .0;
    ensure!(after_first > before);
    let error = session
        .allocate_device_binding(
            "state",
            None::<String>,
            DataType::Float32,
            vec![ROWS, fixture.kind.hidden()],
            vec![ROWS, fixture.kind.hidden()],
        )
        .expect_err("second allocation must fail before exceeding event capacity");
    ensure!(
        error.to_string().contains("event capacity 1")
            && error.to_string().contains("no event was committed"),
        "capacity error was not actionable: {error}"
    );
    ensure!(
        provider
            .device_allocation_counts()
            .context("allocation counters")?
            .0
            == after_first,
        "second CUDA allocation occurred despite failed telemetry preflight"
    );
    drop(first);
    Ok(())
}

#[test]
fn shared_provider_sessions_isolate_exact_recorders_and_default_off_stays_empty() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let fixture = Fixture::create(FixtureKind::ManyExpertHybrid)?;
    let capacity = 2_u64 << 30;
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
        LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
    ));
    let mut provider =
        CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some(fixture.kind.device_budget()),
                ..DeviceOffloadPolicy::default()
            },
            governor,
            ExecutorRouteResidencyConfig::Disabled,
        )?;
    provider.configure_observed_byte_capacity(64)?;
    let provider = Arc::new(provider);
    let mut first = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()?;
    let mut second = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()?;
    first.warmup(&[])?;
    second.warmup(&[])?;
    let first_observed = first
        .provider_artifact_observation::<ObservedByteLedger>()
        .context("first session observation")?;
    let second_observed = second
        .provider_artifact_observation::<ObservedByteLedger>()
        .context("second session observation")?;
    ensure!(first_observed.scope().provider == second_observed.scope().provider);
    ensure!(first_observed.scope().executor != second_observed.scope().executor);
    ensure!(first_observed.scope().logical_session != second_observed.scope().logical_session);
    let second_before = second_observed.snapshot()?.events.len();
    let _first_binding = first.allocate_device_binding(
        "hidden",
        None::<String>,
        DataType::Float32,
        vec![ROWS, fixture.kind.hidden()],
        vec![ROWS, fixture.kind.hidden()],
    )?;
    ensure!(first_observed.snapshot()?.events.len() > 0);
    ensure!(
        second_observed.snapshot()?.events.len() == second_before,
        "first session operation bled into sibling recorder"
    );

    first_observed.reset()?;
    let reset_epoch = first_observed.snapshot()?.epoch;
    let _after_reset = first.allocate_device_binding(
        "state",
        None::<String>,
        DataType::Float32,
        vec![ROWS, fixture.kind.hidden()],
        vec![ROWS, fixture.kind.hidden()],
    )?;
    ensure!(first_observed.snapshot()?.epoch == reset_epoch);
    ensure!(first_observed.snapshot()?.events.len() == 1);
    first_observed.close()?;
    let before_closed_attempt = provider
        .device_allocation_counts()
        .context("allocation counts")?
        .0;
    let closed = first
        .allocate_device_output_binding(
            "output",
            DataType::Float32,
            vec![ROWS, fixture.kind.hidden()],
            vec![ROWS, fixture.kind.hidden()],
        )
        .expect_err("closed exact recorder must reject new owner operations");
    ensure!(
        closed
            .to_string()
            .contains("observed-byte ledger is closed")
    );
    ensure!(
        provider
            .device_allocation_counts()
            .context("allocation counts")?
            .0
            == before_closed_attempt
    );
    let _sibling_still_live = second.allocate_device_binding(
        "hidden",
        None::<String>,
        DataType::Float32,
        vec![ROWS, fixture.kind.hidden()],
        vec![ROWS, fixture.kind.hidden()],
    )?;

    let default_governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
        LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
    ));
    let default_provider = Arc::new(
        CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some(fixture.kind.device_budget()),
                ..DeviceOffloadPolicy::default()
            },
            default_governor,
            ExecutorRouteResidencyConfig::Disabled,
        )?,
    );
    let mut default_session = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(default_provider as Arc<dyn ExecutionProvider>)
        .build()?;
    default_session.warmup(&[])?;
    ensure!(
        default_session
            .provider_artifact_observation::<ObservedByteLedger>()
            .is_none(),
        "default-off session unexpectedly created observation authority"
    );
    Ok(())
}

#[test]
fn direct_provider_runtime_cannot_record_into_a_session_owned_ledger() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let fixture = Fixture::create(FixtureKind::ManyExpertHybrid)?;
    let capacity = 2_u64 << 30;
    let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
        LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
    ));
    let mut provider =
        CudaExecutionProvider::initialized_with_offload_policy_governor_and_route_config(
            0,
            DeviceOffloadPolicy {
                enabled: true,
                device_budget_bytes: Some(fixture.kind.device_budget()),
                ..DeviceOffloadPolicy::default()
            },
            governor,
            ExecutorRouteResidencyConfig::Disabled,
        )?;
    provider.configure_observed_byte_capacity(16)?;
    let before_session = provider.runtime().alloc_raw(4096)?;
    unsafe { provider.runtime().free_raw(before_session)? };
    let provider = Arc::new(provider);
    let mut session = InferenceSession::builder()
        .model(&fixture.model)
        .execution_provider(Arc::clone(&provider) as Arc<dyn ExecutionProvider>)
        .build()?;
    session.warmup(&[])?;
    let observed = session
        .provider_artifact_observation::<ObservedByteLedger>()
        .context("session observation")?;
    ensure!(observed.snapshot()?.events.is_empty());
    let hostile_direct = provider.runtime().alloc_raw(4096)?;
    unsafe { provider.runtime().free_raw(hostile_direct)? };
    ensure!(
        observed.snapshot()?.events.is_empty(),
        "direct provider/runtime work attached to a session-owned recorder"
    );
    Ok(())
}

#[test]
fn exact_binding_h2d_and_d2h_bytes_match_authoritative_cuda_call_totals() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let fixture = Fixture::create(FixtureKind::ManyExpertHybrid)?;
    let mut arm = build_arm(&fixture, ExecutorRouteResidencyConfig::Disabled)?;
    observation(&arm)?.set_phase(ObservedPhase::Verification)?;
    let ledger_before = observation(&arm)?.snapshot()?;
    let raw_before = arm.provider.runtime().transfer_byte_counts();
    let bytes = vec![0x5a; ROWS * fixture.kind.hidden() * 4];

    arm.bindings[0].write_bytes(0, &bytes)?;
    let downloaded = arm.bindings[0].read_bytes()?;
    ensure!(downloaded == bytes);

    let raw_after = arm.provider.runtime().transfer_byte_counts();
    let ledger_after = observation(&arm)?.snapshot()?;
    let expected = bytes.len() as u64;
    let ledger_h2d = ledger_after
        .phase_bytes(
            ObservedPhase::Verification,
            ObservedCategory::H2d,
            ObservedStatus::Completed,
        )
        .checked_sub(ledger_before.phase_bytes(
            ObservedPhase::Verification,
            ObservedCategory::H2d,
            ObservedStatus::Completed,
        ))
        .context("verification H2D ledger delta underflow")?;
    let ledger_d2h = ledger_after
        .phase_bytes(
            ObservedPhase::Verification,
            ObservedCategory::D2h,
            ObservedStatus::Completed,
        )
        .checked_sub(ledger_before.phase_bytes(
            ObservedPhase::Verification,
            ObservedCategory::D2h,
            ObservedStatus::Completed,
        ))
        .context("verification D2H ledger delta underflow")?;
    ensure!(ledger_h2d == expected && ledger_d2h == expected);
    ensure!(raw_after.h2d_completed - raw_before.h2d_completed == ledger_h2d);
    ensure!(raw_after.d2h_completed - raw_before.d2h_completed == ledger_d2h);
    ensure!(raw_after.h2d_attempted - raw_before.h2d_attempted == expected);
    ensure!(raw_after.d2h_attempted - raw_before.d2h_attempted == expected);
    Ok(())
}

#[test]
fn eager_capture_and_replay_publish_only_after_sync_and_validation() -> Result<()> {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let fixture = Fixture::create(FixtureKind::ManyExpertHybrid)?;
    let mut arm = build_arm(&fixture, ExecutorRouteResidencyConfig::Disabled)?;
    let read_outputs = |arm: &mut LiveArm| -> Result<(Vec<u8>, Vec<u8>)> {
        Ok((
            arm.bindings[arm.output_index].read_bytes()?,
            arm.bindings[arm.state_output_index].read_bytes()?,
        ))
    };

    set_routes(&mut arm, fixture.kind, &routes(fixture.kind, 0))?;
    arm.session
        .run_with_device_bindings(&[], &mut arm.bindings)?;
    let mut previous = read_outputs(&mut arm)?;
    ensure!(previous.0 == previous.1);

    let before_eager = observation(&arm)?.snapshot()?;
    set_routes(&mut arm, fixture.kind, &routes(fixture.kind, 1))?;
    arm.provider.fail_next_validation_for_test();
    let eager_error = arm
        .session
        .run_with_device_bindings(&[], &mut arm.bindings)
        .expect_err("eager validation failure must reject publication");
    ensure!(eager_error.to_string().contains("flags=0x40000000"));
    ensure!(read_outputs(&mut arm)? == previous);
    let after_eager = observation(&arm)?.snapshot()?;
    ensure!(
        after_eager.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::Published
        ) == before_eager.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::Published
        )
    );
    ensure!(
        after_eager.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::RolledBack
        ) > before_eager.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::RolledBack
        )
    );
    arm.session
        .run_with_device_bindings(&[], &mut arm.bindings)
        .context("retry eager after validation rollback")?;
    previous = read_outputs(&mut arm)?;
    arm.bindings[arm.state_index].write_bytes(0, &previous.1)?;

    set_routes(&mut arm, fixture.kind, &routes(fixture.kind, 2))?;
    arm.provider.fail_next_sync_for_test();
    let capture_error = match arm
        .session
        .try_capture_with_device_bindings(&[], &mut arm.bindings)
    {
        Ok(_) => anyhow::bail!("capture synchronization failure unexpectedly published"),
        Err(error) => error,
    };
    ensure!(
        capture_error
            .to_string()
            .contains("injected synchronization failure")
    );
    ensure!(read_outputs(&mut arm)? == previous);
    let capture = arm
        .session
        .try_capture_with_device_bindings(&[], &mut arm.bindings)
        .context("retry capture after synchronization rollback")?;
    ensure!(matches!(capture, DeviceGraphCaptureResult::Captured(_)));
    previous = read_outputs(&mut arm)?;
    ensure!(previous.0 == previous.1);
    arm.bindings[arm.state_index].write_bytes(0, &previous.1)?;

    let before_replay = observation(&arm)?.snapshot()?;
    arm.provider.fail_next_validation_for_test();
    let replay_error = arm
        .session
        .replay_device_graph(&mut arm.bindings)
        .expect_err("replay validation failure must reject publication");
    ensure!(replay_error.to_string().contains("flags=0x40000000"));
    ensure!(read_outputs(&mut arm)? == previous);
    let after_replay = observation(&arm)?.snapshot()?;
    ensure!(
        after_replay.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::Published
        ) == before_replay.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::Published
        )
    );
    ensure!(
        after_replay.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::RolledBack
        ) > before_replay.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::RolledBack
        )
    );
    ensure!(
        arm.session
            .replay_device_graph(&mut arm.bindings)
            .context("retry replay after validation rollback")?
    );
    Ok(())
}
