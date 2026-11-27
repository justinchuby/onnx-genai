//! Public-session A/B probe. No kernel wrapper or timed metadata counter.
//! The 64 MiB authority covers workspace/derived weights, not process memory.
use onnx_runtime_ep_cpu::{
    CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory,
};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_api::{DeviceKey, MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryAuthorityId,
    MemoryGovernor, MemoryLease, PressureRegistry, ProcessMemoryManager,
};
use onnx_runtime_session::{InferenceSession, Tensor};
use serde_json::{Value, json};
use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

const WIDTH: usize = 32;
const LIMIT: u64 = 64 << 20;

#[derive(Debug)]
struct Books {
    governor: LedgerGovernor,
    claims: AtomicU64,
    returns: AtomicU64,
}
impl LeaseAccounting for Books {
    fn authority_id(&self) -> Option<MemoryAuthorityId> {
        Some(self.governor.authority_id(Tier::Host))
    }
    fn try_claim(&self, tier: Tier, bytes: u64, role: MemoryRole) -> Result<(), MemoryError> {
        self.claims.fetch_add(1, Ordering::Relaxed);
        self.governor.ledger().try_claim(tier, bytes, role)
    }
    fn release(&self, tier: Tier, bytes: u64) {
        self.returns.fetch_add(1, Ordering::Relaxed);
        self.governor.ledger().release(tier, bytes);
    }
}
#[derive(Clone)]
struct Counting(Arc<Books>);
impl Counting {
    fn counts(&self) -> (u64, u64) {
        (
            self.0.claims.load(Ordering::Relaxed),
            self.0.returns.load(Ordering::Relaxed),
        )
    }
}
impl MemoryGovernor for Counting {
    fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
        self.0.governor.authority_id(tier)
    }
    fn pressure_registry(&self, tier: Tier) -> Option<PressureRegistry> {
        self.0.governor.pressure_registry(tier)
    }
    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.0.try_claim(tier, bytes, role)?;
        Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
    }
    fn available(&self, tier: Tier) -> u64 {
        self.0.governor.available(tier)
    }
    fn used(&self, tier: Tier) -> u64 {
        self.0.governor.used(tier)
    }
    fn oversubscribed_bytes(&self, tier: Tier) -> u64 {
        self.0.governor.oversubscribed_bytes(tier)
    }
}

fn usage() -> (f64, u64) {
    let mut raw = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: success initializes the writable rusage structure.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, raw.as_mut_ptr()) },
        0
    );
    let u = unsafe { raw.assume_init() };
    let cpu = u.ru_utime.tv_sec as f64
        + u.ru_utime.tv_usec as f64 / 1e6
        + u.ru_stime.tv_sec as f64
        + u.ru_stime.tv_usec as f64 / 1e6;
    let rss = if cfg!(target_os = "macos") {
        u.ru_maxrss as u64
    } else {
        (u.ru_maxrss as u64).checked_mul(1024).unwrap()
    };
    (cpu, rss)
}

fn inherited_lock() -> Value {
    let Ok(value) = std::env::var("NXRT_STUDY_LOCK_FD") else {
        return Value::Null;
    };
    let fd = value
        .parse::<libc::c_int>()
        .expect("NXRT_STUDY_LOCK_FD must name an inherited descriptor");
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: inspect only; this probe neither acquires nor releases the lock.
    assert_eq!(
        unsafe { libc::fstat(fd, stat.as_mut_ptr()) },
        0,
        "inherited lock FD is unavailable"
    );
    let stat = unsafe { stat.assume_init() };
    // ino_t width is target-dependent; keep the receipt's stable u64 representation.
    #[allow(clippy::unnecessary_cast)]
    let inode = stat.st_ino as u64;
    if let Ok(expected) = std::env::var("NXRT_STUDY_LOCK_INODE") {
        assert_eq!(
            inode,
            expected.parse::<u64>().unwrap(),
            "inherited shared lock inode changed"
        );
    }
    json!({"fd":fd,"device":stat.st_dev,"inode":inode,"owned_by_probe":false})
}

fn coordinate(row: usize, column: usize) -> i64 {
    if column == 0 {
        127
    } else {
        ((row * 17 + column * 13) % 31) as i64 - 15
    }
}
struct Fixture {
    model: Vec<u8>,
    input: Tensor,
    expected: Vec<u32>,
    rows: usize,
}
impl Fixture {
    fn new(kind: &str, nodes: usize, rows: usize) -> Self {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        graph
            .opset_imports
            .insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(), 1);
        let x = graph.create_named_value("X", DataType::Float32, static_shape([rows, WIDTH]));
        graph.add_input(x);
        let weights = if kind == "q8" {
            let w = graph.create_named_value("W", DataType::Uint8, static_shape([WIDTH, 1, 34]));
            let mut packed = vec![0u8; WIDTH * 34];
            for feature in 0..WIDTH {
                let block = &mut packed[feature * 34..][..34];
                block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
                block[2 + feature] = 1;
            }
            graph.set_initializer(
                w,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Uint8,
                    vec![WIDTH, 1, 34],
                    packed,
                )),
            );
            let b = graph.create_named_value("B", DataType::Float32, static_shape([WIDTH]));
            let bias = (0..WIDTH)
                .map(|c| if c == 0 { 0f32 } else { 1f32 })
                .flat_map(f32::to_le_bytes)
                .collect();
            graph.set_initializer(
                b,
                WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![WIDTH], bias)),
            );
            Some((w, b))
        } else {
            None
        };
        let mut previous = x;
        for index in 0..nodes {
            let y = graph.create_named_value(
                format!("Y{index}"),
                DataType::Float32,
                static_shape([rows, WIDTH]),
            );
            let mut node = Node::new(
                NodeId(u32::try_from(index).unwrap()),
                if kind == "q8" {
                    "BlockQuantizedMatMul"
                } else {
                    "Relu"
                },
                if let Some((w, b)) = weights {
                    vec![Some(previous), Some(w), Some(b)]
                } else {
                    vec![Some(previous)]
                },
                vec![y],
            );
            if kind == "q8" {
                node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
                for (name, value) in [
                    ("K", Attribute::Int(WIDTH as i64)),
                    ("N", Attribute::Int(WIDTH as i64)),
                    ("format", Attribute::String(b"q8_0".to_vec())),
                    ("activation_arithmetic", Attribute::String(b"q8_0".to_vec())),
                    ("block_layout_version", Attribute::Int(2)),
                ] {
                    node.attributes.insert(name.into(), value);
                }
            }
            graph.insert_node(node);
            previous = y;
        }
        graph.add_output(previous);
        let values = (0..rows)
            .flat_map(|r| (0..WIDTH).map(move |c| coordinate(r, c) as f32))
            .collect::<Vec<_>>();
        let expected = (0..rows)
            .flat_map(|r| {
                (0..WIDTH).map(move |c| {
                    let x = coordinate(r, c);
                    let y = if kind == "relu" {
                        x.max(0)
                    } else {
                        x + if c == 0 { 0 } else { nodes as i64 }
                    };
                    (y as f32).to_bits()
                })
            })
            .collect();
        Self {
            model: encode_model(&Model::new(&graph)).unwrap(),
            input: Tensor::from_f32(&[rows, WIDTH], &values).unwrap(),
            expected,
            rows,
        }
    }
}
fn step(session: &mut InferenceSession, f: &Fixture, prepare: bool) {
    let inputs = [("X", &f.input)];
    if prepare {
        session
            .prepare_with_device_bindings(&inputs, &mut [])
            .unwrap()
            .require_complete()
            .unwrap();
    }
    let output = session.run(black_box(&inputs)).unwrap();
    black_box(&output);
    // Every timed call includes destruction of the complete returned Vec and tensors.
    drop(output);
}
fn validate(session: &mut InferenceSession, f: &Fixture, prepare: bool) -> (Vec<u32>, String) {
    let inputs = [("X", &f.input)];
    if prepare {
        session
            .prepare_with_device_bindings(&inputs, &mut [])
            .unwrap()
            .require_complete()
            .unwrap();
    }
    let output = session.run(&inputs).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].shape, [f.rows, WIDTH]);
    let bits = output[0]
        .to_vec_f32()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        bits, f.expected,
        "complete independent coordinate oracle mismatch"
    );
    let digest = bits.iter().fold(0xcbf29ce484222325u64, |hash, &bits| {
        (hash ^ u64::from(bits)).wrapping_mul(0x100000001b3)
    });
    (bits, format!("{digest:016x}"))
}
fn owner(manager: &ProcessMemoryManager, books: &Counting) -> (u64, Vec<String>) {
    let snapshot = manager.snapshot().unwrap();
    let funded = books.used(Tier::Host);
    assert_eq!(manager.process_used(Tier::Host), funded);
    assert_eq!(books.oversubscribed_bytes(Tier::Host), 0);
    assert_eq!(snapshot.unattributed_bytes, 0);
    let identities = snapshot
        .allocations
        .iter()
        .map(|a| {
            assert_eq!(a.physical_bytes.unwrap(), funded);
            format!("{:?}", a.identity)
        })
        .collect::<Vec<_>>();
    assert!(
        identities.len() <= 1,
        "serial graph must retain at most one workspace owner"
    );
    (funded, identities)
}

#[allow(clippy::too_many_arguments)]
fn window(
    session: &mut InferenceSession,
    f: &Fixture,
    prepare: bool,
    nodes: usize,
    reps: usize,
    double: bool,
    index: usize,
    books: &Counting,
    manager: &ProcessMemoryManager,
    expected_owner: &(u64, Vec<String>),
) -> Value {
    let before = session.cache_stats();
    let issuer = books.counts();
    let actual = reps.checked_mul(if double { 2 } else { 1 }).unwrap();
    let cpu_start = usage().0;
    let start = Instant::now();
    for _ in 0..reps {
        for _ in 0..if double { 2 } else { 1 } {
            step(session, f, prepare);
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let cpu = usage().0 - cpu_start;
    let after = session.cache_stats();
    let prebind = after.prebind_hits - before.prebind_hits;
    assert_eq!(
        prebind,
        (nodes * actual) as u64,
        "timed public dispatch must reach every prebound node exactly once"
    );
    assert_eq!(after.entries, nodes);
    assert_eq!(after.misses, before.misses);
    assert_eq!(after.evictions, before.evictions);
    assert_eq!(
        books.counts(),
        issuer,
        "warm issuer callbacks changed, including zero-byte calls"
    );
    assert_eq!(&owner(manager, books), expected_owner);
    // Outputs of timed calls have already been destroyed. One additional run
    // validates complete results outside both timers and counter-delta windows.
    let (_, digest) = validate(session, f, prepare);
    json!({"kind":if double {"double"} else {"single"},"index":index,"outer_calls":reps,"actual_calls":actual,
        "cpu_s":cpu,"wall_s":wall,"prebind_delta":prebind,"digest":digest,"issuer_delta":{"claims":0,"returns":0},"untimed_validation_calls":1})
}

fn study(kind: &str, nodes: usize, rows: usize, mode: &str, reps: usize) -> Value {
    let lock = inherited_lock();
    assert_eq!(rayon::current_num_threads(), 1);
    assert!(rayon::current_thread_index().is_some());
    let books = Counting(Arc::new(Books {
        governor: LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, LIMIT, 0)),
        claims: AtomicU64::new(0),
        returns: AtomicU64::new(0),
    }));
    let zero = books
        .reserve(
            Tier::Host,
            0,
            MemoryRole::Workspace { step_scoped: true },
            HolderId::new(13),
        )
        .unwrap();
    drop(zero);
    books.0.release(Tier::Host, 0);
    assert_eq!(
        books.counts(),
        (1, 1),
        "observer must detect zero-byte admission and release"
    );
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(&books, HolderId::new(19)).unwrap(),
        CpuWeightCacheLimits {
            dense_bytes: 0,
            expert_bytes: 0,
        },
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let cpu = Arc::new(CpuExecutionProvider::initialized_with_resources(resources).unwrap());
    let cold_cpu = usage().0;
    let cold_wall = Instant::now();
    let f = Fixture::new(kind, nodes, rows);
    let mut session = InferenceSession::builder()
        .model_bytes(&f.model)
        .option("optimization", "none")
        .execution_provider(cpu.clone())
        .build()
        .unwrap();
    let prepare = mode == "prepare_run";
    let (output_bits, digest) = validate(&mut session, &f, prepare);
    let cold_wall_s = cold_wall.elapsed().as_secs_f64();
    let cold_cpu_s = usage().0 - cold_cpu;
    assert_eq!(session.cache_stats().entries, nodes);
    let warm_cpu = usage().0;
    let mut warm_calls = 0usize;
    loop {
        for _ in 0..32 {
            step(&mut session, &f, prepare);
            warm_calls += 1;
        }
        if usage().0 - warm_cpu >= 0.15 {
            break;
        }
    }
    validate(&mut session, &f, prepare);
    let expected_owner = owner(&manager, &books);
    assert_eq!(expected_owner.1.len(), usize::from(kind == "q8"));
    assert_eq!(expected_owner.0 > 0, kind == "q8");
    let warm_issuer = books.counts();
    let mut windows = Vec::new();
    let mut calibration_observations = Vec::new();
    // The initial cold validation and the post-warmup validation are separate
    // public runs. Every measured/calibration window adds exactly one more.
    let mut validation_runs = 2usize;
    let calibration_calls = if reps == 0 {
        let mut calls = 1usize;
        loop {
            let observed = window(
                &mut session,
                &f,
                prepare,
                nodes,
                calls,
                false,
                0,
                &books,
                &manager,
                &expected_owner,
            );
            let cpu = observed["cpu_s"].as_f64().unwrap();
            calibration_observations.push(observed);
            validation_runs += 1;
            if cpu >= 0.05 {
                let target = (calls as f64 * 0.15 / cpu).ceil() as usize;
                assert!(target > 0);
                break Some(target);
            }
            calls = calls.checked_mul(2).expect("calibration call overflow");
            assert!(calls <= 1 << 24, "calibration did not reach 50 ms CPU");
        }
    } else {
        for index in 0..9 {
            for double in if index % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                windows.push(window(
                    &mut session,
                    &f,
                    prepare,
                    nodes,
                    reps,
                    double,
                    index,
                    &books,
                    &manager,
                    &expected_owner,
                ));
                validation_runs += 1;
            }
        }
        None
    };
    assert_eq!(books.counts(), warm_issuer);
    assert_eq!(owner(&manager, &books), expected_owner);
    let (whole_before_settlement_cpu_s, peak_rss_bytes) = usage();
    let issuer_before_settlement = books.counts();
    drop(session);
    drop(cpu);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
    assert!(manager.snapshot().unwrap().allocations.is_empty());
    let issuer_after_settlement = books.counts();
    assert_eq!(inherited_lock(), lock);
    json!({"schema_version":1,"kind":kind,"nodes":nodes,"rows":rows,"mode":mode,"reps":reps,"width":WIDTH,
        "case":[kind,nodes,rows,mode],"threads":1,"windows":windows,"calibration_calls":calibration_calls,"calibration_observations":calibration_observations,"cold_cpu_s":cold_cpu_s,"cold_wall_s":cold_wall_s,
        "cold_scope":"model/input creation, session construction and first fully validated run; excludes spawn/provider/pool construction",
        "warmup_calls":warm_calls,"warmup_target_cpu_s":0.15,"whole_before_settlement_cpu_s":whole_before_settlement_cpu_s,
        "funded_bytes":expected_owner.0,"allocation_identities":expected_owner.1,"issuer_delta":{"claims":0,"returns":0},
        "output_bits":output_bits,"digest":digest,"peak_rss_bytes":peak_rss_bytes,"zero_byte_counter_control":true,
        "host_authority_limit_bytes":LIMIT,"scope":"public Session run with full returned-output destruction; optional public prepare per call; scoped workspace/derived cap, not process cap",
        "validation_scope":"additional untimed public run after every window; full bits checked against independent integer-coordinate oracle",
        "validation_runs_total":validation_runs,"warm_issuer_counts":warm_issuer,"issuer_counts_before_settlement":issuer_before_settlement,"issuer_counts_after_settlement":issuer_after_settlement,
        "inherited_lock":lock,"settled":true,"funded_after_settlement":0,"manager_used_after_settlement":0})
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        5,
        "usage: native-workspace-memo-study relu|q8 nodes rows run|prepare_run reps (0=calibrate)"
    );
    let kind = &args[0];
    let nodes = args[1].parse::<usize>().unwrap();
    let rows = args[2].parse::<usize>().unwrap();
    let mode = &args[3];
    let reps = args[4].parse::<usize>().unwrap();
    assert!(
        (kind == "relu" && nodes == 12 && rows == 1 && mode == "run")
            || (kind == "q8"
                && ((rows == 1 && [1, 3, 12].contains(&nodes) && mode == "run")
                    || (rows == 8 && nodes == 3 && mode == "run")
                    || (rows == 1 && nodes == 3 && mode == "prepare_run"))),
        "select one of the six frozen study cases"
    );
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| study(kind, nodes, rows, mode, reps));
    println!("{result}");
}
