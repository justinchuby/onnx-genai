use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspaceContextKey, WorkspaceContextReader, WorkspacePreparation,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId};
use onnx_runtime_memory_api::{DeviceKey, MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryAuthorityId,
    MemoryGovernor, MemoryLease, PressureRegistry, ProcessMemoryManager,
};
use serde_json::{Value, json};
use std::hint::black_box;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

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

struct Fixture {
    rows: usize,
    k: usize,
    n: usize,
    width: usize,
    x: Vec<f32>,
    w: Vec<u8>,
    bias: Vec<f32>,
    oracle: Vec<f32>,
}
fn activation(row: usize, k: usize) -> i64 {
    if k.is_multiple_of(32) {
        127
    } else {
        ((row * 17 + k * 13) % 255) as i64 - 127
    }
}
fn weight(q4: bool, feature: usize, block: usize, lane: usize) -> i64 {
    let code = feature * 7 + block * 3 + lane * 11 + (lane / 16) * 5;
    if q4 {
        (code % 16) as i64 - 8
    } else {
        (code % 255) as i64 - 127
    }
}
impl Fixture {
    fn new(q4: bool, rows: usize, k: usize, n: usize) -> Self {
        assert!(
            rows > 0 && k > 0 && n > 0,
            "study dimensions must be positive; supply rows K N"
        );
        let width = if q4 { 18 } else { 34 };
        let blocks = k.div_ceil(32);
        let x = (0..rows)
            .flat_map(|row| (0..k).map(move |i| activation(row, i) as f32))
            .collect();
        let bias: Vec<_> = (0..n).map(|f| (f % 7) as f32 / 256.).collect();
        let mut w = vec![0u8; n * blocks * width];
        for f in 0..n {
            for b in 0..blocks {
                let bytes = &mut w[(f * blocks + b) * width..][..width];
                bytes[..2].copy_from_slice(&half::f16::from_f32(1. / 256.).to_le_bytes());
                if q4 {
                    for lane in 0..16 {
                        bytes[2 + lane] = (weight(true, f, b, lane) + 8) as u8
                            | (((weight(true, f, b, lane + 16) + 8) as u8) << 4);
                    }
                } else {
                    for lane in 0..32 {
                        bytes[2 + lane] = weight(false, f, b, lane) as i8 as u8;
                    }
                }
            }
        }
        // Integer coordinate oracle never invokes a production decoder. Bounds
        // keep every f32 prefix exact: abs(sum) <= K*127*127 + 6 < 2^24.
        assert!(
            k * 127 * 127 + 6 < 1 << 24,
            "reduce K for this exact-integer fixture"
        );
        let mut oracle = Vec::with_capacity(rows * n);
        for row in 0..rows {
            for f in 0..n {
                let sum: i64 = (0..k)
                    .map(|i| activation(row, i) * weight(q4, f, i / 32, i % 32))
                    .sum();
                oracle.push((sum + (f % 7) as i64) as f32 / 256.);
            }
        }
        Self {
            rows,
            k,
            n,
            width,
            x,
            w,
            bias,
            oracle,
        }
    }
}
struct Arm {
    kernel: Box<dyn Kernel>,
    cached: WorkspacePreparation,
    requirement: WorkspaceRequirement,
    output: Vec<f32>,
    reader: WorkspaceContextReader,
    context: WorkspaceContextKey,
}
impl Arm {
    fn new(
        cpu: &CpuExecutionProvider,
        f: &Fixture,
        q4: bool,
        arithmetic: &str,
        streamed: bool,
    ) -> Self {
        let mut node = Node::new(NodeId(0), "BlockQuantizedMatMul", vec![], vec![]);
        node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
        for (name, value) in [
            ("K", Attribute::Int(f.k as i64)),
            ("N", Attribute::Int(f.n as i64)),
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
                Attribute::Int(i64::from(streamed)),
            ),
        ] {
            node.attributes.insert(name.into(), value);
        }
        let mut kernel = cpu
            .get_kernel(&node, &[], 1)
            .expect("CPU study packed kernel must be supported");
        kernel.set_constant_inputs(&[false, true, true]);
        let a = [f.rows, f.k];
        let b = [f.n, f.k.div_ceil(32), f.width];
        let y = [f.rows, f.n];
        let c = [f.n];
        let metadata = [
            TensorMetadata::new(DataType::Float32, &a, true),
            TensorMetadata::new(DataType::Uint8, &b, true),
            TensorMetadata::new(DataType::Float32, &c, true),
        ];
        let cached = kernel
            .workspace_requirement(&metadata, &[OutputMetadata::tensor(DataType::Float32, &y)])
            .expect("metadata preparation");
        let requirement = cached
            .require_ready()
            .expect("study has static packed geometry");
        let reader = cpu.workspace_preparation_context();
        Self {
            kernel,
            cached,
            requirement,
            output: vec![0.; f.rows * f.n],
            reader,
            context: reader.read(),
        }
    }
    fn run(&mut self, cpu: &CpuExecutionProvider, f: &Fixture, setup_only: bool) {
        assert_eq!(
            self.context,
            self.reader.read(),
            "execution pool changed during the fixed study"
        );
        let a = [f.rows, f.k];
        let b = [f.n, f.k.div_ceil(32), f.width];
        let c = [f.n];
        let y = [f.rows, f.n];
        let a_strides = [f.k as i64, 1];
        let b_strides = [(f.k.div_ceil(32) * f.width) as i64, f.width as i64, 1];
        let c_strides = [1];
        let y_strides = [f.n as i64, 1];
        let device = DeviceId::cpu();
        let inputs = [
            TensorView::new(
                DevicePtr(f.x.as_ptr().cast()),
                DataType::Float32,
                &a,
                &a_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(f.w.as_ptr().cast()),
                DataType::Uint8,
                &b,
                &b_strides,
                device,
            ),
            TensorView::new(
                DevicePtr(f.bias.as_ptr().cast()),
                DataType::Float32,
                &c,
                &c_strides,
                device,
            ),
        ];
        let metadata = [
            TensorMetadata::new(DataType::Float32, &a, true),
            TensorMetadata::new(DataType::Uint8, &b, true),
            TensorMetadata::new(DataType::Float32, &c, true),
        ];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(self.output.as_mut_ptr().cast()),
            DataType::Float32,
            &y,
            &y_strides,
            device,
        )];
        let req = self
            .kernel
            .workspace_requirement_for_execution(
                &inputs,
                &outputs,
                &metadata,
                &[OutputMetadata::tensor(DataType::Float32, &y)],
                Some(self.cached),
            )
            .expect("live execution refinement")
            .require_ready()
            .expect("complete packed requirement");
        assert_eq!(
            req, self.requirement,
            "fixed fixture/context must preserve its admitted bound"
        );
        let mut loan = cpu
            .allocate_workspace(req.bytes as usize, req.alignment, req.role())
            .expect("whole-operation managed lane checkout");
        let workspace = WorkspaceView::new(DevicePtrMut(loan.as_mut_ptr()), loan.len());
        let result = if setup_only {
            Ok(())
        } else {
            self.kernel
                .execute_with_workspace(&inputs, &mut outputs, Some(workspace))
        };
        cpu.deallocate_workspace(loan).expect("managed lane return");
        result.expect("packed execution must complete before output publication");
        black_box(&self.output);
    }
    fn check(&self, f: &Fixture) -> u64 {
        assert_eq!(self.output.len(), f.oracle.len());
        let mut digest = 0xcbf29ce484222325u64;
        for (i, (&got, &want)) in self.output.iter().zip(&f.oracle).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "complete output mismatch at index {i}: {got} != {want}"
            );
            digest = (digest ^ u64::from(got.to_bits())).wrapping_mul(0x100000001b3);
        }
        digest
    }
}
fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: valid writable rusage storage; success initializes it completely.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    let usage = unsafe { usage.assume_init() };
    usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6
}
fn main() {
    let process_start = Instant::now();
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        6,
        "usage: packed-activation-study q4|q8 rows K N check|calibrate|paired|float32|q8_0|setup-float32|setup-q8_0 calls"
    );
    assert!(matches!(args[0].as_str(), "q4" | "q8"));
    let q4 = args[0] == "q4";
    let rows = args[1].parse().expect("rows integer");
    let k = args[2].parse().expect("K integer");
    let n = args[3].parse().expect("N integer");
    let mode = &args[4];
    let calls: usize = args[5].parse().expect("calls integer");
    assert!(calls > 0);
    let f = Fixture::new(q4, rows, k, n);
    let books = Counting(Arc::new(Books {
        governor: LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 64 << 20, 0)),
        claims: AtomicU64::new(0),
        returns: AtomicU64::new(0),
    }));
    // A vacuity control proves the observer sees zero-byte issuer callbacks too.
    let probe = books
        .reserve(
            Tier::Host,
            0,
            MemoryRole::Workspace { step_scoped: true },
            HolderId::new(13),
        )
        .unwrap();
    drop(probe);
    books.0.release(Tier::Host, 0);
    assert_eq!(books.counts(), (1, 1));
    let manager = ProcessMemoryManager::new().unwrap();
    let pool = HostWorkspacePool::governed(&manager, Arc::new(books.clone())).unwrap();
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(&books, HolderId::new(13)).unwrap(),
        Default::default(),
    )
    .unwrap()
    .with_workspace_pool(pool.clone())
    .unwrap();
    let cpu = CpuExecutionProvider::initialized_with_resources(resources).unwrap();
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "study requires RAYON_NUM_THREADS=1 and CPU_DECODE_THREADS=1"
    );
    let mut arms = [
        Arm::new(&cpu, &f, q4, "float32", true),
        Arm::new(&cpu, &f, q4, "float32", true),
        Arm::new(&cpu, &f, q4, "q8_0", false),
    ];
    let peak = arms.iter().map(|a| a.requirement.bytes).max().unwrap();
    let alignment = arms.iter().map(|a| a.requirement.alignment).max().unwrap();
    let lane = cpu
        .allocate_workspace(peak as usize, alignment, arms[0].requirement.role())
        .unwrap();
    cpu.deallocate_workspace(lane).unwrap();
    use onnx_runtime_ep_cpu::kernels::block_quantized_matmul::BLOCK_QUANT_MATMUL_DENSE_EXPANSIONS as EXPANSIONS;
    let before = EXPANSIONS.load(Ordering::Relaxed);
    {
        // Dense positive control: the observer must detect a real expansion.
        let mut dense = Arm::new(&cpu, &f, q4, "float32", false);
        dense.run(&cpu, &f, false);
        dense.check(&f);
    }
    assert!(
        EXPANSIONS.load(Ordering::Relaxed) > before,
        "dense positive control did not expand; set ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES=2097152"
    );
    let expansions = EXPANSIONS.load(Ordering::Relaxed);
    for arm in &mut arms {
        for _ in 0..3 {
            arm.run(&cpu, &f, false);
        }
        arm.check(&f);
    }
    assert_eq!(
        EXPANSIONS.load(Ordering::Relaxed),
        expansions,
        "streamed/Q8 arms must never expand weights"
    );
    let snapshot = manager.snapshot().unwrap();
    assert_eq!(snapshot.allocations.len(), 1);
    let identity = snapshot.allocations[0].identity;
    let physical = snapshot.allocations[0].physical_bytes.unwrap();
    assert_eq!(physical, peak);
    assert_eq!(books.used(Tier::Host), peak);
    let baseline_counts = books.counts();
    let mut windows: Vec<Value> = Vec::new();
    let mut window = |index: usize, count: usize, setup: bool, label: &str| {
        let before_counts = books.counts();
        let cpu_start = cpu_seconds();
        let wall_start = Instant::now();
        // These synchronous CPU windows include cached execution refinement,
        // managed checkout, kernel (unless setup control), and managed return.
        for _ in 0..count {
            arms[index].run(&cpu, &f, setup);
        }
        let wall = wall_start.elapsed().as_secs_f64();
        let cpu_time = cpu_seconds() - cpu_start;
        assert_eq!(
            books.counts(),
            before_counts,
            "warm issuer callbacks changed, including zero-byte calls"
        );
        assert_eq!(EXPANSIONS.load(Ordering::Relaxed), expansions);
        let digest = arms[index].check(&f);
        windows.push(json!({"label":label,"arm":index,"calls":count,"setup_only":setup,"wall_s":wall,"cpu_s":cpu_time,"cpu_per_wall":cpu_time/wall,"output_digest":format!("{digest:016x}"),"output_elements_checked":f.oracle.len(),"claims":0,"returns":0}));
        wall
    };
    let mut calibrated_calls = calls;
    match mode.as_str() {
        "check" => {
            window(0, calls, false, "check");
            window(2, calls, false, "check");
        }
        "calibrate" => loop {
            let a = window(0, calibrated_calls, false, "calibration");
            let b = window(2, calibrated_calls, false, "calibration");
            if a.min(b) >= 0.08 {
                break;
            }
            calibrated_calls = calibrated_calls
                .checked_mul(2)
                .expect("calibration call bound");
            assert!(
                calibrated_calls <= 1 << 22,
                "calibration did not reach 80 ms; inspect the benchmark route"
            );
        },
        "paired" => {
            for (label, sequence) in [
                ("aa", [0, 1, 1, 0]),
                ("ab", [0, 2, 2, 0]),
                ("ba", [2, 0, 0, 2]),
                ("aa_reverse", [1, 0, 0, 1]),
            ] {
                for arm in sequence {
                    window(arm, calls, false, label);
                }
            }
        }
        "float32" => {
            window(0, calls, false, "work");
        }
        "q8_0" => {
            window(2, calls, false, "work");
        }
        "setup-float32" => {
            window(0, calls, true, "setup");
        }
        "setup-q8_0" => {
            window(2, calls, true, "setup");
        }
        _ => panic!("unknown study mode {mode}; select a documented mode"),
    }
    assert_eq!(books.counts(), baseline_counts);
    let final_snapshot = manager.snapshot().unwrap();
    assert_eq!(final_snapshot.allocations.len(), 1);
    assert_eq!(final_snapshot.allocations[0].identity, identity);
    assert_eq!(books.used(Tier::Host), peak);
    assert_eq!(manager.process_used(Tier::Host), peak);
    assert_eq!(books.oversubscribed_bytes(Tier::Host), 0);
    assert_eq!(final_snapshot.unattributed_bytes, 0);
    let requirements: Vec<_> = arms.iter().map(|arm| arm.requirement.bytes).collect();
    // Settlement is outside every measured warm window.
    drop(arms);
    drop(cpu);
    pool.trim_idle(u64::MAX).unwrap();
    assert_eq!(books.used(Tier::Host), 0);
    assert_eq!(manager.process_used(Tier::Host), 0);
    // Give aggregate Mach sampling at least 250 ms of process lifetime.
    // This idle tail is outside all timed windows; parent CPU excludes sleep.
    let minimum = std::time::Duration::from_millis(250);
    if process_start.elapsed() < minimum {
        std::thread::sleep(minimum - process_start.elapsed());
    }
    println!(
        "{}",
        json!({"minimum_process_wall_s":0.25,"format":args[0],"rows":rows,"k":k,"n":n,"mode":mode,"calls":calls,"calibrated_calls":calibrated_calls,"requirements":requirements,"peak_workspace_bytes":peak,"physical_bytes":physical,"allocation_identity":format!("{identity:?}"),"warm_issuer_counts":baseline_counts,"zero_byte_counter_control":true,"dense_expansion_positive_control":true,"window_count":windows.len(),"windows":windows,"settled":true})
    );
}
