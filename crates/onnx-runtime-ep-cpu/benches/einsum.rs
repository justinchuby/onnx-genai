mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{FloatDType, Tensor, assert_close, float_values, make_kernel, process_cpu_time};
use criterion::{BenchmarkId, Criterion, Throughput, black_box};
use onnx_runtime_ep_api::{Kernel, KernelFactory};
use onnx_runtime_ep_cpu::kernels::einsum::{
    EINSUM_MODE_ENV, EinsumFactory, benchmark_execute_route, benchmark_scratch_capacity_bytes,
};
use onnx_runtime_hostmon::{AllowedCpus, Contention};
use onnx_runtime_ir::{Attribute, Node, NodeId};

const EXPECTED_CRITERION_SELECTORS: usize = 12;
const ABSOLUTE_REPS: usize = 3;
const INTERLEAVED_BLOCKS: usize = 3;
const TARGET_WINDOW: Duration = Duration::from_millis(100);
const MAX_RANGE_PCT: f64 = 15.0;
const MAX_CONTROL_RANGE_PCT: f64 = 10.0;
const MAX_NULL_MEDIAN_DELTA_PCT: f64 = 3.0;
const MAX_FREQUENCY_DRIFT_PCT: f64 = 20.0;

struct CountingAllocator;

static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

fn reset_allocations() {
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn allocations() -> (u64, u64) {
    (
        ALLOCATION_CALLS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    equation: &'static str,
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    dtype: FloatDType,
    tolerance: f32,
    expected_route: &'static str,
}

#[derive(Clone, Copy)]
struct FrequencySample {
    min_khz: u64,
    median_khz: u64,
    max_khz: u64,
}

#[derive(Clone, Copy)]
struct Summary {
    median_ns: f64,
    min_ns: f64,
    max_ns: f64,
    range_pct: f64,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "gemm_small",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![4, 16], vec![16, 4]],
            output_shape: vec![4, 4],
            dtype: FloatDType::F32,
            tolerance: 1e-4,
            expected_route: "matmul-direct",
        },
        Case {
            name: "gemm_friendly",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![32, 256], vec![256, 256]],
            output_shape: vec![32, 256],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
            expected_route: "matmul-direct",
        },
        Case {
            name: "gemm_large",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![64, 512], vec![512, 256]],
            output_shape: vec![64, 256],
            dtype: FloatDType::F32,
            tolerance: 5e-3,
            expected_route: "matmul-direct",
        },
        Case {
            name: "transpose_required",
            equation: "ik,jk->ij",
            input_shapes: vec![vec![32, 256], vec![128, 256]],
            output_shape: vec![32, 128],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
            expected_route: "matmul-direct",
        },
        Case {
            name: "broadcast_bmm",
            equation: "...mk,...kn->...mn",
            input_shapes: vec![vec![4, 16, 128], vec![1, 128, 64]],
            output_shape: vec![4, 16, 64],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
            expected_route: "matmul-direct",
        },
        Case {
            name: "gemm_f16",
            equation: "ik,kj->ij",
            input_shapes: vec![vec![32, 256], vec![256, 128]],
            output_shape: vec![32, 128],
            dtype: FloatDType::F16,
            tolerance: 0.25,
            expected_route: "matmul-direct",
        },
        Case {
            name: "reduction",
            equation: "ij->i",
            input_shapes: vec![vec![512, 512]],
            output_shape: vec![512],
            dtype: FloatDType::F32,
            tolerance: 1e-3,
            expected_route: "reduction-native",
        },
        Case {
            name: "elementwise_product",
            equation: "ij,ij->ij",
            input_shapes: vec![vec![512, 512], vec![512, 512]],
            output_shape: vec![512, 512],
            dtype: FloatDType::F32,
            tolerance: 1e-6,
            expected_route: "reduction-native",
        },
        Case {
            name: "multi_axis_permute",
            equation: "abxy,xycd->dcab",
            input_shapes: vec![vec![4, 4, 8, 8], vec![8, 8, 4, 4]],
            output_shape: vec![4, 4, 4, 4],
            dtype: FloatDType::F32,
            tolerance: 2e-3,
            expected_route: "matmul-materialized",
        },
        Case {
            name: "diagonal",
            equation: "ii->i",
            input_shapes: vec![vec![1024, 1024]],
            output_shape: vec![1024],
            dtype: FloatDType::F32,
            tolerance: 0.0,
            expected_route: "view-copy",
        },
    ]
}

fn kernel_with_mode(case: &Case, mode: &str) -> (Box<dyn Kernel>, Duration) {
    struct EnvGuard(Option<OsString>);
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.0.take() {
                    Some(value) => std::env::set_var(EINSUM_MODE_ENV, value),
                    None => std::env::remove_var(EINSUM_MODE_ENV),
                }
            }
        }
    }

    let guard = EnvGuard(std::env::var_os(EINSUM_MODE_ENV));
    unsafe { std::env::set_var(EINSUM_MODE_ENV, mode) };
    let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
    node.attributes.insert(
        "equation".into(),
        Attribute::String(case.equation.as_bytes().to_vec()),
    );
    let start = Instant::now();
    let kernel = EinsumFactory
        .create(&node, &case.input_shapes)
        .expect("Einsum kernel construction must succeed");
    let elapsed = start.elapsed();
    drop(guard);
    (kernel, elapsed)
}

fn inputs(case: &Case) -> Vec<Tensor> {
    case.input_shapes
        .iter()
        .enumerate()
        .map(|(input, shape)| {
            let len = shape.iter().product();
            let values = float_values(len)
                .into_iter()
                .enumerate()
                .map(|(index, value)| value + input as f32 * 0.03125 + (index % 7) as f32 * 0.001)
                .collect::<Vec<_>>();
            Tensor::floats(case.dtype, shape, &values)
        })
        .collect()
}

fn hash_tensors(tensors: &[Tensor]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for tensor in tensors {
        for value in tensor.to_f32() {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

fn hash_values(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn validate_case(
    case: &Case,
    optimized: &dyn Kernel,
    oracle_kernel: &dyn Kernel,
    tensors: &[Tensor],
) {
    let views = tensors.iter().map(Tensor::view).collect::<Vec<_>>();
    let mut fast = Tensor::zeros(case.dtype, &case.output_shape);
    let mut oracle_output = Tensor::zeros(case.dtype, &case.output_shape);
    let route = benchmark_execute_route(optimized, &views, &mut [fast.view_mut()])
        .expect("optimized Einsum route probe must execute")
        .expect("route probe must receive an Einsum kernel");
    oracle_kernel
        .execute(&views, &mut [oracle_output.view_mut()])
        .expect("Einsum correctness oracle must execute");
    let fast = fast.to_f32();
    let oracle = oracle_output.to_f32();
    let nonzero = oracle.iter().filter(|value| **value != 0.0).count();
    assert!(
        nonzero > 0,
        "{} oracle is all-zero; the benchmark correctness check would be vacuous",
        case.name
    );
    assert_eq!(
        route, case.expected_route,
        "{} fired an unexpected native route",
        case.name
    );
    assert_close(&fast, &oracle, case.tolerance);
    let max_abs_error = fast
        .iter()
        .zip(&oracle)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    println!(
        "EINSUM_VALIDATE case={} dtype={} equation={} route={} shared_input_hash={:016x} \
         native_hash={:016x} oracle_hash={:016x} oracle_nonzero={nonzero} \
         max_abs_error={max_abs_error:.9} tolerance={:.9} exact_equal={}",
        case.name,
        case.dtype.name(),
        case.equation,
        route,
        hash_tensors(tensors),
        hash_values(&fast),
        hash_values(&oracle),
        case.tolerance,
        fast == oracle,
    );
}

fn command_text(program: &str, args: &[&str]) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CPU EP crate must live under <repo>/crates");
    let executable = if program.contains('/') {
        root.join(program)
    } else {
        PathBuf::from(program)
    };
    let output = Command::new(executable)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|error| panic!("could not run {program}: {error}"));
    assert!(
        output.status.success(),
        "{program} {} failed:\nstdout:\n{}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("{program} output was not UTF-8: {error}"))
        .trim()
        .to_string()
}

fn require_governance(is_list: bool) -> PathBuf {
    let owner = std::env::var("HOSTLOCK_OWNER").unwrap_or_else(|_| {
        panic!(
            "benchmark refused: HOSTLOCK_OWNER is unset; run the exact cargo bench invocation \
             through scripts/hostlock.sh run --owner <name>"
        )
    });
    let expected_runnable =
        std::env::var("EINSUM_BENCH_EXPECT_RUNNABLE").unwrap_or_else(|_| "4".into());
    let provenance = command_text(
        "scripts/hostlock.sh",
        &[
            "provenance",
            "--oneline",
            "--expect-runnable",
            &expected_runnable,
        ],
    );
    println!("HOSTLOCK_SCRIPT,{provenance}");
    for required in [
        "hostlock_state=HELD",
        "declared=yes",
        "held_owner_source=flag",
        "lock_scope=box",
        "takeover=none",
        "legacy_held_by=none",
    ] {
        assert!(
            format!(" {provenance} ").contains(&format!(" {required} ")),
            "benchmark refused: hostlock provenance lacks {required}: {provenance}"
        );
    }
    if !is_list {
        assert!(
            format!(" {provenance} ").contains(" contended=no "),
            "benchmark refused: hostlock provenance reports load after admission: {provenance}"
        );
    }
    assert!(
        format!(" {provenance} ").contains(" gate=satisfied:"),
        "benchmark refused: hostlock gate was not satisfied: {provenance}"
    );
    assert!(
        format!(" {provenance} ").contains(&format!(" held_by={owner} ")),
        "benchmark refused: HOSTLOCK_OWNER={owner} does not match provenance: {provenance}"
    );

    let tracked = command_text(
        "git",
        &[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ],
    );
    assert!(
        tracked.is_empty(),
        "benchmark refused: tracked worktree changes make commit/tree provenance inexact:\n{tracked}"
    );
    let commit = command_text("git", &["rev-parse", "HEAD"]);
    let tree = command_text("git", &["rev-parse", "HEAD^{tree}"]);
    let branch = command_text("git", &["symbolic-ref", "--short", "HEAD"]);
    println!("BUILD_COMMIT,{commit}");
    println!("BUILD_TREE,{tree}");
    println!("BUILD_BRANCH,{branch}");
    println!(
        "BUILD_RUSTC,{}",
        command_text("rustc", &["--version"]).replace(',', ";")
    );

    let target = std::env::var_os("CARGO_TARGET_DIR").unwrap_or_else(|| {
        panic!(
            "benchmark refused: set CARGO_TARGET_DIR to a dedicated evidence directory so the \
             complete raw Criterion tree is identifiable"
        )
    });
    let target = PathBuf::from(target);
    assert!(
        target.is_absolute(),
        "benchmark refused: CARGO_TARGET_DIR must be absolute so Cargo and the benchmark identify \
         the same raw Criterion tree"
    );
    let criterion = target.join("criterion");
    if !is_list {
        for group in ["einsum", "einsum_view", "einsum_control"] {
            assert!(
                !criterion.join(group).exists(),
                "benchmark refused: {} already exists; use a fresh CARGO_TARGET_DIR so raw \
                 Criterion evidence cannot mix runs",
                criterion.join(group).display()
            );
        }
    }
    println!("CRITERION_RAW_DIR,{}", criterion.display());
    target
}

fn read_cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|rest| rest.split_once(':'))
                .map(|(_, value)| value.trim().replace(',', ";"))
        })
}

fn physical_core(cpu: usize) -> Option<(u64, u64)> {
    let root = PathBuf::from(format!("/sys/devices/system/cpu/cpu{cpu}/topology"));
    let package = fs::read_to_string(root.join("physical_package_id"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let core = fs::read_to_string(root.join("core_id"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some((package, core))
}

fn require_physical_core_affinity() -> AllowedCpus {
    let requested: usize = std::env::var("ONNX_GENAI_CPU_DECODE_THREADS")
        .unwrap_or_else(|_| {
            panic!(
                "benchmark refused: set ONNX_GENAI_CPU_DECODE_THREADS to an explicit physical-core \
                 budget before scripts/hostlock.sh run"
            )
        })
        .parse()
        .expect("ONNX_GENAI_CPU_DECODE_THREADS must be a positive integer");
    assert!(requested > 0, "CPU thread budget must be positive");
    let allowed = AllowedCpus::current()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU affinity is unreadable"));
    assert_eq!(
        allowed.len(),
        requested,
        "benchmark refused: requested {requested} physical cores but realized affinity {}",
        allowed.label()
    );
    let mut physical = BTreeSet::new();
    let mut mapping = Vec::new();
    for &cpu in &allowed.cpus {
        let (package, core) = physical_core(cpu).unwrap_or_else(|| {
            panic!("benchmark refused: physical package/core identity is unreadable for CPU {cpu}")
        });
        assert!(
            physical.insert((package, core)),
            "benchmark refused: affinity {} includes more than one logical CPU from physical \
             package {package} core {core}",
            allowed.label()
        );
        mapping.push(format!("{cpu}:{package}:{core}"));
    }
    println!(
        "CPU_TOPOLOGY,online={},allowed={},physical_cores={},mapping=logical:package:core:{}",
        onnx_runtime_hostmon::online_cpus()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".into()),
        allowed.label(),
        physical.len(),
        mapping.join("|"),
    );
    println!(
        "CPU_MODEL,{}",
        read_cpu_model().unwrap_or_else(|| {
            panic!("benchmark refused: /proc/cpuinfo did not report a CPU model")
        })
    );
    allowed
}

fn read_frequency(allowed: &AllowedCpus) -> Option<FrequencySample> {
    let mut values = Vec::with_capacity(allowed.len());
    for cpu in &allowed.cpus {
        let value = fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        ))
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
        values.push(value);
    }
    values.sort_unstable();
    Some(FrequencySample {
        min_khz: *values.first()?,
        median_khz: values[values.len() / 2],
        max_khz: *values.last()?,
    })
}

fn require_frequency(allowed: &AllowedCpus) -> FrequencySample {
    read_frequency(allowed).unwrap_or_else(|| {
        panic!(
            "benchmark refused: scaling_cur_freq is unreadable for at least one allowed CPU ({})",
            allowed.label()
        )
    })
}

fn finish_contention(
    phase: &str,
    name: &str,
    contention: &Contention,
    lock: &onnx_runtime_hostmon::window::Report,
) {
    assert!(
        lock.is_protected(),
        "benchmark refused: {phase}/{name} was not protected end-to-end: {lock}"
    );
    assert!(
        contention.is_clean(),
        "benchmark refused: {phase}/{name} host window was not clean: measured={} \
         own_time_complete={} foreign_pct={:.1} siblings_known={} sibling_peak_pct={:.1}",
        contention.measured,
        contention.own_time_complete,
        contention.foreign_pct,
        contention.siblings_known,
        contention.sibling_peak_pct,
    );
}

fn frequency_drift_pct(before: FrequencySample, after: FrequencySample) -> f64 {
    (after.median_khz as f64 - before.median_khz as f64).abs() / (before.median_khz.max(1) as f64)
        * 100.0
}

fn measure_window(
    phase: &str,
    name: &str,
    rep: usize,
    order: usize,
    iterations: u64,
    allowed: &AllowedCpus,
    mut run: impl FnMut(),
) -> f64 {
    let lock = onnx_runtime_hostmon::window::Window::open();
    let contention_before = onnx_runtime_hostmon::snapshot();
    let frequency_before = require_frequency(allowed);
    let cpu_before = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"));
    let wall_start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let wall_s = wall_start.elapsed().as_secs_f64();
    let cpu = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"))
        .since(cpu_before);
    let frequency_after = require_frequency(allowed);
    let contention_after = onnx_runtime_hostmon::snapshot();
    let contention =
        onnx_runtime_hostmon::contention(contention_before.as_ref(), contention_after.as_ref());
    let lock = lock.close();
    let drift = frequency_drift_pct(frequency_before, frequency_after);
    let ns_per_iter = wall_s * 1e9 / iterations as f64;
    println!(
        "EINSUM_RAW phase={phase} name={name} rep={rep} order={order} iterations={iterations} \
         wall_s={wall_s:.9} ns_per_iter={ns_per_iter:.3} cpu_user_s={:.6} cpu_sys_s={:.6} \
         process_efficiency={:.4} foreign_pct={:.1} sibling_peak_pct={:.1} \
         frequency_before_khz={}:{}:{} frequency_after_khz={}:{}:{} \
         frequency_drift_pct={drift:.3} {lock}",
        cpu.user_s,
        cpu.sys_s,
        cpu.total_s() / wall_s.max(f64::MIN_POSITIVE),
        contention.foreign_pct,
        contention.sibling_peak_pct,
        frequency_before.min_khz,
        frequency_before.median_khz,
        frequency_before.max_khz,
        frequency_after.min_khz,
        frequency_after.median_khz,
        frequency_after.max_khz,
    );
    finish_contention(phase, name, &contention, &lock);
    assert!(
        drift <= MAX_FREQUENCY_DRIFT_PCT,
        "benchmark refused: {phase}/{name} median CPU frequency drifted {drift:.3}% \
         (limit {MAX_FREQUENCY_DRIFT_PCT:.1}%)"
    );
    ns_per_iter
}

fn monitor_criterion_arm(name: &str, allowed: &AllowedCpus, run: impl FnOnce()) {
    let lock = onnx_runtime_hostmon::window::Window::open();
    let contention_before = onnx_runtime_hostmon::snapshot();
    let frequency_before = require_frequency(allowed);
    let cpu_before = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"));
    let wall_start = Instant::now();
    run();
    let wall_s = wall_start.elapsed().as_secs_f64();
    let cpu = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"))
        .since(cpu_before);
    let frequency_after = require_frequency(allowed);
    let contention_after = onnx_runtime_hostmon::snapshot();
    let contention =
        onnx_runtime_hostmon::contention(contention_before.as_ref(), contention_after.as_ref());
    let lock = lock.close();
    let drift = frequency_drift_pct(frequency_before, frequency_after);
    println!(
        "EINSUM_CRITERION_ARM name={name} wall_s={wall_s:.9} cpu_user_s={:.6} \
         cpu_sys_s={:.6} process_efficiency={:.4} foreign_pct={:.1} \
         sibling_peak_pct={:.1} frequency_before_khz={}:{}:{} \
         frequency_after_khz={}:{}:{} frequency_drift_pct={drift:.3} {lock}",
        cpu.user_s,
        cpu.sys_s,
        cpu.total_s() / wall_s.max(f64::MIN_POSITIVE),
        contention.foreign_pct,
        contention.sibling_peak_pct,
        frequency_before.min_khz,
        frequency_before.median_khz,
        frequency_before.max_khz,
        frequency_after.min_khz,
        frequency_after.median_khz,
        frequency_after.max_khz,
    );
    finish_contention("criterion", name, &contention, &lock);
    assert!(
        drift <= MAX_FREQUENCY_DRIFT_PCT,
        "benchmark refused: criterion/{name} median CPU frequency drifted {drift:.3}% \
         (limit {MAX_FREQUENCY_DRIFT_PCT:.1}%)"
    );
}

fn run_criterion_arm(name: &str, allowed: &AllowedCpus, monitor: bool, run: impl FnOnce()) {
    if monitor {
        monitor_criterion_arm(name, allowed, run);
    } else {
        run();
    }
}

fn calibrate(mut run: impl FnMut()) -> u64 {
    let mut iterations = 1u64;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(20) || iterations >= 5_000_000 {
            let per_iteration = elapsed.as_secs_f64() / iterations as f64;
            return (TARGET_WINDOW.as_secs_f64() / per_iteration.max(1e-9))
                .ceil()
                .clamp(1.0, 5_000_000.0) as u64;
        }
        iterations = iterations.saturating_mul(10).min(5_000_000);
    }
}

fn summary(values: &[f64]) -> Summary {
    assert!(values.len() >= 3, "every reported result requires n >= 3");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median_ns = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    };
    let min_ns = sorted[0];
    let max_ns = sorted[sorted.len() - 1];
    Summary {
        median_ns,
        min_ns,
        max_ns,
        range_pct: (max_ns / min_ns.max(f64::MIN_POSITIVE) - 1.0) * 100.0,
    }
}

fn report_summary(phase: &str, name: &str, values: &[f64], max_range_pct: f64) -> Summary {
    let result = summary(values);
    println!(
        "EINSUM_SUMMARY phase={phase} name={name} n={} median_ns={:.3} min_ns={:.3} \
         max_ns={:.3} range_pct={:.3}",
        values.len(),
        result.median_ns,
        result.min_ns,
        result.max_ns,
        result.range_pct,
    );
    assert!(
        result.range_pct <= max_range_pct,
        "benchmark refused: {phase}/{name} range {:.3}% exceeds stability limit \
         {max_range_pct:.3}%",
        result.range_pct
    );
    result
}

fn absolute_evidence(
    case: &Case,
    kernel: &dyn Kernel,
    views: &[onnx_runtime_ep_api::TensorView<'_>],
    output: &mut Tensor,
    allowed: &AllowedCpus,
) {
    for _ in 0..8 {
        kernel
            .execute(views, &mut [output.view_mut()])
            .expect("absolute evidence warmup must execute");
    }
    let iterations = calibrate(|| {
        kernel
            .execute(views, &mut [output.view_mut()])
            .expect("absolute evidence calibration must execute")
    });
    let mut values = Vec::with_capacity(ABSOLUTE_REPS);
    for rep in 0..ABSOLUTE_REPS {
        values.push(measure_window(
            "absolute",
            case.name,
            rep,
            rep,
            iterations,
            allowed,
            || {
                kernel
                    .execute(black_box(views), black_box(&mut [output.view_mut()]))
                    .expect("absolute evidence dispatch must execute")
            },
        ));
    }
    report_summary("absolute", case.name, &values, MAX_RANGE_PCT);
}

fn interleaved_comparison(allowed: &AllowedCpus) {
    let case = cases()
        .into_iter()
        .find(|case| case.name == "gemm_friendly")
        .expect("friendly GEMM case exists");
    let tensors = inputs(&case);
    let views = tensors.iter().map(Tensor::view).collect::<Vec<_>>();
    let (einsum, einsum_setup) = kernel_with_mode(&case, "optimized");
    let matmul_start = Instant::now();
    let matmul = make_kernel("MatMul", [], &case.input_shapes, 24);
    let matmul_setup = matmul_start.elapsed();
    let mut einsum_output = Tensor::zeros(case.dtype, &case.output_shape);
    let mut matmul_output = Tensor::zeros(case.dtype, &case.output_shape);
    let route = benchmark_execute_route(&*einsum, &views, &mut [einsum_output.view_mut()])
        .expect("comparison Einsum route probe must execute")
        .expect("comparison route probe must receive Einsum");
    matmul
        .execute(&views, &mut [matmul_output.view_mut()])
        .expect("comparison MatMul must execute");
    assert_eq!(
        einsum_output.f32s(),
        matmul_output.f32s(),
        "the equivalent Einsum and MatMul controls must be bit-identical on shared inputs"
    );
    println!(
        "EINSUM_EQUIVALENCE case={} equation={} dtype={} route={} shared_input_hash={:016x} \
         output_hash={:016x} exact_equal=true einsum_setup_us={:.3} matmul_setup_us={:.3}",
        case.name,
        case.equation,
        case.dtype.name(),
        route,
        hash_tensors(&tensors),
        hash_values(einsum_output.f32s()),
        einsum_setup.as_secs_f64() * 1e6,
        matmul_setup.as_secs_f64() * 1e6,
    );

    reset_allocations();
    benchmark_execute_route(&*einsum, &views, &mut [einsum_output.view_mut()])
        .expect("comparison Einsum allocation probe must execute");
    let einsum_allocations = allocations();
    reset_allocations();
    matmul
        .execute(&views, &mut [matmul_output.view_mut()])
        .expect("comparison MatMul allocation probe must execute");
    let matmul_allocations = allocations();
    println!(
        "EINSUM_COMPARISON_ALLOC einsum_calls={} einsum_bytes={} matmul_calls={} matmul_bytes={}",
        einsum_allocations.0, einsum_allocations.1, matmul_allocations.0, matmul_allocations.1,
    );

    for _ in 0..8 {
        einsum
            .execute(&views, &mut [einsum_output.view_mut()])
            .expect("comparison Einsum warmup must execute");
        matmul
            .execute(&views, &mut [matmul_output.view_mut()])
            .expect("comparison MatMul warmup must execute");
    }
    let einsum_iterations = calibrate(|| {
        einsum
            .execute(&views, &mut [einsum_output.view_mut()])
            .expect("comparison Einsum calibration must execute")
    });
    let matmul_iterations = calibrate(|| {
        matmul
            .execute(&views, &mut [matmul_output.view_mut()])
            .expect("comparison MatMul calibration must execute")
    });
    let iterations = einsum_iterations.min(matmul_iterations);
    let patterns = [
        ["einsum", "matmul", "matmul", "einsum"],
        ["matmul", "einsum", "einsum", "matmul"],
        ["einsum", "matmul", "matmul", "einsum"],
    ];
    let mut einsum_values = Vec::new();
    let mut matmul_values = Vec::new();
    let mut order = 0usize;
    for (block, pattern) in patterns.into_iter().enumerate().take(INTERLEAVED_BLOCKS) {
        for arm in pattern {
            let value = if arm == "einsum" {
                measure_window("comparison", arm, block, order, iterations, allowed, || {
                    einsum
                        .execute(
                            black_box(&views),
                            black_box(&mut [einsum_output.view_mut()]),
                        )
                        .expect("comparison Einsum dispatch must execute")
                })
            } else {
                measure_window("comparison", arm, block, order, iterations, allowed, || {
                    matmul
                        .execute(
                            black_box(&views),
                            black_box(&mut [matmul_output.view_mut()]),
                        )
                        .expect("comparison MatMul dispatch must execute")
                })
            };
            if arm == "einsum" {
                einsum_values.push(value);
            } else {
                matmul_values.push(value);
            }
            order += 1;
        }
    }
    let einsum_summary = report_summary(
        "comparison",
        "einsum",
        &einsum_values,
        MAX_CONTROL_RANGE_PCT,
    );
    let matmul_summary = report_summary(
        "comparison",
        "matmul",
        &matmul_values,
        MAX_CONTROL_RANGE_PCT,
    );
    println!(
        "EINSUM_COMPARISON_SUMMARY case={} n_per_arm={} einsum_median_ns={:.3} \
         matmul_median_ns={:.3} einsum_over_matmul={:.6}",
        case.name,
        einsum_values.len(),
        einsum_summary.median_ns,
        matmul_summary.median_ns,
        einsum_summary.median_ns / matmul_summary.median_ns,
    );
}

fn null_control(allowed: &AllowedCpus) {
    let case = cases()
        .into_iter()
        .find(|case| case.name == "gemm_friendly")
        .expect("friendly GEMM case exists");
    let tensors = inputs(&case);
    let views = tensors.iter().map(Tensor::view).collect::<Vec<_>>();
    let left = make_kernel("MatMul", [], &case.input_shapes, 24);
    let right = make_kernel("MatMul", [], &case.input_shapes, 24);
    let mut left_output = Tensor::zeros(case.dtype, &case.output_shape);
    let mut right_output = Tensor::zeros(case.dtype, &case.output_shape);
    for _ in 0..8 {
        left.execute(&views, &mut [left_output.view_mut()])
            .expect("left null warmup must execute");
        right
            .execute(&views, &mut [right_output.view_mut()])
            .expect("right null warmup must execute");
    }
    assert_eq!(left_output.f32s(), right_output.f32s());
    let left_iterations = calibrate(|| {
        left.execute(&views, &mut [left_output.view_mut()])
            .expect("left null calibration must execute")
    });
    let right_iterations = calibrate(|| {
        right
            .execute(&views, &mut [right_output.view_mut()])
            .expect("right null calibration must execute")
    });
    let iterations = left_iterations.min(right_iterations);
    let patterns = [
        ["matmul_a", "matmul_b", "matmul_b", "matmul_a"],
        ["matmul_b", "matmul_a", "matmul_a", "matmul_b"],
        ["matmul_a", "matmul_b", "matmul_b", "matmul_a"],
    ];
    let mut left_values = Vec::new();
    let mut right_values = Vec::new();
    let mut order = 0usize;
    for (block, pattern) in patterns.into_iter().enumerate().take(INTERLEAVED_BLOCKS) {
        for arm in pattern {
            let value = if arm == "matmul_a" {
                measure_window("null", arm, block, order, iterations, allowed, || {
                    left.execute(black_box(&views), black_box(&mut [left_output.view_mut()]))
                        .expect("left null dispatch must execute")
                })
            } else {
                measure_window("null", arm, block, order, iterations, allowed, || {
                    right
                        .execute(black_box(&views), black_box(&mut [right_output.view_mut()]))
                        .expect("right null dispatch must execute")
                })
            };
            if arm == "matmul_a" {
                left_values.push(value);
            } else {
                right_values.push(value);
            }
            order += 1;
        }
    }
    let left_summary = report_summary("null", "matmul_a", &left_values, MAX_CONTROL_RANGE_PCT);
    let right_summary = report_summary("null", "matmul_b", &right_values, MAX_CONTROL_RANGE_PCT);
    let median_delta_pct = (left_summary.median_ns / right_summary.median_ns - 1.0).abs() * 100.0;
    println!(
        "EINSUM_NULL_SUMMARY n_per_arm={} matmul_a_median_ns={:.3} \
         matmul_b_median_ns={:.3} median_delta_pct={median_delta_pct:.3}",
        left_values.len(),
        left_summary.median_ns,
        right_summary.median_ns,
    );
    assert!(
        median_delta_pct <= MAX_NULL_MEDIAN_DELTA_PCT,
        "benchmark refused: A/A MatMul median delta {median_delta_pct:.3}% exceeds \
         {MAX_NULL_MEDIAN_DELTA_PCT:.3}%"
    );
}

fn bench_einsum(c: &mut Criterion, allowed: &AllowedCpus, emit_evidence: bool) {
    let mut group = c.benchmark_group("einsum");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for case in cases() {
        let tensors = inputs(&case);
        let views = tensors.iter().map(Tensor::view).collect::<Vec<_>>();
        let (optimized, optimized_setup) = kernel_with_mode(&case, "optimized");
        let (oracle, oracle_setup) = kernel_with_mode(&case, "oracle");
        validate_case(&case, &*optimized, &*oracle, &tensors);
        let mut allocation_probe_output = Tensor::zeros(case.dtype, &case.output_shape);
        reset_allocations();
        let route = benchmark_execute_route(
            &*optimized,
            &views,
            &mut [allocation_probe_output.view_mut()],
        )
        .expect("allocation route probe must execute")
        .expect("allocation route probe must receive Einsum");
        let (allocation_calls, allocated_bytes) = allocations();
        let workspace = benchmark_scratch_capacity_bytes(&*optimized).unwrap_or(0);
        println!(
            "EINSUM_SETUP case={} native_us={:.3} oracle_us={:.3} route={} \
             reusable_workspace_bytes={workspace} steady_allocations={allocation_calls} \
             steady_allocated_bytes={allocated_bytes}",
            case.name,
            optimized_setup.as_secs_f64() * 1e6,
            oracle_setup.as_secs_f64() * 1e6,
            route,
        );
        if emit_evidence {
            absolute_evidence(
                &case,
                &*optimized,
                &views,
                &mut allocation_probe_output,
                allowed,
            );
        }
        group.throughput(Throughput::Elements(
            case.output_shape.iter().product::<usize>() as u64,
        ));
        let mut optimized_output = Tensor::zeros(case.dtype, &case.output_shape);
        run_criterion_arm(case.name, allowed, emit_evidence, || {
            group.bench_with_input(BenchmarkId::new(case.name, "native"), &(), |bencher, _| {
                bencher.iter(|| {
                    optimized
                        .execute(
                            black_box(&views),
                            black_box(&mut [optimized_output.view_mut()]),
                        )
                        .unwrap()
                });
            });
        });
    }
    group.finish();

    let view_case = Case {
        name: "view_permutation",
        equation: "abc->bca",
        input_shapes: vec![vec![32, 64, 128]],
        output_shape: vec![64, 128, 32],
        dtype: FloatDType::F32,
        tolerance: 0.0,
        expected_route: "view-zero-copy",
    };
    let view_tensors = inputs(&view_case);
    let view_input = view_tensors[0].view();
    let (view_kernel, view_setup) = kernel_with_mode(&view_case, "optimized");
    let specs = view_kernel
        .view_outputs(
            std::slice::from_ref(&view_input),
            std::slice::from_ref(&view_case.output_shape),
            1,
        )
        .expect("view-only Einsum must return a zero-copy output");
    assert_eq!(specs[0].strides, [128, 1, 8192]);
    reset_allocations();
    let allocation_probe = view_kernel
        .view_outputs(
            std::slice::from_ref(&view_input),
            std::slice::from_ref(&view_case.output_shape),
            1,
        )
        .expect("view route must stay reachable");
    black_box(&allocation_probe);
    let (view_allocations, view_allocated_bytes) = allocations();
    println!(
        "EINSUM_VALIDATE case=view_permutation dtype=f32 equation=abc->bca \
         route=view-zero-copy shared_input_hash={:016x} strides=128:1:8192",
        hash_tensors(&view_tensors),
    );
    println!(
        "EINSUM_SETUP case=view_permutation native_us={:.3} oracle_us=n/a \
         route=view-zero-copy reusable_workspace_bytes=0 \
         steady_allocations={view_allocations} steady_allocated_bytes={view_allocated_bytes}",
        view_setup.as_secs_f64() * 1e6,
    );
    if emit_evidence {
        for _ in 0..8 {
            black_box(
                view_kernel
                    .view_outputs(
                        std::slice::from_ref(&view_input),
                        std::slice::from_ref(&view_case.output_shape),
                        1,
                    )
                    .expect("view evidence warmup must execute"),
            );
        }
        let iterations = calibrate(|| {
            black_box(
                view_kernel
                    .view_outputs(
                        std::slice::from_ref(&view_input),
                        std::slice::from_ref(&view_case.output_shape),
                        1,
                    )
                    .expect("view evidence calibration must execute"),
            );
        });
        let mut values = Vec::new();
        for rep in 0..ABSOLUTE_REPS {
            values.push(measure_window(
                "absolute",
                "view_permutation",
                rep,
                rep,
                iterations,
                allowed,
                || {
                    black_box(
                        view_kernel
                            .view_outputs(
                                black_box(std::slice::from_ref(&view_input)),
                                black_box(std::slice::from_ref(&view_case.output_shape)),
                                1,
                            )
                            .expect("view evidence dispatch must execute"),
                    );
                },
            ));
        }
        report_summary("absolute", "view_permutation", &values, MAX_RANGE_PCT);
    }
    let mut views = c.benchmark_group("einsum_view");
    views.sample_size(10);
    views.warm_up_time(Duration::from_secs(1));
    views.measurement_time(Duration::from_secs(2));
    run_criterion_arm("view_permutation", allowed, emit_evidence, || {
        views.bench_function("permutation_metadata_32x64x128", |bencher| {
            bencher.iter(|| {
                black_box(
                    view_kernel
                        .view_outputs(
                            black_box(std::slice::from_ref(&view_input)),
                            black_box(std::slice::from_ref(&view_case.output_shape)),
                            1,
                        )
                        .expect("view route must stay reachable"),
                )
            });
        });
    });
    views.finish();

    let case = cases()
        .into_iter()
        .find(|case| case.name == "gemm_friendly")
        .expect("friendly GEMM case exists");
    let control_tensors = inputs(&case);
    let control_views = control_tensors.iter().map(Tensor::view).collect::<Vec<_>>();
    let control_start = Instant::now();
    let matmul = make_kernel("MatMul", [], &case.input_shapes, 24);
    let control_setup = control_start.elapsed();
    let mut output = Tensor::zeros(case.dtype, &case.output_shape);
    reset_allocations();
    matmul
        .execute(&control_views, &mut [output.view_mut()])
        .expect("MatMul allocation control must execute");
    let (control_allocations, control_allocated_bytes) = allocations();
    println!(
        "EINSUM_CONTROL_SETUP matmul_us={:.3} shared_input_hash={:016x} \
         steady_allocations={control_allocations} steady_allocated_bytes={control_allocated_bytes}",
        control_setup.as_secs_f64() * 1e6,
        hash_tensors(&control_tensors),
    );
    let mut control = c.benchmark_group("einsum_control");
    control.sample_size(10);
    control.warm_up_time(Duration::from_secs(1));
    control.measurement_time(Duration::from_secs(2));
    run_criterion_arm("matmul_32x256x256", allowed, emit_evidence, || {
        control.bench_function("matmul_32x256x256", |bencher| {
            bencher.iter(|| {
                matmul
                    .execute(
                        black_box(&control_views),
                        black_box(&mut [output.view_mut()]),
                    )
                    .unwrap()
            });
        });
    });
    control.finish();
}

fn main() {
    let is_list = std::env::args().any(|argument| argument == "--list");
    let target = require_governance(is_list);
    common::init_decode_topology();
    let allowed = require_physical_core_affinity();
    let frequency = require_frequency(&allowed);
    println!(
        "CPU_FREQUENCY,source=scaling_cur_freq,min_khz={},median_khz={},max_khz={}",
        frequency.min_khz, frequency.median_khz, frequency.max_khz
    );
    println!(
        "EINSUM_CENSUS,expected_criterion_selectors={EXPECTED_CRITERION_SELECTORS} \
         absolute_reps={ABSOLUTE_REPS} interleaved_blocks={INTERLEAVED_BLOCKS}"
    );

    let full_lock = onnx_runtime_hostmon::window::Window::open();
    let contention_before = onnx_runtime_hostmon::snapshot();
    let cpu_before = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"));
    let wall_start = Instant::now();

    if !is_list {
        null_control(&allowed);
        interleaved_comparison(&allowed);
    }
    let mut criterion = Criterion::default().configure_from_args();
    bench_einsum(&mut criterion, &allowed, !is_list);
    criterion.final_summary();

    let wall_s = wall_start.elapsed().as_secs_f64();
    let cpu = process_cpu_time()
        .unwrap_or_else(|| panic!("benchmark refused: process CPU time unreadable"))
        .since(cpu_before);
    let contention_after = onnx_runtime_hostmon::snapshot();
    let contention =
        onnx_runtime_hostmon::contention(contention_before.as_ref(), contention_after.as_ref());
    let full_lock = full_lock.close();
    println!(
        "EINSUM_SWEEP wall_s={wall_s:.6} cpu_user_s={:.6} cpu_sys_s={:.6} \
         process_efficiency={:.4} foreign_pct={:.1} sibling_peak_pct={:.1} {}",
        cpu.user_s,
        cpu.sys_s,
        cpu.total_s() / wall_s.max(f64::MIN_POSITIVE),
        contention.foreign_pct,
        contention.sibling_peak_pct,
        full_lock,
    );
    if is_list {
        assert!(
            full_lock.is_protected(),
            "benchmark census refused: selector listing was not protected end-to-end: {full_lock}"
        );
    } else {
        finish_contention("sweep", "all", &contention, &full_lock);
    }
    if is_list {
        println!(
            "EINSUM_CENSUS_COMPLETE commit={} selector_count={}",
            command_text("git", &["rev-parse", "HEAD"]),
            EXPECTED_CRITERION_SELECTORS,
        );
    } else {
        println!(
            "EINSUM_EVIDENCE_COMPLETE commit={} criterion_raw={} selector_count={}",
            command_text("git", &["rev-parse", "HEAD"]),
            target.join("criterion").display(),
            EXPECTED_CRITERION_SELECTORS,
        );
    }
}
