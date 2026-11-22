//! Real CPU MatMul A/B for derived-weight accounting. Warm-up is excluded.
//! Interleave governed/local arms, a duplicate local A/A arm and an unaffected
//! contiguous-f32 control. Retained and transient policies are separate rows.
//! This is a CPU/tier-scoped measurement, not proof of zero cost on every EP.
//! Run under the host lock with RAYON_NUM_THREADS=1; inspect full ranges/nulls.

use std::hint::black_box;
use std::time::Instant;

use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::{
    CpuCacheRetention, CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory,
};
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId};
use onnx_runtime_memory_api::Tier;
use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};

struct Case {
    kernel: Box<dyn Kernel>,
    a: Vec<f32>,
    b32: Vec<f32>,
    b64: Vec<f64>,
    output: Vec<f32>,
    shapes: [[usize; 2]; 3],
    strides: [[i64; 2]; 3],
    wide: bool,
    governor: LedgerGovernor,
}
impl Case {
    fn new(m: usize, k: usize, n: usize, wide: bool, governed: bool, retained: bool) -> Self {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, (k * n * 16) as u64, 0));
        let memory = if governed {
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap()
        } else {
            DerivedWeightMemory::default()
        };
        let resources = CpuMemoryResources::new(memory, Default::default())
            .unwrap()
            .with_cache_retention(CpuCacheRetention::new(retained));
        let provider = CpuExecutionProvider::with_resources(resources);
        let node = Node::new(NodeId(0), "MatMul", vec![], vec![]);
        let mut kernel = provider
            .get_kernel(&node, &[vec![m, k], vec![k, n]], 17)
            .unwrap();
        kernel.set_constant_inputs(&[false, true]);
        let mut case = Self {
            kernel,
            a: vec![0.25; m * k],
            b32: vec![0.5; k * n],
            b64: vec![0.5; k * n],
            output: vec![0.; m * n],
            shapes: [[m, k], [k, n], [m, n]],
            strides: [[k as i64, 1], [n as i64, 1], [n as i64, 1]],
            wide,
            governor,
        };
        for _ in 0..16 {
            case.execute();
        }
        assert!(case.output.iter().all(|&v| v == k as f32 * 0.125));
        #[cfg(feature = "mlas")]
        let panel_bytes = if retained
            && onnx_runtime_ep_cpu::backend::CpuBackend::auto_detect()
                == onnx_runtime_ep_cpu::backend::CpuBackend::Mlas
        {
            mlas_sys::PackedBPlan::new(n, k).unwrap().owned_heap_bytes() as u64
        } else {
            0
        };
        #[cfg(not(feature = "mlas"))]
        let panel_bytes = 0u64;
        assert_eq!(
            case.governor.used(Tier::Host),
            if governed && retained {
                (if wide { k * n * 4 } else { 0 }) as u64 + panel_bytes
            } else {
                0
            }
        );
        case
    }
    fn execute(&mut self) {
        let a = TensorView::new(
            DevicePtr(self.a.as_ptr().cast()),
            DataType::Float32,
            &self.shapes[0],
            &self.strides[0],
            DeviceId::cpu(),
        );
        let (pointer, dtype) = if self.wide {
            (self.b64.as_ptr().cast(), DataType::Float64)
        } else {
            (self.b32.as_ptr().cast(), DataType::Float32)
        };
        let b = TensorView::new(
            DevicePtr(pointer),
            dtype,
            &self.shapes[1],
            &self.strides[1],
            DeviceId::cpu(),
        );
        let out = TensorMut::new(
            DevicePtrMut(self.output.as_mut_ptr().cast()),
            DataType::Float32,
            &self.shapes[2],
            &self.strides[2],
            DeviceId::cpu(),
        );
        self.kernel.execute(black_box(&[a, b]), &mut [out]).unwrap();
        black_box(&self.output);
    }
    fn batch(&mut self, iterations: usize) -> (f64, f64) {
        let cpu = cpu_seconds();
        let start = Instant::now();
        for _ in 0..iterations {
            self.execute();
        }
        (
            start.elapsed().as_secs_f64() * 1e9 / iterations as f64,
            (cpu_seconds() - cpu) * 1e9 / iterations as f64,
        )
    }
}
#[cfg(unix)]
fn cpu_seconds() -> f64 {
    // SAFETY: getrusage writes into a valid rusage and errors are checked.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
    let seconds = |v: libc::timeval| v.tv_sec as f64 + v.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}
#[cfg(not(unix))]
fn cpu_seconds() -> f64 {
    f64::NAN
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}
fn main() {
    println!(
        "m\tk\tn\tretained\tarm\titerations\twall_ns_median\twall_ns_min\twall_ns_max\tcpu_ns_median"
    );
    for (m, k, n) in [(1, 16, 16), (1, 128, 256), (8, 128, 256)] {
        for retained in [true, false] {
            let mut cases = [
                Case::new(m, k, n, true, false, retained),
                Case::new(m, k, n, true, true, retained),
                Case::new(m, k, n, true, false, retained),
                Case::new(m, k, n, false, false, retained),
                Case::new(m, k, n, false, true, retained),
            ];
            let ns = cases[0].batch(256).0;
            let iterations = ((10_000_000. / ns) as usize).clamp(16, 100_000);
            let mut walls: [Vec<f64>; 5] = Default::default();
            let mut cpus: [Vec<f64>; 5] = Default::default();
            for round in 0..11 {
                for offset in 0..5 {
                    let arm = (round + offset) % 5;
                    let (wall, cpu) = cases[arm].batch(iterations);
                    walls[arm].push(wall);
                    cpus[arm].push(cpu);
                }
            }
            // Keep every sample outside timed regions for noise/null auditing.
            for arm in 0..5 {
                for round in 0..11 {
                    println!(
                        "sample\t{m}\t{k}\t{n}\t{retained}\t{arm}\t{round}\t{:.2}\t{:.2}",
                        walls[arm][round], cpus[arm][round]
                    );
                }
            }
            for (arm, name) in [
                "wide_local",
                "wide_governed",
                "wide_local_null",
                "f32_local_control",
                "f32_governed_control",
            ]
            .iter()
            .enumerate()
            {
                assert!(cases[arm].output.iter().all(|&v| v == k as f32 * 0.125));
                println!(
                    "{m}\t{k}\t{n}\t{retained}\t{name}\t{iterations}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                    median(&walls[arm]),
                    walls[arm].iter().copied().fold(f64::INFINITY, f64::min),
                    walls[arm].iter().copied().fold(0., f64::max),
                    median(&cpus[arm])
                );
            }
            println!(
                "ratio\t{m}\t{k}\t{n}\t{retained}\tgoverned/local={:.4}\tnull/local={:.4}\tf32_control={:.4}",
                median(&walls[1]) / median(&walls[0]),
                median(&walls[2]) / median(&walls[0]),
                median(&walls[4]) / median(&walls[3])
            );
        }
    }
}
