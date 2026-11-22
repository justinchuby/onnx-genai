//! Real CPU MatMul A/B for derived-weight accounting. Warm-up is excluded.
//! Interleave governed/local arms, a duplicate local A/A arm, a constant-f32
//! comparison and an unaffected runtime-only f32 control. Retained and transient policies are separate rows.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WeightFormat {
    F32,
    F64,
    F16,
}

struct Case {
    kernel: Box<dyn Kernel>,
    a: Vec<f32>,
    b32: Vec<f32>,
    b64: Vec<f64>,
    b16: Vec<u16>,
    output: Vec<f32>,
    shapes: [[usize; 2]; 3],
    strides: [[i64; 2]; 3],
    format: WeightFormat,
    governor: LedgerGovernor,
}
impl Case {
    fn new(
        m: usize,
        k: usize,
        n: usize,
        format: WeightFormat,
        governed: bool,
        retained: bool,
        constant: bool,
    ) -> Self {
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
        kernel.set_constant_inputs(&[false, constant]);
        let mut case = Self {
            kernel,
            a: vec![0.25; m * k],
            b32: vec![0.5; k * n],
            b64: vec![0.5; k * n],
            b16: vec![half::f16::from_f32(0.5).to_bits(); k * n],
            output: vec![0.; m * n],
            shapes: [[m, k], [k, n], [m, n]],
            strides: [[k as i64, 1], [n as i64, 1], [n as i64, 1]],
            format,
            governor,
        };
        for _ in 0..16 {
            case.execute();
        }
        assert!(case.output.iter().all(|&v| v == k as f32 * 0.125));
        #[cfg(feature = "mlas")]
        let panel_bytes = if retained
            && constant
            && onnx_runtime_ep_cpu::backend::CpuBackend::auto_detect()
                == onnx_runtime_ep_cpu::backend::CpuBackend::Mlas
        {
            mlas_sys::PackedBPlan::new(n, k).unwrap().owned_heap_bytes() as u64
        } else {
            0
        };
        #[cfg(not(feature = "mlas"))]
        let panel_bytes = 0u64;
        // The F16 switch is also copied unchanged to an older baseline whose
        // transpose is uncharged. Record its actual bytes outside the timer;
        // public provider tests separately require the new half grant.
        println!(
            "footprint\t{m}\t{k}\t{n}\t{format:?}\t{governed}\t{retained}\t{constant}\t{}",
            case.governor.used(Tier::Host)
        );
        // This assertion-only input selects the old or new expected footprint
        // in before/current runs; it never changes runtime policy or compute.
        let expect_transpose = match std::env::var("NXRT_ADMISSION_EXPECT_F32_TRANSPOSE_GRANT")
            .as_deref()
        {
            Ok("0") => false,
            Ok("1") | Err(_) => true,
            Ok(value) => panic!(
                "NXRT_ADMISSION_EXPECT_F32_TRANSPOSE_GRANT={value:?}; choose 0 or 1 for the expected footprint"
            ),
        };
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let transpose_bytes = if constant
            && retained
            && expect_transpose
            && m == 1
            && onnx_runtime_ep_cpu::backend::CpuBackend::auto_detect()
                == onnx_runtime_ep_cpu::backend::CpuBackend::Accelerate
        {
            (k * n * 4) as u64
        } else {
            0
        };
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let transpose_bytes = {
            let _ = expect_transpose;
            0u64
        };
        if format != WeightFormat::F16 {
            assert_eq!(
                case.governor.used(Tier::Host),
                if governed && retained && constant {
                    (if format == WeightFormat::F64 {
                        k * n * 4
                    } else {
                        0
                    }) as u64
                        + panel_bytes
                        + transpose_bytes
                } else {
                    0
                }
            );
        }
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
        let (pointer, dtype) = match self.format {
            WeightFormat::F64 => (self.b64.as_ptr().cast(), DataType::Float64),
            WeightFormat::F16 => (self.b16.as_ptr().cast(), DataType::Float16),
            WeightFormat::F32 => (self.b32.as_ptr().cast(), DataType::Float32),
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
    let format = match std::env::var("NXRT_ADMISSION_WEIGHT_FORMAT").as_deref() {
        Ok("f16") => WeightFormat::F16,
        Ok("f64") | Err(_) => WeightFormat::F64,
        Ok(value) => {
            panic!("NXRT_ADMISSION_WEIGHT_FORMAT={value:?} is unsupported; choose f16 or f64")
        }
    };
    println!(
        "context\tweight_format={format:?}\tbackend={:?}",
        onnx_runtime_ep_cpu::backend::CpuBackend::auto_detect()
    );
    println!(
        "m\tk\tn\tretained\tarm\titerations\twall_ns_median\twall_ns_min\twall_ns_max\tcpu_ns_median"
    );
    for (m, k, n) in [(1, 16, 16), (1, 128, 256), (8, 128, 256)] {
        for retained in [true, false] {
            let mut cases = [
                Case::new(m, k, n, format, false, retained, true),
                Case::new(m, k, n, format, true, retained, true),
                Case::new(m, k, n, format, false, retained, true),
                Case::new(m, k, n, WeightFormat::F32, false, retained, true),
                Case::new(m, k, n, WeightFormat::F32, true, retained, true),
                Case::new(m, k, n, WeightFormat::F32, true, retained, false),
            ];
            let ns = cases[0].batch(256).0;
            let iterations = ((10_000_000. / ns) as usize).clamp(16, 100_000);
            let mut walls: [Vec<f64>; 6] = Default::default();
            let mut cpus: [Vec<f64>; 6] = Default::default();
            for round in 0..11 {
                for offset in 0..6 {
                    let arm = (round + offset) % 6;
                    let (wall, cpu) = cases[arm].batch(iterations);
                    walls[arm].push(wall);
                    cpus[arm].push(cpu);
                }
            }
            // Keep every sample outside timed regions for noise/null auditing.
            for arm in 0..6 {
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
                "runtime_f32_control",
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
