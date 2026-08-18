//! RSS measurement harness for the QLinearMatMul per-thread accumulator scratch
//! (#1133).
//!
//! Parks the `i32` accumulator on every worker thread of a rayon pool, then
//! holds so an external poller can read `PeakWorkingSet64`/`WorkingSet64` by
//! PID. The point is to make the `x threads` multiplier observable: at one
//! thread the retention is one buffer, at N threads it is
//! `min(process_cap, N x buffer)`.
//!
//! Env:
//! * `RAYON_NUM_THREADS`    - pool width (default 1).
//! * `QLINEAR_PROC_CAP_MIB` - process cap in MiB; a huge value reproduces the
//!   pre-fix per-thread-only (unbounded) behaviour (default 128).
//! * `QLINEAR_ITERS`        - decode-like iterations (default 24).

use onnx_runtime_ep_api::{DeviceId, DevicePtr, DevicePtrMut, Kernel, TensorMut, TensorView};
use onnx_runtime_ep_cpu::kernels::qlinear_matmul::QLinearMatMulKernel;
use onnx_runtime_ir::DataType;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn view<'a>(
    bytes: &'a [u8],
    dtype: DataType,
    shape: &'static [usize],
    strides: &'static [i64],
) -> TensorView<'a> {
    TensorView::new(
        DevicePtr(bytes.as_ptr().cast()),
        dtype,
        shape,
        strides,
        DeviceId::cpu(),
    )
}

fn main() {
    let threads = env_usize("RAYON_NUM_THREADS", 1);
    let cap_mib = env_usize("QLINEAR_PROC_CAP_MIB", 128);
    let iters = env_usize("QLINEAR_ITERS", 24);

    onnx_runtime_ep_cpu::set_qlinear_accumulator_budget_admitted(true);
    onnx_runtime_ep_cpu::set_qlinear_accumulator_process_cap_bytes((cap_mib as u64) << 20);

    // A per-thread accumulator of 12 MiB: under the 32 MiB per-thread cap (so a
    // single buffer is admitted), while 16 of them (192 MiB) exceed the 128 MiB
    // process cap and only part fit -- the divergence the measurement shows.
    let (m, k, n) = (768usize, 64usize, 4096usize);
    let acc_mib = (m * n * 4) as f64 / (1u64 << 20) as f64;

    let a = vec![130u8; m * k];
    let a_scale = 0.5f32.to_le_bytes();
    let a_zero = [128u8];
    let b = vec![120u8; k * n];
    let b_scale = 0.25f32.to_le_bytes();
    let b_zero = [127u8];
    let y_scale = 64.0f32.to_le_bytes();
    let y_zero = [100u8];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");
    let kernel = QLinearMatMulKernel::default();

    for _ in 0..iters {
        pool.broadcast(|_| {
            let mut out = vec![0u8; m * n];
            let inputs = [
                view(&a, DataType::Uint8, &[768, 64], &[64, 1]),
                view(&a_scale, DataType::Float32, &[], &[]),
                view(&a_zero, DataType::Uint8, &[], &[]),
                view(&b, DataType::Uint8, &[64, 4096], &[4096, 1]),
                view(&b_scale, DataType::Float32, &[], &[]),
                view(&b_zero, DataType::Uint8, &[], &[]),
                view(&y_scale, DataType::Float32, &[], &[]),
                view(&y_zero, DataType::Uint8, &[], &[]),
            ];
            let out_view = TensorMut::new(
                DevicePtrMut(out.as_mut_ptr().cast()),
                DataType::Uint8,
                &[768, 4096],
                &[4096, 1],
                DeviceId::cpu(),
            );
            kernel.execute(&inputs, &mut [out_view]).expect("execute");
            // `out` drops here, so only the parked accumulators persist between
            // iterations -- the retention the harness measures.
        });
    }

    let live = onnx_runtime_ep_cpu::qlinear_accumulator_live_bytes();
    let cap = onnx_runtime_ep_cpu::qlinear_accumulator_process_cap_bytes();
    println!(
        "threads={threads} proc_cap_mib={cap_mib} per_thread_acc_mib={acc_mib:.1} \
         retained_mib={:.1} cap_in_force_mib={:.1}",
        live as f64 / (1u64 << 20) as f64,
        cap as f64 / (1u64 << 20) as f64,
    );
    // Hold with only the parked accumulators resident so the poller can read a
    // steady working set free of the per-iteration transients.
    std::thread::sleep(std::time::Duration::from_secs(3));
}
