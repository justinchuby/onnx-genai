#![allow(dead_code)]

pub mod decode_workload;

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

#[derive(Clone, Copy, Debug)]
pub enum FloatDType {
    F32,
    F16,
    Bf16,
}

impl FloatDType {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// Bytes per element, so a bandwidth-bound bench can report the traffic
    /// its route actually issues rather than a flop count.
    pub fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::F32 => DataType::Float32,
            Self::F16 => DataType::Float16,
            Self::Bf16 => DataType::BFloat16,
        }
    }
}

enum Storage {
    F32(Vec<f32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<i64>,
    dtype: DataType,
}

impl Tensor {
    pub fn floats(dtype: FloatDType, shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let storage = match dtype {
            FloatDType::F32 => Storage::F32(values.to_vec()),
            FloatDType::F16 => {
                Storage::U16(values.iter().map(|&v| f16::from_f32(v).to_bits()).collect())
            }
            FloatDType::Bf16 => Storage::U16(
                values
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_bits())
                    .collect(),
            ),
        };
        Self::new(storage, dtype.data_type(), shape)
    }

    pub fn zeros(dtype: FloatDType, shape: &[usize]) -> Self {
        let len = shape.iter().product();
        let storage = match dtype {
            FloatDType::F32 => Storage::F32(vec![0.0; len]),
            FloatDType::F16 | FloatDType::Bf16 => Storage::U16(vec![0; len]),
        };
        Self::new(storage, dtype.data_type(), shape)
    }

    pub fn i64(shape: &[usize], values: &[i64]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(Storage::I64(values.to_vec()), DataType::Int64, shape)
    }

    pub fn u8(shape: &[usize], values: &[u8]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(Storage::U8(values.to_vec()), DataType::Uint8, shape)
    }

    pub fn i32(shape: &[usize], values: &[i32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(Storage::I32(values.to_vec()), DataType::Int32, shape)
    }

    pub fn bool(shape: &[usize], values: &[bool]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self::new(
            Storage::U8(values.iter().map(|&b| b as u8).collect()),
            DataType::Bool,
            shape,
        )
    }

    fn new(storage: Storage, dtype: DataType, shape: &[usize]) -> Self {
        Self {
            storage,
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype,
        }
    }

    fn const_ptr(&self) -> *const std::ffi::c_void {
        match &self.storage {
            Storage::F32(values) => values.as_ptr().cast(),
            Storage::U16(values) => values.as_ptr().cast(),
            Storage::U8(values) => values.as_ptr().cast(),
            Storage::I64(values) => values.as_ptr().cast(),
            Storage::I32(values) => values.as_ptr().cast(),
        }
    }

    fn mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match &mut self.storage {
            Storage::F32(values) => values.as_mut_ptr().cast(),
            Storage::U16(values) => values.as_mut_ptr().cast(),
            Storage::U8(values) => values.as_mut_ptr().cast(),
            Storage::I64(values) => values.as_mut_ptr().cast(),
            Storage::I32(values) => values.as_mut_ptr().cast(),
        }
    }

    /// The f32 payload, for benches that need to compare results across cells.
    ///
    /// Panics on any other storage: a hash that silently skipped a non-f32
    /// tensor would compare equal for every input, which is the vacuous-control
    /// shape this is here to avoid.
    pub fn f32s(&self) -> &[f32] {
        match &self.storage {
            Storage::F32(values) => values,
            _ => panic!("f32s() on a non-f32 tensor"),
        }
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.const_ptr()),
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    pub fn view_mut(&mut self) -> TensorMut<'_> {
        let data = DevicePtrMut(self.mut_ptr());
        TensorMut::new(
            data,
            self.dtype,
            &self.shape,
            &self.strides,
            DeviceId::cpu(),
        )
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match (&self.storage, self.dtype) {
            (Storage::F32(values), DataType::Float32) => values.clone(),
            (Storage::U16(values), DataType::Float16) => values
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect(),
            (Storage::U16(values), DataType::BFloat16) => values
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect(),
            _ => panic!("tensor is not a supported floating-point tensor"),
        }
    }
}

pub fn float_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 251) as i32 - 125) as f32 / 64.0)
        .collect()
}

pub fn make_kernel(
    op_type: &str,
    attributes: impl IntoIterator<Item = (&'static str, Attribute)>,
    input_shapes: &[Vec<usize>],
    opset: u64,
) -> Box<dyn Kernel> {
    let mut node = Node::new(NodeId(0), op_type, vec![], vec![]);
    for (name, value) in attributes {
        node.attributes.insert(name.into(), value);
    }
    CpuExecutionProvider::new()
        .get_kernel(&node, input_shapes, opset)
        .unwrap()
}

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected} (tolerance {tolerance})"
        );
    }
}

/// Put this benchmark process into the same decode thread topology a served
/// session runs in.
///
/// `CpuExecutionProvider::initialize` is the earliest per-session hook and the
/// only place an explicit `ONNX_GENAI_CPU_DECODE_THREADS` budget becomes a
/// process-wide bound on prefill/MLAS Rayon parallelism and, on Linux, on CPU
/// affinity. A benchmark that never calls it runs every row on an *unbounded*
/// process, so a `t=N` row measures N decode workers competing with a
/// full-width Rayon pool on every core -- not the configuration production runs
/// (#1749).
///
/// Deliberately the real `CpuExecutionProvider::initialize` rather than a copy
/// of what it currently does. A reimplementation would silently stop matching
/// the moment `initialize` grows a second responsibility, which is exactly the
/// drift that leaves a benchmark measuring a configuration nothing ships.
///
/// A no-op unless a budget is set, and idempotent: the underlying bound latches
/// on first call. Benches that install their own fixed-width Rayon pool (see
/// `kernels.rs`, which sweeps `[1, 8]`) will oversubscribe that pool onto a
/// smaller budget's cores -- which is what production does with the same two
/// settings, and is why this is applied there too rather than special-cased.
/// Process-wide CPU time, split into user and system, in seconds.
#[derive(Clone, Copy, Debug)]
pub struct CpuTime {
    pub user_s: f64,
    pub sys_s: f64,
}

impl CpuTime {
    pub fn total_s(self) -> f64 {
        self.user_s + self.sys_s
    }

    /// `self - earlier`, for bracketing a measured window.
    pub fn since(self, earlier: CpuTime) -> CpuTime {
        CpuTime {
            user_s: self.user_s - earlier.user_s,
            sys_s: self.sys_s - earlier.sys_s,
        }
    }
}

/// CPU seconds consumed by **every thread of this process**, user and system
/// separately.
///
/// Bracket a measured window with two reads and subtract. Wall time answers
/// "how long did it take"; this answers "how much machine did it cost", and the
/// two together decide a question wall time alone cannot: when a width doubling
/// fails to halve the wall, are the extra workers *idle* (CPU flat, the loss is
/// in dispatch, wake or join) or *busy* (CPU up, the loss is inside the kernel
/// or is spin burned waiting)? Those want opposite fixes, and a throughput
/// curve is consistent with both.
///
/// The user/system split is not decoration either. `sched_yield` in a park
/// path is charged to **system**, so a spin-then-park barrier that ramps its
/// yields with width shows up here as a rising `sys` fraction while `user`
/// stays flat -- a signature that names the mechanism rather than just
/// bounding it.
///
/// Robust to contention in a way wall time is not: another agent saturating
/// the box steals wall from us but does not add to our `utime`. `/usr/bin/time`'s
/// `Percent of CPU` is *not* an independent check on a wall-time result -- it is
/// `(user+sys)/wall`, the same wall in the denominator, so it degrades exactly
/// when wall does.
///
/// `/proc/self/stat`'s `utime`/`stime` are thread-group totals, so this covers
/// the decode pool's workers without enumerating them. `comm` is arbitrary
/// bytes in parentheses that may itself contain spaces and parentheses, so the
/// field walk starts at the *last* `)` -- the same convention as
/// `onnx_runtime_hostmon::own_jiffies_of_self_stat`, which this splits rather
/// than sums.
#[cfg(target_os = "linux")]
pub fn process_cpu_time() -> Option<CpuTime> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let tail = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    let hz = onnx_runtime_hostmon::clock_tick_hz();
    if hz <= 0.0 {
        return None;
    }
    Some(CpuTime {
        user_s: utime as f64 / hz,
        sys_s: stime as f64 / hz,
    })
}

/// Non-Linux hosts have no `/proc/self/stat`; a bench that wants this reports
/// the row as unavailable rather than substituting a wall-derived stand-in.
#[cfg(not(target_os = "linux"))]
pub fn process_cpu_time() -> Option<CpuTime> {
    None
}

pub fn init_decode_topology() {
    CpuExecutionProvider::new()
        .initialize(&Default::default())
        .expect("the CPU EP must initialize");
}

/// Print the decode width the run actually realized, next to the width it asked
/// for.
///
/// Call this **after** at least one decode has run. The persistent pool is built
/// lazily at first decode and [`decode_width`] is deliberately non-forcing, so
/// calling it earlier reports `path=unresolved` and no realized width -- and
/// would build the pool if it were forcing, changing the very topology the
/// benchmark is measuring.
///
/// Why every row needs this: three paths silently reduce the realized width
/// below the request -- the pre-clamp to `available_parallelism` in
/// `resolve_persistent_decode_threads_with_override`, `reserve_split_headroom`
/// on a NUMA split, and the single-CPU-cpuset branch that drops decode to the
/// flat path entirely. Only the last is even `NXRT_CALIB_DEBUG`-visible; the
/// other two log nothing. Without this line a `t=N` row is an unverified
/// *label*: a sweep that silently pins every width to the same realized value
/// prints a flat line that reads exactly like "this kernel does not scale"
/// (#1763).
///
/// `reserve_single_group_headroom` is deliberately not in that list. It reduces
/// the *spawned thread* count, but only in the single-group case, which is
/// exactly where the dispatcher takes a shard of its own and adds the lane back.
/// A 2-lane budget on a 2-CPU cpuset spawns one thread and still realizes two
/// lanes -- verified, not assumed.
///
/// Reports rather than asserts. A reduced width is legitimate when the host
/// genuinely cannot honour the request (an 8-lane budget in a 2-CPU cpuset), and
/// aborting there would make a constrained container unable to benchmark at all.
/// The `WIDTH-MISMATCH` token is for the caller -- human or matrix script -- to
/// discard or mark the row, the same way a contended cell is marked UNTRUSTED.
pub fn report_decode_width() {
    let width = onnx_runtime_ep_cpu::decode_spmd::decode_width();
    let show = |v: Option<usize>| v.map_or("unknown".to_string(), |v| v.to_string());
    // Three outcomes, not two: a width that was reduced is a different problem
    // from a width that was never resolved, and a matrix script wants to treat
    // them differently -- the first invalidates the row's label, the second says
    // no decode reached the persistent pool at all.
    let verdict = if width.is_as_requested() {
        "as_requested"
    } else if width.requested.is_some() && width.realized.is_some() {
        "WIDTH-MISMATCH"
    } else {
        "WIDTH-UNRESOLVED"
    };
    println!(
        "decode_width requested={} realized={} path={} {verdict}",
        show(width.requested),
        show(width.realized),
        width.path,
    );
}

/// Report the dispatcher's reserved CPU against the CPU it is actually on.
///
/// The non-vacuity check for `ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN`, and the
/// measurement that motivated it. Two independent facts, never inferred from
/// each other: which CPU the headroom reserve kept clear, and which CPU the
/// dispatching thread is running on now.
///
/// Both come from the pool, which samples the dispatcher from *inside* the
/// dispatch path. Neither can be obtained here. The first version of this read
/// `sched_getcpu()` on the reporting thread and was exactly inverted: the
/// reporter is idle while the pool works, so with the dispatcher unpinned the
/// scheduler parks the reporter on the one free core, and pinning the
/// dispatcher *evicts* it -- so "unpinned" read as on-the-reserved-CPU and
/// "pinned" read as off it. Reading the wrong thread does not merely add noise,
/// it can invert the sign. The second version read the dispatcher's own
/// `/proc/self/task/<tid>/stat`, which parses correctly but almost always
/// returns nothing: the dispatcher is a transient thread and has usually exited
/// by the time a bench reports.
///
/// `PIN-TOOK` only when the knob was asked for and the two agree; `PIN-MISSED`
/// when it was asked for and they do not, which is a failed intervention and
/// must not be scored as a control. With the knob off this is pure
/// observation -- `observed` is where the scheduler left the dispatcher, which
/// is the quantity the experiment is about.
pub fn report_dispatcher_cpu() {
    let pools = onnx_runtime_ep_cpu::decode_spmd::pools();
    let reserved = pools.and_then(|p| p.dispatcher_cpu());
    let observed = pools.and_then(|p| p.dispatcher_observed_cpu());
    let tid = pools.and_then(|p| p.dispatcher_thread_id());
    let moves = pools.map(|p| p.dispatcher_cpu_changes());
    let requested = onnx_runtime_ep_cpu::decode_spmd::dispatcher_pin_requested();
    let verdict = match (requested, reserved, observed) {
        (false, _, _) => "PIN-OFF",
        (true, None, _) => "PIN-UNRESERVED",
        (true, Some(_), None) => "PIN-UNOBSERVABLE",
        (true, Some(r), Some(o)) if r == o => "PIN-TOOK",
        (true, Some(_), Some(_)) => "PIN-MISSED",
    };
    let show = |v: Option<usize>| v.map_or("none".to_string(), |v| v.to_string());
    println!(
        "dispatcher reserved_cpu={} requested={} tid={} observed_cpu={} moves={} {verdict}",
        show(reserved),
        u8::from(requested),
        tid.map_or("none".to_string(), |v| v.to_string()),
        show(observed),
        moves.map_or("none".to_string(), |v| v.to_string()),
    );
}
