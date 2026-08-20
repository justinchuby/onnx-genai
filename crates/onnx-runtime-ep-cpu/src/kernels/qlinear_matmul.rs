//! `QLinearMatMul`: integer matrix multiplication with linear quantization.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    DataType, Dim, Graph, Node, Shape, broadcast_shapes, compute_contiguous_strides,
};
use rayon::prelude::*;

use super::{check_arity, qgemm_native, to_dense_bytes, write_dense_bytes};
use crate::strided::numel;
use std::borrow::Cow;

/// Borrow a tensor's bytes when they are already dense, copy them when they are
/// not.
///
/// `to_dense_bytes` always allocates: it builds a zeroed `Vec` and then
/// overwrites it, so a contiguous host operand costs an allocation, a
/// `memset`, a copy and a `free` on every single call. For `A` that is `m * k`
/// bytes per call -- 256 KiB at a 128x2048 prefill -- and on a busy pool that
/// is far more expensive than it looks: freeing a fresh multi-page mapping
/// forces a TLB shootdown to every core the process is running on, so the cost
/// grows with the thread count. Measured on a 32-vCPU EPYC 9V74, the per-call
/// `A` copy alone was 13 us at one thread and 136 us at sixteen.
///
/// The bytes are only borrowed when the view is host-accessible and
/// contiguous, which is exactly when `to_dense_bytes` would have produced an
/// identical copy of them.
fn dense_bytes<'a>(view: &TensorView<'a>) -> Result<Cow<'a, [u8]>> {
    if view.device.is_host_accessible() && view.is_contiguous() {
        view.validate()?;
        let len = numel(view.shape) * super::elem_size(view.dtype)?;
        // SAFETY: the executor guarantees the view's backing allocation is live
        // for `'a` and in bounds (ep-api safety invariant #1); `validate`
        // accepted the shape/stride pair and `is_contiguous` means those `len`
        // bytes are consecutive from the byte origin. `u8` has no invalid bit
        // patterns.
        let bytes = unsafe { std::slice::from_raw_parts(view.data_ptr::<u8>(), len) };
        #[cfg(test)]
        BORROWED_INPUT_CALLS.with(|calls| calls.set(calls.get() + 1));
        return Ok(Cow::Borrowed(bytes));
    }
    to_dense_bytes(view).map(Cow::Owned)
}

/// Largest accumulator a single thread keeps between calls, in bytes.
///
/// 32 MiB covers every accumulator up to an 8M-element result -- a 2048x4096
/// output and everything smaller -- which is the range where the per-call
/// allocation actually hurts. Anything larger is released rather than parked on
/// a thread for the life of the process: at that size the GEMM itself dwarfs
/// the allocation, so retaining it would trade real memory for nothing.
///
/// This is a **per-thread** figure. The buffer is parked on every worker thread
/// that ever runs the kernel, so the naive process-wide exposure would be
/// `MAX_RETAINED_ACCUMULATOR_BYTES x threads` -- 1 GiB on a 32-vCPU box, 4 GiB
/// on a 128-vCPU server. That multiplier (the thread count) is precisely the
/// variable the reuse optimisation scales with, so it must not be left out of
/// the ceiling. See [`MAX_PROCESS_ACCUMULATOR_BYTES`].
const MAX_RETAINED_ACCUMULATOR_BYTES: usize = 32 << 20;

/// Hard ceiling on the accumulator scratch summed across **all** worker
/// threads.
///
/// The per-thread cap above bounds one buffer; this bounds the sum, so the
/// process-wide exposure is `min(128 MiB, 32 MiB x threads)` -- a flat 128 MiB
/// once four or more threads park a full buffer, and it does *not* grow with the
/// vCPU count. Arithmetic, stated explicitly rather than left implicit:
///
/// * 4-vCPU box:  4 x 32 MiB = 128 MiB (at the cap)
/// * 20-vCPU box: min(128 MiB, 640 MiB) = **128 MiB** (was 640 MiB)
/// * 32-vCPU box: min(128 MiB, 1 GiB)   = **128 MiB** (was 1 GiB)
/// * 128-vCPU box: min(128 MiB, 4 GiB)  = **128 MiB** (was 4 GiB)
///
/// Threads beyond the fourth that want to park a full 32 MiB buffer are refused
/// and recompute the buffer per call (byte-identical output, only slower). 128
/// MiB of transient integer-GEMM scratch is defensible independent of model
/// size and vCPU count, which is the property the per-thread-only bound lacked.
const MAX_PROCESS_ACCUMULATOR_BYTES: usize = 128 << 20;

/// Process-wide, declinable budget governing the parked accumulator scratch
/// across every worker thread (#1056). `live_bytes()` reports the sum actually
/// parked, so a wrong ceiling is detectable in one run rather than argued from a
/// formula -- and a single-thread test cannot move it past one buffer, which is
/// why the test that defends it drives multiple threads.
pub(crate) static ACCUMULATOR_BUDGET:
    crate::kernels::governed_accumulator_budget::GovernedAccumulatorBudget =
    crate::kernels::governed_accumulator_budget::GovernedAccumulatorBudget::new(
        MAX_RETAINED_ACCUMULATOR_BYTES as u64,
        MAX_PROCESS_ACCUMULATOR_BYTES as u64,
    );

thread_local! {
    /// Per-thread scratch for the `i32` accumulator that the integer GEMM writes
    /// before the values are requantized down to the output dtype.
    ///
    /// The accumulator is `m * n` i32 -- 1 MiB at a 128x2048 prefill -- and MLAS
    /// first-touches every page of it from every worker thread. Allocating it per
    /// call therefore buys a fresh mapping, a page fault per page, and on free a
    /// TLB shootdown to every core the process is running on. Measured on a
    /// 32-vCPU EPYC 9V74 at K=N=2048, M=128: re-allocating it per call cost
    /// **301 us** with a 16-thread pool and nothing measurable with one thread,
    /// which is why this only started to matter once the GEMM itself scaled.
    ///
    /// Thread-local rather than shared: `execute` takes `&self` and the executor
    /// may run the same kernel on several threads at once, so a shared buffer would
    /// need a lock that serialises exactly the calls this exists to speed up.
    /// Retention is still bounded process-wide, not just per thread: parking is
    /// admitted through [`ACCUMULATOR_BUDGET`], which caps the sum across all
    /// threads (see [`MAX_PROCESS_ACCUMULATOR_BYTES`]).
    static ACCUMULATOR: std::cell::RefCell<Vec<i32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Tell the CPU EP whether the memory plan admitted the process-wide
/// accumulator scratch budget (#1056). When declined, the kernel parks nothing
/// and reallocates the `i32` accumulator per call -- byte-identical output, only
/// slower -- instead of retaining up to [`MAX_PROCESS_ACCUMULATOR_BYTES`] across
/// the worker pool for the life of the process.
pub fn set_qlinear_accumulator_budget_admitted(admitted: bool) {
    ACCUMULATOR_BUDGET.set_admitted(admitted);
}

/// Bytes the parked accumulator scratch currently holds, summed across every
/// worker thread. This is the figure a predicted ceiling is checked against
/// (#1056); a single-thread run cannot move it past one buffer.
pub fn qlinear_accumulator_live_bytes() -> u64 {
    ACCUMULATOR_BUDGET.live_bytes()
}

/// Retune the process-wide accumulator scratch ceiling. Exposed for deployment
/// tuning and for the RSS measurement harness, which contrasts the bounded
/// ceiling against the pre-fix per-thread-only behaviour (`u64::MAX`).
pub fn set_qlinear_accumulator_process_cap_bytes(bytes: u64) {
    ACCUMULATOR_BUDGET.set_process_cap_bytes(bytes);
}

/// The process-wide accumulator scratch ceiling currently in force.
pub fn qlinear_accumulator_process_cap_bytes() -> u64 {
    ACCUMULATOR_BUDGET.process_cap_bytes()
}

/// Predicted resident bytes the accumulator scratch budget will hold for
/// `graph`, for the memory plan to fold into its resident total.
///
/// The budget is a **fixed** process-wide pool ([`MAX_PROCESS_ACCUMULATOR_BYTES`]),
/// not a per-weight cache, so its ceiling does not scale with the number of
/// `QLinearMatMul` nodes: one node that reaches the parking path can fill the
/// whole pool across the thread count, and more nodes share the same pool. The
/// prediction is therefore the flat process cap when the graph contains any
/// `QLinearMatMul`, and zero otherwise (a model with none never parks). This is
/// the honest ceiling -- the process exposure, not a per-buffer figure silent
/// about the thread multiplier.
pub fn qlinear_accumulator_budget_predicted_bytes(graph: &Graph) -> u64 {
    let has_qlinear_matmul = graph
        .nodes
        .values()
        .any(|node| node.is_default_domain() && node.op_type == "QLinearMatMul");
    if has_qlinear_matmul {
        MAX_PROCESS_ACCUMULATOR_BYTES as u64
    } else {
        0
    }
}

/// Whether the constant-`B` pre-pack is admitted. Stored inverted so an unset
/// process (the atomic's `false` default meaning "not disabled") keeps the fast
/// packed route, matching the sibling `set_mlas_sqnbit_packing_enabled` gate.
static QLINEAR_PACKED_B_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Tell the CPU EP whether the memory plan admitted the constant-`B` MLAS
/// pre-pack (#1056). The pre-pack is a session-lifetime, weight-scaled buffer
/// (`packed_b`); it merged ungoverned in `ac394fd6`. When declined, the kernel
/// densifies `B` and takes the unpacked GEMM per call -- byte-identical output,
/// only slower -- and retains no packed copy. Available regardless of the
/// `mlas` feature so the engine can wire it unconditionally; without `mlas`
/// there is no packing and the flag is inert.
pub fn set_qlinear_packed_b_enabled(enabled: bool) {
    QLINEAR_PACKED_B_DISABLED.store(!enabled, std::sync::atomic::Ordering::Release);
}

#[cfg(feature = "mlas")]
fn qlinear_packed_b_enabled() -> bool {
    !QLINEAR_PACKED_B_DISABLED.load(std::sync::atomic::Ordering::Acquire)
}

/// Heap bytes the constant-`B` pre-packs currently hold across all kernels --
/// the figure the [`qlinear_packed_b_predicted_bytes`] prediction is checked
/// against (#1056). Zero without the `mlas` feature (nothing is ever packed).
pub fn qlinear_packed_b_live_bytes() -> u64 {
    #[cfg(feature = "mlas")]
    {
        mlas_sys::qgemm_packed_live_bytes() as u64
    }
    #[cfg(not(feature = "mlas"))]
    {
        0
    }
}

/// How many times a shape-keyed kernel cache materializes each `packed_b`.
///
/// The executor's kernel cache is keyed by `(node, resolved_input_shapes)`, so
/// an autoregressive decoder compiles a `QLinearMatMul` node once for the
/// prefill shape (`m > 1`) and once for the decode shape (`m == 1`), each a
/// separate `QLinearMatMulKernel` holding its own `packed_b`. Both pack the same
/// constant `B`, so the session holds two packed copies, not one -- the same
/// shape-keyed multiplier #1051 corrected for the MLAS SQNBit buffer. Counting
/// one copy would under-report by 2x, the dangerous direction for an admission
/// gate; a prefill-only run holds one copy, so this over-estimates there, which
/// only makes the gate decline sooner (safe).
#[cfg(feature = "mlas")]
const QLINEAR_PACKED_B_DECODE_INSTANTIATIONS: u64 = 2;

/// Predicted resident bytes the constant-`B` pre-packs will hold for `graph`,
/// for the memory plan to fold into its resident total (#1056).
///
/// For each `QLinearMatMul` whose `B` (input index 3) is a rank-2 constant
/// initializer, this is the exact MLAS packed size for that `k x n` weight times
/// [`QLINEAR_PACKED_B_DECODE_INSTANTIATIONS`]. `A`'s signedness is not a
/// graph-static property (it is an activation), so the size is taken as the max
/// over both `a_signed` possibilities -- the over-predicting (safe) direction.
/// Zero without the `mlas` feature, where nothing is ever packed.
pub fn qlinear_packed_b_predicted_bytes(graph: &Graph) -> u64 {
    #[cfg(not(feature = "mlas"))]
    {
        let _ = graph;
        0
    }
    #[cfg(feature = "mlas")]
    {
        let mut total = 0_u64;
        for node in graph.nodes.values() {
            total = total.saturating_add(node_qlinear_packed_b_bytes(node, graph));
        }
        total
    }
}

/// Per-node contribution to [`qlinear_packed_b_predicted_bytes`]. Mirrors the
/// packing condition in [`QLinearMatMulKernel::pack_key`]: a constant, rank-2
/// `B`. The batched-`B` decline is not visible from a rank-2 initializer, so it
/// need not be re-checked here.
#[cfg(feature = "mlas")]
fn node_qlinear_packed_b_bytes(node: &Node, graph: &Graph) -> u64 {
    if !node.is_default_domain() || node.op_type != "QLinearMatMul" {
        return 0;
    }
    let Some(Some(b_value)) = node.inputs.get(3).copied() else {
        return 0;
    };
    let Some(weight) = graph.initializers.get(&b_value) else {
        return 0;
    };
    let dims = weight.dims();
    if dims.len() != 2 {
        return 0;
    }
    let (k, n) = (dims[0], dims[1]);
    let b_signed = weight.dtype() == DataType::Int8;
    // A is an activation, so its signedness is unknown here; the packed size can
    // differ with it, so take the larger of the two -- the safe (over-predicting)
    // direction per #1056.
    let packed = [false, true]
        .into_iter()
        .filter_map(|a_signed| mlas_sys::qgemm_pack_b_size(n, k, a_signed, b_signed))
        .max()
        .unwrap_or(0) as u64;
    packed.saturating_mul(QLINEAR_PACKED_B_DECODE_INSTANTIATIONS)
}

#[cfg(test)]
thread_local! {
    /// Test-only counters proving which output route a call took. Without them a
    /// test can only check the bytes, which both routes produce identically -- so a
    /// silent regression to always-staging would pass every correctness test.
    ///
    /// Thread-local rather than global: the test harness gives each test its own
    /// thread, so a count read here is this test's own calls and cannot be
    /// perturbed by whatever else is running in parallel.
    static DIRECT_OUTPUT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static STAGED_OUTPUT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Counts the calls where `dense_bytes` handed back a borrow of the caller's
    /// buffer rather than a private copy. A test that checks the caller's input
    /// survived a call is only meaningful if the call actually borrowed it, so
    /// this stops that test passing vacuously.
    static BORROWED_INPUT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Where a batch's requantized output bytes are written.
///
/// A contiguous, host-accessible output view is written **in place**: MLAS
/// requantizes straight into the caller's tensor, so the kernel neither
/// allocates a staging buffer, nor zeroes it, nor copies it back. A strided or
/// non-host view cannot be handed to MLAS at all, so it is staged and
/// scattered once at the end through `write_dense_bytes`, which is what every
/// path did before.
enum OutputSink<'a> {
    Direct(&'a mut [u8]),
    Staged(Vec<u8>),
}

impl OutputSink<'_> {
    /// The `len` bytes this batch owns, starting at `base` in the result.
    fn region(&mut self, base: usize, len: usize) -> &mut [u8] {
        match self {
            Self::Direct(bytes) => &mut bytes[base..base + len],
            Self::Staged(bytes) => {
                // The staged path grows batch by batch exactly as the previous
                // `Vec` did, so a partially-filled result is byte-identical.
                bytes.resize(base + len, 0);
                &mut bytes[base..]
            }
        }
    }
}

/// Multiply-accumulate count below which the integer accumulation runs on the
/// calling thread. A rayon fork costs on the order of microseconds; this is the
/// point where the accumulation itself is comfortably past that, so the guard
/// never trades measurable throughput for it.
const PARALLEL_MIN_WORK: usize = 1 << 16;

/// Identity a pre-packed constant `B` is only valid for.
///
/// MLAS chooses the packed layout from the shape and signedness, and the pack
/// is built from one specific weight buffer, so all five have to match before a
/// cached pack may be reused. `addr` is the weight's base address, which is
/// stable for a graph initializer over the executor's lifetime.
#[cfg(feature = "mlas")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct QgemmPackKey {
    addr: usize,
    k: usize,
    n: usize,
    a_signed: bool,
    b_signed: bool,
    /// Whether the packed bytes are the weight's own bytes or the sign-flipped
    /// ones. A pack built from flipped bytes answers a different question, so
    /// it must never be served to a call that did not flip.
    flip_b: bool,
}

#[derive(Default)]
pub struct QLinearMatMulKernel {
    /// Which operands the graph guarantees are constant initializers. Only
    /// index 3 (`B`) is consulted; the rest are carried so the array lines up
    /// with the operand list.
    constant_inputs: [bool; 8],
    /// `B` pre-packed into MLAS's quantized kernel layout, built at most once.
    ///
    /// `None` inside the `OnceLock` records that MLAS declined to pack this
    /// shape, so the unpacked path is used and no further attempt is made.
    #[cfg(feature = "mlas")]
    packed_b: std::sync::OnceLock<Option<(QgemmPackKey, mlas_sys::QgemmPackedB)>>,
}

pub struct QLinearMatMulFactory;

impl KernelFactory for QLinearMatMulFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QLinearMatMulKernel::default()))
    }
}

impl QLinearMatMulKernel {
    /// The identity a pack for this call would have, or `None` when `B` must
    /// not be packed at all.
    ///
    /// Declines for a non-constant `B`, and for a batched `B` where each batch
    /// is a different weight and one pack could not serve them. Also declines
    /// when the memory plan has not admitted the pre-pack (#1056), in which case
    /// the unpacked GEMM densifies `B` per call and retains nothing.
    #[cfg(feature = "mlas")]
    fn pack_key(
        &self,
        b: &TensorView<'_>,
        geometry: &Geometry,
        plan: &QgemmPlan,
    ) -> Option<QgemmPackKey> {
        if !qlinear_packed_b_enabled() {
            return None;
        }
        if !self.constant_inputs[3] {
            return None;
        }
        if geometry.b_batch.iter().any(|&dimension| dimension != 1) {
            return None;
        }
        Some(QgemmPackKey {
            addr: b.data_ptr::<u8>() as usize,
            k: geometry.k,
            n: geometry.n,
            a_signed: plan.a_signed,
            b_signed: plan.b_signed,
            flip_b: plan.flip_b,
        })
    }

    /// An already-built pack for `key`, without building one.
    ///
    /// Separate from [`pack_build`](Self::pack_build) so a call that already
    /// has a pack can skip densifying `B` entirely -- the dense bytes exist
    /// only to feed the packer.
    #[cfg(feature = "mlas")]
    fn pack_lookup(&self, key: QgemmPackKey) -> Option<&mlas_sys::QgemmPackedB> {
        self.packed_b
            .get()?
            .as_ref()
            .and_then(|(cached, packed)| (*cached == key).then_some(packed))
    }

    /// Build the pack for `key` from `b_bytes`, at most once per kernel.
    ///
    /// A stored `None` records that MLAS declined this shape, so the unpacked
    /// path serves every later call without another attempt.
    #[cfg(feature = "mlas")]
    fn pack_build(&self, key: QgemmPackKey, b_bytes: &[u8]) -> Option<&mlas_sys::QgemmPackedB> {
        if b_bytes.len() != key.k.checked_mul(key.n)? {
            return None;
        }
        let packed = mlas_sys::QgemmPackedB::new(key.n, key.k, b_bytes, key.a_signed, key.b_signed);
        self.packed_b
            .get_or_init(|| packed.map(|packed| (key, packed)))
            .as_ref()
            .and_then(|(cached, packed)| (*cached == key).then_some(packed))
    }
}

/// Return a claim-time denial for metadata the CPU reference kernel cannot run.
pub(crate) fn unsupported_reason(
    input_dtypes: &[DataType],
    input_shapes: &[Shape],
) -> Option<String> {
    if !input_dtypes.is_empty() {
        if input_dtypes.len() != 8 {
            return Some(format!(
                "QLinearMatMul requires 8 inputs, got {}",
                input_dtypes.len()
            ));
        }
        for &(index, name) in &[(0, "A"), (3, "B"), (7, "y_zero_point")] {
            if !is_quantized(input_dtypes[index]) {
                return Some(format!(
                    "QLinearMatMul: {name} must have Int8 or Uint8 dtype, got {:?}",
                    input_dtypes[index]
                ));
            }
        }
        for &(integer, value, name) in &[(0, 2, "a_zero_point"), (3, 5, "b_zero_point")] {
            if input_dtypes[value] != input_dtypes[integer] {
                return Some(format!(
                    "QLinearMatMul: {name} dtype {:?} must match input dtype {:?}",
                    input_dtypes[value], input_dtypes[integer]
                ));
            }
        }
        for &index in &[1, 4, 6] {
            if input_dtypes[index] != DataType::Float32 {
                return Some(format!(
                    "QLinearMatMul: scale input {index} must be Float32, got {:?}",
                    input_dtypes[index]
                ));
            }
        }
    }
    if input_shapes.is_empty() {
        return None;
    }
    if input_shapes.len() != 8 {
        return Some(format!(
            "QLinearMatMul requires 8 input shapes, got {}",
            input_shapes.len()
        ));
    }
    if let Err(reason) = validate_claim_shapes(input_shapes) {
        return Some(reason);
    }
    None
}

fn validate_claim_shapes(shapes: &[Shape]) -> std::result::Result<(), String> {
    let a = &shapes[0];
    let b = &shapes[3];
    if a.is_empty() || b.is_empty() {
        return Err("QLinearMatMul: operands must be at least 1-D".into());
    }
    if !dims_compatible(
        a[a.len() - 1],
        b[if b.len() == 1 { 0 } else { b.len() - 2 }],
    ) {
        return Err("QLinearMatMul: inner dimensions are not provably equal".into());
    }
    validate_batch_broadcast(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    validate_claim_quant_pair("a", &shapes[1], &shapes[2], a, QuantAxis::Row)?;
    validate_claim_quant_pair("b", &shapes[4], &shapes[5], b, QuantAxis::Column)?;
    if shapes[6] != shapes[7] {
        return Err("QLinearMatMul: y_scale and y_zero_point shapes must match".into());
    }
    if !is_claim_scalar_shape(&shapes[6]) {
        return Err("QLinearMatMul: output scale and zero point must be scalar".into());
    }
    Ok(())
}

fn validate_batch_broadcast(a: &[Dim], b: &[Dim]) -> std::result::Result<(), String> {
    let rank = a.len().max(b.len());
    for trailing in 0..rank {
        let a_dim = a
            .len()
            .checked_sub(trailing + 1)
            .map_or(Dim::Static(1), |index| a[index]);
        let b_dim = b
            .len()
            .checked_sub(trailing + 1)
            .map_or(Dim::Static(1), |index| b[index]);
        if !dims_broadcastable(a_dim, b_dim) {
            return Err("QLinearMatMul: batch dimensions are not provably broadcastable".into());
        }
    }
    Ok(())
}

fn validate_claim_quant_pair(
    name: &str,
    scale: &Shape,
    zero_point: &Shape,
    operand: &Shape,
    axis: QuantAxis,
) -> std::result::Result<(), String> {
    if scale != zero_point {
        return Err(format!(
            "QLinearMatMul: {name}_scale and {name}_zero_point shapes must match"
        ));
    }
    if is_claim_scalar_shape(scale) || is_claim_axis_shape(scale, operand, axis) {
        Ok(())
    } else {
        Err(format!(
            "QLinearMatMul: invalid {name} scale/zero-point shape"
        ))
    }
}

fn is_claim_scalar_shape(shape: &[Dim]) -> bool {
    shape.is_empty() || shape == [Dim::Static(1)]
}

fn is_claim_axis_shape(shape: &[Dim], operand: &[Dim], axis: QuantAxis) -> bool {
    match operand.len() {
        0 | 1 => false,
        2 => {
            shape.len() == 1
                && dims_equal(
                    shape[0],
                    operand[match axis {
                        QuantAxis::Row => 0,
                        QuantAxis::Column => 1,
                    }],
                )
        }
        rank => {
            if shape.len() != rank {
                return false;
            }
            let batch = rank - 2;
            if !shape[..batch]
                .iter()
                .zip(&operand[..batch])
                .all(|(&left, &right)| dims_equal(left, right))
            {
                return false;
            }
            match axis {
                QuantAxis::Row => {
                    dims_equal(shape[batch], operand[batch]) && shape[batch + 1] == Dim::Static(1)
                }
                QuantAxis::Column => {
                    shape[batch] == Dim::Static(1)
                        && dims_equal(shape[batch + 1], operand[batch + 1])
                }
            }
        }
    }
}

fn dims_equal(left: Dim, right: Dim) -> bool {
    left == right
}

fn dims_compatible(left: Dim, right: Dim) -> bool {
    dims_equal(left, right)
}

fn dims_broadcastable(left: Dim, right: Dim) -> bool {
    dims_equal(left, right) || left == Dim::Static(1) || right == Dim::Static(1)
}

impl Kernel for QLinearMatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("QLinearMatMul", inputs, outputs, 8, 8, 1)?;
        let a = &inputs[0];
        let b = &inputs[3];
        if !is_quantized(a.dtype) || !is_quantized(b.dtype) || !is_quantized(outputs[0].dtype) {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: A, B, and output must have Int8 or Uint8 dtype".into(),
            ));
        }
        if inputs[2].dtype != a.dtype || inputs[5].dtype != b.dtype {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: each input zero_point must match its quantized input dtype".into(),
            ));
        }
        if inputs[7].dtype != outputs[0].dtype {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: output dtype must match y_zero_point dtype".into(),
            ));
        }
        for &index in &[1, 4, 6] {
            if inputs[index].dtype != DataType::Float32 {
                return Err(EpError::KernelFailed(format!(
                    "QLinearMatMul: scale input {index} must be Float32"
                )));
            }
        }

        let geometry = Geometry::new(a.shape, b.shape)?;
        if outputs[0].shape != geometry.output_shape {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: output shape {:?} must be {:?}",
                outputs[0].shape, geometry.output_shape
            )));
        }
        let a_quant = QuantParams::load("a", &inputs[1], &inputs[2], a.shape, QuantAxis::Row)?;
        let b_quant = QuantParams::load("b", &inputs[4], &inputs[5], b.shape, QuantAxis::Column)?;
        let (y_scale, y_zero_point) = output_quant_params(&inputs[6], &inputs[7])?;

        let a_signed = inputs[0].dtype == DataType::Int8;
        let b_signed = inputs[3].dtype == DataType::Int8;
        let qgemm = QgemmPlan::select(&a_quant, a_signed, b_signed, &geometry);
        // Both routes read the operands as raw bytes, so neither materializes a
        // widened copy: the native kernel widens `A` to `i16` pairs as it packs
        // and reads `B` straight out of the caller's buffer.
        // A constant B is packed once. After that its dense bytes are dead --
        // they exist only to feed the packer -- so skip the copy, which is
        // `k * n` bytes on every call and dominates decode at large K and N.
        #[cfg(feature = "mlas")]
        let pack_key = qgemm
            .as_ref()
            .and_then(|plan| self.pack_key(&inputs[3], &geometry, plan));
        #[cfg(feature = "mlas")]
        let pack_ready = pack_key.is_some_and(|key| self.pack_lookup(key).is_some());
        #[cfg(not(feature = "mlas"))]
        let pack_ready = false;
        #[cfg_attr(not(feature = "mlas"), allow(unused_mut))]
        // Every route reads `A` exactly as the caller laid it out, so a
        // contiguous operand is borrowed rather than copied per call. The one
        // route that rewrites it -- the MLAS sign flip below -- goes through
        // `Cow::to_mut`, which copies a borrowed operand first, so the caller's
        // input is never written through.
        let mut a_bytes = dense_bytes(&inputs[0])?;
        #[cfg_attr(not(feature = "mlas"), allow(unused_mut))]
        let mut b_bytes = if pack_ready {
            Cow::Borrowed(&[][..])
        } else {
            dense_bytes(&inputs[3])?
        };

        // Move any operand MLAS has no kernel for into the sign domain it does,
        // before it is either packed or handed to the unpacked entry point.
        #[cfg(feature = "mlas")]
        if let Some(plan) = &qgemm {
            if plan.flip_a {
                flip_sign_domain(a_bytes.to_mut());
            }
            if plan.flip_b {
                flip_sign_domain(b_bytes.to_mut());
            }
        }
        // Loop-invariant: the pack depends only on the weight, not the batch.
        #[cfg(feature = "mlas")]
        let packed = pack_key.and_then(|key| match self.pack_lookup(key) {
            Some(packed) => Some(packed),
            None => self.pack_build(key, &b_bytes),
        });
        #[cfg(not(feature = "mlas"))]
        let packed: Option<&()> = None;
        let (m, k, n) = (geometry.m, geometry.k, geometry.n);
        // Requantizing into the caller's tensor removes a `result_len`
        // allocation, its zero-fill, and the copy back out of it, per call.
        let out_dtype = outputs[0].dtype;
        let direct_output = outputs[0].device.is_host_accessible() && outputs[0].is_contiguous();
        if direct_output {
            outputs[0].validate()?;
        }
        let mut sink = if direct_output {
            #[cfg(test)]
            DIRECT_OUTPUT_CALLS.with(|calls| calls.set(calls.get() + 1));
            // SAFETY: the executor provides an exclusive, in-bounds output
            // buffer (ep-api safety invariant #1); `validate` accepted the
            // shape/stride pair, `is_contiguous` means the `result_len` bytes
            // are consecutive from the byte origin, and the shape was checked
            // to equal `geometry.output_shape` above. Both supported output
            // dtypes are one byte wide, so `result_len` bytes is the whole
            // tensor.
            OutputSink::Direct(unsafe {
                std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<u8>(), geometry.result_len)
            })
        } else {
            #[cfg(test)]
            STAGED_OUTPUT_CALLS.with(|calls| calls.set(calls.get() + 1));
            OutputSink::Staged(Vec::with_capacity(geometry.result_len))
        };
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        // Hoisted out of the batch loop: both are re-filled per batch, so a
        // many-batch call allocates once rather than once per batch.
        // The accumulator is borrowed from this thread's scratch, so a repeated
        // shape reuses one already-faulted mapping instead of buying a new one.
        // Taking it out of the thread-local returns its reservation to the
        // process-wide budget; it is re-reserved at the end iff it still fits.
        let mut products: Vec<i32> = ACCUMULATOR.with(|cell| cell.take());
        ACCUMULATOR_BUDGET.release((products.capacity() * std::mem::size_of::<i32>()) as u64);
        let mut b_zero_points: Vec<i32> = Vec::new();
        let mut a_zero_points: Vec<i32> = Vec::new();
        let mut b_scales: Vec<f32> = Vec::new();
        #[cfg(feature = "mlas")]
        let mut combined_scales: Vec<f32> = Vec::new();
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            let a_offset = a_batch * m * k;
            let b_offset = b_batch * k * n;
            // `n == 0` produces no output for this batch, and both
            // `par_chunks_mut(0)` and `chunks_exact(0)` panic, so leave early.
            if n == 0 {
                if batch + 1 < geometry.batch_count {
                    next_index(&geometry.batch_shape, &mut batch_index);
                }
                continue;
            }
            b_zero_points.clear();
            b_zero_points.extend((0..n).map(|column| b_quant.at(b_batch, column).1));
            b_scales.clear();
            b_scales.extend((0..n).map(|column| b_quant.at(b_batch, column).0));
            a_zero_points.clear();
            a_zero_points.extend((0..m).map(|row| a_quant.at(a_batch, row).1));

            // Decided before the accumulator is sized because the two paths
            // want different buffers: the scalar gemm below accumulates into
            // `products` and needs it zeroed, while MLAS overwrites every
            // element it is given (the shim leaves `IsAccumulateMode` false, so
            // the first `k` block runs in `ZeroMode`).
            #[cfg(feature = "mlas")]
            let fused = match &qgemm {
                Some(_) => fused_scale(&a_quant, a_batch, &b_scales, y_scale, &mut combined_scales),
                None => None,
            };
            #[cfg(not(feature = "mlas"))]
            let fused: Option<()> = None;

            if fused.is_some() {
                // Only the length matters here. Growing through `vec![0; len]`
                // reaches `alloc_zeroed`, which the allocator serves with fresh
                // zero pages instead of a `memset` at these sizes; re-running
                // the same shape reuses the buffer untouched.
                if products.len() != m * n {
                    products = vec![0i32; m * n];
                }
            } else {
                products.clear();
                products.resize(m * n, 0);
            }

            if let Some(plan) = &qgemm {
                // Requantize inside MLAS when the combined scale allows it, so
                // each output tile is scaled while it is still in cache and the
                // bytes land in `output` directly. Only a non-finite scale --
                // which MLAS maps to the output minimum where the scalar loop
                // maps it to the zero point -- keeps a call off this path.
                #[cfg(feature = "mlas")]
                if let Some(scale) = fused {
                    let region = sink.region(batch * m * n, m * n);
                    plan.run_requantized(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + m * k],
                        match packed {
                            Some(packed) => mlas_sys::QgemmWeights::Packed(packed),
                            None => mlas_sys::QgemmWeights::Dense {
                                bytes: &b_bytes[b_offset..b_offset + k * n],
                                signed: plan.b_signed,
                            },
                        },
                        &b_zero_points,
                        scale,
                        region,
                        out_dtype == DataType::Int8,
                        y_zero_point,
                        &mut products,
                    )?;
                    if batch + 1 < geometry.batch_count {
                        next_index(&geometry.batch_shape, &mut batch_index);
                    }
                    continue;
                }
                match packed {
                    Some(packed) => plan.run_packed(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + m * k],
                        packed,
                        &b_zero_points,
                        &mut products,
                    )?,
                    None => plan.run(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + m * k],
                        &b_bytes[b_offset..b_offset + k * n],
                        &b_zero_points,
                        &mut products,
                    )?,
                }
                requantize_rows(
                    &products,
                    &a_quant,
                    a_batch,
                    &b_scales,
                    n,
                    y_scale,
                    y_zero_point,
                    out_dtype,
                    sink.region(batch * m * n, m * n),
                )?;
                if batch + 1 < geometry.batch_count {
                    next_index(&geometry.batch_shape, &mut batch_index);
                }
                continue;
            }

            // The integer half of the product, on the operand bytes. See
            // [`qgemm_native`]: the zero points are lifted out of the `k` loop
            // by expanding
            //   sum_k (a_k - az) * (b_kn - bz_n)
            //     = sum_k (a_k - az) * b_kn  -  bz_n * sum_k (a_k - az)
            // which is an identity over the integers, so under wrapping
            // arithmetic (exactly arithmetic mod 2^32) both sides reduce to the
            // same `i32`. Wrapping addition is associative and commutative, so
            // the kernel's blocking and its thread count are likewise unable to
            // change a single output bit, on overflow included.
            qgemm_native::qgemm(
                qgemm_native::Operand {
                    bytes: &a_bytes[a_offset..a_offset + m * k],
                    signed: a_signed,
                },
                qgemm_native::Operand {
                    bytes: &b_bytes[b_offset..b_offset + k * n],
                    signed: b_signed,
                },
                &a_zero_points,
                &b_zero_points,
                m,
                k,
                n,
                &mut products,
            );

            requantize_rows(
                &products,
                &a_quant,
                a_batch,
                &b_scales,
                n,
                y_scale,
                y_zero_point,
                out_dtype,
                sink.region(batch * m * n, m * n),
            )?;
            if batch + 1 < geometry.batch_count {
                next_index(&geometry.batch_shape, &mut batch_index);
            }
        }
        // Park the accumulator for the next call on this thread. An early
        // error return skips this and simply drops it, which costs the next
        // call one allocation and cannot affect a result.
        //
        // Parking is admitted through the process-wide budget, not a per-thread
        // constant: it succeeds only when this buffer is within the per-thread
        // cap *and* the sum parked across every worker thread stays within
        // `MAX_PROCESS_ACCUMULATOR_BYTES`. A refused buffer is dropped and
        // recomputed next call (byte-identical), so the process footprint no
        // longer scales with the thread count.
        if ACCUMULATOR_BUDGET.try_park((products.capacity() * std::mem::size_of::<i32>()) as u64) {
            ACCUMULATOR.with(|cell| cell.replace(products));
        }
        match sink {
            // The direct sink already wrote every byte the caller asked for.
            OutputSink::Direct(_) => Ok(()),
            OutputSink::Staged(bytes) => write_dense_bytes(&mut outputs[0], &bytes),
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn is_quantized(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int8 | DataType::Uint8)
}

#[derive(Clone, Copy)]
enum QuantAxis {
    Row,
    Column,
}

struct QuantParams {
    scales: Vec<f32>,
    zero_points: Vec<i32>,
    axis_len: usize,
    per_axis: bool,
}

impl QuantParams {
    fn load(
        name: &str,
        scale: &TensorView,
        zero_point: &TensorView,
        operand_shape: &[usize],
        axis: QuantAxis,
    ) -> Result<Self> {
        if scale.shape != zero_point.shape {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: {name}_scale and {name}_zero_point shapes must match"
            )));
        }
        let per_axis = if is_scalar_shape(scale.shape) {
            false
        } else if is_axis_shape(scale.shape, operand_shape, axis) {
            true
        } else {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: invalid {name} scale/zero-point shape {:?} for operand shape {:?}",
                scale.shape, operand_shape
            )));
        };
        let scales = read_scales(scale)?;
        let zero_points = read_quantized(zero_point)?;
        let axis_len = match axis {
            QuantAxis::Row => {
                if operand_shape.len() == 1 {
                    1
                } else {
                    operand_shape[operand_shape.len() - 2]
                }
            }
            QuantAxis::Column => *operand_shape.last().unwrap_or(&1),
        };
        Ok(Self {
            scales,
            zero_points,
            axis_len,
            per_axis,
        })
    }

    fn at(&self, source_batch: usize, axis_index: usize) -> (f32, i32) {
        let index = if self.per_axis {
            source_batch * self.axis_len + axis_index
        } else {
            0
        };
        (self.scales[index], self.zero_points[index])
    }
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.is_empty() || shape == [1]
}

fn is_axis_shape(shape: &[usize], operand: &[usize], axis: QuantAxis) -> bool {
    match operand.len() {
        0 | 1 => false,
        2 => {
            shape
                == [operand[match axis {
                    QuantAxis::Row => 0,
                    QuantAxis::Column => 1,
                }]]
        }
        rank => {
            if shape.len() != rank || shape[..rank - 2] != operand[..rank - 2] {
                return false;
            }
            match axis {
                QuantAxis::Row => shape[rank - 2] == operand[rank - 2] && shape[rank - 1] == 1,
                QuantAxis::Column => shape[rank - 2] == 1 && shape[rank - 1] == operand[rank - 1],
            }
        }
    }
}

fn output_quant_params(scale: &TensorView, zero_point: &TensorView) -> Result<(f32, i32)> {
    if scale.shape != zero_point.shape {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: y_scale and y_zero_point shapes must match".into(),
        ));
    }
    if !is_scalar_shape(scale.shape) {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: output scale and zero point must be scalar".into(),
        ));
    }
    Ok((read_scales(scale)?[0], read_quantized(zero_point)?[0]))
}

fn read_scales(view: &TensorView) -> Result<Vec<f32>> {
    let bytes = to_dense_bytes(view)?;
    let scales: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if scales
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err(EpError::KernelFailed(
            "QLinearMatMul: scales must be finite and positive".into(),
        ));
    }
    Ok(scales)
}

fn read_quantized(view: &TensorView) -> Result<Vec<i32>> {
    let bytes = to_dense_bytes(view)?;
    match view.dtype {
        DataType::Int8 => Ok(bytes.into_iter().map(|value| value as i8 as i32).collect()),
        DataType::Uint8 => Ok(bytes.into_iter().map(i32::from).collect()),
        other => Err(EpError::KernelFailed(format!(
            "QLinearMatMul: expected Int8 or Uint8 tensor, got {other:?}"
        ))),
    }
}

/// Narrow a zero point back to the operand's storage byte, or `None` if it does
/// not fit.
///
/// Zero points are read from a `u8`/`i8` tensor, so they always fit and this is
/// unreachable for any model that type-checks. It exists so a future caller
/// that synthesizes quantization parameters cannot silently truncate one into
/// a different kernel input; such a case declines to the exact loop instead.
#[cfg(feature = "mlas")]
fn zero_point_byte(value: i32, signed: bool, flipped: bool) -> Option<u8> {
    if flipped {
        // The operand bytes were shifted by +128 in the unsigned domain, so the
        // zero point shifts identically and `a - za` is unchanged.
        i8::try_from(value).ok().map(|value| (value as u8) ^ 0x80)
    } else if signed {
        i8::try_from(value).ok().map(|value| value as u8)
    } else {
        u8::try_from(value).ok()
    }
}

/// Routes the integer accumulation to MLAS's quantized GEMM when that is both
/// applicable and bit-exact.
///
/// MLAS ships tuned `u8`/`i8` GEMM kernels for AVX2, SSE4.1, AMX, NEON, dot
/// product and SMMLA, and they are already compiled into `mlas-sys`; the
/// binding was simply missing, so this kernel accumulated in a scalar loop and
/// lost to ONNX Runtime by more than an order of magnitude.
///
/// Two restrictions keep the result bit-identical to the fallback:
///
/// * MLAS takes a single `ZeroPointA`, so a per-row `a_zero_point` stays on the
///   fallback. Per-column `b_zero_point` is native (`PerColumnZeroPoints`).
/// * On a kernel that pairs products into an `i16` (AVX2 without VNNI), `u8`
///   activations against `i8` weights can saturate. `qgemm_u8s8_is_exact()`
///   probes the running machine for exactly that, so the fast path is taken
///   only where it is exact, and VNNI/AMX hosts get it automatically.
///
/// Operand combinations outside that set are not declined but *translated*: an
/// operand is reinterpreted in the unsigned domain by `XOR 0x80`, with `+128`
/// applied to its zero point. `sum_k (a_k - za)(b_k - zb)` is a difference of
/// integers, so shifting an operand and its zero point by the same constant
/// leaves every accumulator bit-identical, and the call lands on the `u8 x u8`
/// kernel this file already trusts as exact. Without it, `i8` activations sat
/// on the scalar loop and lost to ONNX Runtime by 5-6x.
#[cfg(feature = "mlas")]
struct QgemmPlan {
    /// Signedness handed to MLAS, i.e. *after* any flip.
    a_signed: bool,
    b_signed: bool,
    /// Whether the caller must `XOR 0x80` the operand's bytes first.
    flip_a: bool,
    flip_b: bool,
    zero_point_a: u8,
    b_zero_point_bytes: std::cell::RefCell<Vec<u8>>,
}

/// Reinterpret quantized bytes in the opposite signedness domain.
///
/// `XOR 0x80` is `+128` modulo 256, i.e. exactly the `i8 <-> u8` bijection that
/// preserves order. Applied to an operand *and* its zero point it cancels in
/// `a - za`, which is why it is free of any accuracy cost.
#[cfg(feature = "mlas")]
fn flip_sign_domain(bytes: &mut [u8]) {
    for byte in bytes.iter_mut() {
        *byte ^= 0x80;
    }
}

#[cfg(feature = "mlas")]
impl QgemmPlan {
    fn select(
        a_quant: &QuantParams,
        a_signed: bool,
        b_signed: bool,
        geometry: &Geometry,
    ) -> Option<Self> {
        if geometry.m == 0 || geometry.n == 0 || geometry.k == 0 {
            return None;
        }
        if a_quant.per_axis {
            return None;
        }
        // MLAS documents (mlas.h, `MLAS_GEMM_QUANT_SHAPE_PARAMS`) that signed
        // activations are unsupported off ARM, and on ARM only alongside signed
        // weights. The generic kernel happens to answer correctly outside that
        // envelope today, but relying on it would be relying on an accident.
        //
        // The ARM `i8 x i8` kernels accumulate through non-saturating `vmull` /
        // `vpadalq` / dot-product / SMMLA instructions, so unlike `u8 x i8` on
        // AVX2 they need no exactness probe. That is asserted unconditionally
        // by `qgemm_i32_matches_the_integer_oracle_for_every_signedness`, which
        // runs on every architecture including the aarch64 CI lanes.
        let native = if a_signed {
            cfg!(target_arch = "aarch64") && b_signed
        } else if b_signed {
            mlas_sys::qgemm_u8s8_is_exact()
        } else {
            true
        };
        // Anything outside the native envelope is translated into `u8 x u8`
        // rather than declined. Only the operands actually out of domain move.
        let (flip_a, flip_b) = if native {
            (false, false)
        } else {
            (a_signed, b_signed)
        };
        let zero_point_a = zero_point_byte(a_quant.at(0, 0).1, a_signed, flip_a)?;
        Some(Self {
            a_signed: a_signed && !flip_a,
            b_signed: b_signed && !flip_b,
            flip_a,
            flip_b,
            zero_point_a,
            b_zero_point_bytes: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// The `b_zero_point` column vector as the bytes MLAS expects, in the same
    /// sign domain the weight bytes were handed over in.
    fn b_zero_point_bytes(&self, b_zero_points: &[i32], n: usize) -> Result<()> {
        let mut bytes = self.b_zero_point_bytes.borrow_mut();
        bytes.clear();
        bytes.reserve(n);
        for &value in b_zero_points {
            bytes.push(
                zero_point_byte(value, self.b_signed || self.flip_b, self.flip_b).ok_or_else(
                    || {
                        EpError::KernelFailed(format!(
                            "QLinearMatMul: b_zero_point {value} does not fit the operand dtype"
                        ))
                    },
                )?,
            );
        }
        Ok(())
    }

    fn run(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[u8],
        b: &[u8],
        b_zero_points: &[i32],
        products: &mut [i32],
    ) -> Result<()> {
        self.b_zero_point_bytes(b_zero_points, n)?;
        let bytes = self.b_zero_point_bytes.borrow();
        mlas_sys::qgemm_i32(
            m,
            n,
            k,
            a,
            self.a_signed,
            self.zero_point_a,
            b,
            self.b_signed,
            mlas_sys::QgemmZeroPoints::PerColumn(&bytes),
            products,
        );
        Ok(())
    }

    /// [`run`](Self::run) with the requantization fused into MLAS.
    ///
    /// Writes the final bytes into `output` and leaves `products` holding
    /// unspecified scratch. `products` still has to be `m * n` long -- MLAS
    /// accumulates there and requantizes in place -- but its prior contents are
    /// irrelevant.
    #[allow(clippy::too_many_arguments)]
    fn run_requantized(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[u8],
        b: mlas_sys::QgemmWeights<'_>,
        b_zero_points: &[i32],
        scale: mlas_sys::QgemmScale<'_>,
        output: &mut [u8],
        output_signed: bool,
        output_zero_point: i32,
        products: &mut [i32],
    ) -> Result<()> {
        self.b_zero_point_bytes(b_zero_points, n)?;
        let bytes = self.b_zero_point_bytes.borrow();
        mlas_sys::qgemm_requantize(
            m,
            n,
            k,
            a,
            self.a_signed,
            self.zero_point_a,
            b,
            mlas_sys::QgemmZeroPoints::PerColumn(&bytes),
            scale,
            output,
            output_signed,
            output_zero_point,
            products,
        );
        Ok(())
    }

    /// [`run`](Self::run) against a `B` that was pre-packed once.
    fn run_packed(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[u8],
        packed: &mlas_sys::QgemmPackedB,
        b_zero_points: &[i32],
        products: &mut [i32],
    ) -> Result<()> {
        self.b_zero_point_bytes(b_zero_points, n)?;
        let bytes = self.b_zero_point_bytes.borrow();
        mlas_sys::qgemm_i32_packed(
            m,
            n,
            k,
            a,
            self.a_signed,
            self.zero_point_a,
            packed,
            mlas_sys::QgemmZeroPoints::PerColumn(&bytes),
            products,
        );
        Ok(())
    }
}

#[cfg(not(feature = "mlas"))]
struct QgemmPlan;

#[cfg(not(feature = "mlas"))]
impl QgemmPlan {
    fn select(_: &QuantParams, _: bool, _: bool, _: &Geometry) -> Option<Self> {
        None
    }

    fn run(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[u8],
        _: &[u8],
        _: &[i32],
        _: &mut [i32],
    ) -> Result<()> {
        unreachable!("QgemmPlan::select never yields a plan without the mlas feature")
    }

    /// Unreachable without `mlas`: `packed` is always `None` there, so this
    /// exists only to keep the dispatch in `execute` type-checking.
    #[allow(clippy::too_many_arguments)]
    fn run_packed(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: &[u8],
        _: &(),
        _: &[i32],
        _: &mut [i32],
    ) -> Result<()> {
        unreachable!("no pack exists without the mlas feature")
    }
}

#[allow(clippy::too_many_arguments)]
/// Requantize the `i32` accumulators into the output dtype.
///
/// Rows are independent, so this appends into a pre-sized region and walks the
/// rows in parallel once there is enough work to pay for the fork. Two
/// properties are load-bearing and must not be traded for speed:
///
/// * the per-element arithmetic is still `a_scale * b_scale / y_scale` in that
///   association, so results stay bit-identical to the serial version -- float
///   multiply and divide do not reassociate;
/// * each row writes only its own `n` bytes, so the output is identical
///   whether the rows run serially or in parallel.
///
/// `b_scales` is the per-column scale gathered once per batch by the caller;
/// looking it up per element was a `QuantParams::at` call inside the innermost
/// loop.
/// The scale MLAS's fused requantizer needs, or `None` when this call must keep
/// using the scalar loop.
///
/// The fused processor computes `clamp(round(c * scale)) + zero_point` with one
/// scale per tensor or per column. That covers every call MLAS accepts at all,
/// because a per-row `a_scale` already keeps a call off the MLAS path -- but
/// only for a *finite* scale: MLAS clamps the float before rounding, which
/// agrees with rounding before clamping everywhere except `NaN`. Rather than
/// change any output byte, a non-finite combined scale is declined here and the
/// scalar loop, which maps `NaN` to the zero point, runs instead.
///
/// `buffer` is reused across batches so a per-column scale is not reallocated
/// per call.
#[cfg(feature = "mlas")]
fn fused_scale<'a>(
    a_quant: &QuantParams,
    a_batch: usize,
    b_scales: &[f32],
    y_scale: f32,
    buffer: &'a mut Vec<f32>,
) -> Option<mlas_sys::QgemmScale<'a>> {
    if a_quant.per_axis {
        return None;
    }
    let a_scale = a_quant.at(a_batch, 0).0;
    // A per-tensor `b_scale` was splatted into `b_scales`, so every entry is the
    // same number: fold it once and let MLAS broadcast.
    let uniform = b_scales.iter().all(|scale| *scale == b_scales[0]);
    if uniform {
        let scale = a_scale * b_scales[0] / y_scale;
        return scale
            .is_finite()
            .then_some(mlas_sys::QgemmScale::PerTensor(scale));
    }
    buffer.clear();
    buffer.extend(b_scales.iter().map(|&b_scale| a_scale * b_scale / y_scale));
    buffer
        .iter()
        .all(|scale| scale.is_finite())
        .then_some(mlas_sys::QgemmScale::PerColumn(buffer))
}

fn requantize_rows(
    products: &[i32],
    a_quant: &QuantParams,
    a_batch: usize,
    b_scales: &[f32],
    n: usize,
    y_scale: f32,
    y_zero_point: i32,
    dtype: DataType,
    destination: &mut [u8],
) -> Result<()> {
    // Both supported dtypes are one byte wide, so the output length is known.
    // Reject anything else once here rather than per element.
    if !matches!(dtype, DataType::Int8 | DataType::Uint8) {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: unsupported output dtype {dtype:?}"
        )));
    }
    if destination.len() != products.len() {
        return Err(EpError::KernelFailed(format!(
            "QLinearMatMul: output region of {} bytes cannot hold {} accumulators",
            destination.len(),
            products.len()
        )));
    }

    // `a_scale * b_scale / y_scale` is the same expression for every row when
    // `a_scale` is per tensor, so evaluate it once per column instead of once
    // per element. The association is untouched, so the products are the same
    // `f32` bits -- this removes a division per output element, not a rounding
    // step. A per-row `a_scale` genuinely varies by row and keeps the divide.
    let shared_scales: Option<Vec<f32>> = (!a_quant.per_axis).then(|| {
        let a_scale = a_quant.at(a_batch, 0).0;
        b_scales
            .iter()
            .map(|&b_scale| a_scale * b_scale / y_scale)
            .collect()
    });

    let requantize_row = |row: usize, accumulators: &[i32], bytes: &mut [u8]| {
        let a_scale = a_quant.at(a_batch, row).0;
        let quantize = |accumulated: i32, scale: f32, byte: &mut u8| {
            // `f32 as i64` already saturates, but adding the zero point to
            // `i64::MAX` wraps in release and panics in debug, so a scale large
            // enough to push the product past `2^63` used to produce the
            // *opposite* end of the range.  Saturating here also makes this
            // loop agree with MLAS -- which clamps in `f32`, where there is no
            // such cliff -- for every finite scale, not just the small ones.
            let value = ((accumulated as f32 * scale).round_ties_even() as i64)
                .saturating_add(i64::from(y_zero_point));
            *byte = match dtype {
                DataType::Int8 => value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8,
                _ => value.clamp(u8::MIN as i64, u8::MAX as i64) as u8,
            };
        };
        match &shared_scales {
            Some(scales) => {
                for ((&accumulated, &scale), byte) in
                    accumulators.iter().zip(scales).zip(bytes.iter_mut())
                {
                    quantize(accumulated, scale, byte);
                }
            }
            None => {
                for ((&accumulated, &b_scale), byte) in
                    accumulators.iter().zip(b_scales).zip(bytes.iter_mut())
                {
                    quantize(accumulated, a_scale * b_scale / y_scale, byte);
                }
            }
        }
    };

    if products.len() >= PARALLEL_MIN_WORK {
        destination
            .par_chunks_mut(n)
            .zip(products.par_chunks_exact(n))
            .enumerate()
            .for_each(|(row, (bytes, accumulators))| {
                requantize_row(row, accumulators, bytes);
            });
    } else {
        for (row, (bytes, accumulators)) in destination
            .chunks_mut(n)
            .zip(products.chunks_exact(n))
            .enumerate()
        {
            requantize_row(row, accumulators, bytes);
        }
    }
    Ok(())
}

struct Geometry {
    m: usize,
    k: usize,
    n: usize,
    a_batch: Vec<usize>,
    b_batch: Vec<usize>,
    a_batch_strides: Vec<i64>,
    b_batch_strides: Vec<i64>,
    batch_shape: Vec<usize>,
    batch_count: usize,
    result_len: usize,
    output_shape: Vec<usize>,
}

impl Geometry {
    fn new(a: &[usize], b: &[usize]) -> Result<Self> {
        let a_1d = a.len() == 1;
        let b_1d = b.len() == 1;
        let a = if a_1d { vec![1, a[0]] } else { a.to_vec() };
        let b = if b_1d { vec![b[0], 1] } else { b.to_vec() };
        if a.len() < 2 || b.len() < 2 {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: operands must be at least 1-D".into(),
            ));
        }
        let m = a[a.len() - 2];
        let k = a[a.len() - 1];
        let b_k = b[b.len() - 2];
        let n = b[b.len() - 1];
        if k != b_k {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: inner dims disagree ({k} vs {b_k})"
            )));
        }
        let a_batch = a[..a.len() - 2].to_vec();
        let b_batch = b[..b.len() - 2].to_vec();
        let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
        let batch_count = numel(&batch_shape);
        let mut output_shape = batch_shape.clone();
        if !a_1d {
            output_shape.push(m);
        }
        if !b_1d {
            output_shape.push(n);
        }
        Ok(Self {
            m,
            k,
            n,
            a_batch_strides: compute_contiguous_strides(&a_batch),
            b_batch_strides: compute_contiguous_strides(&b_batch),
            a_batch,
            b_batch,
            batch_shape,
            batch_count,
            result_len: batch_count * m * n,
            output_shape,
        })
    }

    fn a_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.a_batch, &self.a_batch_strides)
    }

    fn b_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.b_batch, &self.b_batch_strides)
    }
}

fn broadcast_offset(batch_index: &[usize], shape: &[usize], strides: &[i64]) -> usize {
    let leading = batch_index.len() - shape.len();
    shape
        .iter()
        .zip(strides)
        .enumerate()
        .map(|(index, (&dimension, &stride))| {
            if dimension == 1 {
                0
            } else {
                batch_index[leading + index] * stride as usize
            }
        })
        .sum()
}

fn next_index(shape: &[usize], index: &mut [usize]) {
    for (dimension, coordinate) in shape.iter().zip(index).rev() {
        *coordinate += 1;
        if *coordinate < *dimension {
            return;
        }
        *coordinate = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::{DeviceId, DevicePtrMut};
    use onnx_runtime_ir::compute_contiguous_strides;

    fn i8(shape: &[usize], values: &[i8]) -> Owned {
        Owned {
            bytes: values.iter().map(|&value| value as u8).collect(),
            shape: shape.to_vec(),
            strides: compute_contiguous_strides(shape),
            dtype: DataType::Int8,
        }
    }

    struct Reference<'a> {
        a: &'a [i32],
        a_shape: &'a [usize],
        a_scales: &'a [f32],
        a_zeros: &'a [i32],
        b: &'a [i32],
        b_shape: &'a [usize],
        b_scales: &'a [f32],
        b_zeros: &'a [i32],
        y_scale: f32,
        y_zero: i32,
        output_dtype: DataType,
    }

    fn reference(input: Reference<'_>) -> Vec<i64> {
        let geometry = Geometry::new(input.a_shape, input.b_shape).unwrap();
        let a_per_row = input.a_scales.len() > 1 || input.a_zeros.len() > 1;
        let b_per_column = input.b_scales.len() > 1 || input.b_zeros.len() > 1;
        let mut batch_index = vec![0; geometry.batch_shape.len()];
        let mut output = Vec::with_capacity(geometry.result_len);
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            for row in 0..geometry.m {
                for column in 0..geometry.n {
                    let a_quant_index = if a_per_row {
                        a_batch * geometry.m + row
                    } else {
                        0
                    };
                    let b_quant_index = if b_per_column {
                        b_batch * geometry.n + column
                    } else {
                        0
                    };
                    let mut product = 0.0f64;
                    for inner in 0..geometry.k {
                        let a_index = a_batch * geometry.m * geometry.k + row * geometry.k + inner;
                        let b_index =
                            b_batch * geometry.k * geometry.n + inner * geometry.n + column;
                        let a = f64::from(input.a[a_index] - input.a_zeros[a_quant_index])
                            * f64::from(input.a_scales[a_quant_index]);
                        let b = f64::from(input.b[b_index] - input.b_zeros[b_quant_index])
                            * f64::from(input.b_scales[b_quant_index]);
                        product += a * b;
                    }
                    let quantized = ((product / f64::from(input.y_scale)).round_ties_even() as i64)
                        .saturating_add(i64::from(input.y_zero));
                    output.push(match input.output_dtype {
                        DataType::Int8 => quantized.clamp(i8::MIN as i64, i8::MAX as i64),
                        DataType::Uint8 => quantized.clamp(0, u8::MAX as i64),
                        _ => unreachable!(),
                    });
                }
            }
            if batch + 1 < geometry.batch_count {
                next_index(&geometry.batch_shape, &mut batch_index);
            }
        }
        output
    }

    fn execute(inputs: [&Owned; 8], output_dtype: DataType, output_shape: &[usize]) -> Owned {
        let mut output = Owned::zeros(output_dtype, output_shape);
        QLinearMatMulKernel::default()
            .execute(&inputs.map(|input| input.view()), &mut [output.view_mut()])
            .unwrap();
        output
    }

    /// The parked accumulator scratch must be bounded across the **whole
    /// process**, not per thread. This is the defect the review named: a per-
    /// thread cap is silent about the `x threads` multiplier, so the real
    /// ceiling was `32 MiB x threads` (1 GiB on a 32-vCPU box).
    ///
    /// A single-instantiation test structurally cannot observe that multiplier
    /// -- that is exactly how #1100 slipped through -- so this drives the kernel
    /// from **many worker threads at once** and checks the summed retention.
    ///
    /// # How to falsify it
    ///
    /// The assertion that fails when the fix is reverted is the process-cap one:
    /// make `GovernedAccumulatorBudget::try_park` ignore `process_cap_bytes`
    /// (the per-thread-only bound the PR replaced) and every one of the N worker
    /// threads parks a full buffer, so `qlinear_accumulator_live_bytes()` climbs
    /// to `N x per_thread_buffer` -- far over the process cap -- and the upper-
    /// bound assertion below fails. Restored, the sum is capped and it passes.
    #[test]
    fn the_parked_accumulator_is_bounded_process_wide_not_per_thread() {
        use std::sync::Mutex;
        // Serialise against any other test that also drives large parked
        // buffers so the reset below opens a clean window; the *upper*-bound
        // assertion holds regardless, being enforced inside `try_park`.
        static SERIALISE: Mutex<()> = Mutex::new(());
        let _guard = SERIALISE
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        // `m * n` just under the 32 MiB per-thread cap so a full buffer is
        // admitted (a buffer *over* the cap is released -- tested separately),
        // while four of them saturate the 128 MiB process cap.
        let (m, k, n) = (2000usize, 64usize, 4096usize);
        let buffer_bytes = (m * n * std::mem::size_of::<i32>()) as u64;
        assert!(
            buffer_bytes < MAX_RETAINED_ACCUMULATOR_BYTES as u64,
            "a single buffer must fit the per-thread cap or nothing parks"
        );

        let threads = 8usize;
        assert!(
            (threads as u64) * buffer_bytes > MAX_PROCESS_ACCUMULATOR_BYTES as u64,
            "the pool must want to park more than the process cap allows, or \
             the bound is never exercised"
        );

        let a = Owned::u8(&[m, k], &vec![130u8; m * k]);
        let a_scale = Owned::f32(&[], &[0.5]);
        let a_zero = Owned::u8(&[], &[128]);
        let b = Owned::u8(&[k, n], &vec![120u8; k * n]);
        let b_scale = Owned::f32(&[], &[0.25]);
        let b_zero = Owned::u8(&[], &[127]);
        let y_scale = Owned::f32(&[], &[64.0]);
        let y_zero = Owned::u8(&[], &[100]);
        let inputs = [
            &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
        ];

        ACCUMULATOR_BUDGET.set_admitted(true);
        ACCUMULATOR_BUDGET.reset_for_test();

        let kernel = QLinearMatMulKernel::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build a dedicated pool");
        // `broadcast` runs the closure once on every worker thread, so each of
        // the `threads` workers runs a full `execute` and parks its own
        // thread-local accumulator -- the multi-thread drive the bound needs.
        pool.broadcast(|_| {
            let mut output = Owned::zeros(DataType::Uint8, &[m, n]);
            kernel
                .execute(&inputs.map(|input| input.view()), &mut [output.view_mut()])
                .expect("kernel runs");
        });

        let live = qlinear_accumulator_live_bytes();

        // The fix: the sum parked across all worker threads is bounded by the
        // process cap, not `threads x per_thread_buffer`.
        assert!(
            live <= MAX_PROCESS_ACCUMULATOR_BYTES as u64,
            "parked accumulator bytes {live} exceeded the process cap {MAX_PROCESS_ACCUMULATOR_BYTES} \
             -- the per-thread bound multiplied by the thread count"
        );
        // Non-vacuity: more than one thread parked, so this genuinely observed
        // the multiplier a single-instantiation test cannot. If the kernel ever
        // stopped parking, this catches it too.
        assert!(
            live >= 2 * buffer_bytes,
            "expected several threads to have parked ({buffer_bytes} bytes each), saw only \
             {live} -- the test is not exercising multi-thread retention"
        );

        ACCUMULATOR_BUDGET.reset_for_test();
    }

    /// A contiguous output must be requantized straight into the caller's
    /// tensor, and a strided one must still be staged and scattered -- with
    /// byte-identical results either way.
    ///
    /// The bytes alone cannot prove this: both routes produce the same answer,
    /// which is the point. So the route counters are asserted too, otherwise a
    /// regression that quietly always staged (re-introducing a `result_len`
    /// allocation, its zero-fill and a copy on every call) would pass.
    #[test]
    fn a_contiguous_output_is_written_in_place_and_a_strided_one_is_staged() {
        let a_values = [10, 14, 7, 20, 3, 11];
        let a = Owned::u8(&[3, 2], &a_values.map(|value| value as u8));
        let a_scale = Owned::f32(&[], &[0.5]);
        let a_zero = Owned::u8(&[], &[8]);
        let b_values = [3, 9, 5, 1];
        let b = Owned::u8(&[2, 2], &b_values.map(|value| value as u8));
        let b_scale = Owned::f32(&[2], &[0.25, 0.5]);
        let b_zero = Owned::u8(&[2], &[2, 4]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[100]);
        let inputs = [
            &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
        ];

        let before_direct = DIRECT_OUTPUT_CALLS.with(std::cell::Cell::get);
        let contiguous = execute(inputs, DataType::Uint8, &[3, 2]);
        assert_eq!(
            DIRECT_OUTPUT_CALLS.with(std::cell::Cell::get),
            before_direct + 1,
            "a contiguous host output must be written in place, not staged"
        );

        // A column-major output view describes the same 3x2 logical tensor but
        // cannot be handed to MLAS, so it must take the staging route.
        let mut strided_bytes = vec![0u8; 6];
        let shape = [3usize, 2];
        let strides = [1i64, 3];
        let before_staged = STAGED_OUTPUT_CALLS.with(std::cell::Cell::get);
        {
            let out = TensorMut::new(
                DevicePtrMut(strided_bytes.as_mut_ptr().cast()),
                DataType::Uint8,
                &shape,
                &strides,
                DeviceId::cpu(),
            );
            QLinearMatMulKernel::default()
                .execute(&inputs.map(|input| input.view()), &mut [out])
                .unwrap();
        }
        assert_eq!(
            STAGED_OUTPUT_CALLS.with(std::cell::Cell::get),
            before_staged + 1,
            "a strided output must be staged, because MLAS cannot scatter"
        );

        // Same logical values, transposed layout: element (row, col) of the
        // strided buffer lives at `row + col * 3`.
        for row in 0..3 {
            for col in 0..2 {
                assert_eq!(
                    strided_bytes[row + col * 3],
                    contiguous.bytes[row * 2 + col],
                    "strided output disagrees with the in-place one at ({row}, {col})"
                );
            }
        }
    }

    /// Borrowing `A` must never let the kernel write through to the caller's
    /// input. The sign-flip route is the one place that rewrites `A`'s bytes,
    /// and it does so through `Cow::to_mut`, which copies a borrowed operand
    /// before touching it. Replace that with a write to the borrowed slice and
    /// this test fails.
    ///
    /// Non-vacuity is asserted too, because the property is only under test
    /// when the call really did borrow `A`: if `dense_bytes` ever stopped
    /// borrowing, the input would trivially survive and the test would pass
    /// while proving nothing. That is asserted on `A` itself rather than on a
    /// count of borrows across all operands, because how many *other* operands
    /// a route borrows is a routing detail this test does not fix.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_sign_flip_route_never_writes_through_to_the_callers_input() {
        // Signed A against unsigned B is the combination that makes a plan flip
        // one operand's sign domain.
        let a_values: Vec<i8> = (0..64).map(|i| (i * 5 - 100) as i8).collect();
        let a = i8(&[8, 8], &a_values);
        let untouched = a.bytes.clone();
        let a_scale = Owned::f32(&[], &[0.5]);
        let a_zero = i8(&[], &[-3]);
        let b_values: Vec<u8> = (0..64).map(|i| ((i * 11 + 3) % 256) as u8).collect();
        let b = Owned::u8(&[8, 8], &b_values);
        let b_scale = Owned::f32(&[], &[0.25]);
        let b_zero = Owned::u8(&[], &[130]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = i8(&[], &[-2]);
        assert!(
            matches!(dense_bytes(&a.view()).unwrap(), Cow::Borrowed(_)),
            "A is no longer borrowed, so this test proves nothing"
        );
        let borrows_before = BORROWED_INPUT_CALLS.with(|calls| calls.get());
        let _ = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[8, 8],
        );
        assert!(
            BORROWED_INPUT_CALLS.with(|calls| calls.get()) > borrows_before,
            "the flip route borrowed nothing, so this test proves nothing"
        );
        assert_eq!(
            a.bytes, untouched,
            "the kernel mutated the caller's A buffer in place"
        );
    }

    /// A batched call writes each batch at its own offset. In place there is no
    /// running `Vec::len` to lean on, so an off-by-one in the offset arithmetic
    /// would scribble batches over each other -- which a single-batch test
    /// cannot see.
    #[test]
    fn a_batched_call_lands_every_batch_at_its_own_offset_in_place() {
        let a_values: Vec<u8> = (0..3 * 2 * 4).map(|i| ((i * 7 + 1) % 251) as u8).collect();
        let a = Owned::u8(&[3, 2, 4], &a_values);
        let a_scale = Owned::f32(&[], &[0.5]);
        let a_zero = Owned::u8(&[], &[8]);
        let b_values: Vec<u8> = (0..3 * 4 * 5).map(|i| ((i * 13 + 5) % 251) as u8).collect();
        let b = Owned::u8(&[3, 4, 5], &b_values);
        let b_scale = Owned::f32(&[], &[0.25]);
        let b_zero = Owned::u8(&[], &[7]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[100]);
        let batched = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[3, 2, 5],
        );

        // Each batch computed on its own must equal that batch's slice.
        for batch in 0..3 {
            let a_slice = Owned::u8(&[2, 4], &a_values[batch * 8..batch * 8 + 8]);
            let b_slice = Owned::u8(&[4, 5], &b_values[batch * 20..batch * 20 + 20]);
            let single = execute(
                [
                    &a_slice, &a_scale, &a_zero, &b_slice, &b_scale, &b_zero, &y_scale, &y_zero,
                ],
                DataType::Uint8,
                &[2, 5],
            );
            assert_eq!(
                &batched.bytes[batch * 10..batch * 10 + 10],
                &single.bytes[..],
                "batch {batch} landed at the wrong offset"
            );
        }
    }

    fn output_values(output: &Owned) -> Vec<i64> {
        match output.dtype {
            DataType::Int8 => output
                .bytes
                .iter()
                .map(|&value| i64::from(value as i8))
                .collect(),
            DataType::Uint8 => output.bytes.iter().map(|&value| i64::from(value)).collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn qlinear_matmul_uint8_per_tensor_matches_dequant_matmul_requant_reference() {
        let a = Owned::u8(&[2, 3], &[130, 125, 140, 120, 135, 128]);
        let a_scale = Owned::f32(&[], &[0.25]);
        let a_zero = Owned::u8(&[], &[128]);
        let b = Owned::u8(&[3, 2], &[131, 126, 120, 140, 128, 130]);
        let b_scale = Owned::f32(&[], &[0.5]);
        let b_zero = Owned::u8(&[], &[128]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[127]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &[130, 125, 140, 120, 135, 128],
            a_shape: &[2, 3],
            a_scales: &[0.25],
            a_zeros: &[128],
            b: &[131, 126, 120, 140, 128, 130],
            b_shape: &[3, 2],
            b_scales: &[0.5],
            b_zeros: &[128],
            y_scale: 0.125,
            y_zero: 127,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_int8_per_column_scales_matches_reference() {
        let a = i8(&[1, 2], &[-2, 5]);
        let a_scale = Owned::f32(&[], &[0.25]);
        let a_zero = i8(&[], &[-1]);
        let b = i8(&[2, 3], &[3, -4, 7, 2, 5, -6]);
        let b_scale = Owned::f32(&[3], &[0.5, 0.25, 0.125]);
        let b_zero = i8(&[3], &[1, -2, 3]);
        let y_scale = Owned::f32(&[], &[0.25]);
        let y_zero = i8(&[], &[2]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[1, 3],
        );
        let expected = reference(Reference {
            a: &[-2, 5],
            a_shape: &[1, 2],
            a_scales: &[0.25],
            a_zeros: &[-1],
            b: &[3, -4, 7, 2, 5, -6],
            b_shape: &[2, 3],
            b_scales: &[0.5, 0.25, 0.125],
            b_zeros: &[1, -2, 3],
            y_scale: 0.25,
            y_zero: 2,
            output_dtype: DataType::Int8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_uint8_per_row_a_scales_matches_reference() {
        let a_values = [10, 14, 7, 20];
        let a = Owned::u8(&[2, 2], &a_values.map(|value| value as u8));
        let a_scale = Owned::f32(&[2], &[0.5, 0.125]);
        let a_zero = Owned::u8(&[2], &[8, 6]);
        let b_values = [3, 9, 5, 1];
        let b = Owned::u8(&[2, 2], &b_values.map(|value| value as u8));
        let b_scale = Owned::f32(&[2], &[0.25, 0.5]);
        let b_zero = Owned::u8(&[2], &[2, 4]);
        let y_scale = Owned::f32(&[], &[0.125]);
        let y_zero = Owned::u8(&[], &[100]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[2, 2],
            a_scales: &[0.5, 0.125],
            a_zeros: &[8, 6],
            b: &b_values,
            b_shape: &[2, 2],
            b_scales: &[0.25, 0.5],
            b_zeros: &[2, 4],
            y_scale: 0.125,
            y_zero: 100,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    #[test]
    fn qlinear_matmul_batched_per_row_and_per_column_broadcasts_match_reference() {
        let a_values = [12, 8, 7, 15, 5, 20, 9, 4];
        let a = Owned::u8(&[2, 2, 2], &a_values.map(|value| value as u8));
        let a_scale = Owned::f32(&[2, 2, 1], &[0.5, 0.25, 0.125, 0.75]);
        let a_zero = Owned::u8(&[2, 2, 1], &[10, 8, 6, 5]);
        let b_values = [3, -4, 6, 2];
        let b = i8(&[1, 2, 2], &b_values);
        let b_scale = Owned::f32(&[1, 1, 2], &[0.5, 0.25]);
        let b_zero = i8(&[1, 1, 2], &[1, -2]);
        let y_scale = Owned::f32(&[1], &[0.125]);
        let y_zero = Owned::u8(&[1], &[120]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2, 2],
        );
        let a_scales = [0.5, 0.25, 0.125, 0.75];
        let a_zeros = [10, 8, 6, 5];
        let b_scales = [0.5, 0.25];
        let b_zeros = [1, -2];
        let mut expected = Vec::with_capacity(8);
        for batch in 0..2 {
            for row in 0..2 {
                for column in 0..2 {
                    let mut product = 0.0f64;
                    for inner in 0..2 {
                        let a_index = batch * 4 + row * 2 + inner;
                        let b_index = inner * 2 + column;
                        let a = f64::from(a_values[a_index] - a_zeros[batch * 2 + row])
                            * a_scales[batch * 2 + row];
                        let b = f64::from(b_values[b_index] - b_zeros[column]) * b_scales[column];
                        product += a * b;
                    }
                    expected.push(((product / 0.125).round_ties_even() as i64 + 120).clamp(0, 255));
                }
            }
        }
        assert_eq!(expected, vec![108, 108, 153, 135, 154, 134, 129, 102]);
        assert_eq!(output_values(&out), expected);
    }

    /// The fused MLAS requantizer runs per output tile, so a width that is not
    /// a multiple of its 16-column block exercises the tail path that a
    /// round-numbered shape would hide. The bytes must still be exactly the
    /// dequantize/matmul/requantize reference.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_fused_requantize_matches_the_reference_on_a_tail_shaped_output() {
        let (m, k, n) = (3usize, 5usize, 19usize);
        let a_values: Vec<i32> = (0..m * k).map(|i| (i * 37 % 251) as i32).collect();
        let b_values: Vec<i32> = (0..k * n).map(|i| (i * 53 % 241) as i32).collect();
        let a = Owned::u8(
            &[m, k],
            &a_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let b = Owned::u8(
            &[k, n],
            &b_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let b_scales: Vec<f32> = (0..n).map(|i| 0.01 * (1.0 + i as f32 * 0.37)).collect();
        let b_zeros: Vec<i32> = (0..n).map(|i| (i * 11 % 97) as i32).collect();
        let a_scale = Owned::f32(&[], &[0.03]);
        let a_zero = Owned::u8(&[], &[128]);
        let b_scale = Owned::f32(&[n], &b_scales);
        let b_zero = Owned::u8(&[n], &b_zeros.iter().map(|&v| v as u8).collect::<Vec<_>>());
        let y_scale = Owned::f32(&[], &[0.5]);
        let y_zero = Owned::u8(&[], &[100]);

        // This is what routes the call: without a `Some` here the assertion
        // below would only be re-testing the scalar loop.
        let quant = QuantParams::load(
            "a",
            &a_scale.view(),
            &a_zero.view(),
            &[m, k],
            QuantAxis::Row,
        )
        .unwrap();
        let mut buffer = Vec::new();
        assert!(
            matches!(
                fused_scale(&quant, 0, &b_scales, 0.5, &mut buffer),
                Some(mlas_sys::QgemmScale::PerColumn(_))
            ),
            "this shape must take the fused path or the test proves nothing"
        );

        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[m, n],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[m, k],
            a_scales: &[0.03],
            a_zeros: &[128],
            b: &b_values,
            b_shape: &[k, n],
            b_scales: &b_scales,
            b_zeros: &b_zeros,
            y_scale: 0.5,
            y_zero: 100,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    /// The per-tensor route is the one an M=1 decode takes, and it is the one
    /// the split into `PerTensor`/`PerColumn` could silently get wrong, so it
    /// gets its own bit-exact end-to-end check rather than riding on the
    /// per-column test.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_fused_requantize_matches_the_reference_with_one_scale_for_the_whole_tensor() {
        let (m, k, n) = (4usize, 6usize, 7usize);
        let a_values: Vec<i32> = (0..m * k).map(|i| (i * 29 % 253) as i32).collect();
        let b_values: Vec<i32> = (0..k * n).map(|i| (i * 61 % 239) as i32).collect();
        let a = Owned::u8(
            &[m, k],
            &a_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let b = Owned::u8(
            &[k, n],
            &b_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let a_scale = Owned::f32(&[], &[0.02]);
        let a_zero = Owned::u8(&[], &[130]);
        let b_scale = Owned::f32(&[], &[0.03]);
        let b_zero = Owned::u8(&[], &[120]);
        let y_scale = Owned::f32(&[], &[0.25]);
        let y_zero = Owned::u8(&[], &[96]);

        let quant = QuantParams::load(
            "a",
            &a_scale.view(),
            &a_zero.view(),
            &[m, k],
            QuantAxis::Row,
        )
        .unwrap();
        let mut buffer = Vec::new();
        assert!(
            matches!(
                fused_scale(&quant, 0, &vec![0.03; n], 0.25, &mut buffer),
                Some(mlas_sys::QgemmScale::PerTensor(_))
            ),
            "this fixture must take the per-tensor fused path or it proves nothing"
        );

        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[m, n],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[m, k],
            a_scales: &[0.02],
            a_zeros: &[130],
            b: &b_values,
            b_shape: &[k, n],
            b_scales: &[0.03],
            b_zeros: &[120],
            y_scale: 0.25,
            y_zero: 96,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(output_values(&out), expected);
    }

    /// Signed output takes a different MLAS pack step and a different saturating
    /// narrow, and `i8` is the half of the range the unsigned tests cannot see.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_fused_requantize_matches_the_reference_for_signed_output() {
        let (m, k, n) = (3usize, 5usize, 9usize);
        let a_values: Vec<i32> = (0..m * k).map(|i| (i * 31 % 251) as i32 - 125).collect();
        let b_values: Vec<i32> = (0..k * n).map(|i| (i * 47 % 251) as i32 - 125).collect();
        let a = i8(
            &[m, k],
            &a_values.iter().map(|&v| v as i8).collect::<Vec<_>>(),
        );
        let b = i8(
            &[k, n],
            &b_values.iter().map(|&v| v as i8).collect::<Vec<_>>(),
        );
        let b_scales: Vec<f32> = (0..n).map(|i| 0.02 * (1.0 + i as f32 * 0.11)).collect();
        let b_zeros: Vec<i32> = (0..n).map(|i| (i * 7 % 61) as i32 - 30).collect();
        let a_scale = Owned::f32(&[], &[0.04]);
        let a_zero = i8(&[], &[-7]);
        let b_scale = Owned::f32(&[n], &b_scales);
        let b_zero = i8(&[n], &b_zeros.iter().map(|&v| v as i8).collect::<Vec<_>>());
        let y_scale = Owned::f32(&[], &[0.5]);
        let y_zero = i8(&[], &[-11]);

        let quant = QuantParams::load(
            "a",
            &a_scale.view(),
            &a_zero.view(),
            &[m, k],
            QuantAxis::Row,
        )
        .unwrap();
        let mut buffer = Vec::new();
        assert!(
            fused_scale(&quant, 0, &b_scales, 0.5, &mut buffer).is_some(),
            "signed output must reach the fused path"
        );

        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[m, n],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[m, k],
            a_scales: &[0.04],
            a_zeros: &[-7],
            b: &b_values,
            b_shape: &[k, n],
            b_scales: &b_scales,
            b_zeros: &b_zeros,
            y_scale: 0.5,
            y_zero: -11,
            output_dtype: DataType::Int8,
        });
        assert_eq!(output_values(&out), expected);
    }

    /// A *finite* combined scale can still push the product past `i64::MAX`.
    /// MLAS clamps in `f32`, where there is no such cliff; the scalar loop has
    /// to saturate to stay bit-identical instead of wrapping to the opposite
    /// end of the range (and panicking in a debug build on the way).  The
    /// per-row `a_scale` keeps this on the scalar loop, which is the path with
    /// the cliff.
    #[test]
    fn a_finite_scale_past_the_i64_range_saturates_on_the_scalar_loop() {
        let a_values = [10, 200, 7, 20];
        let b_values = [3, 9, 5, 1];
        let a = Owned::u8(&[2, 2], &a_values.map(|v| v as u8));
        let b = Owned::u8(&[2, 2], &b_values.map(|v| v as u8));
        let a_scale = Owned::f32(&[2], &[2.0e8, 2.0e8]);
        let a_zero = Owned::u8(&[2], &[8, 8]);
        let b_scale = Owned::f32(&[], &[1.0e8]);
        let b_zero = Owned::u8(&[], &[2]);
        let y_scale = Owned::f32(&[], &[1.0]);
        let y_zero = Owned::u8(&[], &[100]);
        let combined = 2.0e8f32 * 1.0e8 / 1.0;
        assert!(
            combined.is_finite() && (578.0f32 * combined) > 9.3e18,
            "the fixture must stay finite and still exceed the i64 range"
        );

        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[2, 2],
            a_scales: &[2.0e8, 2.0e8],
            a_zeros: &[8, 8],
            b: &b_values,
            b_shape: &[2, 2],
            b_scales: &[1.0e8],
            b_zeros: &[2],
            y_scale: 1.0,
            y_zero: 100,
            output_dtype: DataType::Uint8,
        });
        assert_eq!(expected, [255, 0, 255, 0]);
        assert_eq!(output_values(&out), expected);
    }

    /// MLAS clamps the float before rounding, which agrees with the scalar loop
    /// for every finite scale but maps `NaN` to the output minimum where the
    /// scalar loop maps it to the zero point. Individual scales are already
    /// validated as finite and positive, but their *product* can still overflow
    /// to infinity, so the guard is on the combined scale -- and one overflowing
    /// column has to take the whole call off the fused path, leaving the finite
    /// columns exactly what the scalar loop produced before this path existed.
    #[cfg(feature = "mlas")]
    #[test]
    fn a_non_finite_combined_scale_stays_on_the_scalar_loop() {
        let a_values = [10, 200, 7, 20];
        let b_values = [3, 9, 5, 1];
        let a = Owned::u8(&[2, 2], &a_values.map(|v| v as u8));
        let b = Owned::u8(&[2, 2], &b_values.map(|v| v as u8));
        // `3e8 * 3e30` overflows f32 even though each factor is a legal scale
        // on its own, while the second column stays finite at 0.075 -- so this
        // also pins down that one bad column takes the *whole* call off the
        // fused path instead of only its own column.
        let a_scale = Owned::f32(&[], &[3.0e8]);
        let a_zero = Owned::u8(&[], &[8]);
        let b_scales = [3.0e30f32, 0.25];
        assert!(
            (3.0e8f32 * b_scales[0]).is_infinite(),
            "column 0 must overflow"
        );
        assert!(
            (3.0e8f32 * b_scales[1] / 1.0e9).is_finite(),
            "column 1 must not"
        );
        let b_scale = Owned::f32(&[2], &b_scales);
        let b_zero = Owned::u8(&[2], &[2, 4]);
        let y_scale = Owned::f32(&[], &[1.0e9]);
        let y_zero = Owned::u8(&[], &[100]);

        let quant = QuantParams::load(
            "a",
            &a_scale.view(),
            &a_zero.view(),
            &[2, 2],
            QuantAxis::Row,
        )
        .unwrap();
        let mut buffer = Vec::new();
        assert!(
            fused_scale(&quant, 0, &b_scales, 1.0e9, &mut buffer).is_none(),
            "an infinite combined scale must be declined"
        );

        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[2, 2],
        );
        let expected = reference(Reference {
            a: &a_values,
            a_shape: &[2, 2],
            a_scales: &[3.0e8],
            a_zeros: &[8],
            b: &b_values,
            b_shape: &[2, 2],
            b_scales: &b_scales,
            b_zeros: &[2, 4],
            y_scale: 1.0e9,
            y_zero: 100,
            output_dtype: DataType::Uint8,
        });
        // The finite column must still carry real values, or the comparison
        // would be two saturated columns agreeing about nothing.
        assert_eq!(expected, [255, 58, 255, 97]);
        assert_eq!(output_values(&out), expected);
    }

    /// A per-tensor `b_scale` is splatted across the column vector before it
    /// reaches the requantizer. Folding it back to one number is what lets MLAS
    /// broadcast instead of walking an `n`-long array, so the fold has to
    /// recognise the splat -- and must not mistake a genuinely varying column
    /// scale for one.
    #[cfg(feature = "mlas")]
    #[test]
    fn fused_scale_folds_a_splatted_column_scale_and_keeps_a_varying_one() {
        let quant = QuantParams {
            scales: vec![0.5],
            zero_points: vec![0],
            axis_len: 1,
            per_axis: false,
        };
        let mut buffer = Vec::new();
        match fused_scale(&quant, 0, &[0.25, 0.25, 0.25], 0.5, &mut buffer) {
            Some(mlas_sys::QgemmScale::PerTensor(scale)) => assert_eq!(scale, 0.25),
            other => panic!(
                "a splatted column scale must fold to one number, got {other:?}",
                other = match other {
                    Some(mlas_sys::QgemmScale::PerColumn(values)) =>
                        format!("PerColumn({values:?})"),
                    Some(mlas_sys::QgemmScale::PerTensor(value)) => format!("PerTensor({value})"),
                    None => "None".to_string(),
                }
            ),
        }
        let mut buffer = Vec::new();
        match fused_scale(&quant, 0, &[0.25, 0.5, 0.25], 0.5, &mut buffer) {
            Some(mlas_sys::QgemmScale::PerColumn(values)) => {
                assert_eq!(values, [0.25, 0.5, 0.25]);
            }
            _ => panic!("a varying column scale must stay per column"),
        }
    }

    /// A per-row `a_scale` cannot be expressed as one scale per column, so it
    /// must never reach the fused path -- the same restriction that keeps it
    /// off MLAS's integer GEMM at all.
    #[cfg(feature = "mlas")]
    #[test]
    fn fused_scale_declines_a_per_row_activation_scale() {
        let quant = QuantParams {
            scales: vec![0.5, 0.25],
            zero_points: vec![0, 0],
            axis_len: 2,
            per_axis: true,
        };
        let mut buffer = Vec::new();
        assert!(fused_scale(&quant, 0, &[0.25, 0.25], 0.5, &mut buffer).is_none());
    }

    #[test]
    fn qlinear_matmul_rounds_ties_to_even_and_saturates_int8() {
        let a_values = [1, 1, 1, 1];
        let a = i8(&[1, 4], &a_values);
        let a_scale = Owned::f32(&[], &[1.0]);
        let a_zero = i8(&[], &[0]);
        let b_values = [
            1, 3, 127, -128, 0, 0, 127, -128, 0, 0, 127, -128, 0, 0, 127, -128,
        ];
        let b = i8(&[4, 4], &b_values);
        let b_scale = Owned::f32(&[], &[1.0]);
        let b_zero = i8(&[], &[0]);
        let y_scale = Owned::f32(&[], &[2.0]);
        let y_zero = i8(&[], &[0]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Int8,
            &[1, 4],
        );
        let expected = reference(Reference {
            a: &a_values.map(i32::from),
            a_shape: &[1, 4],
            a_scales: &[1.0],
            a_zeros: &[0],
            b: &b_values.map(i32::from),
            b_shape: &[4, 4],
            b_scales: &[1.0],
            b_zeros: &[0],
            y_scale: 2.0,
            y_zero: 0,
            output_dtype: DataType::Int8,
        });
        assert_eq!(output_values(&out), expected);
        assert_eq!(expected, vec![0, 2, 127, -128]);
    }

    #[test]
    fn qlinear_matmul_rejects_mismatched_scale_and_zero_point_shapes() {
        let a = Owned::u8(&[2, 2], &[1, 2, 3, 4]);
        let a_scale = Owned::f32(&[2], &[0.5, 0.25]);
        let a_zero = Owned::u8(&[], &[0]);
        let b = Owned::u8(&[2, 1], &[1, 1]);
        let b_scale = Owned::f32(&[], &[1.0]);
        let b_zero = Owned::u8(&[], &[0]);
        let y_scale = Owned::f32(&[], &[1.0]);
        let y_zero = Owned::u8(&[], &[0]);
        let mut out = Owned::zeros(DataType::Uint8, &[2, 1]);
        let error = QLinearMatMulKernel::default()
            .execute(
                &[
                    a.view(),
                    a_scale.view(),
                    a_zero.view(),
                    b.view(),
                    b_scale.view(),
                    b_zero.view(),
                    y_scale.view(),
                    y_zero.view(),
                ],
                &mut [out.view_mut()],
            )
            .unwrap_err();
        assert!(error.to_string().contains("shapes must match"), "{error}");
    }

    /// Transcription of the accumulation loop this change replaced, kept as the
    /// oracle for bit-identity. The shared `reference` helper sums in `f64` and
    /// so cannot answer "did we reproduce the *previous kernel* exactly", which
    /// is the property that matters once the `i32` accumulator can wrap.
    ///
    /// Mirrors the removed code: per-`(row, column)` accumulation over `k`,
    /// both zero points subtracted inside the loop, `wrapping_add`, then the
    /// unchanged scale / `round_ties_even` / clamp epilogue.
    #[allow(clippy::too_many_arguments)]
    fn previous_loop_oracle(
        a: &[i32],
        b: &[i32],
        m: usize,
        k: usize,
        n: usize,
        a_scales: &[f32],
        a_zeros: &[i32],
        b_scales: &[f32],
        b_zeros: &[i32],
        y_scale: f32,
        y_zero: i32,
        output_dtype: DataType,
    ) -> Vec<i64> {
        let pick = |values: &[f32], index: usize| values[if values.len() > 1 { index } else { 0 }];
        let pick_zero =
            |values: &[i32], index: usize| values[if values.len() > 1 { index } else { 0 }];
        let mut out = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let a_scale = pick(a_scales, row);
                let a_zero_point = pick_zero(a_zeros, row);
                let b_scale = pick(b_scales, column);
                let b_zero_point = pick_zero(b_zeros, column);
                let mut accumulated = 0i32;
                for inner in 0..k {
                    let av = a[row * k + inner] - a_zero_point;
                    let bv = b[inner * n + column] - b_zero_point;
                    accumulated = accumulated.wrapping_add(av * bv);
                }
                let scale = a_scale * b_scale / y_scale;
                let value =
                    (accumulated as f32 * scale).round_ties_even() as i64 + i64::from(y_zero);
                out.push(match output_dtype {
                    DataType::Int8 => value.clamp(i8::MIN as i64, i8::MAX as i64),
                    DataType::Uint8 => value.clamp(0, u8::MAX as i64),
                    _ => unreachable!(),
                });
            }
        }
        out
    }

    /// The reordered accumulation must be **bit**-identical to the loop it
    /// replaces, not merely close: `QLinearMatMul` output is integer, so a
    /// one-LSB difference is a wrong answer.
    ///
    /// The rewrite lifts the zero points out of the inner loop using
    ///   sum_k (a_k - az) * (b_kn - bz_n)
    ///     = sum_k (a_k - az) * b_kn  -  bz_n * sum_k (a_k - az)
    /// which is an identity over the integers, so under wrapping arithmetic
    /// (arithmetic mod 2^32) both sides reduce to the same `i32`.
    ///
    /// Compared against a transcription of the previous loop over shapes that
    /// miss every tile boundary, both dtypes, and per-tensor as well as
    /// per-axis (per-row `A`, per-column `B`) quantization. Overflow of the
    /// accumulator is covered separately by
    /// `qlinear_matmul_overflowing_accumulator_matches_the_previous_loop`,
    /// which needs a much larger `K` than is practical to sweep here.
    /// The MLAS route `continue`s out of the batch loop before the fallback's
    /// index bookkeeping, so a multi-batch model exercises an advancement path
    /// no per-tensor test reached: every other batched case uses per-row `a`
    /// quantization and therefore stays on the fallback.
    #[test]
    fn qlinear_matmul_batched_per_tensor_uint8_matches_reference() {
        let (batches, m, k, n) = (3usize, 2usize, 4usize, 3usize);
        let a_values: Vec<i32> = (0..batches * m * k)
            .map(|index| ((index * 37 + 11) % 200 + 8) as i32)
            .collect();
        let b_values: Vec<i32> = (0..batches * k * n)
            .map(|index| ((index * 53 + 5) % 200 + 12) as i32)
            .collect();
        let a = Owned::u8(
            &[batches, m, k],
            &a_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let b = Owned::u8(
            &[batches, k, n],
            &b_values.iter().map(|&v| v as u8).collect::<Vec<_>>(),
        );
        let a_scale = Owned::f32(&[], &[0.03]);
        let a_zero = Owned::u8(&[], &[100]);
        let b_scale = Owned::f32(&[], &[0.02]);
        let b_zero = Owned::u8(&[], &[110]);
        let y_scale = Owned::f32(&[], &[0.25]);
        let y_zero = Owned::u8(&[], &[128]);
        let out = execute(
            [
                &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
            ],
            DataType::Uint8,
            &[batches, m, n],
        );

        let mut expected = Vec::with_capacity(batches * m * n);
        for batch in 0..batches {
            for row in 0..m {
                for column in 0..n {
                    let mut product = 0.0f64;
                    for inner in 0..k {
                        let a = f64::from(a_values[batch * m * k + row * k + inner] - 100) * 0.03;
                        let b =
                            f64::from(b_values[batch * k * n + inner * n + column] - 110) * 0.02;
                        product += a * b;
                    }
                    expected.push(((product / 0.25).round_ties_even() as i64 + 128).clamp(0, 255));
                }
            }
        }
        assert_eq!(output_values(&out), expected);
        assert_eq!(expected.len(), batches * m * n);
    }

    #[cfg(feature = "mlas")]
    fn quant_params(per_axis: bool, zero_point: i32, axis_len: usize) -> QuantParams {
        QuantParams {
            scales: vec![0.02; if per_axis { axis_len } else { 1 }],
            zero_points: vec![zero_point; if per_axis { axis_len } else { 1 }],
            axis_len,
            per_axis,
        }
    }

    /// The MLAS integer-GEMM route must be taken for the ordinary `u8 x u8`
    /// case -- otherwise every parity test above silently exercises only the
    /// fallback and proves nothing about the fast path.
    #[cfg(feature = "mlas")]
    #[test]
    fn qgemm_plan_is_selected_for_the_ordinary_uint8_case() {
        let geometry = Geometry::new(&[4, 32], &[32, 8]).unwrap();
        assert!(
            QgemmPlan::select(&quant_params(false, 128, 4), false, false, &geometry).is_some(),
            "u8 activations against u8 weights with a per-tensor zero point is \
             the case the binding exists for"
        );
    }

    /// Every decline below is a correctness rule, not a tuning knob: MLAS takes
    /// a single `ZeroPointA`, and a zero point that does not fit the operand
    /// dtype cannot be truncated.
    #[cfg(feature = "mlas")]
    #[test]
    fn qgemm_plan_declines_every_case_it_cannot_reproduce_exactly() {
        let geometry = Geometry::new(&[4, 32], &[32, 8]).unwrap();
        assert!(
            QgemmPlan::select(&quant_params(true, 128, 4), false, false, &geometry).is_none(),
            "per-row activation zero points have no MLAS equivalent"
        );
        assert!(
            QgemmPlan::select(&quant_params(false, 300, 4), false, false, &geometry).is_none(),
            "a zero point that does not fit the operand dtype must fall back \
             rather than be truncated"
        );
        let empty = Geometry::new(&[0, 32], &[32, 8]).unwrap();
        assert!(
            QgemmPlan::select(&quant_params(false, 128, 4), false, false, &empty).is_none(),
            "an empty shape has nothing to hand MLAS"
        );
    }

    /// Every signedness combination must produce a plan, and every combination
    /// MLAS has no kernel for must be reached by translating the offending
    /// operand into the unsigned domain rather than by declining.
    ///
    /// This is the non-vacuity guard for
    /// `qlinear_matmul_reordered_accumulation_is_bit_identical`: without it
    /// the `i8` half of that sweep could silently be proving only that the
    /// scalar fallback equals itself.
    #[cfg(feature = "mlas")]
    #[test]
    fn signed_operands_are_translated_rather_than_declined() {
        let geometry = Geometry::new(&[4, 32], &[32, 8]).unwrap();
        let signed_activations_native = cfg!(target_arch = "aarch64");
        let cases = [
            // (a_signed, b_signed, expected flip_a, expected flip_b)
            (false, false, false, false),
            (false, true, false, !mlas_sys::qgemm_u8s8_is_exact()),
            (
                true,
                true,
                !signed_activations_native,
                !signed_activations_native,
            ),
            (true, false, true, false),
        ];
        for (a_signed, b_signed, flip_a, flip_b) in cases {
            // The zero point has to be legal in the operand's *own* dtype;
            // that is what the flip then has to carry across.
            let zero_point = if a_signed { -1 } else { 128 };
            let plan = QgemmPlan::select(
                &quant_params(false, zero_point, 4),
                a_signed,
                b_signed,
                &geometry,
            )
            .unwrap_or_else(|| panic!("a_signed={a_signed} b_signed={b_signed} must yield a plan"));
            assert_eq!(
                (plan.flip_a, plan.flip_b),
                (flip_a, flip_b),
                "a_signed={a_signed} b_signed={b_signed}"
            );
            assert_eq!(
                (plan.a_signed, plan.b_signed),
                (a_signed && !flip_a, b_signed && !flip_b),
                "a flipped operand must be handed to MLAS as unsigned"
            );
        }
    }

    /// Returns whether this kernel is holding a built pack.
    #[cfg(all(test, feature = "mlas"))]
    fn qlinear_pack_state(kernel: &QLinearMatMulKernel) -> Option<bool> {
        kernel.packed_b.get().map(|slot| slot.is_some())
    }

    #[cfg(feature = "mlas")]
    struct PackFixture {
        a: Owned,
        b: Owned,
        m: usize,
        n: usize,
    }

    /// Shapes big enough that MLAS actually accepts a pack, with values that
    /// span the whole byte range so a wrong pack cannot coincidentally agree.
    #[cfg(feature = "mlas")]
    fn pack_fixture(m: usize, k: usize, n: usize, seed: usize) -> PackFixture {
        PackFixture {
            a: Owned::u8(
                &[m, k],
                &(0..m * k)
                    .map(|i| ((i * 37 + seed * 11) % 256) as u8)
                    .collect::<Vec<_>>(),
            ),
            b: Owned::u8(
                &[k, n],
                &(0..k * n)
                    .map(|i| ((i * 91 + seed * 53 + 7) % 256) as u8)
                    .collect::<Vec<_>>(),
            ),
            m,
            n,
        }
    }

    /// Run the kernel into a caller-owned output, so a benchmark can separate
    /// the kernel's cost from the cost of allocating its result.
    #[cfg(feature = "mlas")]
    fn run_into(kernel: &QLinearMatMulKernel, fixture: &PackFixture, output: &mut Owned) {
        let (a_scale, a_zero) = (Owned::f32(&[], &[0.02]), Owned::u8(&[], &[128]));
        let (b_scale, b_zero) = (Owned::f32(&[], &[0.01]), Owned::u8(&[], &[127]));
        let (y_scale, y_zero) = (Owned::f32(&[], &[0.05]), Owned::u8(&[], &[120]));
        let inputs = [
            fixture.a.view(),
            a_scale.view(),
            a_zero.view(),
            fixture.b.view(),
            b_scale.view(),
            b_zero.view(),
            y_scale.view(),
            y_zero.view(),
        ];
        kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
    }

    #[cfg(feature = "mlas")]
    fn run_with(kernel: &QLinearMatMulKernel, fixture: &PackFixture) -> Owned {
        let mut output = Owned::zeros(DataType::Uint8, &[fixture.m, fixture.n]);
        let (a_scale, a_zero) = (Owned::f32(&[], &[0.02]), Owned::u8(&[], &[128]));
        let (b_scale, b_zero) = (Owned::f32(&[], &[0.01]), Owned::u8(&[], &[127]));
        let (y_scale, y_zero) = (Owned::f32(&[], &[0.05]), Owned::u8(&[], &[120]));
        let inputs = [
            fixture.a.view(),
            a_scale.view(),
            a_zero.view(),
            fixture.b.view(),
            b_scale.view(),
            b_zero.view(),
            y_scale.view(),
            y_zero.view(),
        ];
        kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
        output
    }

    /// A constant weight must be packed exactly once and then reused, and the
    /// packed answer must equal the unpacked one bit for bit.
    ///
    /// Without the reuse this kernel re-packed and re-copied the whole `k * n`
    /// weight on every call, which is what made decode 20x slower than ONNX
    /// Runtime.
    #[cfg(feature = "mlas")]
    #[test]
    fn a_constant_weight_is_packed_once_and_reused() {
        let fixture = pack_fixture(3, 64, 96, 1);

        let mut unpacked = QLinearMatMulKernel::default();
        unpacked.set_constant_inputs(&[false; 8]);
        let reference = run_with(&unpacked, &fixture);
        assert!(
            qlinear_pack_state(&unpacked).is_none(),
            "a non-constant weight must never be packed: it can change under us"
        );

        let mut packed = QLinearMatMulKernel::default();
        packed.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let first = run_with(&packed, &fixture);
        assert_eq!(
            qlinear_pack_state(&packed),
            Some(true),
            "MLAS must accept a pack for this shape, or the rest of this test \
             proves nothing"
        );
        let second = run_with(&packed, &fixture);

        assert_eq!(first.bytes, reference.bytes, "first packed call");
        assert_eq!(second.bytes, reference.bytes, "reused packed call");
    }

    /// Splits a steady-state call into "inside MLAS" and "everything else", so
    /// a thread-scaling gap can be attributed rather than guessed at.
    ///
    /// `#[ignore]`; run with
    /// `ONNX_GENAI_MLAS_THREADPOOL_THREADS=8 cargo test --release --features mlas \
    ///  qlinear_phase_report -- --ignored --nocapture`.
    ///
    /// `NXRT_QL_KN` picks the (square) K=N, `NXRT_QL_M` a comma-separated list
    /// of M values, so the same report can be pointed at whichever shape a
    /// model actually runs instead of only the built-in default.
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "reports timings; not a pass/fail assertion"]
    fn qlinear_phase_report() {
        let kn: usize = std::env::var("NXRT_QL_KN")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3584);
        let (k, n) = (kn, kn);
        let ms: Vec<usize> = std::env::var("NXRT_QL_M")
            .ok()
            .and_then(|value| {
                value
                    .split(',')
                    .map(|part| part.trim().parse().ok())
                    .collect::<Option<Vec<usize>>>()
            })
            .unwrap_or_else(|| vec![1, 128, 512]);
        for m in ms {
            let fixture = pack_fixture(m, k, n, 1);
            let mut kernel = QLinearMatMulKernel::default();
            kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
            let reps = if m == 1 { 100 } else { 10 };
            for _ in 0..3 {
                let _ = run_with(&kernel, &fixture);
            }
            let start = std::time::Instant::now();
            for _ in 0..reps {
                let _ = run_with(&kernel, &fixture);
            }
            let whole = start.elapsed() / reps;

            // Same kernel call, but the caller's output tensor is allocated
            // once and reused. `whole - whole_reused` is what per-call
            // allocation of the result costs, which is the part a real session
            // pays through ORT's arena rather than through `Owned::zeros`.
            let mut reused = Owned::zeros(DataType::Uint8, &[m, n]);
            let start = std::time::Instant::now();
            for _ in 0..reps {
                run_into(&kernel, &fixture, &mut reused);
            }
            let whole_reused = start.elapsed() / reps;

            let a_bytes = fixture.a.bytes.clone();
            let packed = kernel
                .pack_lookup(
                    kernel
                        .pack_key(
                            &fixture.b.view(),
                            &Geometry::new(&[m, k], &[k, n]).unwrap(),
                            &QgemmPlan::select(
                                &quant_params(false, 128, 1),
                                false,
                                false,
                                &Geometry::new(&[m, k], &[k, n]).unwrap(),
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                )
                .unwrap();
            let zeros = vec![127u8; n];
            let mut products = vec![0i32; m * n];
            for _ in 0..3 {
                mlas_sys::qgemm_i32_packed(
                    m,
                    n,
                    k,
                    &a_bytes,
                    false,
                    128,
                    packed,
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    &mut products,
                );
            }
            let start = std::time::Instant::now();
            for _ in 0..reps {
                mlas_sys::qgemm_i32_packed(
                    m,
                    n,
                    k,
                    &a_bytes,
                    false,
                    128,
                    packed,
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    &mut products,
                );
            }
            let gemm = start.elapsed() / reps;

            // The fused primitive the kernel actually calls since the MLAS
            // requantize landed, timed on its own. `fused - gemm` is what the
            // requantize costs *inside* MLAS's threaded region; `whole - fused`
            // is everything our wrapper adds around it. Without this split the
            // two are indistinguishable and a wrapper cost looks like a slow
            // kernel.
            let mut fused_out = vec![0u8; m * n];
            let fused_scales = vec![0.001f32; n];
            for _ in 0..3 {
                mlas_sys::qgemm_requantize(
                    m,
                    n,
                    k,
                    &a_bytes,
                    false,
                    128,
                    mlas_sys::QgemmWeights::Packed(packed),
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    mlas_sys::QgemmScale::PerColumn(&fused_scales),
                    &mut fused_out,
                    false,
                    120,
                    &mut products,
                );
            }
            let start = std::time::Instant::now();
            for _ in 0..reps {
                mlas_sys::qgemm_requantize(
                    m,
                    n,
                    k,
                    &a_bytes,
                    false,
                    128,
                    mlas_sys::QgemmWeights::Packed(packed),
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    mlas_sys::QgemmScale::PerColumn(&fused_scales),
                    &mut fused_out,
                    false,
                    120,
                    &mut products,
                );
            }
            let fused = start.elapsed() / reps;

            // Same call, but A is re-materialised per iteration the way
            // `execute` re-materialises it through `to_dense_bytes`. If this
            // costs what the kernel costs, the per-call input copy is the gap.
            let start = std::time::Instant::now();
            for _ in 0..reps {
                let fresh_a = a_bytes.clone();
                mlas_sys::qgemm_requantize(
                    m,
                    n,
                    k,
                    &fresh_a,
                    false,
                    128,
                    mlas_sys::QgemmWeights::Packed(packed),
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    mlas_sys::QgemmScale::PerColumn(&fused_scales),
                    &mut fused_out,
                    false,
                    120,
                    &mut products,
                );
            }
            let fused_fresh_a = start.elapsed() / reps;

            // Same call, but the i32 accumulator is re-allocated per iteration
            // the way `execute` re-allocates it. MLAS first-touches every page
            // of it from every worker thread, so this is the other per-call
            // buffer whose cost scales with the pool.
            let start = std::time::Instant::now();
            for _ in 0..reps {
                let mut fresh_c = vec![0i32; m * n];
                mlas_sys::qgemm_requantize(
                    m,
                    n,
                    k,
                    &a_bytes,
                    false,
                    128,
                    mlas_sys::QgemmWeights::Packed(packed),
                    mlas_sys::QgemmZeroPoints::PerColumn(&zeros),
                    mlas_sys::QgemmScale::PerColumn(&fused_scales),
                    &mut fused_out,
                    false,
                    120,
                    &mut fresh_c,
                );
            }
            let fused_fresh_c = start.elapsed() / reps;

            let a_quant = quant_params(false, 128, 1);
            let b_scales = vec![0.01f32; n];
            let mut output = vec![0u8; m * n];
            let start = std::time::Instant::now();
            for _ in 0..reps {
                requantize_rows(
                    &products,
                    &a_quant,
                    0,
                    &b_scales,
                    n,
                    0.05,
                    120,
                    DataType::Uint8,
                    &mut output,
                )
                .unwrap();
            }
            let requant = start.elapsed() / reps;

            let start = std::time::Instant::now();
            for _ in 0..reps {
                // Deliberately the same `clear` + `resize` shape the kernel
                // uses, so this measures what the kernel pays.
                let mut scratch: Vec<i32> = Vec::new();
                scratch.clear();
                scratch.resize(m * n, 0);
                std::hint::black_box(&scratch);
            }
            let alloc = start.elapsed() / reps;

            // The partition counters are printed, not asserted, because the
            // docs cite them as the reason the pool is exonerated. A reader
            // reproducing this should see `serial_fallback=0` and
            // `sched_per_call >= threads`.
            let stats = mlas_sys::mlas_threading_stats();
            let sched_per_call = stats
                .scheduled_iterations
                .checked_div(stats.parallel_for_calls)
                .unwrap_or(0);
            println!(
                "m={m} threads={}: whole={whole:?} whole_reused={whole_reused:?} gemm={gemm:?} fused={fused:?} fused_fresh_a={fused_fresh_a:?} fused_fresh_c={fused_fresh_c:?} \
                 mlas-requant={:?} wrapper={:?} scalar-requant={requant:?} \
                 products-alloc={alloc:?} \
                 sched_per_call={sched_per_call} serial_fallback={}",
                stats.pool_threads,
                fused.saturating_sub(gemm),
                whole_reused.saturating_sub(fused),
                stats.serial_fallback_calls,
            );
        }
    }

    /// Reports the one-time cost of the pack against a steady-state call, at a
    /// real decode shape, because the benchmark harness runs a parity check
    /// before it starts timing and therefore never sees a first call.
    ///
    /// `#[ignore]` so CI does not pay for a 3584x3584 pack; run with
    /// `cargo test --release --features mlas qlinear_pack_cost -- --ignored --nocapture`.
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "reports timings; not a pass/fail assertion"]
    fn qlinear_pack_cost_report() {
        let fixture = pack_fixture(1, 3584, 3584, 1);
        let mut kernel = QLinearMatMulKernel::default();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);

        let start = std::time::Instant::now();
        let _ = run_with(&kernel, &fixture);
        let cold = start.elapsed();
        assert_eq!(qlinear_pack_state(&kernel), Some(true));

        let start = std::time::Instant::now();
        for _ in 0..20 {
            let _ = run_with(&kernel, &fixture);
        }
        let steady = start.elapsed() / 20;

        let mut unpacked = QLinearMatMulKernel::default();
        unpacked.set_constant_inputs(&[false; 8]);
        let start = std::time::Instant::now();
        for _ in 0..20 {
            let _ = run_with(&unpacked, &fixture);
        }
        let never_packed = start.elapsed() / 20;

        println!(
            "k=n=3584 m=1: cold(first call incl. pack)={cold:?} steady={steady:?} \
             never-packed={never_packed:?}"
        );
    }

    /// The pack is keyed on the weight's identity, so a second weight must
    /// never be served the first one's pack. `addr` alone is not enough --
    /// hence the shape and signedness in the key -- and a stale pack would be
    /// a silent wrong answer, not a slowdown.
    #[cfg(feature = "mlas")]
    #[test]
    fn a_different_weight_is_never_served_the_cached_pack() {
        let first = pack_fixture(3, 64, 96, 1);
        let second = pack_fixture(3, 64, 96, 2);
        assert_ne!(
            first.b.bytes, second.b.bytes,
            "the two weights must actually differ"
        );

        let mut kernel = QLinearMatMulKernel::default();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let _ = run_with(&kernel, &first);
        assert_eq!(qlinear_pack_state(&kernel), Some(true));
        let served = run_with(&kernel, &second);

        let mut fresh = QLinearMatMulKernel::default();
        fresh.set_constant_inputs(&[false; 8]);
        assert_eq!(
            served.bytes,
            run_with(&fresh, &second).bytes,
            "the second weight must be computed from itself, not from the \
             cached pack of the first"
        );
    }

    /// The two pack guards shadow each other through `execute` -- a batched `B`
    /// is refused by `pack_key` before `pack_build` can refuse its byte count
    /// -- so each is falsified directly here instead of through a compound
    /// injection.
    ///
    /// The length guard is not decoration: `QgemmPackedB::new` *asserts* on a
    /// buffer that is not `k * n`, because it hands the pointer to MLAS, so
    /// dropping the guard turns a graceful decline into a panic.
    #[cfg(feature = "mlas")]
    #[test]
    fn each_pack_guard_declines_on_its_own() {
        let mut kernel = QLinearMatMulKernel::default();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let plan_geometry = Geometry::new(&[3, 64], &[64, 96]).unwrap();
        let plan =
            QgemmPlan::select(&quant_params(false, 128, 4), false, false, &plan_geometry).unwrap();

        let batched = Owned::u8(&[2, 64, 96], &vec![7u8; 2 * 64 * 96]);
        let batched_geometry = Geometry::new(&[2, 3, 64], &[2, 64, 96]).unwrap();
        assert!(
            kernel
                .pack_key(&batched.view(), &batched_geometry, &plan)
                .is_none(),
            "a per-batch weight has no single pack"
        );

        let flat = Owned::u8(&[64, 96], &vec![7u8; 64 * 96]);
        let key = kernel
            .pack_key(&flat.view(), &plan_geometry, &plan)
            .expect("a constant 2-D weight is packable, or the rest is vacuous");
        assert!(
            kernel.pack_build(key, &[7u8; 8]).is_none(),
            "a byte count that is not k * n must be declined, not packed"
        );
        assert!(
            kernel.pack_build(key, &vec![7u8; 64 * 96]).is_some(),
            "MLAS must accept this shape, or the decline above proves nothing"
        );
    }

    /// A batched `B` is a different weight per batch, so one pack cannot serve
    /// it: the kernel must decline to build one, and every batch must still be
    /// computed from its own weight.
    ///
    /// Two independent guards refuse this -- the batch check in `pack_key` and
    /// the `k * n` length check in `pack_build` -- so the assertion that
    /// matters, and the one a single injection can falsify, is that batch 1 is
    /// not answered with batch 0's weights.
    #[cfg(feature = "mlas")]
    #[test]
    fn a_batched_weight_declines_the_pack() {
        let (batches, m, k, n) = (2usize, 3usize, 64usize, 96usize);
        let a = Owned::u8(
            &[batches, m, k],
            &(0..batches * m * k)
                .map(|i| ((i * 37 + 11) % 256) as u8)
                .collect::<Vec<_>>(),
        );
        let b = Owned::u8(
            &[batches, k, n],
            &(0..batches * k * n)
                .map(|i| ((i * 91 + 7) % 256) as u8)
                .collect::<Vec<_>>(),
        );
        let mut kernel = QLinearMatMulKernel::default();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let mut output = Owned::zeros(DataType::Uint8, &[batches, m, n]);
        let (a_scale, a_zero) = (Owned::f32(&[], &[0.02]), Owned::u8(&[], &[128]));
        let (b_scale, b_zero) = (Owned::f32(&[], &[0.01]), Owned::u8(&[], &[127]));
        let (y_scale, y_zero) = (Owned::f32(&[], &[0.05]), Owned::u8(&[], &[120]));
        kernel
            .execute(
                &[
                    a.view(),
                    a_scale.view(),
                    a_zero.view(),
                    b.view(),
                    b_scale.view(),
                    b_zero.view(),
                    y_scale.view(),
                    y_zero.view(),
                ],
                &mut [output.view_mut()],
            )
            .unwrap();
        assert!(
            qlinear_pack_state(&kernel).is_none(),
            "a per-batch weight has no single pack"
        );

        let mut unpacked = QLinearMatMulKernel::default();
        unpacked.set_constant_inputs(&[false; 8]);
        let mut expected = Owned::zeros(DataType::Uint8, &[batches, m, n]);
        unpacked
            .execute(
                &[
                    a.view(),
                    a_scale.view(),
                    a_zero.view(),
                    b.view(),
                    b_scale.view(),
                    b_zero.view(),
                    y_scale.view(),
                    y_zero.view(),
                ],
                &mut [expected.view_mut()],
            )
            .unwrap();
        assert_eq!(
            output.bytes, expected.bytes,
            "every batch must use its own weight"
        );
        assert_ne!(
            output.bytes[..m * n],
            output.bytes[m * n..],
            "the two batches must actually differ, or serving batch 0's pack \
             to batch 1 would be undetectable"
        );
    }

    /// `XOR 0x80` must move the operand and its zero point by the same amount,
    /// or the shift stops cancelling in `a - za` and every accumulator is off
    /// by a multiple of `k`.
    #[cfg(feature = "mlas")]
    #[test]
    fn the_sign_flip_moves_bytes_and_zero_points_together() {
        for value in i8::MIN..=i8::MAX {
            let mut byte = [value as u8];
            flip_sign_domain(&mut byte);
            assert_eq!(
                i32::from(byte[0]),
                i32::from(value) + 128,
                "flipping {value} must land on its unsigned image"
            );
            assert_eq!(
                zero_point_byte(i32::from(value), true, true),
                Some(byte[0]),
                "the zero point of {value} must move with it"
            );
        }
        assert_eq!(
            zero_point_byte(128, true, true),
            None,
            "a zero point outside the signed dtype must still be rejected, not \
             wrapped by the flip"
        );
    }

    #[test]
    fn qlinear_matmul_reordered_accumulation_is_bit_identical() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (1, 37, 65),
            (3, 64, 16),
            (5, 129, 33),
            // Above `PARALLEL_MIN_WORK`, so `requantize_rows` forks. The
            // parallel and serial row walks must agree bit for bit.
            (96, 40, 900),
            // `m <= 4` *and* above `PARALLEL_MIN_WORK`, so the pack-free
            // kernel's column split runs and its products then flow through
            // the forked `requantize_rows`. Without this the fused-parallel
            // path was only ever checked at the kernel level, never end to
            // end through requantization.
            (4, 1029, 1100),
        ] {
            for per_axis in [false, true] {
                // --- Uint8 ---
                let a_u8: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 11) % 256) as u8).collect();
                let b_u8: Vec<u8> = (0..k * n).map(|i| ((i * 91 + 7) % 256) as u8).collect();
                let (a_scales, a_zeros): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..m).map(|i| 0.02 + i as f32 * 0.003).collect(),
                        (0..m).map(|i| ((i * 13) % 256) as i32).collect(),
                    )
                } else {
                    (vec![0.02], vec![128])
                };
                let (b_scales, b_zeros): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..n).map(|i| 0.01 + i as f32 * 0.002).collect(),
                        (0..n).map(|i| ((i * 29 + 3) % 256) as i32).collect(),
                    )
                } else {
                    (vec![0.01], vec![127])
                };
                let axis_shape = |len: usize| if len > 1 { vec![len] } else { Vec::new() };
                let output = execute(
                    [
                        &Owned::u8(&[m, k], &a_u8),
                        &Owned::f32(&axis_shape(a_scales.len()), &a_scales),
                        &Owned::u8(
                            &axis_shape(a_zeros.len()),
                            &a_zeros.iter().map(|&z| z as u8).collect::<Vec<_>>(),
                        ),
                        &Owned::u8(&[k, n], &b_u8),
                        &Owned::f32(&axis_shape(b_scales.len()), &b_scales),
                        &Owned::u8(
                            &axis_shape(b_zeros.len()),
                            &b_zeros.iter().map(|&z| z as u8).collect::<Vec<_>>(),
                        ),
                        &Owned::f32(&[], &[0.05]),
                        &Owned::u8(&[], &[120]),
                    ],
                    DataType::Uint8,
                    &[m, n],
                );
                let a_i32: Vec<i32> = a_u8.iter().map(|&v| i32::from(v)).collect();
                let b_i32: Vec<i32> = b_u8.iter().map(|&v| i32::from(v)).collect();
                assert_eq!(
                    output_values(&output),
                    previous_loop_oracle(
                        &a_i32,
                        &b_i32,
                        m,
                        k,
                        n,
                        &a_scales,
                        &a_zeros,
                        &b_scales,
                        &b_zeros,
                        0.05,
                        120,
                        DataType::Uint8
                    ),
                    "u8 m={m} k={k} n={n} per_axis={per_axis}"
                );

                // --- Int8 ---
                let a_i8: Vec<i8> = (0..m * k)
                    .map(|i| (((i * 37 + 11) % 256) as i32 - 128) as i8)
                    .collect();
                let b_i8: Vec<i8> = (0..k * n)
                    .map(|i| (((i * 91 + 7) % 256) as i32 - 128) as i8)
                    .collect();
                let (a_scales_i, a_zeros_i): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..m).map(|i| 0.02 + i as f32 * 0.003).collect(),
                        (0..m).map(|i| ((i * 13) % 256) as i32 - 128).collect(),
                    )
                } else {
                    (vec![0.02], vec![-5])
                };
                let (b_scales_i, b_zeros_i): (Vec<f32>, Vec<i32>) = if per_axis {
                    (
                        (0..n).map(|i| 0.01 + i as f32 * 0.002).collect(),
                        (0..n).map(|i| ((i * 29 + 3) % 256) as i32 - 128).collect(),
                    )
                } else {
                    (vec![0.01], vec![7])
                };
                let output = execute(
                    [
                        &i8(&[m, k], &a_i8),
                        &Owned::f32(&axis_shape(a_scales_i.len()), &a_scales_i),
                        &i8(
                            &axis_shape(a_zeros_i.len()),
                            &a_zeros_i.iter().map(|&z| z as i8).collect::<Vec<_>>(),
                        ),
                        &i8(&[k, n], &b_i8),
                        &Owned::f32(&axis_shape(b_scales_i.len()), &b_scales_i),
                        &i8(
                            &axis_shape(b_zeros_i.len()),
                            &b_zeros_i.iter().map(|&z| z as i8).collect::<Vec<_>>(),
                        ),
                        &Owned::f32(&[], &[0.05]),
                        &i8(&[], &[-3]),
                    ],
                    DataType::Int8,
                    &[m, n],
                );
                let a_i32: Vec<i32> = a_i8.iter().map(|&v| i32::from(v)).collect();
                let b_i32: Vec<i32> = b_i8.iter().map(|&v| i32::from(v)).collect();
                assert_eq!(
                    output_values(&output),
                    previous_loop_oracle(
                        &a_i32,
                        &b_i32,
                        m,
                        k,
                        n,
                        &a_scales_i,
                        &a_zeros_i,
                        &b_scales_i,
                        &b_zeros_i,
                        0.05,
                        -3,
                        DataType::Int8
                    ),
                    "i8 m={m} k={k} n={n} per_axis={per_axis}"
                );
            }
        }
    }

    /// `N == 0` produces an empty output. `par_chunks_mut(0)` and
    /// `chunks_exact(0)` both panic, so the kernel has to skip the batch
    /// outright -- the previous `for column in 0..0` loop degenerated
    /// harmlessly and this must keep doing so.
    #[test]
    fn qlinear_matmul_zero_width_output_is_empty_not_a_panic() {
        let out = execute(
            [
                &Owned::u8(&[2, 3], &[1, 2, 3, 4, 5, 6]),
                &Owned::f32(&[], &[0.5]),
                &Owned::u8(&[], &[3]),
                &Owned::u8(&[3, 0], &[]),
                &Owned::f32(&[], &[0.25]),
                &Owned::u8(&[], &[2]),
                &Owned::f32(&[], &[0.125]),
                &Owned::u8(&[], &[10]),
            ],
            DataType::Uint8,
            &[2, 0],
        );
        assert!(output_values(&out).is_empty());
    }

    /// `K` large enough that the `i32` accumulator wraps.
    ///
    /// The removed loop accumulated with `wrapping_add`, so on overflow its
    /// answer is defined but not the mathematical dot product -- the shared
    /// `reference` helper here sums in `f64` and legitimately disagrees. What
    /// must hold is that the rewrite reproduces the *old kernel* exactly,
    /// because the zero-point identity it relies on is only valid modulo 2^32.
    /// So this compares against a transcription of the old inner loop.
    #[test]
    fn qlinear_matmul_overflowing_accumulator_matches_the_previous_loop() {
        let (m, k, n) = (2usize, 40_000usize, 3usize);
        let a_values: Vec<u8> = (0..m * k)
            .map(|i| if i % 3 == 0 { 255 } else { 254 })
            .collect();
        let b_values: Vec<u8> = (0..k * n)
            .map(|i| if i % 5 == 0 { 255 } else { 253 })
            .collect();
        let (a_zero, b_zero) = (17i32, 250i32);
        let (a_scale, b_scale, y_scale, y_zero) = (1.0f32, 1.0f32, 1.0e6f32, 40i32);

        let output = execute(
            [
                &Owned::u8(&[m, k], &a_values),
                &Owned::f32(&[], &[a_scale]),
                &Owned::u8(&[], &[a_zero as u8]),
                &Owned::u8(&[k, n], &b_values),
                &Owned::f32(&[], &[b_scale]),
                &Owned::u8(&[], &[b_zero as u8]),
                &Owned::f32(&[], &[y_scale]),
                &Owned::u8(&[], &[y_zero as u8]),
            ],
            DataType::Uint8,
            &[m, n],
        );

        // The zero-point correction term this rewrite introduces is
        // `a_sum * b_zero_point`, and the test is only meaningful if that term
        // actually leaves `i32`.
        let a_sum: i64 = (0..k)
            .map(|inner| i64::from(a_values[inner]) - i64::from(a_zero))
            .sum();
        assert!(
            a_sum * i64::from(b_zero) > i64::from(i32::MAX),
            "test data no longer overflows the correction term ({})",
            a_sum * i64::from(b_zero)
        );

        // Transcription of the loop this change replaced.
        let mut expected = Vec::with_capacity(m * n);
        for row in 0..m {
            for column in 0..n {
                let mut accumulated = 0i32;
                for inner in 0..k {
                    let av = i32::from(a_values[row * k + inner]) - a_zero;
                    let bv = i32::from(b_values[inner * n + column]) - b_zero;
                    accumulated = accumulated.wrapping_add(av * bv);
                }
                let scale = a_scale * b_scale / y_scale;
                let value =
                    (accumulated as f32 * scale).round_ties_even() as i64 + i64::from(y_zero);
                expected.push(value.clamp(0, i64::from(u8::MAX)));
            }
        }
        assert_eq!(
            output_values(&output),
            expected,
            "the zero-point identity did not survive i32 overflow"
        );
    }

    /// Row parallelism must not make the result depend on the thread count:
    /// each output row is accumulated by exactly one worker, so repeated runs
    /// -- and runs under a narrower pool -- have to agree exactly.
    #[test]
    fn qlinear_matmul_is_deterministic_across_repeated_runs() {
        let (m, k, n) = (9usize, 71usize, 23usize);
        let a_values: Vec<u8> = (0..m * k).map(|i| ((i * 53 + 3) % 256) as u8).collect();
        let b_values: Vec<u8> = (0..k * n).map(|i| ((i * 17 + 29) % 256) as u8).collect();
        let a = Owned::u8(&[m, k], &a_values);
        let b = Owned::u8(&[k, n], &b_values);
        let a_scale = Owned::f32(&[], &[0.03]);
        let a_zero = Owned::u8(&[], &[130]);
        let b_scale = Owned::f32(&[], &[0.007]);
        let b_zero = Owned::u8(&[], &[110]);
        let y_scale = Owned::f32(&[], &[0.04]);
        let y_zero = Owned::u8(&[], &[100]);
        let inputs = [
            &a, &a_scale, &a_zero, &b, &b_scale, &b_zero, &y_scale, &y_zero,
        ];

        let first = output_values(&execute(inputs, DataType::Uint8, &[m, n]));
        for round in 1..4 {
            let again = output_values(&execute(inputs, DataType::Uint8, &[m, n]));
            assert_eq!(first, again, "round {round} disagreed with the first run");
        }

        // Vary the pool width so the row partition genuinely changes. Row
        // accumulation is sequential and integer, so the answer must not depend
        // on how many workers split the rows -- including the serial path the
        // single-thread pool takes.
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let under_pool =
                pool.install(|| output_values(&execute(inputs, DataType::Uint8, &[m, n])));
            assert_eq!(
                first, under_pool,
                "a {threads}-thread pool produced a different result"
            );
        }
    }
}
