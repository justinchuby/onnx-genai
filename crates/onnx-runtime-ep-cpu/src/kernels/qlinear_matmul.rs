//! `QLinearMatMul`: integer matrix multiplication with linear quantization.

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView, WorkspaceLayout,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Dim, Graph, Node, Shape, broadcast_shapes};
use rayon::prelude::*;

use super::{check_arity, qgemm_native, to_dense_bytes, write_dense_bytes};
use crate::DerivedWeightMemory;
#[cfg(test)]
use crate::kernels::testutil::PreparedTestKernel;
#[cfg(feature = "mlas")]
use crate::resources::{DerivedWeightGrant, IdleWeightCache, WeightAdmission};
use crate::strided::numel;
use smallvec::SmallVec;
use std::borrow::Cow;
#[cfg(feature = "mlas")]
use std::sync::{Arc, Mutex};
use std::{alloc::Layout, mem::MaybeUninit};

/// The payload is dropped before its Host grant, including after cache eviction.
#[cfg(feature = "mlas")]
struct GrantedQgemmPack {
    packed: mlas_sys::QgemmPackedB,
    _grant: DerivedWeightGrant,
}

#[cfg(feature = "mlas")]
struct CachedQgemmPack {
    key: QgemmPackKey,
    // None records a reference layout refusal, never a temporary capacity refusal.
    packed: Option<Arc<GrantedQgemmPack>>,
}

/// Erased by the existing executor seam, which shares only within one node.
/// Keep the immutable source beside the cache so adoption can verify ownership.
#[cfg(feature = "mlas")]
struct SharedQgemmPackState {
    cache: Arc<QgemmPackCache>,
    memory: DerivedWeightMemory,
}

/// Admission may synchronously ask this cache to reclaim, so the builder lock
/// is independent of the state lock and callbacks only acquire the latter.
#[cfg(feature = "mlas")]
#[derive(Default)]
struct QgemmPackCache {
    slot: Mutex<Option<CachedQgemmPack>>,
    build: Mutex<()>,
}

#[cfg(feature = "mlas")]
impl QgemmPackCache {
    fn state(&self) -> Result<std::sync::MutexGuard<'_, Option<CachedQgemmPack>>> {
        self.slot.lock().map_err(|_| EpError::KernelFailed(
            "QLinearMatMul packed-weight cache lock was poisoned; recreate the CPU kernel before retrying".into()
        ))
    }
}

#[cfg(feature = "mlas")]
impl IdleWeightCache for QgemmPackCache {
    fn reclaim_idle(&self, want: usize) -> Result<usize> {
        if want == 0 {
            return Ok(0);
        }
        let retired = {
            let mut slot = self.state()?;
            let idle = slot
                .as_ref()
                .and_then(|cached| cached.packed.as_ref())
                .is_some_and(|packed| Arc::strong_count(packed) == 1);
            if idle { slot.take() } else { None }
        };
        let bytes = retired
            .as_ref()
            .and_then(|cached| cached.packed.as_ref())
            .map_or(0, |payload| payload.packed.owned_heap_bytes());
        // Release payload and its external accounting only after unlocking.
        drop(retired);
        Ok(bytes)
    }
}

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

/// Predicted resident packed-B bytes for `graph` under the session's node-scoped
/// constant-state sharing. Activation shape variants share one cache entry;
/// distinct graph nodes remain independent. A cache retains at most one layout,
/// so bound it by the largest signedness layout, including exact sign-domain
/// translation, rather than multiplying by a fixed number of shape variants.
/// Zero without the optional MLAS research reference.
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

/// One logical B matrix: a vector is K×1, and leading unit dimensions do
/// not create distinct weight batches. Shared by admission prediction and use.
#[cfg(feature = "mlas")]
fn single_matrix_shape(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [k] => Some((*k, 1)),
        [batch @ .., k, n] if batch.iter().all(|&dimension| dimension == 1) => Some((*k, *n)),
        _ => None,
    }
}

/// Per-node contribution, using the same single-matrix rule as pack lookup.
#[cfg(feature = "mlas")]
fn node_qlinear_packed_b_bytes(node: &Node, graph: &Graph) -> u64 {
    if !node.is_default_domain() || node.op_type != "QLinearMatMul" {
        return 0;
    }
    let Some(Some(b_value)) = node.inputs.get(3).copied() else {
        return 0;
    };
    if !graph.is_constant_initializer(b_value) {
        return 0;
    }
    let Some(weight) = graph.initializers.get(&b_value) else {
        return 0;
    };
    if !is_quantized(weight.dtype()) {
        return 0;
    }
    let Some((k, n)) = single_matrix_shape(weight.dims()) else {
        return 0;
    };
    [false, true]
        .into_iter()
        .flat_map(|a_signed| [false, true].map(|b_signed| (a_signed, b_signed)))
        .filter_map(|(a_signed, b_signed)| mlas_sys::qgemm_pack_b_size(n, k, a_signed, b_signed))
        .max()
        .unwrap_or(0) as u64
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
    static PRODUCT_INITIALIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

pub struct QLinearMatMulKernel {
    #[cfg(feature = "mlas")]
    retain_caches: bool,
    /// Which operands the graph guarantees are constant initializers. Only
    /// index 3 (`B`) is consulted; the rest are carried so the array lines up
    /// with the operand list.
    constant_inputs: [bool; 8],
    /// One constant weight pack; construction is serialized before allocation.
    /// Active calls own an Arc, so its grant follows the payload's last owner.
    #[cfg(feature = "mlas")]
    packed_b: Arc<QgemmPackCache>,
    #[cfg(feature = "mlas")]
    weight_memory: DerivedWeightMemory,
}

pub struct QLinearMatMulFactory {
    retain_caches: bool,
    weight_memory: DerivedWeightMemory,
}
impl QLinearMatMulFactory {
    /// Bind retained reference packs to the provider's immutable Host grant source.
    pub fn new(retain_caches: bool, weight_memory: DerivedWeightMemory) -> Self {
        Self {
            retain_caches,
            weight_memory,
        }
    }
}
impl Default for QLinearMatMulFactory {
    fn default() -> Self {
        Self::new(true, DerivedWeightMemory::default())
    }
}
impl Default for QLinearMatMulKernel {
    fn default() -> Self {
        Self::new(true, DerivedWeightMemory::default())
    }
}

impl KernelFactory for QLinearMatMulFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(QLinearMatMulKernel::new(
            self.retain_caches,
            self.weight_memory.clone(),
        )))
    }
}

impl QLinearMatMulKernel {
    fn accumulator_layout(geometry: &Geometry) -> Result<WorkspaceLayout<1>> {
        let layout = Layout::array::<i32>(geometry.tile_len).map_err(|_| EpError::KernelFailed(format!(
            "QLinearMatMul accumulator [{}, {}] exceeds addressable i32 workspace; reduce the matrix dimensions", geometry.m, geometry.n
        )))?;
        WorkspaceLayout::new([layout])
    }

    fn new(retain_caches: bool, weight_memory: DerivedWeightMemory) -> Self {
        #[cfg(not(feature = "mlas"))]
        let _ = (retain_caches, weight_memory);
        #[cfg(feature = "mlas")]
        let packed_b = Arc::new(QgemmPackCache::default());
        #[cfg(feature = "mlas")]
        if retain_caches {
            weight_memory.register_idle_cache(&packed_b);
        }
        Self {
            #[cfg(feature = "mlas")]
            retain_caches,
            constant_inputs: [false; 8],
            #[cfg(feature = "mlas")]
            packed_b,
            #[cfg(feature = "mlas")]
            weight_memory,
        }
    }
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
        if !self.retain_caches {
            return None;
        }
        if !self.constant_inputs[3] {
            return None;
        }
        if single_matrix_shape(b.shape) != Some((geometry.k, geometry.n)) {
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
    fn pack_lookup(&self, key: QgemmPackKey) -> Result<Option<Arc<GrantedQgemmPack>>> {
        let cache = self.packed_b.state()?;
        Ok(cache
            .as_ref()
            .filter(|cached| cached.key == key)
            .and_then(|cached| cached.packed.clone()))
    }

    /// Serialize construction, then keep its grant with the payload.
    /// A capacity refusal leaves the slot empty so a later call can retry.
    /// A different weight never builds a throwaway pack or uses the cached one.
    #[cfg(feature = "mlas")]
    fn pack_build(
        &self,
        key: QgemmPackKey,
        b_bytes: &[u8],
    ) -> Result<Option<Arc<GrantedQgemmPack>>> {
        if key.k.checked_mul(key.n) != Some(b_bytes.len()) {
            return Ok(None);
        }
        let _build = self.packed_b.build.lock().map_err(|_| EpError::KernelFailed(
            "QLinearMatMul packed-weight construction lock was poisoned; recreate the CPU kernel before retrying".into()
        ))?;
        {
            let cache = self.packed_b.state()?;
            if let Some(cached) = cache.as_ref() {
                return Ok((cached.key == key).then(|| cached.packed.clone()).flatten());
            }
        }
        // No state lock spans admission: the Host authority can synchronously
        // reclaim this cache as well as another kernel's idle weight.
        let Some(bytes) = mlas_sys::qgemm_pack_b_size(key.n, key.k, key.a_signed, key.b_signed)
        else {
            *self.packed_b.state()? = Some(CachedQgemmPack { key, packed: None });
            return Ok(None);
        };
        let grant = match self.weight_memory.try_grant(bytes)? {
            WeightAdmission::Granted(grant) => grant,
            WeightAdmission::Declined(_) => return Ok(None),
        };
        let packed = mlas_sys::QgemmPackedB::new(key.n, key.k, b_bytes, key.a_signed, key.b_signed)
            .map(|packed| {
                Arc::new(GrantedQgemmPack {
                    packed,
                    _grant: grant,
                })
            });
        if packed
            .as_ref()
            .is_some_and(|payload| payload.packed.owned_heap_bytes() != bytes)
        {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul packed B allocation disagrees with its {bytes}-byte Host grant for K={} N={}; fix the reference size query before retaining the pack",
                key.k, key.n
            )));
        }
        *self.packed_b.state()? = Some(CachedQgemmPack {
            key,
            packed: packed.clone(),
        });
        Ok(packed)
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

    #[cfg(feature = "mlas")]
    fn shareable_constant_state(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        (self.retain_caches && self.constant_inputs[3]).then(|| {
            Arc::new(SharedQgemmPackState {
                cache: self.packed_b.clone(),
                memory: self.weight_memory.clone(),
            }) as Arc<dyn std::any::Any + Send + Sync>
        })
    }

    #[cfg(feature = "mlas")]
    fn adopt_shareable_constant_state(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<bool> {
        let Ok(shared) = Arc::downcast::<SharedQgemmPackState>(state) else {
            return Ok(false);
        };
        if !self.retain_caches || !self.constant_inputs[3] {
            return Ok(false);
        }
        if !self.weight_memory.shares_source_with(&shared.memory) {
            return Err(EpError::KernelFailed(
                "QLinearMatMul cannot adopt a pack from a different derived-weight grant source; construct this node's shape variants from clones of the same CpuMemoryResources".into()
            ));
        }
        self.packed_b = shared.cache.clone();
        Ok(true)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        if inputs.len() != 8 || !inputs[0].present || !inputs[3].present {
            return Err(EpError::KernelFailed("QLinearMatMul workspace needs eight input slots with present A (0) and B (3); provide complete quantized-matmul metadata".into()));
        }
        let geometry = Geometry::new(inputs[0].shape, inputs[3].shape)?;
        if geometry.tile_len == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let layout = Self::accumulator_layout(&geometry)?;
        Ok(WorkspaceRequirement {
            bytes: layout.bytes() as u64,
            alignment: layout.alignment(),
            ..WorkspaceRequirement::NONE
        })
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl QLinearMatMulKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
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
        if outputs[0].shape != geometry.output_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: output shape {:?} must be {:?}",
                outputs[0].shape, geometry.output_shape
            )));
        }
        let [region] = Self::accumulator_layout(&geometry)?.bind(workspace)?;
        let products_storage: &mut [MaybeUninit<i32>] = if region.bytes() == 0 {
            &mut []
        } else {
            // SAFETY: the executor exclusively lends complete, live Host scratch
            // through this synchronous call. Binding checked capacity/alignment;
            // fresh bytes are not assumed initialized until the selected route fills them.
            unsafe {
                std::slice::from_raw_parts_mut(
                    region.ptr().as_ptr::<MaybeUninit<i32>>(),
                    geometry.tile_len,
                )
            }
        };
        let a_quant = QuantParams::load("a", &inputs[1], &inputs[2], a.shape, QuantAxis::Row)?;
        let b_quant = QuantParams::load("b", &inputs[4], &inputs[5], b.shape, QuantAxis::Column)?;
        let (y_scale, y_zero_point) = output_quant_params(&inputs[6], &inputs[7])?;
        if geometry.result_len == 0 {
            return Ok(());
        }

        let a_signed = inputs[0].dtype == DataType::Int8;
        let b_signed = inputs[3].dtype == DataType::Int8;
        let qgemm = QgemmPlan::select(&a_quant, a_signed, b_signed, &geometry);
        // Both routes read the operands as raw bytes, so neither materializes a
        // widened copy: the native kernel widens `A` to `i16` pairs as it packs
        // and reads `B` straight out of the caller's buffer.
        // A constant B is packed once. After that its dense bytes are dead --
        // they exist only to feed the packer -- so skip the copy, which is
        // `geometry.b_matrix_len` bytes on every call and dominates decode at large K and N.
        #[cfg(feature = "mlas")]
        let pack_key = qgemm
            .as_ref()
            .and_then(|plan| self.pack_key(&inputs[3], &geometry, plan));
        #[cfg(feature = "mlas")]
        let existing_pack = pack_key
            .map(|key| self.pack_lookup(key))
            .transpose()?
            .flatten();
        #[cfg(feature = "mlas")]
        let pack_ready = existing_pack.is_some();
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
        let packed_owner = match existing_pack {
            Some(packed) => Some(packed),
            None => pack_key
                .map(|key| self.pack_build(key, &b_bytes))
                .transpose()?
                .flatten(),
        };
        #[cfg(feature = "mlas")]
        let packed = packed_owner.as_deref().map(|owner| &owner.packed);
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
        let mut b_zero_points: Vec<i32> = Vec::new();
        let mut a_zero_points: Vec<i32> = Vec::new();
        let mut b_scales: Vec<f32> = Vec::new();
        #[cfg(feature = "mlas")]
        let mut combined_scales: Vec<f32> = Vec::new();
        for batch in 0..geometry.batch_count {
            let a_batch = geometry.a_batch_offset(&batch_index);
            let b_batch = geometry.b_batch_offset(&batch_index);
            let a_offset = a_batch * geometry.a_matrix_len;
            let b_offset = b_batch * geometry.b_matrix_len;
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

            if let Some(plan) = &qgemm {
                // Requantize inside MLAS when the combined scale allows it, so
                // each output tile is scaled while it is still in cache and the
                // bytes land in `output` directly. Only a non-finite scale --
                // which MLAS maps to the output minimum where the scalar loop
                // maps it to the zero point -- keeps a call off this path.
                #[cfg(feature = "mlas")]
                if let Some(scale) = fused {
                    let region = sink.region(batch * geometry.tile_len, geometry.tile_len);
                    plan.run_requantized(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + geometry.a_matrix_len],
                        match packed {
                            Some(packed) => mlas_sys::QgemmWeights::Packed(packed),
                            None => mlas_sys::QgemmWeights::Dense {
                                bytes: &b_bytes[b_offset..b_offset + geometry.b_matrix_len],
                                signed: plan.b_signed,
                            },
                        },
                        &b_zero_points,
                        scale,
                        region,
                        out_dtype == DataType::Int8,
                        y_zero_point,
                        products_storage,
                    )?;
                    if batch + 1 < geometry.batch_count {
                        next_index(&geometry.batch_shape, &mut batch_index);
                    }
                    continue;
                }
                let products = initialize_products(products_storage);
                match packed {
                    Some(packed) => plan.run_packed(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + geometry.a_matrix_len],
                        packed,
                        &b_zero_points,
                        products,
                    )?,
                    None => plan.run(
                        m,
                        n,
                        k,
                        &a_bytes[a_offset..a_offset + geometry.a_matrix_len],
                        &b_bytes[b_offset..b_offset + geometry.b_matrix_len],
                        &b_zero_points,
                        products,
                    )?,
                }
                requantize_rows(
                    products,
                    &a_quant,
                    a_batch,
                    &b_scales,
                    n,
                    y_scale,
                    y_zero_point,
                    out_dtype,
                    sink.region(batch * geometry.tile_len, geometry.tile_len),
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
            let products = initialize_products(products_storage);
            qgemm_native::qgemm(
                qgemm_native::Operand {
                    bytes: &a_bytes[a_offset..a_offset + geometry.a_matrix_len],
                    signed: a_signed,
                },
                qgemm_native::Operand {
                    bytes: &b_bytes[b_offset..b_offset + geometry.b_matrix_len],
                    signed: b_signed,
                },
                &a_zero_points,
                &b_zero_points,
                m,
                k,
                n,
                products,
            );

            requantize_rows(
                products,
                &a_quant,
                a_batch,
                &b_scales,
                n,
                y_scale,
                y_zero_point,
                out_dtype,
                sink.region(batch * geometry.tile_len, geometry.tile_len),
            )?;
            if batch + 1 < geometry.batch_count {
                next_index(&geometry.batch_shape, &mut batch_index);
            }
        }
        match sink {
            // The direct sink already wrote every byte the caller asked for.
            OutputSink::Direct(_) => Ok(()),
            OutputSink::Staged(bytes) => write_dense_bytes(&mut outputs[0], &bytes),
        }
    }
}

fn initialize_products(storage: &mut [MaybeUninit<i32>]) -> &mut [i32] {
    #[cfg(test)]
    PRODUCT_INITIALIZATIONS.with(|count| count.set(count.get() + 1));
    for element in storage.iter_mut() {
        element.write(0);
    }
    // SAFETY: every element was initialized to a valid i32 above.
    unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast(), storage.len()) }
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
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
        products: &mut [MaybeUninit<i32>],
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

/// The scale MLAS's fused requantizer needs, or `None` when this call must keep
/// using the native epilogue.
///
/// The fused processor computes `clamp(round(c * scale)) + zero_point` with one
/// scale per tensor or per column. That covers every call MLAS accepts at all,
/// because a per-row `a_scale` already keeps a call off the MLAS path -- but
/// only for a *finite* scale: MLAS clamps the float before rounding, which
/// agrees with rounding before clamping everywhere except `NaN`. Rather than
/// change any output byte, a non-finite combined scale is declined here and the
/// native epilogue, which maps `NaN` to the zero point, runs instead.
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

/// The scalar contract also covers non-finite scales: Rust maps NaN to zero
/// during the saturating float-to-integer cast, before adding the zero point.
#[inline]
fn requantize_byte(accumulated: i32, scale: f32, zero: i32, dtype: DataType) -> u8 {
    let value =
        ((accumulated as f32 * scale).round_ties_even() as i64).saturating_add(i64::from(zero));
    match dtype {
        DataType::Int8 => value.clamp(i8::MIN as i64, i8::MAX as i64) as i8 as u8,
        _ => value.clamp(0, u8::MAX as i64) as u8,
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
static QLINEAR_REQUANT_NEON_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(all(test, target_arch = "aarch64"))]
static QLINEAR_ROW_REQUANT_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Directory-level proof counts one real vector dispatch per batch. The local
/// witnesses below additionally prevent another thread/mode satisfying a test.
#[cfg(all(test, target_arch = "aarch64"))]
fn record_requantization_dispatch<const RESCALE: bool>(row: usize, prefix: usize) {
    if row == 0 && prefix > 0 {
        if RESCALE {
            QLINEAR_ROW_REQUANT_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            QLINEAR_REQUANT_NEON_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
thread_local! {
    // Separate calling-thread witnesses for shared and per-row scale routes.
    static QLINEAR_REQUANT_NEON_THREAD_HITS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

/// Convert eight lanes at a time, returning the prefix length written. Tail
/// elements use the same scalar contract on every target. No padding is read.
///
/// # Safety
/// NEON must be available, the three slices must have equal lengths, and zero
/// must lie in the output dtype's range. The caller validates those properties.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn requantize_neon_prefix<const RESCALE: bool>(
    accumulators: &[i32],
    scales: &[f32],
    a_scale: f32,
    y_scale: f32,
    zero: i32,
    signed: bool,
    bytes: &mut [u8],
) -> usize {
    use std::arch::aarch64::*;
    let (minimum, maximum) = if signed { (-128, 127) } else { (0, 255) };
    let low = vdupq_n_f32((minimum - zero) as f32);
    let high = vdupq_n_f32((maximum - zero) as f32);
    let zeros = vdupq_n_f32(0.0);
    let offset = vdupq_n_s32(zero);
    let mut index = 0;
    while index + 8 <= accumulators.len() {
        // SAFETY: each load/store covers eight in-bounds elements in slices
        // of equal length; NEON is enabled by this function's contract.
        unsafe {
            let convert = |at: usize| {
                let mut scale = vld1q_f32(scales.as_ptr().add(at));
                if RESCALE {
                    // Preserve the scalar association and both f32 rounding
                    // steps. A reciprocal or reassociation changes output ties.
                    scale = vdivq_f32(vmulq_f32(vdupq_n_f32(a_scale), scale), vdupq_n_f32(y_scale));
                }
                let values = vmulq_f32(
                    vcvtq_f32_s32(vld1q_s32(accumulators.as_ptr().add(at))),
                    scale,
                );
                // Match Rust's NaN -> 0 conversion before clipping; propagating
                // NaN through a platform min/max is not a portable substitute.
                let values = vbslq_f32(vceqq_f32(values, values), values, zeros);
                let clipped = vminq_f32(vmaxq_f32(values, low), high);
                // Integral clipping bounds commute with ties-to-even rounding.
                // Conversion and zero-point addition now cannot overflow i32.
                vaddq_s32(vcvtnq_s32_f32(clipped), offset)
            };
            let lanes = vcombine_s16(vqmovn_s32(convert(index)), vqmovn_s32(convert(index + 4)));
            let packed = if signed {
                vreinterpret_u8_s8(vqmovn_s16(lanes))
            } else {
                vqmovun_s16(lanes)
            };
            vst1_u8(bytes.as_mut_ptr().add(index), packed);
        }
        index += 8;
    }
    index
}

#[inline]
fn requantize_portable_row(
    accumulators: &[i32],
    scales: &[f32],
    zero: i32,
    dtype: DataType,
    bytes: &mut [u8],
) {
    for ((&accumulated, &scale), byte) in accumulators.iter().zip(scales).zip(bytes) {
        *byte = requantize_byte(accumulated, scale, zero, dtype);
    }
}

#[inline]
fn requantize_per_row_portable_row(
    accumulators: &[i32],
    b_scales: &[f32],
    a_scale: f32,
    y_scale: f32,
    zero: i32,
    dtype: DataType,
    bytes: &mut [u8],
) {
    for ((&accumulated, &b_scale), byte) in accumulators.iter().zip(b_scales).zip(bytes) {
        *byte = requantize_byte(accumulated, a_scale * b_scale / y_scale, zero, dtype);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
fn requantize_neon_row(
    accumulators: &[i32],
    scales: &[f32],
    zero: i32,
    dtype: DataType,
    bytes: &mut [u8],
) {
    let _ = requantize_neon_row_impl::<false>(accumulators, scales, 1.0, 1.0, zero, dtype, bytes);
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
fn requantize_per_row_neon_row(
    accumulators: &[i32],
    b_scales: &[f32],
    a_scale: f32,
    y_scale: f32,
    zero: i32,
    dtype: DataType,
    bytes: &mut [u8],
) {
    let _ = requantize_neon_row_impl::<true>(
        accumulators,
        b_scales,
        a_scale,
        y_scale,
        zero,
        dtype,
        bytes,
    );
}

#[cfg(target_arch = "aarch64")]
fn requantize_neon_row_impl<const RESCALE: bool>(
    accumulators: &[i32],
    scales: &[f32],
    a_scale: f32,
    y_scale: f32,
    zero: i32,
    dtype: DataType,
    bytes: &mut [u8],
) -> usize {
    let mut start = 0;
    if scales.len() == accumulators.len() && bytes.len() == accumulators.len() {
        let valid_zero = match dtype {
            DataType::Int8 => (-128..=127).contains(&zero),
            DataType::Uint8 => (0..=255).contains(&zero),
            _ => false,
        };
        if valid_zero && std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection and the length/range checks above
            // establish every precondition; ordinary ONNX zero points satisfy
            // the range check by their i8/u8 dtype.
            start = unsafe {
                requantize_neon_prefix::<RESCALE>(
                    accumulators,
                    scales,
                    a_scale,
                    y_scale,
                    zero,
                    dtype == DataType::Int8,
                    bytes,
                )
            };
            #[cfg(test)]
            if start > 0 {
                QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| {
                    let mut counts = hits.get();
                    counts[RESCALE as usize] += 1;
                    hits.set(counts);
                });
            }
        }
    }
    if RESCALE {
        requantize_per_row_portable_row(
            &accumulators[start..],
            &scales[start..],
            a_scale,
            y_scale,
            zero,
            dtype,
            &mut bytes[start..],
        );
    } else {
        requantize_portable_row(
            &accumulators[start..],
            &scales[start..],
            zero,
            dtype,
            &mut bytes[start..],
        );
    }
    start
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

    match shared_scales {
        Some(scales) => {
            // Classify once per batch. Sub-vector rows keep the original
            // portable loop without entering a vector dispatcher per row.
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                apply_requantization_rows(products, destination, n, |row, acc, out| {
                    let prefix = requantize_neon_row_impl::<false>(
                        acc,
                        &scales,
                        1.0,
                        1.0,
                        y_zero_point,
                        dtype,
                        out,
                    );
                    #[cfg(test)]
                    record_requantization_dispatch::<false>(row, prefix);
                    #[cfg(not(test))]
                    let _ = (row, prefix);
                });
                return Ok(());
            }
            apply_requantization_rows(products, destination, n, |_, acc, out| {
                requantize_portable_row(acc, &scales, y_zero_point, dtype, out);
            });
        }
        None => {
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                apply_requantization_rows(products, destination, n, |row, acc, out| {
                    let prefix = requantize_neon_row_impl::<true>(
                        acc,
                        b_scales,
                        a_quant.at(a_batch, row).0,
                        y_scale,
                        y_zero_point,
                        dtype,
                        out,
                    );
                    #[cfg(test)]
                    record_requantization_dispatch::<true>(row, prefix);
                    #[cfg(not(test))]
                    let _ = prefix;
                });
                return Ok(());
            }
            apply_requantization_rows(products, destination, n, |row, acc, out| {
                requantize_per_row_portable_row(
                    acc,
                    b_scales,
                    a_quant.at(a_batch, row).0,
                    y_scale,
                    y_zero_point,
                    dtype,
                    out,
                );
            });
        }
    }
    Ok(())
}

#[inline]
fn apply_requantization_rows(
    products: &[i32],
    destination: &mut [u8],
    n: usize,
    row: impl Fn(usize, &[i32], &mut [u8]) + Sync,
) {
    if products.len() >= PARALLEL_MIN_WORK {
        destination
            .par_chunks_mut(n)
            .zip(products.par_chunks_exact(n))
            .enumerate()
            .for_each(|(index, (bytes, accumulators))| row(index, accumulators, bytes));
    } else {
        for (index, (bytes, accumulators)) in destination
            .chunks_mut(n)
            .zip(products.chunks_exact(n))
            .enumerate()
        {
            row(index, accumulators, bytes);
        }
    }
}

struct Geometry {
    m: usize,
    k: usize,
    n: usize,
    a_matrix_len: usize,
    b_matrix_len: usize,
    tile_len: usize,
    a_batch: SmallVec<[usize; 4]>,
    b_batch: SmallVec<[usize; 4]>,
    batch_shape: Vec<usize>,
    batch_count: usize,
    result_len: usize,
    output_shape: SmallVec<[usize; 4]>,
}

fn checked_elements(shape: &[usize], name: &str) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .filter(|&count| count <= isize::MAX as usize)
        .ok_or_else(|| EpError::KernelFailed(format!("QLinearMatMul {name} shape {shape:?} exceeds addressable element storage; reduce its dimensions")))
}

impl Geometry {
    fn new(a: &[usize], b: &[usize]) -> Result<Self> {
        if a.is_empty() || b.is_empty() {
            return Err(EpError::KernelFailed(
                "QLinearMatMul: operands must be at least 1-D".into(),
            ));
        }
        checked_elements(a, "A")?;
        checked_elements(b, "B")?;
        let a_1d = a.len() == 1;
        let b_1d = b.len() == 1;
        let m = if a_1d { 1 } else { a[a.len() - 2] };
        let k = a[a.len() - 1];
        let b_k = if b_1d { b[0] } else { b[b.len() - 2] };
        let n = if b_1d { 1 } else { b[b.len() - 1] };
        if k != b_k {
            return Err(EpError::KernelFailed(format!(
                "QLinearMatMul: inner dims disagree ({k} vs {b_k})"
            )));
        }
        let a_batch: SmallVec<[usize; 4]> =
            a[..a.len().saturating_sub(2)].iter().copied().collect();
        let b_batch: SmallVec<[usize; 4]> =
            b[..b.len().saturating_sub(2)].iter().copied().collect();
        let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
        let batch_count = if m == 0 || n == 0 {
            0
        } else {
            checked_elements(&batch_shape, "broadcast batch")?
        };
        let tile_len = if batch_count == 0 {
            0
        } else {
            checked_elements(&[m, n], "accumulator")?
        };
        let result_len = checked_elements(&[batch_count, tile_len], "output")?;
        let a_matrix_len = if batch_count == 0 {
            0
        } else {
            checked_elements(&[m, k], "A matrix")?
        };
        let b_matrix_len = if batch_count == 0 {
            0
        } else {
            checked_elements(&[k, n], "B matrix")?
        };
        let mut output_shape: SmallVec<[usize; 4]> = batch_shape.iter().copied().collect();
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
            a_matrix_len,
            b_matrix_len,
            tile_len,
            a_batch,
            b_batch,
            batch_shape,
            batch_count,
            result_len,
            output_shape,
        })
    }
    fn a_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.a_batch)
    }
    fn b_batch_offset(&self, batch_index: &[usize]) -> usize {
        broadcast_offset(batch_index, &self.b_batch)
    }
}

fn broadcast_offset(batch_index: &[usize], shape: &[usize]) -> usize {
    let leading = batch_index.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &dimension)| {
            offset * dimension
                + if dimension == 1 {
                    0
                } else {
                    batch_index[leading + index]
                }
        })
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

    fn requantize_shared_row(
        acc: &[i32],
        scales: &[f32],
        zero: i32,
        dtype: DataType,
        out: &mut [u8],
    ) {
        #[cfg(target_arch = "aarch64")]
        if acc.len() >= 8 {
            requantize_neon_row(acc, scales, zero, dtype, out);
            return;
        }
        requantize_portable_row(acc, scales, zero, dtype, out);
    }

    fn requantization_oracle(acc: i32, scale: f32, zero: i32, dtype: DataType) -> u8 {
        let rounded = (acc as f32 * scale).round_ties_even() as i64;
        let shifted = rounded.saturating_add(i64::from(zero));
        let (low, high) = if dtype == DataType::Int8 {
            (-128, 127)
        } else {
            (0, 255)
        };
        shifted.clamp(low, high) as u8
    }

    #[test]
    fn shared_requantization_is_byte_exact_for_tails_and_extreme_values() {
        let values = [
            i32::MIN,
            i32::MAX,
            -16_777_217,
            16_777_217,
            -511,
            511,
            -5,
            -3,
            -1,
            0,
            1,
            3,
            5,
            127,
            128,
            255,
        ];
        let scales = [
            0.0,
            -0.0,
            0.5,
            1.5,
            -0.5,
            1.0 / 255.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fa0_0001),
        ];
        for n in (0..=33).chain([65, 257]) {
            // Offset every slice so SIMD cannot accidentally require alignment.
            let acc: Vec<i32> = (0..n + 2).map(|i| values[i % values.len()]).collect();
            for shift in 0..scales.len() {
                let factors: Vec<f32> = (0..n + 2)
                    .map(|i| scales[(i + shift) % scales.len()])
                    .collect();
                for (dtype, zeros) in [
                    (DataType::Int8, [-128, -1, 0, 127]),
                    (DataType::Uint8, [0, 1, 128, 255]),
                ] {
                    // Also defend the scalar fallback for out-of-dtype zero points.
                    for zero in zeros.into_iter().chain([i32::MIN, i32::MAX]) {
                        let mut bytes = vec![0xa5; n + 2];
                        requantize_shared_row(
                            &acc[1..n + 1],
                            &factors[1..n + 1],
                            zero,
                            dtype,
                            &mut bytes[1..n + 1],
                        );
                        assert_eq!((bytes[0], bytes[n + 1]), (0xa5, 0xa5));
                        for index in 0..n {
                            assert_eq!(
                                bytes[index + 1],
                                requantization_oracle(
                                    acc[index + 1],
                                    factors[index + 1],
                                    zero,
                                    dtype
                                ),
                                "n={n} shift={shift} index={index} zero={zero} dtype={dtype:?}"
                            );
                        }
                    }
                }
            }
        }
        let acc = [1, 3, 5, -1, -3, -5, 255, -255];
        let mut out = [0; 8];
        requantize_shared_row(&acc, &[0.5; 8], 0, DataType::Int8, &mut out);
        assert_eq!(
            out,
            [0, 2, 2, 0, 254, 254, 127, 128],
            "ties round to even before clipping"
        );
    }

    #[test]
    fn requantization_rows_preserve_batch_scales_and_parallel_results() {
        for (m, n) in [(3, 17), (257, 257)] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .unwrap();
            let products: Vec<i32> = (0..m * n)
                .map(|i| ((i * 7919) as i32).wrapping_sub(40_000))
                .collect();
            let b_scales: Vec<f32> = (0..n).map(|i| (i % 13 + 1) as f32 * 0.003125).collect();
            for per_axis in [false, true] {
                let a_quant = QuantParams {
                    scales: if per_axis {
                        (0..2 * m).map(|i| (i % 7 + 1) as f32 * 0.125).collect()
                    } else {
                        vec![0.375]
                    },
                    zero_points: vec![0; if per_axis { 2 * m } else { 1 }],
                    axis_len: m,
                    per_axis,
                };
                for dtype in [DataType::Int8, DataType::Uint8] {
                    let zero = if dtype == DataType::Int8 { -7 } else { 137 };
                    let expected: Vec<u8> = products
                        .iter()
                        .enumerate()
                        .map(|(i, &acc)| {
                            let scale = a_quant.at(1, i / n).0 * b_scales[i % n] / 0.0625;
                            requantization_oracle(acc, scale, zero, dtype)
                        })
                        .collect();
                    let mut actual = vec![0; products.len()];
                    pool.install(|| {
                        requantize_rows(
                            &products,
                            &a_quant,
                            1,
                            &b_scales,
                            n,
                            0.0625,
                            zero,
                            dtype,
                            &mut actual,
                        )
                    })
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "m={m} n={n} per_axis={per_axis} dtype={dtype:?}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn shared_requantization_reaches_neon_for_native_rows() {
        use std::sync::atomic::Ordering;
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let before = QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| hits.get()[0]);
        let dispatch_before = QLINEAR_REQUANT_NEON_TEST_HITS.load(Ordering::Relaxed);
        let a_quant = QuantParams {
            scales: vec![0.5],
            zero_points: vec![0],
            axis_len: 1,
            per_axis: false,
        };
        let mut actual = [0; 17];
        requantize_rows(
            &[3; 17],
            &a_quant,
            0,
            &[1.0; 17],
            17,
            1.0,
            0,
            DataType::Uint8,
            &mut actual,
        )
        .unwrap();
        assert_eq!(actual, [2; 17]);
        assert!(QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| hits.get()[0]) > before);
        assert!(QLINEAR_REQUANT_NEON_TEST_HITS.load(Ordering::Relaxed) > dispatch_before);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn per_row_requantization_reaches_neon_with_exact_tail_outputs() {
        use std::sync::atomic::Ordering;
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let before = QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| hits.get()[1]);
        let dispatch_before = QLINEAR_ROW_REQUANT_TEST_HITS.load(Ordering::Relaxed);
        let a_quant = QuantParams {
            scales: vec![0.5, 2., 0.25],
            zero_points: vec![0; 3],
            axis_len: 3,
            per_axis: true,
        };
        let mut backing = [0xa5; 53];
        requantize_rows(
            &[3; 51],
            &a_quant,
            0,
            &[1.; 17],
            17,
            1.,
            127,
            DataType::Uint8,
            &mut backing[1..52],
        )
        .unwrap();
        assert_eq!((backing[0], backing[52]), (0xa5, 0xa5));
        for (row, value) in [129u8, 133, 128].into_iter().enumerate() {
            assert!(
                backing[1 + row * 17..1 + (row + 1) * 17]
                    .iter()
                    .all(|&byte| byte == value)
            );
        }
        assert!(
            QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| hits.get()[1]) > before,
            "native per-row scales must execute a vector block, not just the scalar loop"
        );
        assert!(QLINEAR_ROW_REQUANT_TEST_HITS.load(Ordering::Relaxed) > dispatch_before);
    }

    #[test]
    fn per_row_requantization_matches_scalar_for_tails_and_nonfinite_scales() {
        let values = [
            i32::MIN,
            i32::MAX,
            -16_777_217,
            16_777_217,
            -511,
            511,
            -5,
            -3,
            -1,
            0,
            1,
            3,
            5,
            127,
            255,
        ];
        let factors = [
            0.,
            -0.,
            0.5,
            -0.5,
            1.5,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fa0_0001),
        ];
        for n in (0..=33).chain([65, 257]) {
            let acc: Vec<_> = (0..n + 2).map(|i| values[i % values.len()]).collect();
            for shift in 0..factors.len() {
                let columns: Vec<_> = (0..n + 2)
                    .map(|i| factors[(i + shift) % factors.len()])
                    .collect();
                for a_scale in [
                    0.,
                    -0.,
                    0.003125,
                    1.5,
                    f32::MIN_POSITIVE,
                    f32::MAX,
                    f32::INFINITY,
                    f32::NAN,
                ] {
                    for y_scale in [
                        1.,
                        0.0625,
                        0.,
                        -0.,
                        f32::MIN_POSITIVE,
                        f32::from_bits(1),
                        f32::MAX,
                        f32::INFINITY,
                        f32::NAN,
                    ] {
                        for (dtype, zeros) in [
                            (DataType::Int8, [-128, -1, 0, 127]),
                            (DataType::Uint8, [0, 1, 128, 255]),
                        ] {
                            for zero in zeros.into_iter().chain([i32::MIN, i32::MAX]) {
                                let mut out = vec![0xa5; n + 2];
                                #[cfg(target_arch = "aarch64")]
                                requantize_per_row_neon_row(
                                    &acc[1..n + 1],
                                    &columns[1..n + 1],
                                    a_scale,
                                    y_scale,
                                    zero,
                                    dtype,
                                    &mut out[1..n + 1],
                                );
                                #[cfg(not(target_arch = "aarch64"))]
                                requantize_per_row_portable_row(
                                    &acc[1..n + 1],
                                    &columns[1..n + 1],
                                    a_scale,
                                    y_scale,
                                    zero,
                                    dtype,
                                    &mut out[1..n + 1],
                                );
                                assert_eq!((out[0], out[n + 1]), (0xa5, 0xa5));
                                for i in 0..n {
                                    let expected = requantization_oracle(
                                        acc[i + 1],
                                        a_scale * columns[i + 1] / y_scale,
                                        zero,
                                        dtype,
                                    );
                                    assert_eq!(
                                        out[i + 1],
                                        expected,
                                        "n={n} i={i} a={a_scale:?} y={y_scale:?} zero={zero} dtype={dtype:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn per_row_requantization_preserves_scale_association_at_rounding_boundaries() {
        for (a_bits, b_bits, y_bits, acc) in [
            (1080352412, 1168457022, 1329431408, 10677323),
            (1153449446, 982523135, 1221068150, 15383253),
        ] {
            let a = f32::from_bits(a_bits);
            let b = f32::from_bits(b_bits);
            let y = f32::from_bits(y_bits);
            let expected = requantization_oracle(acc, a * b / y, 0, DataType::Uint8);
            assert_eq!(expected, 63);
            if acc == 10677323 {
                assert_eq!(
                    requantization_oracle(acc, (a * b) * (1. / y), 0, DataType::Uint8),
                    64,
                    "reciprocal control must distinguish the output"
                );
                assert_eq!(
                    (((acc as f32 * a) * b / y).round_ties_even() as i64).clamp(0, 255),
                    64,
                    "accumulator-first control must distinguish the output"
                );
            } else {
                assert_eq!(
                    requantization_oracle(acc, a * (b / y), 0, DataType::Uint8),
                    64,
                    "reassociation control must distinguish the output"
                );
            }
            let mut out = [0; 17];
            #[cfg(target_arch = "aarch64")]
            requantize_per_row_neon_row(&[acc; 17], &[b; 17], a, y, 0, DataType::Uint8, &mut out);
            #[cfg(not(target_arch = "aarch64"))]
            requantize_per_row_portable_row(
                &[acc; 17],
                &[b; 17],
                a,
                y,
                0,
                DataType::Uint8,
                &mut out,
            );
            assert_eq!(out, [expected; 17]);
        }
    }

    /// Direct synchronous row epilogue comparison; no GEMM/model/pool claim.
    /// Retain every rotating portable/native-A/native-B sample and its output
    /// checksum. Test-only TLS witness bookkeeping is included in native arms.
    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    #[ignore = "manual per-row epilogue diagnostic; run under the host lock"]
    fn bench_per_row_requantization_neon_ab() {
        fn cpu_seconds() -> f64 {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            // SAFETY: getrusage initializes the writable object on success.
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
                0
            );
            // SAFETY: successful getrusage initialized usage.
            let usage = unsafe { usage.assume_init() };
            (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
                + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 * 1e-6
        }
        use std::hint::black_box;
        use std::time::Instant;
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        for n in [7, 8, 9, 17, 65, 257, 2048] {
            for dtype in [DataType::Int8, DataType::Uint8] {
                let acc: Vec<_> = (0..n)
                    .map(|i| (i as i32 * 7919) % 65_537 - 32_768)
                    .collect();
                let scales: Vec<_> = (0..n).map(|i| (i % 7 + 1) as f32 / 4096.).collect();
                let zero = if dtype == DataType::Int8 { -7 } else { 137 };
                let (a, y) = (0.375, 0.625);
                let expected: Vec<_> = acc
                    .iter()
                    .zip(&scales)
                    .map(|(&v, &b)| requantization_oracle(v, a * b / y, zero, dtype))
                    .collect();
                assert!(
                    expected.windows(2).any(|pair| pair[0] != pair[1]),
                    "the fixture must not collapse to one clamped byte"
                );
                let checksum = expected.iter().fold(0xcbf29ce484222325u64, |hash, &value| {
                    (hash ^ u64::from(value)).wrapping_mul(0x100000001b3)
                });
                let mut buffers = [vec![0; n], vec![0; n], vec![0; n]];
                let iterations = (2_000_000 / n).max(1);
                let before = QLINEAR_REQUANT_NEON_THREAD_HITS.with(|hits| hits.get()[1]);
                for repetition in 0..21 {
                    for offset in 0..3 {
                        let arm = (repetition + offset) % 3;
                        let kernel = if arm == 0 || n < 8 {
                            requantize_per_row_portable_row
                        } else {
                            requantize_per_row_neon_row
                        };
                        let run = |out: &mut [u8]| {
                            kernel(
                                black_box(&acc),
                                black_box(&scales),
                                black_box(a),
                                black_box(y),
                                zero,
                                dtype,
                                out,
                            )
                        };
                        run(&mut buffers[arm]);
                        assert_eq!(buffers[arm], expected);
                        let cpu = cpu_seconds();
                        let wall = Instant::now();
                        for _ in 0..iterations {
                            run(black_box(&mut buffers[arm]));
                        }
                        let wall_ns = wall.elapsed().as_nanos();
                        let cpu_ns = (cpu_seconds() - cpu) * 1e9;
                        assert!(wall_ns > 0 && cpu_ns > 0.);
                        assert_eq!(buffers[arm], expected);
                        println!(
                            "QRE_SAMPLE,1,{n},{dtype:?},{repetition},{arm},{iterations},{cpu_ns:.0},{wall_ns},{checksum}"
                        );
                    }
                }
                let hits = QLINEAR_REQUANT_NEON_THREAD_HITS.with(|counts| counts.get()[1]) - before;
                assert_eq!(
                    hits > 0,
                    n >= 8,
                    "the selected native arm must actually execute vectors"
                );
                println!("QRE_WITNESS,1,{n},{dtype:?},{hits}");
            }
        }
    }

    /// Same-process row epilogue diagnostic, not a GEMM or end-to-end claim.
    /// Run under the host lock in release mode; records rotating portable/native/
    /// repeated-native arms with exact output checks before timing.
    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    #[ignore = "manual row epilogue measurement; run explicitly under the host lock"]
    fn bench_requantization_neon_ab() {
        fn cpu_seconds() -> f64 {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            // SAFETY: the OS initializes the writable rusage on success.
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
                0
            );
            // SAFETY: successful getrusage initialized usage.
            let usage = unsafe { usage.assume_init() };
            (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
                + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 * 1e-6
        }
        use std::hint::black_box;
        use std::time::Instant;
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        println!("n,dtype,arm,cpu_ns_per_element,wall_ns_per_element,cpu_min,cpu_max");
        for n in [7, 8, 17, 128, 257, 2048, 5120] {
            for dtype in [DataType::Int8, DataType::Uint8] {
                let acc: Vec<i32> = (0..n)
                    .map(|i| (i as i32 * 7919) % 65_537 - 32_768)
                    .collect();
                let scales: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32 / 1024.0).collect();
                let zero = if dtype == DataType::Int8 { -7 } else { 137 };
                let expected: Vec<u8> = acc
                    .iter()
                    .zip(&scales)
                    .map(|(&a, &s)| requantization_oracle(a, s, zero, dtype))
                    .collect();
                let mut buffers = [vec![0; n], vec![0; n], vec![0; n]];
                let mut cpus: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
                let mut walls = cpus.clone();
                let iterations = (1_000_000 / n).max(1);
                for repetition in 0..21 {
                    for offset in 0..3 {
                        let arm = (repetition + offset) % 3;
                        // Like production, select once outside the row loop.
                        // Widths below a vector block use the identical kernel
                        // in all arms; no per-row dispatcher is being timed.
                        let kernel = if arm == 0 || n < 8 {
                            requantize_portable_row
                        } else {
                            requantize_neon_row
                        };
                        let run = |out: &mut [u8]| {
                            kernel(black_box(&acc), black_box(&scales), zero, dtype, out);
                        };
                        run(&mut buffers[arm]);
                        assert_eq!(buffers[arm], expected);
                        let cpu = cpu_seconds();
                        let wall = Instant::now();
                        for _ in 0..iterations {
                            run(black_box(&mut buffers[arm]));
                        }
                        let elapsed = wall.elapsed().as_secs_f64();
                        let cpu_elapsed = cpu_seconds() - cpu;
                        assert_eq!(buffers[arm], expected);
                        let elements = (iterations * n) as f64;
                        cpus[arm].push(cpu_elapsed * 1e9 / elements);
                        walls[arm].push(elapsed * 1e9 / elements);
                    }
                }
                for arm in 0..3 {
                    cpus[arm].sort_by(f64::total_cmp);
                    walls[arm].sort_by(f64::total_cmp);
                    let label = ["portable", "native", "native_control"][arm];
                    println!(
                        "{n},{dtype:?},{label},{:.4},{:.4},{:.4},{:.4}",
                        cpus[arm][10], walls[arm][10], cpus[arm][0], cpus[arm][20]
                    );
                }
            }
        }
    }
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
            .execute_prepared(&inputs.map(|input| input.view()), &mut [output.view_mut()])
            .unwrap();
        output
    }

    fn workspace_inputs(batches: usize) -> [Owned; 8] {
        let values: Vec<u8> = (0..batches * 6)
            .map(|i| (1 + i % 6 + i / 6) as u8)
            .collect();
        [
            Owned::u8(&[batches, 2, 3], &values),
            Owned::f32(&[], &[1.]),
            Owned::u8(&[], &[0]),
            Owned::u8(&[3, 2], &[1, 0, 0, 1, 1, 1]),
            Owned::f32(&[], &[1.]),
            Owned::u8(&[], &[0]),
            Owned::f32(&[], &[1.]),
            Owned::u8(&[], &[0]),
        ]
    }

    #[test]
    fn workspace_bounds_are_checked_and_do_not_multiply_by_batch_count() {
        let kernel = QLinearMatMulKernel::default();
        let requirement = |a: &[usize], b: &[usize]| {
            let mut metadata = [TensorMetadata::new(DataType::Uint8, &[], true); 8];
            metadata[0].shape = a;
            metadata[3].shape = b;
            kernel.workspace_requirement(&metadata)
        };
        for batches in [1, 2, 64] {
            assert_eq!(requirement(&[batches, 2, 3], &[3, 4]).unwrap().bytes, 32);
        }
        for (a, b) in [
            (vec![0, 2, 3], vec![3, 4]),
            (vec![0, 3], vec![3, 4]),
            (vec![2, 3], vec![3, 0]),
            (
                vec![0, usize::MAX, usize::MAX],
                vec![0, usize::MAX, usize::MAX],
            ),
        ] {
            assert_eq!(requirement(&a, &b).unwrap(), WorkspaceRequirement::NONE);
        }
        assert_eq!(requirement(&[2, 0], &[0, 3]).unwrap().bytes, 24);
        assert_eq!(requirement(&[3], &[3]).unwrap().bytes, 4);
        for (a, b) in [
            (vec![usize::MAX, 2], vec![2, 3]),
            (vec![isize::MAX as usize, 2, 0], vec![0, 2]),
            (vec![isize::MAX as usize / 4 + 1, 0], vec![0, 1]),
        ] {
            let error = requirement(&a, &b).unwrap_err().to_string();
            assert!(
                error.contains("QLinearMatMul") && error.contains("reduce"),
                "{error}"
            );
        }
    }

    #[test]
    fn missing_workspace_preserves_output_and_batches_refill_one_accumulator() {
        let data = workspace_inputs(2);
        let inputs = data.each_ref().map(|value| value.view());
        let kernel = QLinearMatMulKernel::default();
        let mut output = Owned::u8(&[2, 2, 2], &[73; 8]);
        let error = kernel
            .execute(&inputs, &mut [output.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("16 bytes") && error.contains("prepare and pass"),
            "{error}"
        );
        assert_eq!(output.bytes, [73; 8]);
        kernel
            .execute_prepared(&inputs, &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.bytes, [4, 5, 10, 11, 6, 7, 12, 13]);
    }

    #[test]
    fn governed_accumulator_reuses_real_backing_without_warm_issuer_calls() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let books = Arc::new(RecordingGovernor::new_host(1 << 20));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = crate::CpuMemoryResources::new(
            books.weight_memory(HolderId::new(13)),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let mut kernel = provider
            .get_kernel(
                &Node::new(onnx_runtime_ir::NodeId(0), "QLinearMatMul", vec![], vec![]),
                &[],
                10,
            )
            .unwrap();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let data = workspace_inputs(2);
        let inputs = data.each_ref().map(|value| value.view());
        let metadata = inputs.map(|input| TensorMetadata::new(input.dtype, input.shape, true));
        let request = kernel.workspace_requirement(&metadata).unwrap();
        assert_eq!(request.bytes, 16);
        let mut allocation = provider
            .allocate_workspace(16, request.alignment, request.role())
            .unwrap();
        let identity = allocation.managed_owner().unwrap().identity();
        let mut output = Owned::zeros(DataType::Uint8, &[2, 2, 2]);
        let workspace = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
            allocation.len(),
        );
        kernel
            .execute_with_workspace(&inputs, &mut [output.view_mut()], Some(workspace))
            .unwrap();
        provider.deallocate_workspace(allocation).unwrap();
        let packed_bytes = books.used(Tier::Host) - manager.process_used(Tier::Host);
        #[cfg(feature = "mlas")]
        if mlas_sys::qgemm_pack_b_size(2, 3, false, false).is_some() {
            assert!(
                packed_bytes > 0,
                "the retained-pack arm must actually prepare a weight"
            );
        }
        let calls = (books.claims(), books.releases());
        for _ in 0..4096 {
            let mut allocation = provider
                .allocate_workspace(16, request.alignment, request.role())
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            );
            kernel
                .execute_with_workspace(&inputs, &mut [output.view_mut()], Some(workspace))
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        assert_eq!(output.bytes, [4, 5, 10, 11, 6, 7, 12, 13]);
        assert_eq!((books.claims(), books.releases()), calls);
        assert_eq!(books.used(Tier::Host), 16 + packed_bytes);
        assert_eq!(manager.process_used(Tier::Host), 16);
        drop(kernel);
        assert_eq!(books.used(Tier::Host), 16);
        pool.trim_idle(16).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn empty_output_needs_no_workspace_and_zero_k_requantizes_zero() {
        let kernel = QLinearMatMulKernel::default();
        for (a_shape, b_shape, output_shape, bytes) in [
            (vec![0, 2, 3], vec![3, 2], vec![0, 2, 2], 0),
            (vec![0, 3], vec![3, 2], vec![0, 2], 0),
            (vec![2, 3], vec![3, 0], vec![2, 0], 0),
            (vec![2, 0], vec![0, 3], vec![2, 3], 6),
        ] {
            let mut data = workspace_inputs(1);
            data[0] = Owned::u8(&a_shape, &vec![1; a_shape.iter().product()]);
            data[3] = Owned::u8(&b_shape, &vec![1; b_shape.iter().product()]);
            data[7] = Owned::u8(&[], &[100]);
            let mut output = Owned::zeros(DataType::Uint8, &output_shape);
            let inputs = data.each_ref().map(|value| value.view());
            if bytes == 0 {
                kernel.execute(&inputs, &mut [output.view_mut()]).unwrap();
            } else {
                kernel
                    .execute_prepared(&inputs, &mut [output.view_mut()])
                    .unwrap();
            }
            assert_eq!(output.bytes, vec![100; bytes]);
        }
    }

    #[test]
    fn concurrent_accumulators_are_funded_and_exclusive_until_completion() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        let books = Arc::new(RecordingGovernor::new_host(48));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let resources = crate::CpuMemoryResources::new(
            books.weight_memory(HolderId::new(13)),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let provider = crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let kernel = QLinearMatMulKernel::default();
        let data = workspace_inputs(2);
        let ready = Barrier::new(5);
        let release = Barrier::new(5);
        let refused = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let (provider, kernel, data, ready, release, refused) =
                    (&provider, &kernel, &data, &ready, &release, &refused);
                scope.spawn(move || {
                    let request = WorkspaceRequirement {
                        bytes: 16,
                        alignment: 4,
                        ..WorkspaceRequirement::NONE
                    };
                    let allocation = provider.allocate_workspace(16, 4, request.role());
                    if allocation.is_err() {
                        refused.fetch_add(1, Ordering::Relaxed);
                    }
                    ready.wait();
                    release.wait();
                    if let Ok(mut allocation) = allocation {
                        let workspace = WorkspaceView::new(
                            onnx_runtime_ep_api::DevicePtrMut(allocation.as_mut_ptr()),
                            allocation.len(),
                        );
                        let mut output = Owned::zeros(DataType::Uint8, &[2, 2, 2]);
                        kernel
                            .execute_with_workspace(
                                &data.each_ref().map(|value| value.view()),
                                &mut [output.view_mut()],
                                Some(workspace),
                            )
                            .unwrap();
                        assert_eq!(output.bytes, [4, 5, 10, 11, 6, 7, 12, 13]);
                        provider.deallocate_workspace(allocation).unwrap();
                    }
                });
            }
            ready.wait();
            let refused_now = refused.load(Ordering::Relaxed);
            let used_now = (books.used(Tier::Host), manager.process_used(Tier::Host));
            let snapshot = manager.snapshot();
            release.wait();
            assert_eq!(
                refused_now, 1,
                "only three 16-byte active lanes fit real Host quota"
            );
            assert_eq!(used_now, (48, 48));
            let allocations = snapshot.unwrap().allocations;
            assert_eq!(allocations.len(), 3);
            let identities = allocations
                .iter()
                .map(|allocation| allocation.identity)
                .collect::<std::collections::HashSet<_>>();
            assert_eq!(
                identities.len(),
                3,
                "concurrent calls never alias one mutable lane"
            );
        });
        pool.trim_idle(48).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn fused_uninitialized_scratch_and_native_accumulation_reuse_safely() {
        let mut data = workspace_inputs(3);
        data[0] = Owned::u8(&[3, 2, 3], &[0; 18]);
        data[1] = Owned::f32(&[], &[1e30]);
        data[3] = Owned::u8(&[3, 3, 2], &[1; 18]);
        data[4] = Owned::f32(&[3, 1, 2], &[1e-30, 1e-30, 1e30, 1e30, 1e-30, 1e-30]);
        data[5] = Owned::u8(&[3, 1, 2], &[0; 6]);
        data[7] = Owned::u8(&[], &[100]);
        let kernel = QLinearMatMulKernel::default();
        let mut output = Owned::zeros(DataType::Uint8, &[3, 2, 2]);
        let mut scratch = [MaybeUninit::<i32>::uninit(); 4];
        let workspace = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
            16,
        );
        PRODUCT_INITIALIZATIONS.with(|count| count.set(0));
        kernel
            .execute_with_workspace(
                &data.each_ref().map(|value| value.view()),
                &mut [output.view_mut()],
                Some(workspace),
            )
            .unwrap();
        assert_eq!(output.bytes, [100; 12]);
        assert_eq!(
            PRODUCT_INITIALIZATIONS.with(|count| count.get()),
            1,
            "fused batches do not add a memset"
        );
        data[1] = Owned::f32(&[3, 2, 1], &[1.; 6]);
        data[2] = Owned::u8(&[3, 2, 1], &[0; 6]);
        PRODUCT_INITIALIZATIONS.with(|count| count.set(0));
        kernel
            .execute_with_workspace(
                &data.each_ref().map(|value| value.view()),
                &mut [output.view_mut()],
                Some(workspace),
            )
            .unwrap();
        assert_eq!(output.bytes, [100; 12]);
        assert_eq!(
            PRODUCT_INITIALIZATIONS.with(|count| count.get()),
            3,
            "native batches reset scratch after fused use"
        );
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
                .execute_prepared(&inputs.map(|input| input.view()), &mut [out])
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
            .execute_prepared(
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
        kernel
            .packed_b
            .state()
            .unwrap()
            .as_ref()
            .map(|slot| slot.packed.is_some())
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

    /// Run into caller-owned output/workspace to exclude their allocation
    /// from the steady execution diagnostic.
    #[cfg(feature = "mlas")]
    fn run_into(
        kernel: &QLinearMatMulKernel,
        fixture: &PackFixture,
        output: &mut Owned,
        workspace: Option<WorkspaceView>,
    ) {
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
        kernel
            .execute_with_workspace(&inputs, &mut [output.view_mut()], workspace)
            .unwrap();
    }

    #[cfg(feature = "mlas")]
    fn run_with(kernel: &dyn Kernel, fixture: &PackFixture) -> Owned {
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
        kernel
            .execute_prepared(&inputs, &mut [output.view_mut()])
            .unwrap();
        output
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn provider_qlinear_pack_owns_its_host_grant() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let fixture = pack_fixture(3, 64, 96, 1);
        let bytes = mlas_sys::qgemm_pack_b_size(96, 64, false, false)
            .expect("the fixture must support an actual packed allocation");
        let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
        let resources = crate::CpuMemoryResources::new(
            crate::DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
            crate::CpuWeightCacheLimits::default(),
        )
        .unwrap();
        let provider = crate::CpuExecutionProvider::with_resources(resources);
        let mut kernel = provider
            .get_kernel(
                &Node::new(onnx_runtime_ir::NodeId(0), "QLinearMatMul", vec![], vec![]),
                &[],
                10,
            )
            .unwrap();
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        assert_eq!(governor.used(Tier::Host), 0);
        let actual = run_with(kernel.as_ref(), &fixture);
        assert_eq!(
            governor.used(Tier::Host),
            bytes as u64,
            "the provider's Host books must fund the actual pack"
        );
        assert_eq!(
            actual.bytes,
            run_with(&QLinearMatMulKernel::default(), &fixture).bytes
        );
        assert_eq!(run_with(kernel.as_ref(), &fixture).bytes, actual.bytes);
        assert_eq!(
            governor.used(Tier::Host),
            bytes as u64,
            "a warm hit must not claim a second pack"
        );
        drop(kernel);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn qlinear_pack_capacity_refusal_retries_and_active_owner_keeps_grant() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let fixture = pack_fixture(3, 64, 96, 9);
        let bytes = mlas_sys::qgemm_pack_b_size(96, 64, false, false).unwrap();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
        let memory = crate::DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let mut kernel = QLinearMatMulKernel::new(true, memory);
        kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        let kv = governor
            .reserve(
                Tier::Host,
                bytes as u64,
                MemoryRole::KvCache,
                HolderId::new(7),
            )
            .unwrap();
        let unpacked = run_with(&kernel, &fixture);
        assert!(
            qlinear_pack_state(&kernel).is_none(),
            "capacity refusal must leave a retryable empty slot"
        );
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 0);
        let packed = run_with(&kernel, &fixture);
        assert_eq!(
            packed.bytes, unpacked.bytes,
            "refused and admitted routes must agree bitwise"
        );
        assert_eq!(qlinear_pack_state(&kernel), Some(true));
        let key = kernel.packed_b.state().unwrap().as_ref().unwrap().key;
        let live = kernel.pack_lookup(key).unwrap().unwrap();
        assert_eq!(live.packed.owned_heap_bytes(), bytes);
        drop(kernel);
        assert_eq!(
            governor.used(Tier::Host),
            bytes as u64,
            "an active call must keep its payload charged after kernel drop"
        );
        assert!(
            governor
                .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(7))
                .is_err()
        );
        drop(live);
        assert_eq!(governor.used(Tier::Host), 0);
        let _kv = governor
            .reserve(
                Tier::Host,
                bytes as u64,
                MemoryRole::KvCache,
                HolderId::new(7),
            )
            .unwrap();
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn qlinear_pack_concurrent_builders_share_one_granted_payload() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let fixture = pack_fixture(3, 64, 96, 9);
        let bytes = mlas_sys::qgemm_pack_b_size(96, 64, false, false).unwrap();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
        let memory = crate::DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let mut kernel = QLinearMatMulKernel::new(true, memory.clone());
        let constant = [false, false, false, true, false, false, false, false];
        kernel.set_constant_inputs(&constant);
        let shared = kernel.shareable_constant_state().unwrap();
        let builders: Vec<_> = (0..8)
            .map(|_| {
                let mut builder = QLinearMatMulKernel::new(true, memory.clone());
                builder.set_constant_inputs(&constant);
                assert!(
                    builder
                        .adopt_shareable_constant_state(shared.clone())
                        .unwrap()
                );
                builder
            })
            .collect();
        drop(shared);
        let key = QgemmPackKey {
            addr: fixture.b.bytes.as_ptr() as usize,
            k: 64,
            n: 96,
            a_signed: false,
            b_signed: false,
            flip_b: false,
        };
        let barrier = std::sync::Barrier::new(8);
        let owners = std::thread::scope(|scope| {
            let threads: Vec<_> = builders
                .iter()
                .map(|builder| {
                    scope.spawn(|| {
                        barrier.wait();
                        builder.pack_build(key, &fixture.b.bytes).unwrap().unwrap()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(owners.len(), 8);
        assert!(owners.iter().all(|owner| Arc::ptr_eq(owner, &owners[0])));
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        let other = QgemmPackKey {
            addr: key.addr + 1,
            ..key
        };
        assert!(
            kernel
                .pack_build(other, &fixture.b.bytes)
                .unwrap()
                .is_none()
        );
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(builders);
        drop(kernel);
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(owners);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn qlinear_pack_pressure_reclaims_idle_without_deadlocking_builders() {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        let name = "kernels::qlinear_matmul::tests::qlinear_pack_pressure_child";
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--ignored", "--nocapture"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let timed_out = loop {
            if child.try_wait().unwrap().is_some() {
                break false;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                break true;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !timed_out,
            "{name} deadlocked during Host pressure; stdout={stdout}; stderr={stderr}"
        );
        assert!(
            output.status.success() && stdout.contains(&format!("test {name} ... ok")),
            "the named pressure child must run and pass; stdout={stdout}; stderr={stderr}"
        );
    }

    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "run by the bounded pressure parent to detect admission/reclaim deadlocks"]
    fn qlinear_pack_pressure_child() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let fixture = pack_fixture(3, 64, 96, 9);
        let bytes = mlas_sys::qgemm_pack_b_size(96, 64, false, false).unwrap();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
        let memory = crate::DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        assert!(memory.supports_pressure_reclaim());
        // Register the builder before the donor. Its own empty cache must be
        // visited while admission holds the construction lock, without deadlock.
        let mut builder = QLinearMatMulKernel::new(true, memory.clone());
        let mut donor = QLinearMatMulKernel::new(true, memory);
        for kernel in [&mut builder, &mut donor] {
            kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        }
        let reference = run_with(&donor, &fixture);
        assert_eq!(qlinear_pack_state(&donor), Some(true));
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        let unrelated = LedgerGovernor::new(LeaseLedger::new(0, 0, 0));
        assert!(
            unrelated
                .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(7))
                .is_err()
        );
        assert_eq!(
            qlinear_pack_state(&donor),
            Some(true),
            "another authority must not evict this cache"
        );
        // The first pressure request visits the builder's empty cache before
        // the donor. This constructs the self-callback case, before any round-
        // robin snapshot advances the source's cursor.
        eprintln!("pressure child: entering self-cache admission");
        assert_eq!(run_with(&builder, &fixture).bytes, reference.bytes);
        assert!(qlinear_pack_state(&donor).is_none());
        assert_eq!(qlinear_pack_state(&builder), Some(true));
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        assert_eq!(builder.packed_b.reclaim_idle(0).unwrap(), 0);
        assert_eq!(qlinear_pack_state(&builder), Some(true));
        let kv = governor
            .reserve(
                Tier::Host,
                bytes as u64,
                MemoryRole::KvCache,
                HolderId::new(7),
            )
            .expect("ordinary Host KV reserve must reclaim the idle packed weight");
        assert!(qlinear_pack_state(&builder).is_none());
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        assert_eq!(run_with(&builder, &fixture).bytes, reference.bytes);
        assert!(qlinear_pack_state(&builder).is_none());
        drop(kv);
        assert_eq!(run_with(&builder, &fixture).bytes, reference.bytes);
        assert_eq!(qlinear_pack_state(&builder), Some(true));
        let key = QgemmPackKey {
            addr: fixture.b.bytes.as_ptr() as usize,
            k: 64,
            n: 96,
            a_signed: false,
            b_signed: false,
            flip_b: false,
        };
        let active = builder.pack_lookup(key).unwrap().unwrap();
        assert!(
            governor
                .reserve(Tier::Host, 1, MemoryRole::KvCache, HolderId::new(7))
                .is_err()
        );
        assert!(Arc::ptr_eq(
            &active,
            &builder.pack_lookup(key).unwrap().unwrap()
        ));
        drop(active);
        assert_eq!(run_with(&donor, &fixture).bytes, reference.bytes);
        assert!(qlinear_pack_state(&builder).is_none());
        assert_eq!(qlinear_pack_state(&donor), Some(true));
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        let mut kv = governor
            .reserve(Tier::Host, 0, MemoryRole::KvCache, HolderId::new(7))
            .unwrap();
        kv.grow(bytes as u64)
            .expect("Host KV growth must also reclaim an idle pack");
        assert!(qlinear_pack_state(&donor).is_none());
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(kv);
        let builder_weak = Arc::downgrade(&builder.packed_b);
        let donor_weak = Arc::downgrade(&donor.packed_b);
        drop(builder);
        drop(donor);
        assert!(
            builder_weak.upgrade().is_none() && donor_weak.upgrade().is_none(),
            "pressure registration must not retain dead kernel caches"
        );
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn shared_qlinear_state_keeps_source_policy_and_lifetime() {
        use onnx_runtime_memory_api::{MemoryRole, Tier};
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let fixture = pack_fixture(3, 64, 96, 11);
        let bytes = mlas_sys::qgemm_pack_b_size(96, 64, false, false).unwrap();
        let governor = LedgerGovernor::new(LeaseLedger::new(0, bytes as u64, 0));
        let memory = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let mut first = QLinearMatMulKernel::new(true, memory.clone());
        let mut second = QLinearMatMulKernel::new(true, memory);
        for kernel in [&mut first, &mut second] {
            kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
        }
        let expected = run_with(&first, &fixture);
        assert!(
            second
                .adopt_shareable_constant_state(first.shareable_constant_state().unwrap())
                .unwrap()
        );
        assert!(Arc::ptr_eq(&first.packed_b, &second.packed_b));
        assert_eq!(run_with(&second, &fixture).bytes, expected.bytes);
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(first);
        let kv = governor
            .reserve(
                Tier::Host,
                bytes as u64,
                MemoryRole::KvCache,
                HolderId::new(7),
            )
            .unwrap();
        assert!(
            qlinear_pack_state(&second).is_none(),
            "shared cache registration must survive its originating kernel"
        );
        drop(kv);
        assert_eq!(run_with(&second, &fixture).bytes, expected.bytes);
        let key = QgemmPackKey {
            addr: fixture.b.bytes.as_ptr() as usize,
            k: 64,
            n: 96,
            a_signed: false,
            b_signed: false,
            flip_b: false,
        };
        let live = second.pack_lookup(key).unwrap().unwrap();
        drop(second);
        assert_eq!(governor.used(Tier::Host), bytes as u64);
        drop(live);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn shared_qlinear_state_rejects_foreign_sources_and_respects_retention() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 1 << 20, 0));
        let other = LedgerGovernor::new(LeaseLedger::new(0, 1 << 20, 0));
        let memory = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let constant = [false, false, false, true, false, false, false, false];
        let mut producer = QLinearMatMulKernel::new(true, memory.clone());
        producer.set_constant_inputs(&constant);
        let state = producer.shareable_constant_state().unwrap();
        for source in [
            DerivedWeightMemory::default(),
            DerivedWeightMemory::governed(&other, HolderId::new(13)).unwrap(),
            DerivedWeightMemory::governed(&governor, HolderId::new(14)).unwrap(),
            DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
        ] {
            let mut receiver = QLinearMatMulKernel::new(true, source);
            receiver.set_constant_inputs(&constant);
            let error = receiver
                .adopt_shareable_constant_state(state.clone())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("QLinearMatMul")
                    && error.contains("grant source")
                    && error.contains("CpuMemoryResources"),
                "{error}"
            );
            assert!(!Arc::ptr_eq(&receiver.packed_b, &producer.packed_b));
        }
        for (retention, constants) in [(false, constant), (true, [false; 8])] {
            let mut receiver = QLinearMatMulKernel::new(retention, memory.clone());
            receiver.set_constant_inputs(&constants);
            assert!(receiver.shareable_constant_state().is_none());
            assert!(
                !receiver
                    .adopt_shareable_constant_state(state.clone())
                    .unwrap()
            );
            assert!(!Arc::ptr_eq(&receiver.packed_b, &producer.packed_b));
        }
        let mut receiver = QLinearMatMulKernel::new(true, memory);
        receiver.set_constant_inputs(&constant);
        assert!(
            !receiver
                .adopt_shareable_constant_state(Arc::new(7usize))
                .unwrap()
        );
        assert!(receiver.adopt_shareable_constant_state(state).unwrap());
        assert!(Arc::ptr_eq(&receiver.packed_b, &producer.packed_b));
        assert_eq!(
            governor.used(Tier::Host),
            0,
            "sharing an empty state must reserve no payload"
        );
        assert_eq!(other.used(Tier::Host), 0);
        let mut local = QLinearMatMulKernel::default();
        let mut local_peer = QLinearMatMulKernel::default();
        local.set_constant_inputs(&constant);
        local_peer.set_constant_inputs(&constant);
        assert!(
            local_peer
                .adopt_shareable_constant_state(local.shareable_constant_state().unwrap())
                .unwrap()
        );
        assert!(Arc::ptr_eq(&local.packed_b, &local_peer.packed_b));
        assert!(
            receiver
                .adopt_shareable_constant_state(local.shareable_constant_state().unwrap())
                .is_err()
        );
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn qlinear_prediction_covers_single_matrix_shapes_and_declines_distinct_batches() {
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor};
        let mut observed_pack = false;
        for shape in [
            vec![64],
            vec![64, 96],
            vec![1, 64, 96],
            vec![1, 1, 64, 96],
            vec![2, 64, 96],
        ] {
            let n = if shape.len() == 1 { 1 } else { 96 };
            let mut fixture = pack_fixture(3, 64, n, 5);
            fixture.b = Owned::u8(&shape, &vec![129; shape.iter().product()]);
            let mut graph = Graph::new();
            let b = graph.create_named_value(
                "b",
                DataType::Uint8,
                shape.iter().map(|&d| Dim::Static(d)).collect(),
            );
            graph.set_initializer(
                b,
                onnx_runtime_ir::WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                    DataType::Uint8,
                    shape.clone(),
                    fixture.b.bytes.clone(),
                )),
            );
            let mut node = Node::new(
                onnx_runtime_ir::NodeId(0),
                "QLinearMatMul",
                vec![None, None, None, Some(b)],
                vec![],
            );
            let predicted = node_qlinear_packed_b_bytes(&node, &graph);
            let governor = LedgerGovernor::new(LeaseLedger::new(0, 1 << 20, 0));
            let mut kernel = QLinearMatMulKernel::new(
                true,
                DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap(),
            );
            kernel.set_constant_inputs(&[false, false, false, true, false, false, false, false]);
            let output_shape = Geometry::new(&fixture.a.shape, &fixture.b.shape)
                .unwrap()
                .output_shape;
            let mut output = Owned::zeros(DataType::Uint8, &output_shape);
            let mut scratch = vec![MaybeUninit::<i32>::uninit(); output.bytes.len()];
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
                scratch.len() * 4,
            );
            run_into(&kernel, &fixture, &mut output, Some(workspace));
            let mut reference = Owned::zeros(DataType::Uint8, &output_shape);
            run_into(
                &QLinearMatMulKernel::default(),
                &fixture,
                &mut reference,
                Some(workspace),
            );
            assert_eq!(output.bytes, reference.bytes);
            let actual = governor.used(Tier::Host);
            assert!(
                predicted >= actual,
                "shape={shape:?} prediction={predicted} actual={actual}"
            );
            if shape[0] == 2 {
                assert_eq!(predicted, 0);
                assert!(qlinear_pack_state(&kernel).is_none());
            } else if qlinear_pack_state(&kernel) == Some(true) {
                assert!(actual > 0);
                observed_pack = true;
            }
            graph.add_input(b);
            assert_eq!(
                node_qlinear_packed_b_bytes(&node, &graph),
                0,
                "overridable initializers are not immutable weights"
            );
            node.domain = "custom".into();
            assert_eq!(node_qlinear_packed_b_bytes(&node, &graph), 0);
        }
        assert!(
            observed_pack,
            "the bound must be checked against an actual packed allocation"
        );
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

            // Reuse both output and workspace. The fresh-buffer control above
            // allocates both per call; their difference measures that combined
            // lifecycle, not just result allocation.
            let mut reused = Owned::zeros(DataType::Uint8, &[m, n]);
            let mut scratch = vec![MaybeUninit::<i32>::uninit(); m * n];
            let workspace = WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(scratch.as_mut_ptr().cast()),
                scratch.len() * 4,
            );
            let start = std::time::Instant::now();
            for _ in 0..reps {
                run_into(&kernel, &fixture, &mut reused, Some(workspace));
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
                .unwrap()
                .unwrap();
            let packed = &packed.packed;
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
            let mut fused_products = vec![MaybeUninit::<i32>::uninit(); m * n];
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
                    &mut fused_products,
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
                    &mut fused_products,
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
                    &mut fused_products,
                );
            }
            let fused_fresh_a = start.elapsed() / reps;

            // Same call, but the i32 accumulator is re-allocated per iteration
            // as an explicit forced-cold control. The kernel now borrows its
            // accumulator workspace, so this allocation is outside that path.
            let start = std::time::Instant::now();
            for _ in 0..reps {
                let mut fresh_c = vec![MaybeUninit::<i32>::uninit(); m * n];
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
            kernel.pack_build(key, &[7u8; 8]).unwrap().is_none(),
            "a byte count that is not k * n must be declined, not packed"
        );
        assert!(
            kernel
                .pack_build(key, &vec![7u8; 64 * 96])
                .unwrap()
                .is_some(),
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
            .execute_prepared(
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
            .execute_prepared(
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
