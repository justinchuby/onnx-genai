//! Residency-preserving tensor algebra for interpreter constructs.
//!
//! An interpreter-level construct — a speculative proposal chain, above all —
//! needs to narrow, truncate, gather, assemble and argmax tensors that a
//! component just produced. When that component ran on a device, those tensors
//! are device-resident, and the obvious implementation (copy to host, operate,
//! copy back) is correct and catastrophically slow: a per-draft-token argmax of
//! a `[1, 1, vocab]` f16 row is ≈300 KiB down the PCIe bus to learn one
//! four-byte token id, and a per-step `concat(embed(token), carry)` assembled
//! on the host is that same bus twice more.
//!
//! # The rule
//!
//! Every method here returns a value in the **same residency** as its principal
//! input, or fails saying why. There is deliberately no fallback: a silent host
//! round trip is exactly the behaviour this exists to remove, and one that only
//! shows up as a throughput number is the kind of regression nobody attributes.
//! Mixing residencies within one operation — scattering a device carry into a
//! host buffer — is rejected for the same reason.
//!
//! # One primitive, three narrowings
//!
//! [`ResidentTensorOps::slice_axis`] is the only narrowing implemented;
//! `last_along_axis` and `truncate_axis` are the two windows a proposal chain
//! actually asks for, expressed in terms of it. That is deliberate: two code
//! paths answering "which elements survive" would be duplicated state, and the
//! one that is exercised less is the one that would drift.
//!
//! # The one exception to residency preservation
//!
//! A result with *no elements* borrows no bytes and has no device address to
//! publish — CUDA cannot allocate zero bytes, and ORT is entitled to hand back
//! a null data pointer for such a tensor. An empty result is therefore
//! published host-resident, which is how the component boundary already
//! publishes an empty output. Emptiness is checked before anything else, so
//! this answer never depends on which branch happened to be reachable.
//!
//! # What is and is not zero-copy
//!
//! Narrowing an axis whose leading axes are all extent 1 keeps a contiguous
//! window, so it is a pointer view — free on any backend, via
//! [`Value::alias_with_offset`] — provided the value has an owner whose
//! lifetime the view can borrow. Narrowing with a non-unit leading extent is
//! strided and is not expressible as a view; it is a copy performed *on
//! whatever owns the buffer*: host bytes for a host tensor, device-to-device
//! copies for a device tensor. Neither case crosses the bus.

use anyhow::Context as _;
use onnx_genai_ort::{DataType, Value};

/// The workflow value currency's element type for a graph element type.
///
/// Partial on purpose: a graph may declare element types the interpreter's
/// value pool does not carry (`Float64`, `String`, sub-byte ints). `None` says
/// so, and lets each caller name the artifact, initializer or port it was
/// reading rather than emitting one context-free diagnostic for all of them.
pub(crate) fn value_dtype_from_ir(dtype: onnx_runtime_ir::DataType) -> Option<DataType> {
    use onnx_runtime_ir::DataType as Ir;
    Some(match dtype {
        Ir::Float32 => DataType::Float32,
        Ir::Float16 => DataType::Float16,
        Ir::BFloat16 => DataType::BFloat16,
        // ORT's single-precision f8 spellings are the ONNX FN/(non-UZ) variants.
        Ir::Float8E4M3FN => DataType::Float8E4M3,
        Ir::Float8E5M2 => DataType::Float8E5M2,
        Ir::Int8 => DataType::Int8,
        Ir::Int16 => DataType::Int16,
        Ir::Int32 => DataType::Int32,
        Ir::Int64 => DataType::Int64,
        Ir::Uint8 => DataType::Uint8,
        Ir::Uint16 => DataType::Uint16,
        Ir::Uint32 => DataType::Uint32,
        Ir::Uint64 => DataType::Uint64,
        Ir::Bool => DataType::Bool,
        _ => return None,
    })
}

/// Where a value lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Residency {
    Host,
    /// Constructed only by the CUDA implementation, which a build without the
    /// `ort-cuda` feature does not compile.
    #[cfg_attr(not(feature = "ort-cuda"), allow(dead_code))]
    Cuda(i32),
}

impl Residency {
    /// A stable key for caching a value mirrored into this residency.
    ///
    /// Host is `-1` because no CUDA ordinal is negative, so host and device
    /// mirrors of the same table can never collide in one map.
    pub(crate) fn cache_key(self) -> i32 {
        match self {
            Residency::Host => -1,
            Residency::Cuda(device) => device,
        }
    }
}

impl std::fmt::Display for Residency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Residency::Host => write!(f, "host"),
            Residency::Cuda(device) => write!(f, "CUDA device {device}"),
        }
    }
}

/// Where `value`'s bytes live.
pub(crate) fn residency_of(value: &Value) -> anyhow::Result<Residency> {
    if value.is_host_resident()? {
        return Ok(Residency::Host);
    }
    Ok(Residency::Cuda(value.device_id()?))
}

/// Tensor operations that never change a value's residency.
pub(crate) trait ResidentTensorOps {
    /// Where values produced by this backend live.
    ///
    /// Read by the invariant every method below is written to: what comes out
    /// is resident where what went in was. A caller asserting that is asserting
    /// the property, not the implementation.
    fn residency(&self) -> Residency;

    /// Bring `value` into this residency, copying it if it is not already here.
    ///
    /// The one sanctioned crossing, and the reason every other method refuses
    /// to make one: a construct that must operate on a value produced
    /// elsewhere says so, once, at a point a reader can see — rather than each
    /// operation quietly staging its operands. A chain whose seed the target
    /// left on the host adopts it once per proposal; what it must never do is
    /// pay that crossing per draft token.
    fn adopt(&self, value: &Value) -> anyhow::Result<Value>;

    /// A tensor of `shape` and `dtype`, zero-filled, in this residency.
    ///
    /// The destination half of "assemble where the operands are": a fused
    /// proposer input allocated on the host and uploaded is a host round trip
    /// under another name.
    fn zeros(&self, shape: &[i64], dtype: DataType) -> anyhow::Result<Value>;

    /// Keep `len` indices of `axis` starting at `start`, preserving rank.
    fn slice_axis(
        &self,
        value: &Value,
        axis: usize,
        start: usize,
        len: usize,
    ) -> anyhow::Result<Value>;

    /// Row-gather `table[id]` for each id, into a `[ids.len(), hidden]` value.
    ///
    /// `table` is a rank-2 `[vocab, hidden]` value in this residency. An id
    /// outside the table is an error rather than a clamp onto row 0: a proposer
    /// that drafted an id the table cannot embed has left the declared
    /// vocabulary, and silently embedding row 0 would hide that.
    fn gather_rows(&self, table: &Value, ids: &[i64]) -> anyhow::Result<Value>;

    /// Write `src` into `dst[.., feature_offset .. feature_offset + src_width]`
    /// in place, where `src_width` is `src`'s trailing extent.
    ///
    /// This is how a fused input is assembled without a host buffer: each half
    /// is written into its own segment of a destination that never moves. Both
    /// values must be in this residency and hold the same number of rows.
    fn scatter_into_last_axis(
        &self,
        dst: &Value,
        feature_offset: usize,
        src: &Value,
    ) -> anyhow::Result<()>;

    /// Argmax each contiguous `vocab`-wide row, returning token ids.
    ///
    /// The only sanctioned device→host transfer in a proposal loop: four bytes
    /// per row, rather than a vocabulary per row.
    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>>;

    /// Keep only the final index of `axis`, preserving rank.
    fn last_along_axis(&self, value: &Value, axis: usize) -> anyhow::Result<Value> {
        let extent = axis_extent(value, axis)?;
        anyhow::ensure!(extent > 0, "cannot take the last position of an empty axis");
        self.slice_axis(value, axis, extent - 1, 1)
    }

    /// Keep the leading `length` indices of `axis`, preserving rank.
    fn truncate_axis(&self, value: &Value, axis: usize, length: usize) -> anyhow::Result<Value> {
        let extent = axis_extent(value, axis)?;
        anyhow::ensure!(
            length <= extent,
            "cannot truncate to {length} positions along axis {axis}: the tensor holds {extent}"
        );
        self.slice_axis(value, axis, 0, length)
    }
}

/// `value`'s extent along `axis`, or an actionable error.
fn axis_extent(value: &Value, axis: usize) -> anyhow::Result<usize> {
    let shape = value.shape();
    anyhow::ensure!(
        axis < shape.len(),
        "axis {axis} is out of range for a rank-{} tensor with shape {shape:?}",
        shape.len()
    );
    usize::try_from(shape[axis]).context("negative tensor extent")
}

/// The `(outer, extent, inner)` element decomposition of `axis`.
///
/// `outer` is how many independent blocks the axis repeats within, and `inner`
/// how many contiguous elements each of its indices spans. A window is
/// contiguous — and therefore a view — exactly when `outer == 1`.
fn axis_layout(shape: &[i64], axis: usize) -> anyhow::Result<(usize, usize, usize)> {
    let dims = shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).context("negative tensor extent"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let outer = dims[..axis]
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
    let inner = dims[axis + 1..]
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
    Ok((
        outer.context("tensor element count overflows usize")?,
        dims[axis],
        inner.context("tensor element count overflows usize")?,
    ))
}

/// Validate a slice request and return the narrowed shape plus its layout.
fn slice_plan(
    value: &Value,
    axis: usize,
    start: usize,
    len: usize,
) -> anyhow::Result<(Vec<i64>, usize, usize, usize)> {
    let shape = value.shape().to_vec();
    let extent = axis_extent(value, axis)?;
    let end = start
        .checked_add(len)
        .context("slice window start + length overflows usize")?;
    anyhow::ensure!(
        end <= extent,
        "slice window {start}..{end} along axis {axis} leaves a tensor of shape {shape:?}, whose \
         axis holds {extent}"
    );
    let (outer, _, inner) = axis_layout(&shape, axis)?;
    let mut narrowed = shape;
    narrowed[axis] = i64::try_from(len).context("slice length exceeds i64")?;
    Ok((narrowed, outer, inner, extent))
}

/// A zero-copy view of a contiguous window, when the value can lend its buffer.
///
/// A value that owns its allocation (or is already a view of one) can be
/// aliased in O(1), and the alias keeps the owner — and with it any device
/// allocation or IO binding — alive for as long as the view exists. A value
/// with no shareable owner has no lifetime to borrow, so it must be copied.
fn contiguous_view(
    value: &Value,
    element_offset: usize,
    shape: &[i64],
) -> anyhow::Result<Option<Value>> {
    let Some(aliased) = value.try_alias_clone() else {
        return Ok(None);
    };
    let aliased =
        aliased.map_err(|error| anyhow::anyhow!("failed to alias a resident value: {error}"))?;
    // `Value` is deliberately neither Send nor Sync — an OrtValue belongs to the
    // thread that made it — and `Arc` is what the alias backing takes, so this
    // is the shape the production path already uses.
    #[allow(clippy::arc_with_non_send_sync)]
    let owner = std::sync::Arc::new(aliased);
    Value::alias_with_offset(owner, element_offset, shape)
        .map(Some)
        .map_err(|error| anyhow::anyhow!("failed to view a resident value: {error}"))
}

/// Clone a value without changing where it lives.
///
/// A value backed by an allocation it owns is aliased in O(1), so a device
/// tensor stays on the device; only an owned-host-backing value is deep-copied.
fn clone_resident(value: &Value) -> anyhow::Result<Value> {
    if let Some(aliased) = value.try_alias_clone() {
        return aliased.map_err(|error| anyhow::anyhow!("failed to alias a value: {error}"));
    }
    value
        .clone_owned()
        .map_err(|error| anyhow::anyhow!("failed to clone a value: {error}"))
}

/// Rows and trailing width of a `[.., width]` value.
fn rows_and_width(value: &Value, role: &str) -> anyhow::Result<(usize, usize)> {
    let shape = value.shape();
    let width = usize::try_from(
        *shape
            .last()
            .with_context(|| format!("the {role} has rank 0, so it has no feature axis"))?,
    )
    .context("negative tensor extent")?;
    anyhow::ensure!(
        width > 0,
        "the {role} has a zero-width feature axis (shape {shape:?})"
    );
    Ok((value.numel() / width, width))
}

/// Validate a `[vocab, hidden]` gather table and the ids being gathered.
fn gather_plan(table: &Value, ids: &[i64]) -> anyhow::Result<(usize, usize)> {
    let shape = table.shape();
    anyhow::ensure!(
        shape.len() == 2,
        "a gather table must be a rank-2 [vocab, hidden] matrix, found shape {shape:?}"
    );
    let vocab = usize::try_from(shape[0]).context("negative vocabulary extent")?;
    let hidden = usize::try_from(shape[1]).context("negative hidden extent")?;
    anyhow::ensure!(
        vocab > 0 && hidden > 0,
        "a gather table must have a non-empty [vocab, hidden] shape, found {shape:?}"
    );
    for id in ids {
        let index = usize::try_from(*id).ok().filter(|index| *index < vocab);
        anyhow::ensure!(
            index.is_some(),
            "token id {id} has no row in a [{vocab}, {hidden}] gather table"
        );
    }
    Ok((vocab, hidden))
}

/// Validate a scatter and return `(rows, dst_width, src_width)`.
fn scatter_plan(
    dst: &Value,
    feature_offset: usize,
    src: &Value,
) -> anyhow::Result<(usize, usize, usize)> {
    anyhow::ensure!(
        dst.dtype() == src.dtype(),
        "a scatter cannot change element type: the destination is {:?} and the source is {:?}",
        dst.dtype(),
        src.dtype()
    );
    let (dst_rows, dst_width) = rows_and_width(dst, "scatter destination")?;
    let (src_rows, src_width) = rows_and_width(src, "scatter source")?;
    anyhow::ensure!(
        dst_rows == src_rows,
        "a scatter writes one source row per destination row, but the destination has {dst_rows} \
         (shape {:?}) and the source has {src_rows} (shape {:?})",
        dst.shape(),
        src.shape()
    );
    let end = feature_offset
        .checked_add(src_width)
        .context("scatter window overflows usize")?;
    anyhow::ensure!(
        end <= dst_width,
        "a scatter of {src_width} features at offset {feature_offset} runs past the destination's \
         {dst_width}-wide feature axis (shape {:?})",
        dst.shape()
    );
    Ok((dst_rows, dst_width, src_width))
}

/// Host-resident implementation.
pub(crate) struct HostTensorOps;

impl HostTensorOps {
    fn ensure_host(&self, value: &Value, role: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            value.is_host_resident()?,
            "the host tensor operations were asked to use a {role} resident on {}; an operation \
             never moves a value between residencies. Run this component on the host backend, or \
             keep the whole operation on the device that produced this value.",
            residency_of(value)?
        );
        Ok(())
    }
}

impl ResidentTensorOps for HostTensorOps {
    fn residency(&self) -> Residency {
        Residency::Host
    }

    fn adopt(&self, value: &Value) -> anyhow::Result<Value> {
        match residency_of(value)? {
            Residency::Host => clone_resident(value),
            Residency::Cuda(device) => value.to_host_from_cuda(device).map_err(|error| {
                anyhow::anyhow!(
                    "failed to adopt a {:?} {:?} tensor from CUDA device {device} onto the host: \
                     {error}",
                    value.shape(),
                    value.dtype()
                )
            }),
        }
    }

    fn zeros(&self, shape: &[i64], dtype: DataType) -> anyhow::Result<Value> {
        let numel = shape
            .iter()
            .try_fold(1usize, |total, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| total.checked_mul(dimension))
            })
            .with_context(|| format!("unusable tensor shape {shape:?}"))?;
        Value::from_raw_bytes(vec![0u8; numel * dtype.size_of()], shape, dtype)
            .map_err(|error| anyhow::anyhow!("failed to allocate a host tensor: {error}"))
    }

    fn slice_axis(
        &self,
        value: &Value,
        axis: usize,
        start: usize,
        len: usize,
    ) -> anyhow::Result<Value> {
        self.ensure_host(value, "slice input")?;
        let (narrowed, outer, inner, extent) = slice_plan(value, axis, start, len)?;
        if outer == 1
            && let Some(view) = contiguous_view(value, start * inner, &narrowed)?
        {
            return Ok(view);
        }
        let element = value.dtype().size_of();
        let bytes = value.as_raw_bytes()?;
        let block = extent * inner * element;
        let keep = len * inner * element;
        let skip = start * inner * element;
        let mut narrowed_bytes = Vec::with_capacity(outer * keep);
        for index in 0..outer {
            let begin = index * block + skip;
            narrowed_bytes.extend_from_slice(&bytes[begin..begin + keep]);
        }
        Value::from_raw_bytes(narrowed_bytes, &narrowed, value.dtype())
            .map_err(|error| anyhow::anyhow!("failed to narrow a host value: {error}"))
    }

    fn gather_rows(&self, table: &Value, ids: &[i64]) -> anyhow::Result<Value> {
        self.ensure_host(table, "gather table")?;
        let (_, hidden) = gather_plan(table, ids)?;
        let element = table.dtype().size_of();
        let bytes = table.as_raw_bytes()?;
        let mut gathered = Vec::with_capacity(ids.len() * hidden * element);
        for id in ids {
            let start = (*id as usize) * hidden * element;
            gathered.extend_from_slice(&bytes[start..start + hidden * element]);
        }
        let shape = [
            i64::try_from(ids.len()).context("gathered row count exceeds i64")?,
            i64::try_from(hidden).context("gathered row width exceeds i64")?,
        ];
        Value::from_raw_bytes(gathered, &shape, table.dtype())
            .map_err(|error| anyhow::anyhow!("failed to materialize gathered rows: {error}"))
    }

    fn scatter_into_last_axis(
        &self,
        dst: &Value,
        feature_offset: usize,
        src: &Value,
    ) -> anyhow::Result<()> {
        self.ensure_host(dst, "scatter destination")?;
        self.ensure_host(src, "scatter source")?;
        let (rows, dst_width, src_width) = scatter_plan(dst, feature_offset, src)?;
        let element = dst.dtype().size_of();
        let source = src.as_raw_bytes()?;
        for row in 0..rows {
            let begin = row * src_width * element;
            dst.write_raw_bytes_at(
                row * dst_width + feature_offset,
                &source[begin..begin + src_width * element],
            )
            .map_err(|error| anyhow::anyhow!("failed to write a scattered row: {error}"))?;
        }
        Ok(())
    }

    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>> {
        self.ensure_host(logits, "argmax input")?;
        anyhow::ensure!(
            rows == 1,
            "the host argmax path selects one row at a time; {rows} were requested"
        );
        Ok(vec![logits.argmax_last_row()?])
    }
}

/// CUDA-resident implementation.
///
/// Holds only the device ordinal: every operation either produces a pointer
/// view over the caller's own value, copies device-to-device, or launches a
/// kernel that reads the ORT device pointer directly.
#[cfg(feature = "ort-cuda")]
pub(crate) struct CudaTensorOps {
    device: i32,
}

#[cfg(feature = "ort-cuda")]
impl CudaTensorOps {
    pub(crate) fn new(device: i32) -> Self {
        Self { device }
    }

    /// Reject a value that is not on this implementation's own device.
    ///
    /// Both halves matter: a host value here would be read through a device
    /// pointer, and a value on a *different* device would be copied across a
    /// peer link this seam never established.
    fn ensure_device(&self, value: &Value, role: &str) -> anyhow::Result<()> {
        let residency = residency_of(value)?;
        anyhow::ensure!(
            residency == Residency::Cuda(self.device),
            "the CUDA device {} tensor operations were asked to use a {role} resident on {}; an \
             operation never moves a value between residencies. Route both operands to the same \
             device, or run this component on the backend that owns this value.",
            self.device,
            residency
        );
        Ok(())
    }

    /// Copy `count` bytes device-to-device on this implementation's device.
    ///
    /// Unfenced: the caller issues its whole write batch and then calls
    /// [`Self::fence`] once. Fencing per copy would turn a row-wise gather into
    /// one device-wide barrier per row.
    fn copy(&self, destination: usize, source: usize, count: usize) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        let _guard = onnx_genai_ort::cuda_rt::DeviceGuard::set(self.device)?;
        onnx_genai_ort::cuda_rt::memcpy_device_to_device(destination, source, count)
            .map_err(Into::into)
    }

    /// Make every device write this operation issued visible to the execution
    /// provider's kernels.
    ///
    /// The copies above run on cudart's *legacy default stream*; both CUDA
    /// execution providers run kernels on streams created with
    /// `cudaStreamNonBlocking`, which are exempt from the legacy stream's
    /// implicit ordering. Without this barrier a kernel launched immediately
    /// after — the proposer step reading the fused input this seam just
    /// assembled, or the target reading the KV cell a rejection just truncated
    /// — can read a buffer that is still being filled. That is a silent wrong
    /// answer proportional to how much of the copy is outstanding: never on a
    /// 128-byte fixture, megabytes on a real cache.
    ///
    /// The same barrier, for the same reason, brackets the shared-KV grow copy;
    /// see `onnx_genai_ort::cuda_rt::device_synchronize`.
    ///
    /// The other direction needs nothing here: every device value that *enters*
    /// this seam was published by the component boundary, which drains the
    /// producing stream before the value exists (`value_from_output_binding`
    /// and `device_tensor_to_value` in `native_component.rs`), so a producer
    /// kernel is never still writing what these copies read.
    ///
    /// The barrier is per *operation*, not per proposal step, which is two to
    /// three more than correctness strictly needs — the writes of one step all
    /// land on the same legacy stream, so only the last batch before a launch
    /// has to be waited on. Hoisting it to the caller was rejected: a fence a
    /// caller can forget buys microseconds and pays for them in a class of bug
    /// that presents as wrong output on large models and as nothing at all on
    /// the fixtures. What it replaces is two to three orders of magnitude
    /// larger anyway — a vocabulary-wide download per draft token and a whole
    /// KV cache per rejection. The way to make it free is to issue the copies
    /// on the execution provider's own stream so they are kernel-ordered
    /// without any barrier; that needs the provider's stream handle at this
    /// seam, which it does not have today.
    fn fence(&self) -> anyhow::Result<()> {
        let _guard = onnx_genai_ort::cuda_rt::DeviceGuard::set(self.device)?;
        onnx_genai_ort::cuda_rt::device_synchronize().map_err(Into::into)
    }

    /// A zeroed device tensor, without the fence.
    ///
    /// Used as the destination of a copy batch the caller fences at the end, so
    /// allocating a destination costs no extra barrier.
    fn zeroed_unfenced(&self, shape: &[i64], dtype: DataType) -> anyhow::Result<Value> {
        let value = Value::empty_cuda(shape, dtype, self.device).with_context(|| {
            format!(
                "failed to allocate a {shape:?} {dtype:?} tensor on CUDA device {}",
                self.device
            )
        })?;
        value.fill_zero_device(self.device)?;
        Ok(value)
    }
}

#[cfg(feature = "ort-cuda")]
impl ResidentTensorOps for CudaTensorOps {
    fn residency(&self) -> Residency {
        Residency::Cuda(self.device)
    }

    fn adopt(&self, value: &Value) -> anyhow::Result<Value> {
        if residency_of(value)? == Residency::Cuda(self.device) {
            return clone_resident(value);
        }
        let adopted =
            Value::empty_cuda(value.shape(), value.dtype(), self.device).with_context(|| {
                format!(
                    "failed to allocate a {:?} {:?} tensor on CUDA device {}",
                    value.shape(),
                    value.dtype(),
                    self.device
                )
            })?;
        adopted
            .copy_from_cuda(value, self.device)
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to adopt a {:?} {:?} tensor from {} onto CUDA device {}: {error}",
                    value.shape(),
                    value.dtype(),
                    residency_of(value)
                        .map(|residency| residency.to_string())
                        .unwrap_or_else(|_| "an unreadable residency".to_string()),
                    self.device
                )
            })?;
        self.fence()?;
        Ok(adopted)
    }

    fn zeros(&self, shape: &[i64], dtype: DataType) -> anyhow::Result<Value> {
        // A tensor with no elements has no device address to publish; the
        // module's one exception to residency preservation applies here too, so
        // that the rule holds in every branch rather than in most of them.
        if shape.contains(&0) {
            return HostTensorOps.zeros(shape, dtype);
        }
        let value = self.zeroed_unfenced(shape, dtype)?;
        self.fence()?;
        Ok(value)
    }

    fn slice_axis(
        &self,
        value: &Value,
        axis: usize,
        start: usize,
        len: usize,
    ) -> anyhow::Result<Value> {
        self.ensure_device(value, "slice input")?;
        let (narrowed, outer, inner, extent) = slice_plan(value, axis, start, len)?;
        // Emptiness is decided before anything else so the answer cannot depend
        // on which branch happened to be reachable; see the module's note on the
        // one exception to residency preservation.
        if len == 0 || inner == 0 || outer == 0 {
            return HostTensorOps.zeros(&narrowed, value.dtype());
        }
        // A contiguous window is a pointer view: no bytes move at all, on any
        // backend. This is what makes a rejection rollback free.
        if outer == 1
            && let Some(view) = contiguous_view(value, start * inner, &narrowed)?
        {
            return Ok(view);
        }
        // Strided: one device-to-device copy per leading block. Still no host
        // round trip, which is the property that matters.
        let element = value.dtype().size_of();
        let destination = self.zeroed_unfenced(&narrowed, value.dtype())?;
        let source_base = value.data_ptr_addr()?;
        let destination_base = destination.data_ptr_addr()?;
        let block = extent * inner * element;
        let keep = len * inner * element;
        let skip = start * inner * element;
        for index in 0..outer {
            self.copy(
                destination_base + index * keep,
                source_base + index * block + skip,
                keep,
            )?;
        }
        self.fence()?;
        Ok(destination)
    }

    fn gather_rows(&self, table: &Value, ids: &[i64]) -> anyhow::Result<Value> {
        self.ensure_device(table, "gather table")?;
        let (_, hidden) = gather_plan(table, ids)?;
        let element = table.dtype().size_of();
        let shape = [
            i64::try_from(ids.len()).context("gathered row count exceeds i64")?,
            i64::try_from(hidden).context("gathered row width exceeds i64")?,
        ];
        if ids.is_empty() {
            return HostTensorOps.zeros(&shape, table.dtype());
        }
        let gathered = self.zeroed_unfenced(&shape, table.dtype())?;
        let table_base = table.data_ptr_addr()?;
        let gathered_base = gathered.data_ptr_addr()?;
        let row_bytes = hidden * element;
        for (row, id) in ids.iter().enumerate() {
            self.copy(
                gathered_base + row * row_bytes,
                table_base + (*id as usize) * row_bytes,
                row_bytes,
            )?;
        }
        self.fence()?;
        Ok(gathered)
    }

    fn scatter_into_last_axis(
        &self,
        dst: &Value,
        feature_offset: usize,
        src: &Value,
    ) -> anyhow::Result<()> {
        self.ensure_device(dst, "scatter destination")?;
        self.ensure_device(src, "scatter source")?;
        let (rows, dst_width, src_width) = scatter_plan(dst, feature_offset, src)?;
        let element = dst.dtype().size_of();
        let destination_base = dst.data_ptr_addr()?;
        let source_base = src.data_ptr_addr()?;
        for row in 0..rows {
            self.copy(
                destination_base + (row * dst_width + feature_offset) * element,
                source_base + row * src_width * element,
                src_width * element,
            )?;
        }
        self.fence()
    }

    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>> {
        self.ensure_device(logits, "argmax input")?;
        let shape = logits.shape();
        let vocab = usize::try_from(*shape.last().context("logits have no trailing axis")?)
            .context("negative vocabulary extent")?;
        anyhow::ensure!(
            vocab > 0 && logits.numel() == rows * vocab,
            "device argmax expects {rows} contiguous rows of {vocab}, but the value has shape \
             {shape:?}"
        );
        onnx_genai_ort::device_sampler::device_argmax_rows(
            usize::try_from(self.device).context("negative CUDA device ordinal")?,
            logits.dtype(),
            logits.data_ptr_addr()?,
            rows,
            vocab,
        )
        .map_err(Into::into)
    }
}

/// The operations for `residency`, or a refusal naming both remedies.
///
/// Rule 4: a device-resident value with no device implementation available is
/// an error, never a quiet copy to the host. The two things an operator can
/// actually do — select the host backend, or build with the CUDA feature — are
/// both named, because "unsupported" alone sends someone hunting a bug that is
/// really a build configuration.
pub(crate) fn tensor_ops_for_residency(
    residency: Residency,
) -> anyhow::Result<Box<dyn ResidentTensorOps>> {
    match residency {
        Residency::Host => Ok(Box::new(HostTensorOps)),
        #[cfg(feature = "ort-cuda")]
        Residency::Cuda(device) => Ok(Box::new(CudaTensorOps::new(device))),
        #[cfg(not(feature = "ort-cuda"))]
        Residency::Cuda(device) => anyhow::bail!(
            "this value is resident on CUDA device {device}, and this build has no device tensor \
             operations to narrow, gather or score it without copying it to the host. Rebuild \
             with the `ort-cuda` (or `native-cuda`) feature, or run this package on the host \
             backend."
        ),
    }
}

/// The operations that preserve `value`'s residency.
pub(crate) fn tensor_ops_for(value: &Value) -> anyhow::Result<Box<dyn ResidentTensorOps>> {
    tensor_ops_for_residency(residency_of(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host implementation narrows the axis it was asked for.
    #[test]
    fn host_last_along_axis_keeps_the_final_index() -> anyhow::Result<()> {
        let value = Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2])?;
        let last = HostTensorOps.last_along_axis(&value, 1)?;
        assert_eq!(last.shape(), &[1, 1, 2]);
        assert_eq!(last.to_vec_f32_lossy()?, vec![5.0, 6.0]);
        Ok(())
    }

    /// Narrowing an inner axis with a non-unit leading extent is strided, and
    /// the host implementation still produces the right elements.
    #[test]
    fn host_slice_axis_handles_a_strided_window() -> anyhow::Result<()> {
        // [2, 3, 2]: two blocks of three positions of two features.
        let value = Value::from_slice_f32(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            &[2, 3, 2],
        )?;
        let middle = HostTensorOps.slice_axis(&value, 1, 1, 2)?;
        assert_eq!(middle.shape(), &[2, 2, 2]);
        assert_eq!(
            middle.to_vec_f32_lossy()?,
            vec![2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0]
        );
        Ok(())
    }

    /// `last_along_axis` and `truncate_axis` are windows of one primitive, so a
    /// change to the primitive cannot make them disagree with it.
    #[test]
    fn the_narrowings_are_windows_of_one_primitive() -> anyhow::Result<()> {
        let value = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 3, 2])?;
        assert_eq!(
            HostTensorOps
                .last_along_axis(&value, 1)?
                .to_vec_f32_lossy()?,
            HostTensorOps
                .slice_axis(&value, 1, 2, 1)?
                .to_vec_f32_lossy()?
        );
        assert_eq!(
            HostTensorOps
                .truncate_axis(&value, 1, 2)?
                .to_vec_f32_lossy()?,
            HostTensorOps
                .slice_axis(&value, 1, 0, 2)?
                .to_vec_f32_lossy()?
        );
        Ok(())
    }

    /// A host value resolves to host operations, so nothing about a CPU-only
    /// run changes.
    #[test]
    fn a_host_value_resolves_to_host_operations() -> anyhow::Result<()> {
        let value = Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?;
        assert_eq!(tensor_ops_for(&value)?.residency(), Residency::Host);
        Ok(())
    }

    /// Host argmax matches the value's own convention, so a device
    /// implementation substituting for it cannot change which token is chosen.
    #[test]
    fn host_argmax_agrees_with_the_value_convention() -> anyhow::Result<()> {
        let logits = Value::from_slice_f32(&[0.1, 0.9, 0.4], &[1, 1, 3])?;
        assert_eq!(
            HostTensorOps.argmax_rows(&logits, 1)?,
            vec![logits.argmax_last_row()?]
        );
        Ok(())
    }

    /// The gather takes the rows it was asked for, in the order it was asked.
    #[test]
    fn host_gather_rows_selects_declared_rows() -> anyhow::Result<()> {
        let table = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2])?;
        let gathered = HostTensorOps.gather_rows(&table, &[2, 0, 2])?;
        assert_eq!(gathered.shape(), &[3, 2]);
        assert_eq!(
            gathered.to_vec_f32_lossy()?,
            vec![4.0, 5.0, 0.0, 1.0, 4.0, 5.0]
        );
        Ok(())
    }

    /// An id outside the table is an error, not a clamp onto row 0.
    #[test]
    fn a_gather_id_outside_the_table_is_an_error() -> anyhow::Result<()> {
        let table = Value::from_slice_f32(&[0.0, 1.0, 2.0, 3.0], &[2, 2])?;
        let Err(error) = HostTensorOps.gather_rows(&table, &[2]) else {
            panic!("an out-of-range id must be rejected");
        };
        let message = error.to_string();
        assert!(message.contains("token id 2"), "{message}");
        assert!(message.contains("[2, 2]"), "{message}");
        assert!(
            HostTensorOps.gather_rows(&table, &[-1]).is_err(),
            "a negative id must be rejected"
        );
        Ok(())
    }

    /// Two scatters assemble a fused row without allocating a third buffer.
    #[test]
    fn host_scatter_assembles_both_halves_in_place() -> anyhow::Result<()> {
        let fused = HostTensorOps.zeros(&[1, 1, 4], DataType::Float32)?;
        let embed = Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?;
        let carry = Value::from_slice_f32(&[3.0, 4.0], &[1, 2])?;
        HostTensorOps.scatter_into_last_axis(&fused, 0, &embed)?;
        HostTensorOps.scatter_into_last_axis(&fused, 2, &carry)?;
        assert_eq!(fused.to_vec_f32_lossy()?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    /// A scatter that would run past the destination's feature axis is
    /// rejected, naming both widths.
    #[test]
    fn a_scatter_past_the_feature_axis_is_rejected() -> anyhow::Result<()> {
        let fused = HostTensorOps.zeros(&[1, 1, 3], DataType::Float32)?;
        let source = Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?;
        let message = HostTensorOps
            .scatter_into_last_axis(&fused, 2, &source)
            .expect_err("an overrunning scatter must be rejected")
            .to_string();
        assert!(message.contains("runs past"), "{message}");
        Ok(())
    }

    /// A scatter between values holding different row counts is rejected
    /// rather than silently writing the rows it can.
    #[test]
    fn a_scatter_with_mismatched_rows_is_rejected() -> anyhow::Result<()> {
        let fused = HostTensorOps.zeros(&[2, 1, 4], DataType::Float32)?;
        let source = Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?;
        let message = HostTensorOps
            .scatter_into_last_axis(&fused, 0, &source)
            .expect_err("a mismatched scatter must be rejected")
            .to_string();
        assert!(
            message.contains("one source row per destination row"),
            "{message}"
        );
        Ok(())
    }
}

/// The CUDA implementation, against the host implementation it substitutes for.
///
/// Every case here is a differential test: the device answer must be the host
/// answer, element for element. That is the only thing that makes substituting
/// one for the other safe, and it is what a token-level parity test cannot see —
/// a device path that produced *nearly* the right bytes would still decode the
/// same short sequence on a tiny fixture.
#[cfg(feature = "ort-cuda")]
#[cfg(test)]
mod cuda_tests {
    use super::*;

    /// CUDA device 0, or `None` on a machine that has none.
    ///
    /// The feature says this build *can* reach a device, not that one is
    /// present; a developer building with CUDA on a laptop still runs the
    /// suite. The probe is deliberately *not* one of the primitives under test:
    /// gating `empty_cuda`'s own tests on `empty_cuda` succeeding would turn a
    /// regression in it into seven silently green tests.
    fn device() -> Option<i32> {
        onnx_genai_ort::cuda_rt::device_memory_info(0)
            .ok()
            .map(|_| 0)
    }

    /// A host value copied onto the device, for use as a device operand.
    fn upload(host: &Value, device: i32) -> anyhow::Result<Value> {
        let resident = Value::empty_cuda(host.shape(), host.dtype(), device)?;
        resident.copy_from_cuda(host, device)?;
        Ok(resident)
    }

    /// `value`'s elements, wherever it lives.
    ///
    /// A zero-element tensor borrows no bytes and has no address to copy from,
    /// on either side, so it reads as the empty vector it is.
    fn read(value: &Value, device: i32) -> anyhow::Result<Vec<f32>> {
        if value.numel() == 0 {
            return Ok(Vec::new());
        }
        match value.is_host_resident()? {
            true => Ok(value.to_vec_f32()?),
            false => Ok(value.to_host_from_cuda(device)?.to_vec_f32()?),
        }
    }

    /// Narrowing on the device produces exactly what narrowing on the host
    /// produces, contiguous windows and strided ones alike.
    #[test]
    fn device_slices_match_host_slices() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        // Shapes chosen so every axis is exercised both as a contiguous window
        // (leading extents all 1) and as a strided one (a non-unit leading
        // extent), which are the two implementations the device path picks
        // between.
        let cases: &[&[i64]] = &[&[1, 5, 3], &[2, 3, 4], &[1, 2, 4, 2], &[3, 1, 2]];
        for shape in cases {
            let numel: usize = shape.iter().map(|dimension| *dimension as usize).product();
            let data = (0..numel).map(|index| index as f32).collect::<Vec<_>>();
            let host = Value::from_slice_f32(&data, shape)?;
            let resident = upload(&host, device)?;
            for axis in 0..shape.len() {
                let extent = shape[axis] as usize;
                for start in 0..extent {
                    for len in 0..=(extent - start) {
                        let expected = HostTensorOps.slice_axis(&host, axis, start, len)?;
                        let actual = ops.slice_axis(&resident, axis, start, len)?;
                        assert_eq!(
                            actual.shape(),
                            expected.shape(),
                            "shape {shape:?} axis {axis} window {start}..{}",
                            start + len
                        );
                        assert_eq!(
                            read(&actual, device)?,
                            read(&expected, device)?,
                            "shape {shape:?} axis {axis} window {start}..{}",
                            start + len
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// A device gather selects the rows it was asked for, in order.
    #[test]
    fn device_gather_matches_host_gather() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        let data = (0..24).map(|index| index as f32).collect::<Vec<_>>();
        let host = Value::from_slice_f32(&data, &[8, 3])?;
        let resident = upload(&host, device)?;
        let ids = [7i64, 0, 3, 3, 1];
        let expected = HostTensorOps.gather_rows(&host, &ids)?;
        let actual = ops.gather_rows(&resident, &ids)?;
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(read(&actual, device)?, read(&expected, device)?);
        Ok(())
    }

    /// Two device scatters assemble the same fused row the host assembles.
    #[test]
    fn device_scatter_matches_host_scatter() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        let embed_host = Value::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let carry_host = Value::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;

        let expected = HostTensorOps.zeros(&[2, 1, 4], DataType::Float32)?;
        HostTensorOps.scatter_into_last_axis(&expected, 0, &embed_host)?;
        HostTensorOps.scatter_into_last_axis(&expected, 2, &carry_host)?;

        let actual = ops.zeros(&[2, 1, 4], DataType::Float32)?;
        ops.scatter_into_last_axis(&actual, 0, &upload(&embed_host, device)?)?;
        ops.scatter_into_last_axis(&actual, 2, &upload(&carry_host, device)?)?;

        assert_eq!(read(&actual, device)?, read(&expected, device)?);
        Ok(())
    }

    /// The device argmax picks the token the host argmax picks, ties and NaN
    /// included.
    ///
    /// Ties resolve to the lowest index and NaN is ignored on both sides; a
    /// device kernel that disagreed on either would silently change which token
    /// a proposal drafts, which is a decoding difference no shape or count
    /// assertion can catch.
    #[test]
    fn device_argmax_matches_host_argmax_on_ties_and_nan() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        let rows: &[Vec<f32>] = &[
            vec![0.1, 0.9, 0.4],
            // A tie at the maximum: the lowest index must win.
            vec![0.5, 0.5, 0.5],
            vec![f32::NAN, 0.2, 0.1],
            vec![0.2, f32::NAN, 0.9],
            vec![-1.0, -2.0, -0.5],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0],
        ];
        for row in rows {
            let shape = [1, 1, row.len() as i64];
            let host = Value::from_slice_f32(row, &shape)?;
            let resident = upload(&host, device)?;
            assert_eq!(
                ops.argmax_rows(&resident, 1)?,
                HostTensorOps.argmax_rows(&host, 1)?,
                "device and host argmax disagree on {row:?}"
            );
        }
        Ok(())
    }

    /// A device value resolves to device operations, and a host operand handed
    /// to them is refused rather than read through a device pointer.
    #[test]
    fn mixed_residency_fails_closed() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let host = Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?;
        let resident = upload(&host, device)?;
        assert_eq!(
            tensor_ops_for(&resident)?.residency(),
            Residency::Cuda(device)
        );

        // Device destination, host source.
        let fused = CudaTensorOps::new(device).zeros(&[1, 1, 4], DataType::Float32)?;
        let message = CudaTensorOps::new(device)
            .scatter_into_last_axis(&fused, 0, &host)
            .expect_err("a host source must not be scattered into a device buffer")
            .to_string();
        assert!(message.contains("never moves a value"), "{message}");
        assert!(message.contains("host"), "{message}");

        // Host destination, device source.
        let host_fused = HostTensorOps.zeros(&[1, 1, 4], DataType::Float32)?;
        let message = HostTensorOps
            .scatter_into_last_axis(&host_fused, 0, &resident)
            .expect_err("a device source must not be scattered into a host buffer")
            .to_string();
        assert!(message.contains("never moves a value"), "{message}");
        assert!(message.contains("CUDA device"), "{message}");
        Ok(())
    }

    /// Adoption is the one crossing, and it is exact in both directions.
    #[test]
    fn adoption_moves_a_value_between_residencies_exactly() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        let host = Value::from_slice_f32(&[1.0, -2.0, 3.5, 0.0], &[2, 2])?;

        let adopted = ops.adopt(&host)?;
        assert!(
            !adopted.is_host_resident()?,
            "adoption must land on the device"
        );
        assert_eq!(adopted.device_id()?, device);
        assert_eq!(read(&adopted, device)?, host.to_vec_f32()?);

        let returned = HostTensorOps.adopt(&adopted)?;
        assert!(
            returned.is_host_resident()?,
            "adoption must land on the host"
        );
        assert_eq!(returned.to_vec_f32()?, host.to_vec_f32()?);

        // Adopting a value already here is a no-op that keeps it here.
        let same = ops.adopt(&adopted)?;
        assert!(!same.is_host_resident()?);
        assert_eq!(same.device_id()?, device);
        Ok(())
    }

    /// A device allocation lives exactly as long as the views borrowing it.
    ///
    /// The narrowed carry a proposal step holds is an alias over the previous
    /// step's output buffer; if the alias did not keep that buffer alive the
    /// next step would read freed device memory, which is a correctness bug
    /// that presents as noise rather than as a crash.
    #[test]
    fn a_view_outlives_the_value_it_was_taken_from() -> anyhow::Result<()> {
        let Some(device) = device() else {
            return Ok(());
        };
        let ops = CudaTensorOps::new(device);
        let data = (0..6).map(|index| index as f32).collect::<Vec<_>>();
        let view = {
            let owner = upload(&Value::from_slice_f32(&data, &[1, 3, 2])?, device)?;
            ops.last_along_axis(&owner, 1)?
        };
        assert_eq!(read(&view, device)?, vec![4.0, 5.0]);
        Ok(())
    }
}
