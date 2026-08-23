//! Residency-preserving tensor algebra for interpreter constructs.
//!
//! An interpreter-level construct — a speculative proposal chain, above all —
//! needs to narrow, truncate and argmax tensors that a component just produced.
//! When that component ran on a device, those tensors are device-resident, and
//! the obvious implementation (copy to host, operate, copy back) is correct and
//! catastrophically slow: a per-draft-token argmax of a `[1, 1, vocab]` f16 row
//! is ≈300 KiB down the PCIe bus to learn one four-byte token id.
//!
//! # The rule
//!
//! Every method here returns a value in the **same residency** as its principal
//! input, or fails saying why. There is deliberately no fallback: a silent host
//! round-trip is exactly the behaviour this exists to remove, and one that only
//! shows up as a throughput number is the kind of regression nobody attributes.
//!
//! # What is and is not zero-copy
//!
//! Narrowing the *outermost* non-unit axis of a contiguous tensor is a
//! contiguous window, so it is a pointer view — free on any backend, via
//! [`Value::alias_with_offset`]. Narrowing an inner axis is strided and is not
//! expressible as a view; it needs a copy on whatever owns the buffer. Host ops
//! do that copy on the host; the CUDA ops refuse rather than staging, because a
//! strided device copy belongs to the execution provider that owns the stream
//! and pretending otherwise here would reintroduce the round trip under a
//! different name.

#[cfg(feature = "ort-cuda")]
use std::sync::Arc;

#[cfg(feature = "ort-cuda")]
use anyhow::Context as _;
use onnx_genai_ort::Value;

/// Where a value lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Residency {
    Host,
    /// Constructed only by the CUDA implementation, which a build without the
    /// `ort-cuda` feature does not compile.
    #[cfg_attr(not(feature = "ort-cuda"), allow(dead_code))]
    Cuda(i32),
}

/// Tensor operations that never change a value's residency.
pub(crate) trait ResidentTensorOps {
    /// Keep only the final index of `axis`, preserving rank.
    fn last_along_axis(&self, value: &Value, axis: usize) -> anyhow::Result<Value>;

    /// Argmax each contiguous `vocab`-wide row, returning token ids.
    ///
    /// The only sanctioned device→host transfer in a proposal loop: four bytes
    /// per row, rather than a vocabulary per row.
    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>>;

    /// Where values produced by this backend live.
    ///
    /// Read by the invariant every method above is written to: what comes out
    /// is resident where what went in was. A caller asserting that is asserting
    /// the property, not the implementation.
    #[cfg_attr(not(test), allow(dead_code))]
    fn residency(&self) -> Residency;
}

/// Host-resident implementation.
///
/// The bodies here are the ones the speculative driver always used; moving them
/// behind the trait is what lets the CUDA implementation exist beside them
/// without a second copy of the semantics.
pub(crate) struct HostTensorOps;

impl ResidentTensorOps for HostTensorOps {
    fn last_along_axis(&self, value: &Value, axis: usize) -> anyhow::Result<Value> {
        super::speculative::last_position_along(value, axis)
    }

    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(
            rows == 1,
            "the host argmax path selects one row at a time; {rows} were requested"
        );
        Ok(vec![logits.argmax_last_row()?])
    }

    fn residency(&self) -> Residency {
        Residency::Host
    }
}

/// CUDA-resident implementation.
///
/// Holds only the device ordinal: every operation either produces a pointer
/// view over the caller's own value or launches a kernel that reads the ORT
/// device pointer directly, so there is no allocation or stream to own here.
#[cfg(feature = "ort-cuda")]
pub(crate) struct CudaTensorOps {
    device: i32,
}

#[cfg(feature = "ort-cuda")]
impl CudaTensorOps {
    pub(crate) fn new(device: i32) -> Self {
        Self { device }
    }
}

#[cfg(feature = "ort-cuda")]
impl ResidentTensorOps for CudaTensorOps {
    fn last_along_axis(&self, value: &Value, axis: usize) -> anyhow::Result<Value> {
        let shape = value.shape().to_vec();
        anyhow::ensure!(
            axis < shape.len(),
            "position axis {axis} is out of range for a rank-{} tensor",
            shape.len()
        );
        let extent = usize::try_from(shape[axis]).context("negative tensor extent")?;
        anyhow::ensure!(extent > 0, "cannot take the last position of an empty axis");
        let outer = shape[..axis].iter().try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension).ok().map(|d| total * d)
        });
        let outer = outer.context("negative tensor extent")?;
        // Only a contiguous window is a view. With more than one leading block
        // the last position of each block is strided, and a device-side gather
        // belongs to the provider that owns the stream — not to a host staging
        // copy dressed up as an operation.
        anyhow::ensure!(
            outer == 1,
            "narrowing axis {axis} of a device-resident tensor with shape {shape:?} needs a \
             strided copy, which this seam does not perform. Either declare the sequence axis \
             outermost, or run this component on the host backend."
        );
        let inner = shape[axis + 1..]
            .iter()
            .try_fold(1usize, |total, dimension| {
                usize::try_from(*dimension).ok().map(|d| total * d)
            })
            .context("negative tensor extent")?;
        let mut narrowed = shape.clone();
        narrowed[axis] = 1;
        let owner = Arc::new(value.try_alias_clone().transpose()?.context(
            "a device-resident value must be aliasable to be narrowed without a copy; \
                     this one owns its buffer directly",
        )?);
        Value::alias_with_offset(owner, (extent - 1) * inner, &narrowed).map_err(Into::into)
    }

    fn argmax_rows(&self, logits: &Value, rows: usize) -> anyhow::Result<Vec<u32>> {
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

    fn residency(&self) -> Residency {
        Residency::Cuda(self.device)
    }
}

/// The operations that preserve `value`'s residency, or a refusal naming both
/// remedies.
///
/// Rule 4: a device-resident value with no device implementation available is
/// an error, never a quiet copy to the host. The two things an operator can
/// actually do — select the host backend, or build with the CUDA feature — are
/// both named, because "unsupported" alone sends someone hunting a bug that is
/// really a build configuration.
pub(crate) fn tensor_ops_for(value: &Value) -> anyhow::Result<Box<dyn ResidentTensorOps>> {
    if value.is_host_resident()? {
        return Ok(Box::new(HostTensorOps));
    }
    let device = value.device_id()?;
    #[cfg(feature = "ort-cuda")]
    {
        return Ok(Box::new(CudaTensorOps::new(device)));
    }
    #[cfg(not(feature = "ort-cuda"))]
    anyhow::bail!(
        "this value is resident on CUDA device {device}, and this build has no device tensor \
         operations to narrow or score it without copying it to the host. Rebuild with the \
         `cuda` feature, or run this package on the host backend."
    )
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
}
