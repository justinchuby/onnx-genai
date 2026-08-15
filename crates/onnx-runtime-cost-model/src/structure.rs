//! The **machine-independent** half of the cost model: the structure an EP
//! reports for a kernel, and the estimated [`Cost`] that results once a
//! [`crate::DeviceProfile`] supplies the rates.
//!
//! `cost = structure / rate`. [`KernelStructure`] is exactly the portable facts
//! an execution provider knows from a kernel and its shapes — bytes moved,
//! FLOPs, launch count — with no machine constant anywhere. It mirrors the
//! fields an EP fills on `onnx_runtime_ep_api::Cost` (`bytes_moved`) plus
//! `Kernel::estimated_flops`, so the numbers the EP already produces flow
//! straight in.

use std::time::Duration;

use onnx_runtime_ir::DataType;
use serde::{Deserialize, Serialize};

/// The portable, machine-independent cost structure of one kernel invocation.
///
/// FLOPs are `Option` because they are genuinely unknown for some kernels (a
/// dynamic batch dimension the EP could not resolve at match time). An unknown
/// FLOP count must stay unknown so the model declines a compute-bound estimate
/// rather than inventing one — the same discipline the rate side follows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KernelStructure {
    /// Total memory traffic (bytes) read + written by the kernel. Derived from
    /// the real dtypes and shapes (sub-byte-aware), never assumed `elems * 4`.
    pub bytes_moved: u64,
    /// Floating-point operations, or `None` if the EP could not derive it.
    pub flops: Option<u64>,
    /// The dtype whose compute-throughput rate governs this kernel (the compute
    /// dtype), or `None` if the kernel is not compute-bound / dtype is unknown.
    #[serde(with = "opt_dtype_serde")]
    pub compute_dtype: Option<DataType>,
    /// Number of device launches / dispatches (for launch-overhead accounting).
    pub launch_count: u32,
}

impl KernelStructure {
    /// A structure that only knows its memory traffic.
    pub fn from_bytes(bytes_moved: u64) -> Self {
        Self {
            bytes_moved,
            launch_count: 1,
            ..Self::default()
        }
    }

    /// Set the FLOP count.
    pub fn with_flops(mut self, flops: u64) -> Self {
        self.flops = Some(flops);
        self
    }

    /// Set the governing compute dtype.
    pub fn with_compute_dtype(mut self, dtype: DataType) -> Self {
        self.compute_dtype = Some(dtype);
        self
    }

    /// Set the launch count.
    pub fn with_launch_count(mut self, launch_count: u32) -> Self {
        self.launch_count = launch_count;
        self
    }
}

/// Serde adapter for `Option<DataType>`: `DataType` lives in the IR crate and
/// does not derive serde, so it is stored as its stable ONNX dtype integer.
mod opt_dtype_serde {
    use onnx_runtime_ir::DataType;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<DataType>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.map(DataType::to_onnx).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<DataType>, D::Error> {
        let raw = Option::<i32>::deserialize(deserializer)?;
        Ok(raw.and_then(DataType::from_onnx))
    }
}

/// An estimated cost: wall time plus the memory traffic it accounts for.
///
/// This is the §6.3 `Cost { time, memory_bytes }`. It is produced *only* by
/// combining a [`KernelStructure`] with a device's rates, so it never contains
/// a fabricated constant — if a rate is missing the model returns `None` and no
/// `Cost` is produced at all.
///
/// # Lower bound vs. point estimate
///
/// `time` is **not always a point estimate**. Some formulas legitimately omit a
/// term the model could not price without inventing a constant — an unmeasured
/// transfer `latency_base` (treated as zero), or the compute term of an op whose
/// FLOPs or dtype throughput are unknown (leaving a memory-bound-only figure).
/// In those cases the true cost is **≥** `time`, and [`is_lower_bound`] is set.
///
/// This distinction is load-bearing for placement, not cosmetic. The omitted
/// terms are all *optimistic*: a lower-bound transfer time under-states the cost
/// of moving data, which biases a placement search toward moving it. Issue #994
/// exists precisely because we move data we should not (an embedding gather that
/// ships 389 MB to produce 10 KB), so a consumer that treats a lower bound as an
/// estimate can "confirm" a move the real cost would have rejected. A reader of a
/// [`Cost`] must therefore be unable to mistake the two: check [`is_lower_bound`]
/// before using a cost to justify moving data.
///
/// [`is_lower_bound`]: Cost::is_lower_bound
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Estimated wall time. A point estimate when `is_lower_bound` is `false`;
    /// otherwise a lower bound (the true time is `>=` this).
    pub time: Duration,
    /// Bytes of memory traffic this cost accounts for.
    pub memory_bytes: u64,
    /// Whether `time` is a lower bound rather than a point estimate, because the
    /// formula omitted an unpriceable (and optimistic) term. See the type-level
    /// docs. Never use a lower-bound cost to justify moving data.
    pub is_lower_bound: bool,
}

impl Cost {
    /// The zero cost (a free op / same-device transfer). Exact, not a bound.
    pub fn zero() -> Self {
        Self {
            time: Duration::ZERO,
            memory_bytes: 0,
            is_lower_bound: false,
        }
    }

    /// A point-estimate cost: every term the formula needs was priced from a
    /// measured rate.
    pub fn estimate(time: Duration, memory_bytes: u64) -> Self {
        Self {
            time,
            memory_bytes,
            is_lower_bound: false,
        }
    }

    /// A lower-bound cost: an unpriceable, optimistic term was omitted (e.g. an
    /// unmeasured transfer latency, or a missing compute term). The true cost is
    /// `>=` `time`; callers must not treat it as a point estimate.
    pub fn lower_bound(time: Duration, memory_bytes: u64) -> Self {
        Self {
            time,
            memory_bytes,
            is_lower_bound: true,
        }
    }
}
