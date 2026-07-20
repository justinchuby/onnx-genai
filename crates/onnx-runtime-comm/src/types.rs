//! Communicator supporting types (`docs/DISTRIBUTED_RUNTIME.md` §3.1).
//!
//! These are the identity, buffer, and result types referenced by the
//! [`Communicator`](crate::Communicator) trait. For the in-process backend they
//! are reference projections: `DeviceBuffer` is host-backed, and
//! `DType`/`DeviceType` are the minimal set the in-process backend needs. Real
//! hardware backends (NCCL, Gloo, Thunderbolt) reuse the same identities but
//! provide device-native buffers.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use onnx_runtime_protocol_trace::PhysicalAllocationId;

/// Immutable world-rank identity. Group-local vector positions are not ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RankId(pub u32);

/// Identity of a pre-compiled communication group. The full world is group 0.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct GroupId(pub u32);

/// Monotonic execution identity. Not reused while any operation from the
/// execution is outstanding.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ExecutionId(pub u64);

/// Frozen, group-local submission position assigned by the plan compiler.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CommSequenceId(pub u32);

/// Runtime identity for one communication operation. The pair is unique within
/// a [`GroupId`] and also acts as the send/recv message tag.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CommInstanceId {
    pub execution: ExecutionId,
    pub sequence: CommSequenceId,
}

impl CommInstanceId {
    /// Convenience constructor.
    pub fn new(execution: u64, sequence: u32) -> Self {
        Self {
            execution: ExecutionId(execution),
            sequence: CommSequenceId(sequence),
        }
    }
}

/// Reduction operation for collective reduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}

/// The minimal element dtype set the in-process backend understands. This is a
/// 1b transport projection; the reduction *semantics* over these dtypes are a
/// implemented deterministically by the in-process oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}

/// Broad device class a communicator backend can move data to/from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Metal,
}

/// A machine-global device identity (node + local device). For the in-process
/// backend every rank is a logical CPU device on node 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlobalDeviceId {
    pub node: u32,
    pub local: u32,
}

/// Describes what a communicator backend can access directly
/// (`docs/DISTRIBUTED_RUNTIME.md` §3.3).
#[derive(Clone, Debug)]
pub struct TransportCapability {
    /// Device types this backend can send FROM directly.
    pub send_from: Vec<DeviceType>,
    /// Device types this backend can receive INTO directly.
    pub recv_into: Vec<DeviceType>,
    /// Device to stage through when a buffer's device is unsupported.
    pub staging_device: GlobalDeviceId,
}

/// A host-backed buffer for the in-process reference backend.
///
/// Every buffer carries a process-unique [`PhysicalAllocationId`]; two
/// `DeviceBuffer` views that share an id share a physical allocation and are
/// therefore subject to lease-conflict detection even if their byte ranges
/// differ. Real backends provide device-native storage behind the same
/// allocation-id contract.
#[derive(Debug)]
pub struct DeviceBuffer {
    id: PhysicalAllocationId,
    data: Vec<u8>,
}

impl DeviceBuffer {
    /// Creates a zeroed buffer of `len_bytes` bytes with the given allocation id.
    pub fn zeroed(id: PhysicalAllocationId, len_bytes: usize) -> Self {
        Self {
            id,
            data: vec![0u8; len_bytes],
        }
    }

    /// Creates a buffer from existing bytes.
    pub fn from_bytes(id: PhysicalAllocationId, data: Vec<u8>) -> Self {
        Self { id, data }
    }

    /// This buffer's physical allocation identity (the lease key).
    pub fn allocation(&self) -> PhysicalAllocationId {
        self.id
    }

    /// Length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable byte view.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Mutable byte view.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// A device stream/queue a completion can be attached to. For the in-process
/// backend attachment is a no-op; real backends record a cross-stream
/// dependency so the stream does not proceed past the collective.
pub trait DeviceStream: Send + Sync {
    /// Backend-defined stream identity, for diagnostics.
    fn stream_id(&self) -> u64;
}

/// Terminal transport/enqueue failures.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CommError {
    /// An argument failed validation before enqueue.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A buffer-ownership lease could not be acquired or released.
    #[error("buffer ownership error: {0}")]
    Ownership(#[from] crate::registry::OwnershipError),
    /// The rank named in a point-to-point or root argument is not a member of
    /// this communicator's group.
    #[error("rank {0:?} is not a member of this group")]
    UnknownRank(RankId),
    /// A peer mailbox held no message for a `recv` (ordering is the caller's /
    /// sequencer's responsibility; the in-process backend does not queue).
    #[error("no pending message from rank {0:?} for this instance")]
    NoPendingMessage(RankId),
    /// The communicator was aborted; all outstanding handles are terminal-error.
    #[error("communicator aborted: {0}")]
    Aborted(String),
    /// A checked byte-extent computation overflowed or exceeded a reservation.
    #[error("checked extent error: {0}")]
    Extent(String),
    /// A later backend capability is not implemented by the in-process oracle.
    #[error("collective capability '{0}' is not implemented")]
    CollectiveDeferred(&'static str),
    /// A rank attempted to diverge from its group's established transport order.
    #[error("collective ordering violation: {0}")]
    Ordering(String),
}

/// Convenience result alias.
pub type CommResult<T> = Result<T, CommError>;

/// A completion that backs a [`CommHandle`]: an awaitable terminal fence that
/// can also be attached to a device stream.
pub trait CommCompletion: Future<Output = CommResult<()>> + Send {
    /// Attach as a dependency on a device stream/queue.
    fn attach_to_stream(&self, stream: &dyn DeviceStream) -> CommResult<()>;
}

/// Completion handle for one rank's asynchronous operation. Awaiting it yields
/// the rank-local terminal result.
pub struct CommHandle {
    inner: Pin<Box<dyn CommCompletion>>,
}

impl std::fmt::Debug for CommHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommHandle").finish_non_exhaustive()
    }
}

impl CommHandle {
    /// Wraps a completion.
    pub fn new(completion: impl CommCompletion + 'static) -> Self {
        Self {
            inner: Box::pin(completion),
        }
    }

    /// An already-terminal handle (in-process operations complete synchronously).
    pub fn ready(result: CommResult<()>) -> Self {
        Self::new(ImmediateCompletion {
            result: Some(result),
        })
    }

    /// Attach this completion as a dependency on a device stream.
    pub fn attach_to_stream(&self, stream: &dyn DeviceStream) -> CommResult<()> {
        self.inner.attach_to_stream(stream)
    }
}

impl Future for CommHandle {
    type Output = CommResult<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// A completion that is already terminal. Used by the in-process reference
/// backend, whose operations complete synchronously (~0 latency).
struct ImmediateCompletion {
    result: Option<CommResult<()>>,
}

impl Future for ImmediateCompletion {
    type Output = CommResult<()>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(
            self.result
                .take()
                .expect("ImmediateCompletion polled after completion"),
        )
    }
}

impl CommCompletion for ImmediateCompletion {
    fn attach_to_stream(&self, _stream: &dyn DeviceStream) -> CommResult<()> {
        // In-process: nothing to order against. Real backends record the fence.
        Ok(())
    }
}

/// Wire codec negotiated and frozen at plan compilation
/// (`docs/DISTRIBUTED_RUNTIME.md` §3.1). Block-quantized reduction semantics are
/// a slice-1c/Phase-3 capability.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireCodec {
    Identity,
    BlockQuantized {
        block_size: usize,
        scale_dtype: DType,
    },
}

/// Transport projection of a negotiated boundary tensor format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireTensorSpec {
    pub logical_dtype: DType,
    pub wire_dtype: DType,
    pub codec: WireCodec,
}

impl WireTensorSpec {
    /// A full-precision (identity codec) spec.
    pub fn identity(dtype: DType) -> Self {
        Self {
            logical_dtype: dtype,
            wire_dtype: dtype,
            codec: WireCodec::Identity,
        }
    }

    /// Checked encoded byte length for `logical_elements`. Identity codec is
    /// `elements * wire_dtype.size()` with overflow rejected; codec-aware
    /// (block-quantized) sizing is deferred to the codec-capability slice.
    pub fn encoded_bytes(&self, logical_elements: usize) -> CommResult<usize> {
        match self.codec {
            WireCodec::Identity => logical_elements
                .checked_mul(self.wire_dtype.size())
                .ok_or_else(|| CommError::Extent("encoded byte length overflow".into())),
            WireCodec::BlockQuantized { .. } => Err(CommError::CollectiveDeferred(
                "block-quantized encoded_bytes",
            )),
        }
    }

    /// Validates that `byte_offset` respects codec block alignment. Identity
    /// codec imposes element-size alignment.
    pub fn validate_segment_alignment(&self, byte_offset: usize) -> CommResult<()> {
        match self.codec {
            WireCodec::Identity => {
                if byte_offset.is_multiple_of(self.wire_dtype.size()) {
                    Ok(())
                } else {
                    Err(CommError::Extent(format!(
                        "offset {byte_offset} not aligned to element size {}",
                        self.wire_dtype.size()
                    )))
                }
            }
            WireCodec::BlockQuantized { .. } => {
                Err(CommError::CollectiveDeferred("block-quantized alignment"))
            }
        }
    }
}

/// Opaque, single-use result of a count exchange for a variable-size all-to-all.
/// Consumed exactly once by `all_to_all_v`.
#[derive(Debug)]
pub struct AllToAllVTicket {
    pub(crate) group: GroupId,
    pub(crate) instance: CommInstanceId,
    pub(crate) send_counts: Vec<usize>,
    pub(crate) recv_counts: Vec<usize>,
    pub(crate) spec: WireTensorSpec,
}

impl AllToAllVTicket {
    /// The group this ticket is bound to.
    pub fn group(&self) -> GroupId {
        self.group
    }

    /// The composite collective instance this ticket is bound to.
    pub fn instance(&self) -> CommInstanceId {
        self.instance
    }

    /// Per-peer send counts (logical elements).
    pub fn send_counts(&self) -> &[usize] {
        &self.send_counts
    }

    /// Per-peer receive counts (logical elements).
    pub fn recv_counts(&self) -> &[usize] {
        &self.recv_counts
    }

    /// The frozen wire spec.
    pub fn spec(&self) -> &WireTensorSpec {
        &self.spec
    }
}

/// A subset of ranks that participate in a collective.
#[derive(Clone, Debug)]
pub struct GroupSpec {
    pub id: GroupId,
    /// Sorted, unique world ranks; this order defines all peer-vector indices.
    pub ranks: Vec<RankId>,
}
