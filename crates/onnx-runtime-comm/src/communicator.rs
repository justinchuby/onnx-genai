//! The runtime-level [`Communicator`] abstraction
//! (`docs/DISTRIBUTED_RUNTIME.md` §3.1).
//!
//! The [`InProcessCommunicator`](crate::InProcessCommunicator) is the
//! single-process correctness oracle: it implements every method, deterministic
//! reduction math, and per-group collective ordering.

use async_trait::async_trait;

use crate::types::{
    AllToAllVTicket, CommHandle, CommInstanceId, CommResult, DType, DeviceBuffer, RankId, ReduceOp,
    WireTensorSpec,
};

/// Runtime-level communication abstraction for distributed inference.
///
/// Each participant in a distributed execution group holds a `Communicator`
/// handle. The communicator manages point-to-point and collective operations
/// across devices, nodes, or processes. It is **not** part of the EP trait: EPs
/// produce tensors; the communicator moves them between devices.
///
/// # Handle lifetime and buffer reuse
///
/// Data-transfer operations return a [`CommHandle`] representing an
/// asynchronous, rank-local completion fence. The caller MUST NOT reuse, free,
/// or mutate input buffers until the handle reaches a terminal state: the
/// backend's outstanding-operation registry
/// ([`OwnershipRegistry`](crate::OwnershipRegistry)) retains the buffer leases
/// until terminal, so dropping the Rust handle cannot stop progress or release
/// storage. Dropping a handle detaches the caller; it NEVER cancels a
/// collective. Abort is an explicit communicator-wide operation.
#[async_trait]
pub trait Communicator: Send + Sync {
    // ── Identity ──

    /// This participant's immutable world-rank identity.
    fn rank(&self) -> RankId;

    /// Identity of this communicator's pre-compiled group.
    fn group_id(&self) -> crate::types::GroupId;

    /// Group members in the frozen peer-vector order.
    fn members(&self) -> &[RankId];

    /// Number of members in this communicator's group.
    fn group_size(&self) -> usize {
        self.members().len()
    }

    /// Human-readable backend name (e.g., "inprocess", "nccl", "gloo").
    fn backend_name(&self) -> &str;

    // ── Collective operations ──

    /// In-place all-reduce: every rank ends with the element-wise reduction of
    /// all inputs.
    async fn all_reduce(
        &self,
        instance: CommInstanceId,
        tensor: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> CommResult<CommHandle>;

    /// All-to-all: each rank sends a distinct chunk to every other rank and
    /// receives a distinct chunk from every other rank.
    async fn all_to_all(
        &self,
        instance: CommInstanceId,
        send_bufs: &[&DeviceBuffer],
        recv_bufs: &mut [&mut DeviceBuffer],
        chunk_sizes: &[usize],
        dtype: DType,
    ) -> CommResult<CommHandle>;

    /// Prepare one variable-size all-to-all invocation by exchanging per-peer
    /// counts. The returned ticket is consumed exactly once by `all_to_all_v`.
    async fn exchange_counts(
        &self,
        instance: CommInstanceId,
        send_counts: &[usize],
        spec: &WireTensorSpec,
    ) -> CommResult<AllToAllVTicket>;

    /// Variable-size all-to-all data transfer.
    async fn all_to_all_v(
        &self,
        send_buf: &DeviceBuffer,
        send_offsets: &[usize],
        recv_buf: &mut DeviceBuffer,
        recv_offsets: &[usize],
        ticket: AllToAllVTicket,
    ) -> CommResult<CommHandle>;

    /// All-gather: each rank contributes a chunk; every rank receives the
    /// concatenation of all chunks.
    async fn all_gather(
        &self,
        instance: CommInstanceId,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
    ) -> CommResult<CommHandle>;

    /// Broadcast: rank `root` sends; all other ranks receive.
    async fn broadcast(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        root: RankId,
    ) -> CommResult<CommHandle>;

    /// Reduce-scatter: reduce + scatter in one step. Each rank ends with
    /// `1/group_size` of the reduced result.
    async fn reduce_scatter(
        &self,
        instance: CommInstanceId,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> CommResult<CommHandle>;

    // ── Point-to-point (in-process plumbing, slice 1b) ──

    /// Send a buffer to a specific rank.
    async fn send(
        &self,
        instance: CommInstanceId,
        buffer: &DeviceBuffer,
        len: usize,
        dtype: DType,
        dest: RankId,
    ) -> CommResult<CommHandle>;

    /// Receive a buffer from a specific rank.
    async fn recv(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        source: RankId,
    ) -> CommResult<CommHandle>;

    // ── Synchronization (in-process plumbing, slice 1b) ──

    /// Asynchronous barrier. Its local handle becomes terminal only after every
    /// rank in this communicator has entered the same barrier instance.
    async fn barrier(&self, instance: CommInstanceId) -> CommResult<CommHandle>;

    /// Abort all outstanding operations on this communicator. Idempotent;
    /// transitions every outstanding handle to a terminal error.
    async fn abort(&self, cause: crate::types::CommError) -> CommResult<()>;
}
