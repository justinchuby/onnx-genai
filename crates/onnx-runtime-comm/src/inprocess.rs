//! In-process communicator reference backend
//! (`docs/DISTRIBUTED_RUNTIME.md` §4.6, Phase 1).
//!
//! All "ranks" live in one process; each rank holds its own
//! [`InProcessCommunicator`] handle over shared state. This is the reference
//! backend and test oracle for real transports (NCCL/Gloo/Thunderbolt): the
//! trait shape and correctness matter, not throughput.
//!
//! Slice 1b implements the plumbing — identity/topology, point-to-point
//! `send`/`recv` via per-instance mailboxes, a cross-thread `barrier`, `abort`,
//! and buffer-ownership registry integration. Collective *algorithms* are slice
//! 1c (see [`crate::communicator`]).
//!
//! Each rank is expected to run on its own thread (a logical device); the
//! blocking `barrier` coordinates those threads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use async_trait::async_trait;

use onnx_runtime_protocol_trace::{PhysicalAllocationId, ProtocolSourceId, ProtocolTraceSink};

use crate::communicator::Communicator;
use crate::registry::{CommLeaseSet, OwnershipRegistry};
use crate::types::{
    AllToAllVTicket, CommError, CommHandle, CommInstanceId, CommResult, DType, DeviceBuffer,
    DeviceType, GlobalDeviceId, GroupId, RankId, ReduceOp, TransportCapability, WireTensorSpec,
};

/// Mailbox key: one slot per (instance, sender position, receiver position).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MailKey {
    instance: CommInstanceId,
    src_pos: usize,
    dst_pos: usize,
}

/// Cross-thread barrier state for one instance.
struct BarrierState {
    arrived: usize,
    released: bool,
}

/// Shared state across all simulated ranks in one process.
struct InProcessSharedState {
    /// The single backend buffer-ownership registry, shared by all ranks.
    registry: OwnershipRegistry,
    world_size: usize,
    /// Point-to-point mailboxes indexed by frozen member positions.
    mailboxes: Mutex<HashMap<MailKey, Vec<u8>>>,
    /// Async barriers keyed by the runtime operation instance.
    barriers: Mutex<HashMap<CommInstanceId, BarrierState>>,
    barrier_cv: Condvar,
    /// Set once `abort` is called; every subsequent operation fails.
    aborted: Mutex<Option<String>>,
    /// Process-unique, never-reused allocation-id source for `DeviceBuffer`s.
    next_alloc: AtomicU64,
}

/// In-process communicator for testing and simulation.
#[derive(Clone)]
pub struct InProcessCommunicator {
    shared: Arc<InProcessSharedState>,
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
}

impl InProcessCommunicator {
    /// Builds a `world_size`-rank in-process world (group 0) and returns one
    /// communicator handle per rank, ranks `0..world_size` in frozen order.
    /// All handles share one buffer-ownership registry with a no-op trace sink.
    pub fn world(world_size: usize) -> Vec<InProcessCommunicator> {
        Self::world_with_sink(
            world_size,
            OwnershipRegistry::new(ProtocolSourceId::new(1), 0),
        )
    }

    /// Like [`InProcessCommunicator::world`] but installs a shared registry
    /// (typically one with a lossless trace sink) so a conformance campaign can
    /// observe every rank's buffer-ownership linearization events.
    pub fn world_with_registry(
        world_size: usize,
        registry: OwnershipRegistry,
    ) -> Vec<InProcessCommunicator> {
        Self::world_with_sink(world_size, registry)
    }

    /// Convenience: build a world whose registry emits to `sink`.
    pub fn world_traced(
        world_size: usize,
        source: ProtocolSourceId,
        topology_epoch: u64,
        sink: Arc<dyn ProtocolTraceSink>,
    ) -> Vec<InProcessCommunicator> {
        Self::world_with_sink(
            world_size,
            OwnershipRegistry::with_sink(source, topology_epoch, sink),
        )
    }

    fn world_with_sink(
        world_size: usize,
        registry: OwnershipRegistry,
    ) -> Vec<InProcessCommunicator> {
        assert!(world_size >= 1, "world_size must be positive");
        let members: Arc<[RankId]> = (0..world_size as u32).map(RankId).collect();
        let shared = Arc::new(InProcessSharedState {
            registry,
            world_size,
            mailboxes: Mutex::new(HashMap::new()),
            barriers: Mutex::new(HashMap::new()),
            barrier_cv: Condvar::new(),
            aborted: Mutex::new(None),
            next_alloc: AtomicU64::new(1),
        });
        (0..world_size as u32)
            .map(|r| InProcessCommunicator {
                shared: shared.clone(),
                group_id: GroupId(0),
                members: members.clone(),
                rank: RankId(r),
            })
            .collect()
    }

    /// The shared backend buffer-ownership registry (for the invariant gate and
    /// integration with higher layers).
    pub fn registry(&self) -> &OwnershipRegistry {
        &self.shared.registry
    }

    /// Allocates a zeroed host-backed [`DeviceBuffer`] of `len_bytes` bytes with
    /// a fresh, process-unique [`PhysicalAllocationId`].
    pub fn allocate_buffer(&self, len_bytes: usize) -> DeviceBuffer {
        DeviceBuffer::zeroed(self.next_allocation(), len_bytes)
    }

    /// Allocates a host-backed [`DeviceBuffer`] initialized from `data`.
    pub fn allocate_from(&self, data: Vec<u8>) -> DeviceBuffer {
        DeviceBuffer::from_bytes(self.next_allocation(), data)
    }

    /// This backend's transport capability: it moves host (CPU) buffers only.
    pub fn capability(&self) -> TransportCapability {
        TransportCapability {
            send_from: vec![DeviceType::Cpu],
            recv_into: vec![DeviceType::Cpu],
            staging_device: GlobalDeviceId { node: 0, local: 0 },
        }
    }

    fn next_allocation(&self) -> PhysicalAllocationId {
        PhysicalAllocationId::new(self.shared.next_alloc.fetch_add(1, Ordering::Relaxed))
    }

    fn position(&self, rank: RankId) -> Option<usize> {
        self.members.iter().position(|m| *m == rank)
    }

    fn self_position(&self) -> usize {
        self.position(self.rank).expect("self is a member")
    }

    fn check_not_aborted(&self) -> CommResult<()> {
        match &*self.shared.aborted.lock().expect("aborted lock poisoned") {
            Some(cause) => Err(CommError::Aborted(cause.clone())),
            None => Ok(()),
        }
    }

    /// Checked byte extent = `len * dtype.size()`, bounded by the buffer length.
    fn checked_extent(len: usize, dtype: DType, buf_len: usize) -> CommResult<usize> {
        let bytes = len
            .checked_mul(dtype.size())
            .ok_or_else(|| CommError::Extent("len * dtype.size() overflow".into()))?;
        if bytes > buf_len {
            return Err(CommError::InvalidArgument(format!(
                "transfer extent {bytes} exceeds buffer length {buf_len}"
            )));
        }
        Ok(bytes)
    }
}

#[async_trait]
impl Communicator for InProcessCommunicator {
    fn rank(&self) -> RankId {
        self.rank
    }

    fn group_id(&self) -> GroupId {
        self.group_id
    }

    fn members(&self) -> &[RankId] {
        &self.members
    }

    fn backend_name(&self) -> &str {
        "inprocess"
    }

    async fn all_reduce(
        &self,
        _instance: CommInstanceId,
        _tensor: &mut DeviceBuffer,
        _len: usize,
        _dtype: DType,
        _op: ReduceOp,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("all_reduce"))
    }

    async fn all_to_all(
        &self,
        _instance: CommInstanceId,
        _send_bufs: &[&DeviceBuffer],
        _recv_bufs: &mut [&mut DeviceBuffer],
        _chunk_sizes: &[usize],
        _dtype: DType,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("all_to_all"))
    }

    async fn exchange_counts(
        &self,
        _instance: CommInstanceId,
        _send_counts: &[usize],
        _spec: &WireTensorSpec,
    ) -> CommResult<AllToAllVTicket> {
        Err(CommError::CollectiveDeferred("exchange_counts"))
    }

    async fn all_to_all_v(
        &self,
        _send_buf: &DeviceBuffer,
        _send_offsets: &[usize],
        _recv_buf: &mut DeviceBuffer,
        _recv_offsets: &[usize],
        _ticket: AllToAllVTicket,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("all_to_all_v"))
    }

    async fn all_gather(
        &self,
        _instance: CommInstanceId,
        _send_buf: &DeviceBuffer,
        _recv_buf: &mut DeviceBuffer,
        _count: usize,
        _dtype: DType,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("all_gather"))
    }

    async fn broadcast(
        &self,
        _instance: CommInstanceId,
        _buffer: &mut DeviceBuffer,
        _len: usize,
        _dtype: DType,
        _root: RankId,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("broadcast"))
    }

    async fn reduce_scatter(
        &self,
        _instance: CommInstanceId,
        _send_buf: &DeviceBuffer,
        _recv_buf: &mut DeviceBuffer,
        _count: usize,
        _dtype: DType,
        _op: ReduceOp,
    ) -> CommResult<CommHandle> {
        Err(CommError::CollectiveDeferred("reduce_scatter"))
    }

    async fn send(
        &self,
        instance: CommInstanceId,
        buffer: &DeviceBuffer,
        len: usize,
        dtype: DType,
        dest: RankId,
    ) -> CommResult<CommHandle> {
        self.check_not_aborted()?;
        let dst_pos = self.position(dest).ok_or(CommError::UnknownRank(dest))?;
        let bytes = Self::checked_extent(len, dtype, buffer.len_bytes())?;

        // Submit: the registry owns the read lease before "transport enqueue".
        let lease = self
            .shared
            .registry
            .submit(&CommLeaseSet::read_only([buffer.allocation()]))?;

        // In-process transport: copy into the destination's mailbox slot.
        {
            let mut mailboxes = self.shared.mailboxes.lock().expect("mailbox poisoned");
            mailboxes.insert(
                MailKey {
                    instance,
                    src_pos: self.self_position(),
                    dst_pos,
                },
                buffer.as_slice()[..bytes].to_vec(),
            );
        }

        // Terminal: local input buffer is safe to reuse; release the read lease.
        self.shared.registry.complete_success(lease.operation())?;
        drop(lease);
        Ok(CommHandle::ready(Ok(())))
    }

    async fn recv(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        source: RankId,
    ) -> CommResult<CommHandle> {
        self.check_not_aborted()?;
        let src_pos = self
            .position(source)
            .ok_or(CommError::UnknownRank(source))?;
        let bytes = Self::checked_extent(len, dtype, buffer.len_bytes())?;
        let key = MailKey {
            instance,
            src_pos,
            dst_pos: self.self_position(),
        };

        // Take the message first: the in-process backend does not queue, so
        // ordering is the caller's/sequencer's responsibility (slice 1c).
        let message = {
            let mut mailboxes = self.shared.mailboxes.lock().expect("mailbox poisoned");
            mailboxes.remove(&key)
        };
        let message = message.ok_or(CommError::NoPendingMessage(source))?;
        if message.len() != bytes {
            return Err(CommError::InvalidArgument(format!(
                "recv extent {bytes} does not match sent extent {}",
                message.len()
            )));
        }

        // Submit: the registry owns the write lease before the copy.
        let lease = self
            .shared
            .registry
            .submit(&CommLeaseSet::write_only([buffer.allocation()]))?;
        buffer.as_mut_slice()[..bytes].copy_from_slice(&message);

        // Terminal: output visible, write lease released.
        self.shared.registry.complete_success(lease.operation())?;
        drop(lease);
        Ok(CommHandle::ready(Ok(())))
    }

    async fn barrier(&self, instance: CommInstanceId) -> CommResult<CommHandle> {
        self.check_not_aborted()?;
        let world = self.shared.world_size;
        let mut barriers = self.shared.barriers.lock().expect("barrier poisoned");
        let state = barriers.entry(instance).or_insert(BarrierState {
            arrived: 0,
            released: false,
        });
        state.arrived += 1;
        if state.arrived >= world {
            state.released = true;
            self.shared.barrier_cv.notify_all();
        } else {
            while !barriers.get(&instance).map(|s| s.released).unwrap_or(true) {
                barriers = self
                    .shared
                    .barrier_cv
                    .wait(barriers)
                    .expect("barrier wait poisoned");
            }
        }
        Ok(CommHandle::ready(Ok(())))
    }

    async fn abort(&self, cause: CommError) -> CommResult<()> {
        // Idempotent: record the first cause; later aborts are no-ops.
        let mut aborted = self.shared.aborted.lock().expect("aborted lock poisoned");
        if aborted.is_none() {
            *aborted = Some(cause.to_string());
        }
        // In-process operations complete synchronously, so no lease is
        // outstanding at an abort boundary; the registry drives its own
        // BeginAbort/QuiesceAbort path when an operation is genuinely in flight
        // (exercised by the buffer-ownership conformance campaigns).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_on;
    use crate::types::CommError;

    fn instance() -> CommInstanceId {
        CommInstanceId::new(1, 0)
    }

    #[test]
    fn identity_and_topology() {
        let world = InProcessCommunicator::world(3);
        assert_eq!(world.len(), 3);
        for (i, comm) in world.iter().enumerate() {
            assert_eq!(comm.rank(), RankId(i as u32));
            assert_eq!(comm.group_id(), GroupId(0));
            assert_eq!(comm.members().len(), 3);
            assert_eq!(comm.backend_name(), "inprocess");
        }
    }

    #[test]
    fn allocations_are_unique() {
        let world = InProcessCommunicator::world(1);
        let comm = &world[0];
        let a = comm.allocate_buffer(8);
        let b = comm.allocate_buffer(8);
        assert_ne!(a.allocation(), b.allocation());
        assert_eq!(a.len_bytes(), 8);
    }

    #[test]
    fn send_recv_roundtrip_single_thread() {
        let world = InProcessCommunicator::world(2);
        let src = world[0].allocate_from(vec![1u8, 2, 3, 4]);
        let mut dst = world[1].allocate_buffer(4);
        block_on(world[0].send(instance(), &src, 4, DType::U8, RankId(1))).unwrap();
        block_on(world[1].recv(instance(), &mut dst, 4, DType::U8, RankId(0))).unwrap();
        assert_eq!(dst.as_slice(), &[1u8, 2, 3, 4]);
        // Registry drained back to empty after both terminal completions.
        assert_eq!(world[0].registry().snapshot().unwrap().active, 0);
    }

    #[test]
    fn recv_without_message_reports_no_pending() {
        let world = InProcessCommunicator::world(2);
        let mut dst = world[1].allocate_buffer(4);
        let err =
            block_on(world[1].recv(instance(), &mut dst, 4, DType::U8, RankId(0))).unwrap_err();
        assert!(matches!(err, CommError::NoPendingMessage(RankId(0))));
    }

    #[test]
    fn send_to_unknown_rank_is_rejected() {
        let world = InProcessCommunicator::world(2);
        let src = world[0].allocate_from(vec![0u8; 4]);
        let err = block_on(world[0].send(instance(), &src, 4, DType::U8, RankId(9))).unwrap_err();
        assert!(matches!(err, CommError::UnknownRank(RankId(9))));
    }

    #[test]
    fn barrier_synchronizes_all_ranks() {
        use std::sync::Arc;
        let world = InProcessCommunicator::world(4);
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = world
            .into_iter()
            .map(|comm| {
                let counter = counter.clone();
                std::thread::spawn(move || {
                    counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    block_on(comm.barrier(instance())).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[test]
    fn abort_makes_subsequent_ops_terminal() {
        let world = InProcessCommunicator::world(2);
        block_on(world[0].abort(CommError::Aborted("test".into()))).unwrap();
        let src = world[0].allocate_from(vec![0u8; 4]);
        let err = block_on(world[0].send(instance(), &src, 4, DType::U8, RankId(1))).unwrap_err();
        assert!(matches!(err, CommError::Aborted(_)));
    }

    #[test]
    fn collectives_are_deferred_to_1c() {
        let world = InProcessCommunicator::world(2);
        let mut buf = world[0].allocate_buffer(16);
        let err = block_on(world[0].all_reduce(instance(), &mut buf, 4, DType::F32, ReduceOp::Sum))
            .unwrap_err();
        assert!(matches!(err, CommError::CollectiveDeferred("all_reduce")));
    }
}
