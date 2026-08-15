//! Single-process, multi-rank reference communicator.
//!
//! Collective calls rendezvous through shared host state. The implementation is
//! intentionally direct rather than optimized: it is the correctness oracle for
//! later device transports, with deterministic rank-order reductions and a
//! `CollectiveOrdering.tla` submit sequencer.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use async_trait::async_trait;
use onnx_runtime_protocol_trace::{
    CollectiveDecision, CollectiveKind, PhysicalAllocationId, ProtocolSourceId, ProtocolTraceSink,
};

use crate::communicator::Communicator;
use crate::ordering::CollectiveSequencer;
use crate::reduction::reduce_buffers;
use crate::registry::{CommLeaseSet, OperationLease, OwnershipRegistry};
use crate::types::{
    AllToAllVTicket, CommError, CommHandle, CommInstanceId, CommResult, DType, DeviceBuffer,
    DeviceType, ExecutionId, GlobalDeviceId, GroupId, GroupSpec, RankId, ReduceOp,
    TransportCapability, WireCodec, WireTensorSpec,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MailKey {
    group: GroupId,
    instance: CommInstanceId,
    src_pos: usize,
    dst_pos: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BarrierKey {
    group: GroupId,
    instance: CommInstanceId,
}

struct BarrierState {
    arrived: BTreeSet<RankId>,
    departed: usize,
    released: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CollectivePhase {
    Data,
    Counts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CollectiveKey {
    group: GroupId,
    instance: CommInstanceId,
    phase: CollectivePhase,
}

type RankSegments = Vec<Vec<u8>>;
type CollectiveOutputs = Vec<RankSegments>;

struct PendingCollective {
    output: RankSegments,
    lease: OperationLease,
}

struct DataCollectiveState {
    signature: u64,
    contributions: Vec<Option<RankSegments>>,
    outputs: Option<CommResult<CollectiveOutputs>>,
    consumed: usize,
}

struct CountCollectiveState {
    signature: u64,
    contributions: Vec<Option<Vec<usize>>>,
    outputs: Option<Vec<Vec<usize>>>,
    consumed: usize,
}

#[cfg(test)]
struct TestWaitHook {
    reached: std::sync::mpsc::SyncSender<()>,
    proceed: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
#[derive(Default)]
struct InProcessTestHooks {
    barrier_before_wait: Mutex<Option<TestWaitHook>>,
    data_before_wait: Mutex<Option<TestWaitHook>>,
    abort_ready: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    barrier_notified: Mutex<Option<std::sync::mpsc::Sender<()>>>,
    data_notified: Mutex<Option<std::sync::mpsc::Sender<()>>>,
}

#[cfg(test)]
impl InProcessTestHooks {
    fn pause_before_wait(slot: &Mutex<Option<TestWaitHook>>) {
        if let Some(hook) = slot.lock().expect("test hook poisoned").take() {
            hook.reached.send(()).expect("test observer dropped");
            hook.proceed.recv().expect("test controller dropped");
        }
    }

    fn signal(slot: &Mutex<Option<std::sync::mpsc::Sender<()>>>) {
        if let Some(sender) = slot.lock().expect("test hook poisoned").take() {
            sender.send(()).expect("test observer dropped");
        }
    }
}

struct InProcessSharedState {
    registry: OwnershipRegistry,
    sequencer: CollectiveSequencer,
    world_members: Arc<[RankId]>,
    mailboxes: Mutex<HashMap<MailKey, Vec<u8>>>,
    barriers: Mutex<HashMap<BarrierKey, BarrierState>>,
    barrier_cv: Condvar,
    data_collectives: Mutex<HashMap<CollectiveKey, DataCollectiveState>>,
    data_cv: Condvar,
    count_collectives: Mutex<HashMap<CollectiveKey, CountCollectiveState>>,
    count_cv: Condvar,
    aborted: Mutex<Option<String>>,
    next_alloc: AtomicU64,
    #[cfg(test)]
    test_hooks: InProcessTestHooks,
}

/// In-process communicator for testing, simulation, and conformance campaigns.
#[derive(Clone)]
pub struct InProcessCommunicator {
    shared: Arc<InProcessSharedState>,
    group_id: GroupId,
    members: Arc<[RankId]>,
    rank: RankId,
}

impl InProcessCommunicator {
    /// Builds a `world_size`-rank world communicator (group 0).
    pub fn world(world_size: usize) -> Vec<Self> {
        Self::world_with_components(
            world_size,
            OwnershipRegistry::new(ProtocolSourceId::new(1), 0),
            CollectiveSequencer::new(),
        )
    }

    /// Builds a world using a caller-provided shared ownership registry.
    pub fn world_with_registry(world_size: usize, registry: OwnershipRegistry) -> Vec<Self> {
        Self::world_with_components(world_size, registry, CollectiveSequencer::new())
    }

    /// Builds a world whose ownership and ordering protocols emit to one
    /// lossless sink. Ordering uses distinct coordinator/rank sources.
    pub fn world_traced(
        world_size: usize,
        source: ProtocolSourceId,
        topology_epoch: u64,
        sink: Arc<dyn ProtocolTraceSink>,
    ) -> Vec<Self> {
        Self::world_with_components(
            world_size,
            OwnershipRegistry::with_sink(source, topology_epoch, sink.clone()),
            CollectiveSequencer::with_sink(source, topology_epoch, sink),
        )
    }

    fn world_with_components(
        world_size: usize,
        registry: OwnershipRegistry,
        sequencer: CollectiveSequencer,
    ) -> Vec<Self> {
        assert!(world_size >= 1, "world_size must be positive");
        let members: Arc<[RankId]> = (0..world_size as u32).map(RankId).collect();
        sequencer
            .register_group(GroupId(0), &members)
            .expect("world group is valid");
        let shared = Arc::new(InProcessSharedState {
            registry,
            sequencer,
            world_members: members.clone(),
            mailboxes: Mutex::new(HashMap::new()),
            barriers: Mutex::new(HashMap::new()),
            barrier_cv: Condvar::new(),
            data_collectives: Mutex::new(HashMap::new()),
            data_cv: Condvar::new(),
            count_collectives: Mutex::new(HashMap::new()),
            count_cv: Condvar::new(),
            aborted: Mutex::new(None),
            next_alloc: AtomicU64::new(1),
            #[cfg(test)]
            test_hooks: InProcessTestHooks::default(),
        });
        members
            .iter()
            .copied()
            .map(|rank| Self {
                shared: shared.clone(),
                group_id: GroupId(0),
                members: members.clone(),
                rank,
            })
            .collect()
    }

    /// Creates handles for a precompiled subgroup over the same in-process
    /// topology. Returned handles follow `spec.ranks` order.
    pub fn create_group(world: &[Self], spec: GroupSpec) -> CommResult<Vec<Self>> {
        let Some(first) = world.first() else {
            return Err(CommError::InvalidArgument(
                "cannot create a group from an empty world".into(),
            ));
        };
        if spec.id == GroupId(0) {
            return Err(CommError::InvalidArgument(
                "group 0 is reserved for the world".into(),
            ));
        }
        if world
            .iter()
            .any(|comm| !Arc::ptr_eq(&comm.shared, &first.shared))
        {
            return Err(CommError::InvalidArgument(
                "group handles must come from one in-process world".into(),
            ));
        }
        if !spec.ranks.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(CommError::InvalidArgument(
                "group ranks must be sorted and unique".into(),
            ));
        }
        for rank in &spec.ranks {
            if first.shared.world_members.binary_search(rank).is_err() {
                return Err(CommError::UnknownRank(*rank));
            }
        }
        first
            .shared
            .sequencer
            .register_group(spec.id, &spec.ranks)?;
        let members: Arc<[RankId]> = spec.ranks.into();
        Ok(members
            .iter()
            .copied()
            .map(|rank| Self {
                shared: first.shared.clone(),
                group_id: spec.id,
                members: members.clone(),
                rank,
            })
            .collect())
    }

    /// Explicitly publishes the next admitted execution decision.
    pub fn admit_execution(&self, execution: ExecutionId) -> CommResult<()> {
        self.shared
            .sequencer
            .decide(execution, CollectiveDecision::Admitted)
    }

    /// Explicitly publishes the next skipped execution decision.
    pub fn skip_execution(&self, execution: ExecutionId) -> CommResult<()> {
        self.shared
            .sequencer
            .decide(execution, CollectiveDecision::Skipped)
    }

    /// Advances this rank's per-group cursor over one skipped plan slot without
    /// calling the backend transport.
    pub fn observe_skipped(&self, instance: CommInstanceId) -> CommResult<()> {
        self.shared
            .sequencer
            .observe_skip(self.group_id, self.rank, instance)
    }

    pub fn registry(&self) -> &OwnershipRegistry {
        &self.shared.registry
    }

    pub fn allocate_buffer(&self, len_bytes: usize) -> DeviceBuffer {
        DeviceBuffer::zeroed(self.next_allocation(), len_bytes)
    }

    pub fn allocate_from(&self, data: Vec<u8>) -> DeviceBuffer {
        DeviceBuffer::from_bytes(self.next_allocation(), data)
    }

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
        self.members.iter().position(|member| *member == rank)
    }

    fn self_position(&self) -> usize {
        self.position(self.rank).expect("self is a group member")
    }

    fn check_not_aborted(&self) -> CommResult<()> {
        match &*self.shared.aborted.lock().expect("aborted lock poisoned") {
            Some(cause) => Err(CommError::Aborted(cause.clone())),
            None => Ok(()),
        }
    }

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

    fn signature(words: &[u64]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in words.iter().flat_map(|word| word.to_le_bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn dtype_code(dtype: DType) -> u64 {
        match dtype {
            DType::F32 => 1,
            DType::F16 => 2,
            DType::BF16 => 3,
            DType::I64 => 4,
            DType::I32 => 5,
            DType::U8 => 6,
        }
    }

    fn op_code(op: ReduceOp) -> u64 {
        match op {
            ReduceOp::Sum => 1,
            ReduceOp::Product => 2,
            ReduceOp::Min => 3,
            ReduceOp::Max => 4,
        }
    }

    fn retire_error(&self, lease: &OperationLease) {
        if self.shared.registry.begin_abort(lease.operation()).is_ok() {
            let _ = self.shared.registry.quiesce_abort(lease.operation());
        }
    }

    fn finish_success(
        &self,
        instance: CommInstanceId,
        pending: PendingCollective,
    ) -> CommResult<RankSegments> {
        self.shared
            .registry
            .complete_success(pending.lease.operation())?;
        self.shared
            .sequencer
            .complete(self.group_id, self.rank, instance, true)?;
        Ok(pending.output)
    }

    fn finish_error(&self, instance: CommInstanceId, lease: &OperationLease) {
        self.retire_error(lease);
        let _ = self
            .shared
            .sequencer
            .complete(self.group_id, self.rank, instance, false);
    }

    /// Runs one synchronous in-process collective rendezvous.
    ///
    /// The `compute` callback, including reduction work, runs while the
    /// rendezvous mutex is held. That is acceptable only for this in-process
    /// correctness oracle. Real asynchronous transports (for example, NCCL)
    /// must not copy this pattern into their hot path; they must publish
    /// rendezvous state and perform transport/reduction work outside the mutex.
    fn run_collective<F>(
        &self,
        instance: CommInstanceId,
        ordering: (CollectiveKind, u64, bool),
        leases: CommLeaseSet,
        contribution: RankSegments,
        compute: F,
    ) -> CommResult<PendingCollective>
    where
        F: FnOnce(&[RankSegments]) -> CommResult<CollectiveOutputs>,
    {
        let (collective, signature, submit_order) = ordering;
        self.check_not_aborted()?;
        let lease = self.shared.registry.submit(&leases)?;
        if submit_order {
            if let Err(error) = self.shared.sequencer.submit(
                self.group_id,
                self.rank,
                instance,
                collective,
                signature,
            ) {
                self.retire_error(&lease);
                return Err(error);
            }
        } else if !self
            .shared
            .sequencer
            .is_active(self.group_id, self.rank, instance)
        {
            self.retire_error(&lease);
            return Err(CommError::Ordering(
                "all_to_all_v data phase has no active count-exchange slot".into(),
            ));
        }

        let key = CollectiveKey {
            group: self.group_id,
            instance,
            phase: CollectivePhase::Data,
        };
        let rank_pos = self.self_position();
        let group_size = self.group_size();
        let mut states = self
            .shared
            .data_collectives
            .lock()
            .expect("collective lock poisoned");
        let state = states.entry(key).or_insert_with(|| DataCollectiveState {
            signature,
            contributions: (0..group_size).map(|_| None).collect(),
            outputs: None,
            consumed: 0,
        });
        if state.signature != signature {
            drop(states);
            self.retire_error(&lease);
            let _ = self
                .shared
                .sequencer
                .complete(self.group_id, self.rank, instance, false);
            return Err(CommError::Ordering(
                "collective rendezvous signature diverged".into(),
            ));
        }
        if state.contributions[rank_pos]
            .replace(contribution)
            .is_some()
        {
            drop(states);
            self.retire_error(&lease);
            return Err(CommError::Ordering(
                "rank contributed to one collective twice".into(),
            ));
        }
        if state.contributions.iter().all(Option::is_some) {
            let inputs: Vec<RankSegments> = state
                .contributions
                .iter()
                .map(|input| input.clone().expect("all contributions present"))
                .collect();
            state.outputs = Some(compute(&inputs));
            self.shared.data_cv.notify_all();
        }
        while states
            .get(&key)
            .and_then(|state| state.outputs.as_ref())
            .is_none()
            && !self.shared.sequencer.is_aborted()
        {
            #[cfg(test)]
            InProcessTestHooks::pause_before_wait(&self.shared.test_hooks.data_before_wait);
            states = self
                .shared
                .data_cv
                .wait(states)
                .expect("collective wait poisoned");
        }

        if self.shared.sequencer.is_aborted() {
            drop(states);
            self.retire_error(&lease);
            return Err(CommError::Aborted(
                self.shared
                    .aborted
                    .lock()
                    .expect("aborted lock poisoned")
                    .clone()
                    .unwrap_or_else(|| "collective aborted".into()),
            ));
        }

        let result = states
            .get(&key)
            .and_then(|state| state.outputs.as_ref())
            .expect("completed collective has outputs")
            .as_ref()
            .map(|outputs| outputs[rank_pos].clone())
            .map_err(Clone::clone);
        let remove = {
            let state = states.get_mut(&key).expect("collective state exists");
            state.consumed += 1;
            state.consumed == group_size
        };
        if remove {
            states.remove(&key);
        }
        drop(states);

        match result {
            Ok(output) => Ok(PendingCollective { output, lease }),
            Err(error) => {
                self.finish_error(instance, &lease);
                Err(error)
            }
        }
    }

    fn validate_peer_vectors(&self, lengths: &[usize]) -> CommResult<()> {
        if lengths.iter().any(|length| *length != self.group_size()) {
            return Err(CommError::InvalidArgument(format!(
                "per-peer vector length must equal group size {}",
                self.group_size()
            )));
        }
        Ok(())
    }

    fn checked_span(
        offset: usize,
        count: usize,
        spec: &WireTensorSpec,
        capacity: usize,
    ) -> CommResult<std::ops::Range<usize>> {
        spec.validate_segment_alignment(offset)?;
        let bytes = spec.encoded_bytes(count)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| CommError::Extent("segment offset + extent overflow".into()))?;
        if end > capacity {
            return Err(CommError::Extent(format!(
                "segment {offset}..{end} exceeds buffer capacity {capacity}"
            )));
        }
        Ok(offset..end)
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
        instance: CommInstanceId,
        tensor: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> CommResult<CommHandle> {
        let bytes = Self::checked_extent(len, dtype, tensor.len_bytes())?;
        let signature = Self::signature(&[
            CollectiveKind::AllReduce as u64,
            len as u64,
            Self::dtype_code(dtype),
            Self::op_code(op),
        ]);
        let pending = self.run_collective(
            instance,
            (CollectiveKind::AllReduce, signature, true),
            CommLeaseSet {
                reads: vec![tensor.allocation()],
                writes: vec![tensor.allocation()],
            },
            vec![tensor.as_slice()[..bytes].to_vec()],
            |inputs| {
                let rank_inputs: Vec<Vec<u8>> =
                    inputs.iter().map(|segments| segments[0].clone()).collect();
                let reduced = reduce_buffers(&rank_inputs, dtype, op)?;
                Ok((0..inputs.len()).map(|_| vec![reduced.clone()]).collect())
            },
        )?;
        tensor.as_mut_slice()[..bytes].copy_from_slice(&pending.output[0]);
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
    }

    async fn all_to_all(
        &self,
        instance: CommInstanceId,
        send_bufs: &[&DeviceBuffer],
        recv_bufs: &mut [&mut DeviceBuffer],
        chunk_sizes: &[usize],
        dtype: DType,
    ) -> CommResult<CommHandle> {
        self.validate_peer_vectors(&[send_bufs.len(), recv_bufs.len(), chunk_sizes.len()])?;
        let mut contribution = Vec::with_capacity(self.group_size());
        for (buffer, count) in send_bufs.iter().zip(chunk_sizes) {
            let bytes = Self::checked_extent(*count, dtype, buffer.len_bytes())?;
            contribution.push(buffer.as_slice()[..bytes].to_vec());
        }
        for (buffer, count) in recv_bufs.iter().zip(chunk_sizes) {
            Self::checked_extent(*count, dtype, buffer.len_bytes())?;
        }
        let signature_words = [CollectiveKind::AllToAll as u64, Self::dtype_code(dtype)];
        let signature = Self::signature(&signature_words);
        let leases = CommLeaseSet {
            reads: send_bufs.iter().map(|buffer| buffer.allocation()).collect(),
            writes: recv_bufs.iter().map(|buffer| buffer.allocation()).collect(),
        };
        let pending = self.run_collective(
            instance,
            (CollectiveKind::AllToAll, signature, true),
            leases,
            contribution,
            |inputs| {
                Ok((0..inputs.len())
                    .map(|destination| {
                        inputs
                            .iter()
                            .map(|source| source[destination].clone())
                            .collect()
                    })
                    .collect())
            },
        )?;
        for ((buffer, bytes), count) in recv_bufs.iter_mut().zip(&pending.output).zip(chunk_sizes) {
            let extent = Self::checked_extent(*count, dtype, buffer.len_bytes())?;
            if bytes.len() != extent {
                self.finish_error(instance, &pending.lease);
                return Err(CommError::InvalidArgument(
                    "all_to_all remote chunk size mismatch".into(),
                ));
            }
            buffer.as_mut_slice()[..extent].copy_from_slice(bytes);
        }
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
    }

    async fn exchange_counts(
        &self,
        instance: CommInstanceId,
        send_counts: &[usize],
        spec: &WireTensorSpec,
    ) -> CommResult<AllToAllVTicket> {
        self.check_not_aborted()?;
        self.validate_peer_vectors(&[send_counts.len()])?;
        if !matches!(spec.codec, WireCodec::Identity) {
            return Err(CommError::CollectiveDeferred(
                "block-quantized all_to_all_v",
            ));
        }
        for count in send_counts {
            spec.encoded_bytes(*count)?;
        }
        let signature = Self::signature(&[
            CollectiveKind::AllToAllV as u64,
            Self::dtype_code(spec.logical_dtype),
            Self::dtype_code(spec.wire_dtype),
        ]);
        self.shared.sequencer.submit(
            self.group_id,
            self.rank,
            instance,
            CollectiveKind::AllToAllV,
            signature,
        )?;
        let key = CollectiveKey {
            group: self.group_id,
            instance,
            phase: CollectivePhase::Counts,
        };
        let rank_pos = self.self_position();
        let group_size = self.group_size();
        let mut states = self
            .shared
            .count_collectives
            .lock()
            .expect("count collective lock poisoned");
        let state = states.entry(key).or_insert_with(|| CountCollectiveState {
            signature,
            contributions: vec![None; group_size],
            outputs: None,
            consumed: 0,
        });
        if state.signature != signature || state.contributions[rank_pos].is_some() {
            return Err(CommError::Ordering(
                "count exchange rendezvous diverged or duplicated".into(),
            ));
        }
        state.contributions[rank_pos] = Some(send_counts.to_vec());
        if state.contributions.iter().all(Option::is_some) {
            let counts: Vec<Vec<usize>> = state
                .contributions
                .iter()
                .map(|entry| entry.clone().expect("all counts present"))
                .collect();
            state.outputs = Some(
                (0..group_size)
                    .map(|destination| counts.iter().map(|source| source[destination]).collect())
                    .collect(),
            );
            self.shared.count_cv.notify_all();
        }
        while states
            .get(&key)
            .and_then(|state| state.outputs.as_ref())
            .is_none()
            && !self.shared.sequencer.is_aborted()
        {
            states = self
                .shared
                .count_cv
                .wait(states)
                .expect("count collective wait poisoned");
        }
        if self.shared.sequencer.is_aborted() {
            return Err(CommError::Aborted("count exchange aborted".into()));
        }
        let recv_counts = states
            .get(&key)
            .and_then(|state| state.outputs.as_ref())
            .expect("count exchange outputs exist")[rank_pos]
            .clone();
        let remove = {
            let state = states.get_mut(&key).expect("count state exists");
            state.consumed += 1;
            state.consumed == group_size
        };
        if remove {
            states.remove(&key);
        }
        Ok(AllToAllVTicket {
            group: self.group_id,
            instance,
            send_counts: send_counts.to_vec(),
            recv_counts,
            spec: spec.clone(),
        })
    }

    async fn all_to_all_v(
        &self,
        send_buf: &DeviceBuffer,
        send_offsets: &[usize],
        recv_buf: &mut DeviceBuffer,
        recv_offsets: &[usize],
        ticket: AllToAllVTicket,
    ) -> CommResult<CommHandle> {
        self.validate_peer_vectors(&[send_offsets.len(), recv_offsets.len()])?;
        if ticket.group != self.group_id {
            return Err(CommError::InvalidArgument(
                "all_to_all_v ticket belongs to another group".into(),
            ));
        }
        let instance = ticket.instance;
        let mut contribution = Vec::with_capacity(self.group_size());
        for ((offset, count), _) in send_offsets
            .iter()
            .zip(&ticket.send_counts)
            .zip(self.members.iter())
        {
            let span = Self::checked_span(*offset, *count, &ticket.spec, send_buf.len_bytes())?;
            contribution.push(send_buf.as_slice()[span].to_vec());
        }
        let mut receive_spans = Vec::with_capacity(self.group_size());
        for (offset, count) in recv_offsets.iter().zip(&ticket.recv_counts) {
            receive_spans.push(Self::checked_span(
                *offset,
                *count,
                &ticket.spec,
                recv_buf.len_bytes(),
            )?);
        }
        for (index, left) in receive_spans.iter().enumerate() {
            for right in receive_spans.iter().skip(index + 1) {
                if left.start < right.end && right.start < left.end {
                    return Err(CommError::InvalidArgument(
                        "all_to_all_v receive spans overlap".into(),
                    ));
                }
            }
        }
        let signature = Self::signature(&[
            CollectiveKind::AllToAllV as u64,
            Self::dtype_code(ticket.spec.logical_dtype),
            Self::dtype_code(ticket.spec.wire_dtype),
        ]);
        let pending = self.run_collective(
            instance,
            (CollectiveKind::AllToAllV, signature, false),
            CommLeaseSet {
                reads: vec![send_buf.allocation()],
                writes: vec![recv_buf.allocation()],
            },
            contribution,
            |inputs| {
                Ok((0..inputs.len())
                    .map(|destination| {
                        inputs
                            .iter()
                            .map(|source| source[destination].clone())
                            .collect()
                    })
                    .collect())
            },
        )?;
        for ((span, bytes), count) in receive_spans
            .into_iter()
            .zip(&pending.output)
            .zip(ticket.recv_counts)
        {
            let expected = ticket.spec.encoded_bytes(count)?;
            if bytes.len() != expected {
                self.finish_error(instance, &pending.lease);
                return Err(CommError::InvalidArgument(
                    "all_to_all_v remote encoded extent mismatch".into(),
                ));
            }
            recv_buf.as_mut_slice()[span].copy_from_slice(bytes);
        }
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
    }

    async fn all_gather(
        &self,
        instance: CommInstanceId,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
    ) -> CommResult<CommHandle> {
        let send_bytes = Self::checked_extent(count, dtype, send_buf.len_bytes())?;
        let total_count = count
            .checked_mul(self.group_size())
            .ok_or_else(|| CommError::Extent("all_gather total count overflow".into()))?;
        let recv_bytes = Self::checked_extent(total_count, dtype, recv_buf.len_bytes())?;
        let signature = Self::signature(&[
            CollectiveKind::AllGather as u64,
            count as u64,
            Self::dtype_code(dtype),
        ]);
        let pending = self.run_collective(
            instance,
            (CollectiveKind::AllGather, signature, true),
            CommLeaseSet {
                reads: vec![send_buf.allocation()],
                writes: vec![recv_buf.allocation()],
            },
            vec![send_buf.as_slice()[..send_bytes].to_vec()],
            |inputs| {
                let gathered: Vec<u8> = inputs
                    .iter()
                    .flat_map(|segments| segments[0].iter().copied())
                    .collect();
                Ok((0..inputs.len()).map(|_| vec![gathered.clone()]).collect())
            },
        )?;
        recv_buf.as_mut_slice()[..recv_bytes].copy_from_slice(&pending.output[0]);
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
    }

    async fn broadcast(
        &self,
        instance: CommInstanceId,
        buffer: &mut DeviceBuffer,
        len: usize,
        dtype: DType,
        root: RankId,
    ) -> CommResult<CommHandle> {
        let root_pos = self.position(root).ok_or(CommError::UnknownRank(root))?;
        let bytes = Self::checked_extent(len, dtype, buffer.len_bytes())?;
        let signature = Self::signature(&[
            CollectiveKind::Broadcast as u64,
            len as u64,
            Self::dtype_code(dtype),
            u64::from(root.0),
        ]);
        let contribution = if self.rank == root {
            vec![buffer.as_slice()[..bytes].to_vec()]
        } else {
            vec![Vec::new()]
        };
        let leases = if self.rank == root {
            CommLeaseSet {
                reads: vec![buffer.allocation()],
                writes: vec![buffer.allocation()],
            }
        } else {
            CommLeaseSet::write_only([buffer.allocation()])
        };
        let pending = self.run_collective(
            instance,
            (CollectiveKind::Broadcast, signature, true),
            leases,
            contribution,
            |inputs| {
                let root_bytes = inputs[root_pos][0].clone();
                Ok((0..inputs.len())
                    .map(|_| vec![root_bytes.clone()])
                    .collect())
            },
        )?;
        buffer.as_mut_slice()[..bytes].copy_from_slice(&pending.output[0]);
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
    }

    async fn reduce_scatter(
        &self,
        instance: CommInstanceId,
        send_buf: &DeviceBuffer,
        recv_buf: &mut DeviceBuffer,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> CommResult<CommHandle> {
        let total_count = count
            .checked_mul(self.group_size())
            .ok_or_else(|| CommError::Extent("reduce_scatter total count overflow".into()))?;
        let send_bytes = Self::checked_extent(total_count, dtype, send_buf.len_bytes())?;
        let recv_bytes = Self::checked_extent(count, dtype, recv_buf.len_bytes())?;
        let signature = Self::signature(&[
            CollectiveKind::ReduceScatter as u64,
            count as u64,
            Self::dtype_code(dtype),
            Self::op_code(op),
        ]);
        let pending = self.run_collective(
            instance,
            (CollectiveKind::ReduceScatter, signature, true),
            CommLeaseSet {
                reads: vec![send_buf.allocation()],
                writes: vec![recv_buf.allocation()],
            },
            vec![send_buf.as_slice()[..send_bytes].to_vec()],
            |inputs| {
                let rank_inputs: Vec<Vec<u8>> =
                    inputs.iter().map(|segments| segments[0].clone()).collect();
                let reduced = reduce_buffers(&rank_inputs, dtype, op)?;
                Ok((0..inputs.len())
                    .map(|rank| {
                        let start = rank * recv_bytes;
                        vec![reduced[start..start + recv_bytes].to_vec()]
                    })
                    .collect())
            },
        )?;
        recv_buf.as_mut_slice()[..recv_bytes].copy_from_slice(&pending.output[0]);
        self.finish_success(instance, pending)?;
        Ok(CommHandle::ready(Ok(())))
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
        let lease = self
            .shared
            .registry
            .submit(&CommLeaseSet::read_only([buffer.allocation()]))?;
        self.shared
            .mailboxes
            .lock()
            .expect("mailbox poisoned")
            .insert(
                MailKey {
                    group: self.group_id,
                    instance,
                    src_pos: self.self_position(),
                    dst_pos,
                },
                buffer.as_slice()[..bytes].to_vec(),
            );
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
        let message = self
            .shared
            .mailboxes
            .lock()
            .expect("mailbox poisoned")
            .remove(&MailKey {
                group: self.group_id,
                instance,
                src_pos,
                dst_pos: self.self_position(),
            })
            .ok_or(CommError::NoPendingMessage(source))?;
        if message.len() != bytes {
            return Err(CommError::InvalidArgument(format!(
                "recv extent {bytes} does not match sent extent {}",
                message.len()
            )));
        }
        let lease = self
            .shared
            .registry
            .submit(&CommLeaseSet::write_only([buffer.allocation()]))?;
        buffer.as_mut_slice()[..bytes].copy_from_slice(&message);
        self.shared.registry.complete_success(lease.operation())?;
        drop(lease);
        Ok(CommHandle::ready(Ok(())))
    }

    async fn barrier(&self, instance: CommInstanceId) -> CommResult<CommHandle> {
        self.check_not_aborted()?;
        let key = BarrierKey {
            group: self.group_id,
            instance,
        };
        let group_size = self.group_size();
        let mut barriers = self.shared.barriers.lock().expect("barrier poisoned");
        let state = barriers.entry(key).or_insert_with(|| BarrierState {
            arrived: BTreeSet::new(),
            departed: 0,
            released: false,
        });
        if !state.arrived.insert(self.rank) {
            return Err(CommError::Ordering(
                "rank entered one barrier generation twice".into(),
            ));
        }
        if state.arrived.len() == group_size {
            state.released = true;
            self.shared.barrier_cv.notify_all();
        }
        while !barriers
            .get(&key)
            .map(|state| state.released)
            .unwrap_or(true)
            && self
                .shared
                .aborted
                .lock()
                .expect("aborted lock poisoned")
                .is_none()
        {
            #[cfg(test)]
            InProcessTestHooks::pause_before_wait(&self.shared.test_hooks.barrier_before_wait);
            barriers = self
                .shared
                .barrier_cv
                .wait(barriers)
                .expect("barrier wait poisoned");
        }
        let aborted = self
            .shared
            .aborted
            .lock()
            .expect("aborted lock poisoned")
            .clone();
        if let Some(state) = barriers.get_mut(&key) {
            state.departed += 1;
            if state.departed == state.arrived.len() {
                barriers.remove(&key);
            }
        }
        drop(barriers);
        if let Some(cause) = aborted {
            return Err(CommError::Aborted(cause));
        }
        Ok(CommHandle::ready(Ok(())))
    }

    async fn abort(&self, cause: CommError) -> CommResult<()> {
        self.shared.sequencer.abort();
        let mut aborted = self.shared.aborted.lock().expect("aborted lock poisoned");
        if aborted.is_none() {
            *aborted = Some(cause.to_string());
        }
        drop(aborted);
        #[cfg(test)]
        InProcessTestHooks::signal(&self.shared.test_hooks.abort_ready);
        {
            let _barriers = self.shared.barriers.lock().expect("barrier poisoned");
            self.shared.barrier_cv.notify_all();
        }
        #[cfg(test)]
        InProcessTestHooks::signal(&self.shared.test_hooks.barrier_notified);
        {
            let _states = self
                .shared
                .data_collectives
                .lock()
                .expect("collective lock poisoned");
            self.shared.data_cv.notify_all();
        }
        #[cfg(test)]
        InProcessTestHooks::signal(&self.shared.test_hooks.data_notified);
        {
            let _states = self
                .shared
                .count_collectives
                .lock()
                .expect("count collective lock poisoned");
            self.shared.count_cv.notify_all();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_on;
    use std::sync::mpsc;
    use std::time::Duration;

    fn instance(sequence: u32) -> CommInstanceId {
        CommInstanceId::new(1, sequence)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn read_f32(buffer: &DeviceBuffer) -> Vec<f32> {
        buffer
            .as_slice()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn identity_and_topology() {
        let world = InProcessCommunicator::world(3);
        assert_eq!(world.len(), 3);
        for (i, comm) in world.iter().enumerate() {
            assert_eq!(comm.rank(), RankId(i as u32));
            assert_eq!(comm.group_id(), GroupId(0));
            assert_eq!(comm.members().len(), 3);
        }
    }

    #[test]
    fn send_recv_roundtrip_single_thread() {
        let world = InProcessCommunicator::world(2);
        let src = world[0].allocate_from(vec![1u8, 2, 3, 4]);
        let mut dst = world[1].allocate_buffer(4);
        block_on(world[0].send(instance(0), &src, 4, DType::U8, RankId(1))).unwrap();
        block_on(world[1].recv(instance(0), &mut dst, 4, DType::U8, RankId(0))).unwrap();
        assert_eq!(dst.as_slice(), &[1u8, 2, 3, 4]);
    }

    #[test]
    fn barrier_synchronizes_and_reaps_generation() {
        let world = InProcessCommunicator::world(4);
        let shared = world[0].shared.clone();
        let handles: Vec<_> = world
            .into_iter()
            .map(|comm| std::thread::spawn(move || block_on(comm.barrier(instance(0))).unwrap()))
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(shared.barriers.lock().unwrap().is_empty());
    }

    #[test]
    fn abort_wakes_barrier_waiter() {
        let world = InProcessCommunicator::world(2);
        let waiter = world[1].clone();
        let thread = std::thread::spawn(move || block_on(waiter.barrier(instance(0))));
        while world[0]
            .shared
            .barriers
            .lock()
            .unwrap()
            .get(&BarrierKey {
                group: GroupId(0),
                instance: instance(0),
            })
            .map(|state| state.arrived.contains(&RankId(1)))
            != Some(true)
        {
            std::thread::yield_now();
        }
        block_on(world[0].abort(CommError::Aborted("test".into()))).unwrap();
        assert!(matches!(thread.join().unwrap(), Err(CommError::Aborted(_))));
    }

    fn wait_hook() -> (TestWaitHook, mpsc::Receiver<()>, mpsc::SyncSender<()>) {
        let (reached_tx, reached_rx) = mpsc::sync_channel(0);
        let (proceed_tx, proceed_rx) = mpsc::sync_channel(0);
        (
            TestWaitHook {
                reached: reached_tx,
                proceed: proceed_rx,
            },
            reached_rx,
            proceed_tx,
        )
    }

    fn abort_barrier_in_predicate_wait_window() {
        let world = InProcessCommunicator::world(2);
        let shared = world[0].shared.clone();
        let (hook, reached, proceed) = wait_hook();
        *shared.test_hooks.barrier_before_wait.lock().unwrap() = Some(hook);
        let (abort_ready_tx, abort_ready_rx) = mpsc::channel();
        *shared.test_hooks.abort_ready.lock().unwrap() = Some(abort_ready_tx);
        let (notified_tx, notified_rx) = mpsc::channel();
        *shared.test_hooks.barrier_notified.lock().unwrap() = Some(notified_tx);

        let waiter = world[1].clone();
        let (result_tx, result_rx) = mpsc::channel();
        let waiter_thread = std::thread::spawn(move || {
            result_tx
                .send(block_on(waiter.barrier(instance(0))))
                .unwrap();
        });
        reached.recv_timeout(Duration::from_secs(1)).unwrap();

        let aborter = world[0].clone();
        let abort_thread =
            std::thread::spawn(move || block_on(aborter.abort(CommError::Aborted("test".into()))));
        abort_ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let notified_before_park = notified_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        proceed.send(()).unwrap();

        let result = match result_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(result) => Some(result),
            Err(_) => {
                shared.barrier_cv.notify_all();
                result_rx.recv_timeout(Duration::from_secs(1)).ok()
            }
        };
        assert!(abort_thread.join().unwrap().is_ok());
        waiter_thread.join().unwrap();
        assert!(
            !notified_before_park,
            "abort notified the barrier before the waiter parked"
        );
        assert!(matches!(result, Some(Err(CommError::Aborted(_)))));
    }

    fn abort_data_collective_in_predicate_wait_window() {
        let world = InProcessCommunicator::world(2);
        let shared = world[0].shared.clone();
        let (hook, reached, proceed) = wait_hook();
        *shared.test_hooks.data_before_wait.lock().unwrap() = Some(hook);
        let (abort_ready_tx, abort_ready_rx) = mpsc::channel();
        *shared.test_hooks.abort_ready.lock().unwrap() = Some(abort_ready_tx);
        let (notified_tx, notified_rx) = mpsc::channel();
        *shared.test_hooks.data_notified.lock().unwrap() = Some(notified_tx);

        let waiter = world[0].clone();
        let (result_tx, result_rx) = mpsc::channel();
        let waiter_thread = std::thread::spawn(move || {
            let mut buffer = waiter.allocate_from(1i32.to_le_bytes().to_vec());
            result_tx
                .send(block_on(waiter.all_reduce(
                    instance(0),
                    &mut buffer,
                    1,
                    DType::I32,
                    ReduceOp::Sum,
                )))
                .unwrap();
        });
        reached.recv_timeout(Duration::from_secs(1)).unwrap();

        let aborter = world[1].clone();
        let abort_thread =
            std::thread::spawn(move || block_on(aborter.abort(CommError::Aborted("test".into()))));
        abort_ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let notified_before_park = notified_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        proceed.send(()).unwrap();

        let result = match result_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(result) => Some(result),
            Err(_) => {
                shared.data_cv.notify_all();
                result_rx.recv_timeout(Duration::from_secs(1)).ok()
            }
        };
        assert!(abort_thread.join().unwrap().is_ok());
        waiter_thread.join().unwrap();
        assert!(
            !notified_before_park,
            "abort notified the data collective before the waiter parked"
        );
        assert!(matches!(result, Some(Err(CommError::Aborted(_)))));
    }

    #[test]
    fn abort_serializes_notify_with_waiter_park() {
        abort_barrier_in_predicate_wait_window();
        abort_data_collective_in_predicate_wait_window();
    }

    #[test]
    fn all_reduce_sum_is_fixed_rank_order() {
        let world = InProcessCommunicator::world(3);
        let handles: Vec<_> = world
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                std::thread::spawn(move || {
                    let mut buffer = comm
                        .allocate_from(f32_bytes(&[rank as f32 + 1.0, (rank as f32 + 1.0) * 10.0]));
                    block_on(comm.all_reduce(
                        instance(0),
                        &mut buffer,
                        2,
                        DType::F32,
                        ReduceOp::Sum,
                    ))
                    .unwrap();
                    read_f32(&buffer)
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), vec![6.0, 60.0]);
        }
    }

    #[test]
    fn mismatched_collective_order_is_rejected() {
        let world = InProcessCommunicator::world(2);
        let mut first = world[0].allocate_buffer(4);
        let mut second = world[1].allocate_buffer(4);
        let rank0 = world[0].clone();
        let thread = std::thread::spawn(move || {
            block_on(rank0.broadcast(instance(0), &mut first, 1, DType::I32, RankId(0)))
        });
        while !world[1]
            .shared
            .sequencer
            .is_active(GroupId(0), RankId(0), instance(0))
        {
            std::thread::yield_now();
        }
        let error =
            block_on(world[1].all_reduce(instance(0), &mut second, 1, DType::I32, ReduceOp::Sum))
                .unwrap_err();
        assert!(matches!(error, CommError::Ordering(_)));
        block_on(world[1].abort(CommError::Aborted("cleanup".into()))).unwrap();
        assert!(thread.join().unwrap().is_err());
    }

    #[test]
    fn distributed_all_reduce_matches_single_device_bitwise() {
        let shards = [
            vec![1.0f32, 2.5, -3.0, 4.0],
            vec![0.5f32, -1.5, 7.0, 2.0],
            vec![3.0f32, 4.0, -2.0, 1.0],
        ];
        let mut expected = shards[0].clone();
        for shard in &shards[1..] {
            for (value, partial) in expected.iter_mut().zip(shard) {
                *value += *partial;
            }
        }
        let expected_bytes = f32_bytes(&expected);
        let world = InProcessCommunicator::world(shards.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(shards)
            .map(|(comm, shard)| {
                std::thread::spawn(move || {
                    let mut buffer = comm.allocate_from(f32_bytes(&shard));
                    block_on(comm.all_reduce(
                        instance(0),
                        &mut buffer,
                        shard.len(),
                        DType::F32,
                        ReduceOp::Sum,
                    ))
                    .unwrap();
                    buffer.as_slice().to_vec()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected_bytes);
        }
    }

    #[test]
    fn distributed_reduce_scatter_matches_single_device_bitwise() {
        let shards = [
            vec![1.0f32, 2.5, -3.0, 4.0, 8.0, -0.5],
            vec![0.5f32, -1.5, 7.0, 2.0, -3.0, 2.5],
            vec![3.0f32, 4.0, -2.0, 1.0, 0.25, 6.0],
        ];
        let mut reduced = shards[0].clone();
        for shard in &shards[1..] {
            for (value, partial) in reduced.iter_mut().zip(shard) {
                *value += *partial;
            }
        }
        let expected: Vec<Vec<u8>> = reduced.chunks_exact(2).map(f32_bytes).collect();
        let world = InProcessCommunicator::world(shards.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(shards)
            .map(|(comm, shard)| {
                std::thread::spawn(move || {
                    let send = comm.allocate_from(f32_bytes(&shard));
                    let mut recv = comm.allocate_buffer(2 * DType::F32.size());
                    block_on(comm.reduce_scatter(
                        instance(0),
                        &send,
                        &mut recv,
                        2,
                        DType::F32,
                        ReduceOp::Sum,
                    ))
                    .unwrap();
                    recv.as_slice().to_vec()
                })
            })
            .collect();
        let actual: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn distributed_all_gather_matches_single_device_bitwise() {
        let shards = [vec![1u8, 2, 3], vec![10u8, 11, 12], vec![20u8, 21, 22]];
        let expected: Vec<u8> = shards.iter().flatten().copied().collect();
        let recv_len = expected.len();
        let world = InProcessCommunicator::world(shards.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(shards)
            .map(|(comm, shard)| {
                std::thread::spawn(move || {
                    let send = comm.allocate_from(shard);
                    let mut recv = comm.allocate_buffer(recv_len);
                    block_on(comm.all_gather(instance(0), &send, &mut recv, 3, DType::U8)).unwrap();
                    recv.as_slice().to_vec()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }

    #[test]
    fn distributed_broadcast_matches_single_device_bitwise() {
        let inputs = [vec![0u8, 0, 0, 0], vec![0u8, 0, 0, 0], vec![7u8, 8, 9, 10]];
        let expected = inputs[2].clone();
        let world = InProcessCommunicator::world(inputs.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(inputs)
            .map(|(comm, input)| {
                std::thread::spawn(move || {
                    let mut buffer = comm.allocate_from(input);
                    block_on(comm.broadcast(instance(0), &mut buffer, 4, DType::U8, RankId(2)))
                        .unwrap();
                    buffer.as_slice().to_vec()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }

    #[test]
    fn distributed_all_to_all_matches_single_device_bitwise() {
        let sends = [
            [vec![0u8, 1], vec![10, 11], vec![20, 21]],
            [vec![30u8, 31], vec![40, 41], vec![50, 51]],
            [vec![60u8, 61], vec![70, 71], vec![80, 81]],
        ];
        let expected: Vec<Vec<Vec<u8>>> = (0..sends.len())
            .map(|destination| {
                sends
                    .iter()
                    .map(|source| source[destination].clone())
                    .collect()
            })
            .collect();
        let world = InProcessCommunicator::world(sends.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(sends)
            .map(|(comm, chunks)| {
                std::thread::spawn(move || {
                    let sends: Vec<_> = chunks
                        .into_iter()
                        .map(|chunk| comm.allocate_from(chunk))
                        .collect();
                    let send_refs: Vec<_> = sends.iter().collect();
                    let mut recvs: Vec<_> = (0..comm.group_size())
                        .map(|_| comm.allocate_buffer(2))
                        .collect();
                    let mut recv_refs: Vec<_> = recvs.iter_mut().collect();
                    block_on(comm.all_to_all(
                        instance(0),
                        &send_refs,
                        &mut recv_refs,
                        &[2, 2, 2],
                        DType::U8,
                    ))
                    .unwrap();
                    recvs
                        .into_iter()
                        .map(|buffer| buffer.as_slice().to_vec())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let actual: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn distributed_all_to_all_v_matches_single_device_bitwise() {
        let sends = [
            [vec![0u8], vec![10, 11], vec![20]],
            [vec![30u8, 31], vec![40], vec![50, 51]],
            [vec![60u8], vec![70, 71], vec![80]],
        ];
        let expected: Vec<Vec<u8>> = (0..sends.len())
            .map(|destination| {
                sends
                    .iter()
                    .flat_map(|source| source[destination].iter().copied())
                    .collect()
            })
            .collect();
        let world = InProcessCommunicator::world(sends.len());
        let handles: Vec<_> = world
            .into_iter()
            .zip(sends)
            .map(|(comm, chunks)| {
                std::thread::spawn(move || {
                    let counts: Vec<_> = chunks.iter().map(Vec::len).collect();
                    let send_offsets: Vec<_> = counts
                        .iter()
                        .scan(0, |offset, count| {
                            let current = *offset;
                            *offset += count;
                            Some(current)
                        })
                        .collect();
                    let send = comm.allocate_from(chunks.into_iter().flatten().collect());
                    let ticket = block_on(comm.exchange_counts(
                        instance(0),
                        &counts,
                        &WireTensorSpec::identity(DType::U8),
                    ))
                    .unwrap();
                    let recv_offsets: Vec<_> = ticket
                        .recv_counts()
                        .iter()
                        .scan(0, |offset, count| {
                            let current = *offset;
                            *offset += count;
                            Some(current)
                        })
                        .collect();
                    let recv_len = ticket.recv_counts().iter().sum();
                    let mut recv = comm.allocate_buffer(recv_len);
                    block_on(comm.all_to_all_v(
                        &send,
                        &send_offsets,
                        &mut recv,
                        &recv_offsets,
                        ticket,
                    ))
                    .unwrap();
                    recv.as_slice().to_vec()
                })
            })
            .collect();
        let actual: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn all_gather_concatenates_in_member_order() {
        let world = InProcessCommunicator::world(3);
        let handles: Vec<_> = world
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                std::thread::spawn(move || {
                    let send = comm.allocate_from(vec![rank as u8, rank as u8 + 10]);
                    let mut recv = comm.allocate_buffer(6);
                    block_on(comm.all_gather(instance(0), &send, &mut recv, 2, DType::U8)).unwrap();
                    recv.as_slice().to_vec()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), vec![0, 10, 1, 11, 2, 12]);
        }
    }

    #[test]
    fn broadcast_copies_root_bytes() {
        let world = InProcessCommunicator::world(3);
        let handles: Vec<_> = world
            .into_iter()
            .map(|comm| {
                std::thread::spawn(move || {
                    let initial = if comm.rank() == RankId(1) {
                        vec![7, 8, 9]
                    } else {
                        vec![0, 0, 0]
                    };
                    let mut buffer = comm.allocate_from(initial);
                    block_on(comm.broadcast(instance(0), &mut buffer, 3, DType::U8, RankId(1)))
                        .unwrap();
                    buffer.as_slice().to_vec()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), vec![7, 8, 9]);
        }
    }

    #[test]
    fn reduce_scatter_reduces_then_slices_by_rank() {
        let world = InProcessCommunicator::world(2);
        let handles: Vec<_> = world
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                std::thread::spawn(move || {
                    let values = if rank == 0 {
                        vec![1i32, 2, 3, 4]
                    } else {
                        vec![10i32, 20, 30, 40]
                    };
                    let send = comm.allocate_from(
                        values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect(),
                    );
                    let mut recv = comm.allocate_buffer(8);
                    block_on(comm.reduce_scatter(
                        instance(0),
                        &send,
                        &mut recv,
                        2,
                        DType::I32,
                        ReduceOp::Sum,
                    ))
                    .unwrap();
                    recv.as_slice()
                        .chunks_exact(4)
                        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results, vec![vec![11, 22], vec![33, 44]]);
    }

    #[test]
    fn all_to_all_supports_pairwise_chunk_sizes() {
        let world = InProcessCommunicator::world(2);
        let handles: Vec<_> = world
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                std::thread::spawn(move || {
                    let (send_data, sizes) = if rank == 0 {
                        (vec![vec![10], vec![11, 12]], vec![1, 2])
                    } else {
                        (vec![vec![20, 21], vec![22]], vec![2, 1])
                    };
                    let sends: Vec<_> = send_data
                        .into_iter()
                        .map(|bytes| comm.allocate_from(bytes))
                        .collect();
                    let send_refs: Vec<_> = sends.iter().collect();
                    let mut recvs: Vec<_> = sizes
                        .iter()
                        .map(|size| comm.allocate_buffer(*size))
                        .collect();
                    let mut recv_refs: Vec<_> = recvs.iter_mut().collect();
                    block_on(comm.all_to_all(
                        instance(0),
                        &send_refs,
                        &mut recv_refs,
                        &sizes,
                        DType::U8,
                    ))
                    .unwrap();
                    recvs
                        .into_iter()
                        .map(|buffer| buffer.as_slice().to_vec())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results[0], vec![vec![10], vec![20, 21]]);
        assert_eq!(results[1], vec![vec![11, 12], vec![22]]);
    }

    #[test]
    fn all_to_all_v_uses_exchanged_counts_and_offsets() {
        let world = InProcessCommunicator::world(2);
        let handles: Vec<_> = world
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                std::thread::spawn(move || {
                    let (send, counts, send_offsets, recv_offsets, recv_len) = if rank == 0 {
                        (vec![1, 2, 3], vec![1, 2], vec![0, 1], vec![0, 1], 3)
                    } else {
                        (vec![4, 5, 6], vec![2, 1], vec![0, 2], vec![0, 2], 3)
                    };
                    let ticket = block_on(comm.exchange_counts(
                        instance(0),
                        &counts,
                        &WireTensorSpec::identity(DType::U8),
                    ))
                    .unwrap();
                    let send = comm.allocate_from(send);
                    let mut recv = comm.allocate_buffer(recv_len);
                    block_on(comm.all_to_all_v(
                        &send,
                        &send_offsets,
                        &mut recv,
                        &recv_offsets,
                        ticket,
                    ))
                    .unwrap();
                    recv.as_slice().to_vec()
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results[0], vec![1, 4, 5]);
        assert_eq!(results[1], vec![2, 3, 6]);
    }

    #[test]
    fn subgroup_orders_are_independent() {
        let world = InProcessCommunicator::world(3);
        let group_a = InProcessCommunicator::create_group(
            &world,
            GroupSpec {
                id: GroupId(1),
                ranks: vec![RankId(0), RankId(1)],
            },
        )
        .unwrap();
        let group_b = InProcessCommunicator::create_group(
            &world,
            GroupSpec {
                id: GroupId(2),
                ranks: vec![RankId(0), RankId(2)],
            },
        )
        .unwrap();
        assert_eq!(group_a[0].members(), &[RankId(0), RankId(1)]);
        assert_eq!(group_b[0].members(), &[RankId(0), RankId(2)]);
    }
}
