//! Runtime communicator abstraction, in-process reference backend, and the
//! backend buffer-ownership lease registry and collective ordering.
//!
//! See `docs/COMMUNICATOR_BUFFER_IMPL.md` and
//! `docs/COLLECTIVE_ORDERING_IMPL.md`.
//!
//! # What this crate provides
//!
//! * [`Communicator`] — the runtime-level collective/point-to-point trait
//!   (`docs/DISTRIBUTED_RUNTIME.md` §3.1).
//! * [`InProcessCommunicator`] — the Phase-1 single-process, multi-rank
//!   reference backend and test oracle (§4.6), including all collectives.
//! * [`OwnershipRegistry`] — the backend buffer-ownership lease registry,
//!   refined against `specs/tla/BufferOwnership.tla`, emitting contract-revision
//!   trace events at each linearization point for the independent replay
//!   checker.

#![forbid(unsafe_code)]

pub mod communicator;
pub mod inprocess;
pub mod ordering;
mod reduction;
pub mod registry;
pub mod types;

pub use communicator::Communicator;
pub use inprocess::InProcessCommunicator;
pub use ordering::{CollectiveSequencer, membership_hash};
pub use registry::{
    CommLeaseSet, OperationLease, OwnershipError, OwnershipRegistry, OwnershipSnapshot,
};
pub use types::{
    AllToAllVTicket, CommCompletion, CommError, CommHandle, CommInstanceId, CommResult,
    CommSequenceId, DType, DeviceBuffer, DeviceStream, DeviceType, ExecutionId, GlobalDeviceId,
    GroupId, GroupSpec, RankId, ReduceOp, TransportCapability, WireCodec, WireTensorSpec,
};

use std::future::Future;
use std::task::{Context, Poll, Waker};

/// A minimal single-thread executor for the in-process reference backend.
///
/// The in-process communicator completes every operation synchronously, so its
/// futures are ready on the first poll. This `block_on` drives such a future to
/// completion without pulling in a full async runtime. Each simulated rank
/// typically runs its own `block_on` on its own thread (a logical device).
pub fn block_on<F: Future>(future: F) -> F::Output {
    // A no-op waker: our reference futures never register real wakeups.
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            // A ready-only backend should never pend; yield the thread if it
            // somehow does, so a mis-modeled future cannot hot-spin.
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
