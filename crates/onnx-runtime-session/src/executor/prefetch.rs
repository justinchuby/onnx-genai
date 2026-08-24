//! Phase-4 double-buffered weight-prefetch **strategy** (executor half of
//! `docs/memory/WEIGHT_OFFLOAD.md` §4 and issue #87).
//!
//! The compute/transfer overlap for MoE expert paging is split across two
//! layers, per the issue's layering contract:
//!
//! * **EP mechanism** (in `onnx-runtime-ep-cuda`): a real stream-ordered async
//!   H2D copy on a dedicated transfer stream, pinned host staging, and a genuine
//!   CUDA completion event behind [`ExecutionProvider::copy_async`] /
//!   [`ExecutionProvider::wait_fence`]. That half decides *how* a transfer is
//!   made to overlap and *how* a consumer is ordered after it.
//!
//! * **Executor strategy** (this module): *when* to prefetch — the
//!   double-buffering schedule that keeps the transfer stream one expert ahead
//!   of the compute stream. While expert `N` computes, expert `N+1`'s weights
//!   are already being staged into the *other* device buffer, so the transfer
//!   latency of `N+1` is hidden behind the compute of `N`.
//!
//! This module is deliberately EP-agnostic: it drives any `&dyn
//! ExecutionProvider` through the generic `copy_async` + `wait_fence` contract,
//! so the same schedule works for the CUDA EP (real overlap) and degrades to a
//! correct sequential run on a synchronous EP (whose `copy_async` completes
//! inline and whose `wait_fence` is a no-op).
//!
//! ## Ordering guarantees encoded here
//!
//! For every expert `n` the schedule guarantees, in order:
//!   1. `copy_async(n)` is issued (the transfer starts on the copy stream), and
//!   2. `copy_async(n+1)` is issued *before* expert `n` is consumed (overlap),
//!      then
//!   3. `wait_fence(n)` is awaited *before* `compute(n)` (RAW: compute never
//!      reads bytes the copy is still transferring).
//!
//! ## WAR safety on buffer reuse
//!
//! With only two device buffers, slot `s` is reused every second expert, so the
//! copy that refills `s` for expert `n+1` must not overwrite it while expert
//! `n-1`'s compute (which shares slot `s`) is still reading it — a
//! write-after-read hazard. [`drive_double_buffer`] enforces this itself: it
//! records a compute fence over each consumer ([`ExecutionProvider::record_compute_fence`])
//! and makes the transfer stream wait on the prior consumer's fence
//! ([`ExecutionProvider::copy_wait_fence`]) *before* issuing the reuse copy.
//! This is done generically over any `&dyn ExecutionProvider`; on the CUDA EP it
//! becomes a real cross-stream `cuStreamWaitEvent`, and on a synchronous EP the
//! fences are already-signalled and the waits are no-ops. The GPU regression
//! test `drive_double_buffer_war_safe_across_waves` (session `cuda` feature)
//! drives this public path across enough waves that both slots are reused and
//! fails if the driver's WAR fence is removed.
//!
//! ## What this module does **not** own
//!
//! Wiring this scheduler into the live MoE decode loop depends on Phase-3b
//! (live device weight binding), which is not yet landed; until then this is a
//! standalone, unit-tested strategy component (issue #87 follow-up tracks the
//! live wiring).

use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider, Fence, Result};

/// One step of a double-buffered prefetch schedule.
///
/// [`plan_double_buffer`] renders the whole schedule as a `Vec<PrefetchStep>`,
/// which is what makes the *strategy* (the interleaving of transfers and
/// computes) inspectable and unit-testable without any device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStep {
    /// Enqueue expert `expert`'s async weight transfer into device `buffer`
    /// (`buffer` is `0` or `1` — the double-buffer slot).
    Prefetch { expert: usize, buffer: usize },
    /// Await the completion fence of expert `expert`'s transfer before its
    /// weights are consumed (RAW ordering).
    Await { expert: usize },
    /// Consume (compute over) expert `expert`'s now-resident weights in device
    /// `buffer`.
    Compute { expert: usize, buffer: usize },
}

/// Render the double-buffered prefetch schedule for `num_experts`.
///
/// The schedule primes expert `0`'s transfer, then for each expert `n` issues
/// expert `n+1`'s prefetch (into the alternate buffer) *before* awaiting and
/// computing expert `n` — so the transfer of `n+1` overlaps the compute of `n`.
/// Buffers alternate `n % 2`, so at most two device staging buffers are live.
///
/// This is a pure function of `num_experts`; it allocates nothing on device and
/// is the canonical description of the executor's Phase-4 strategy.
pub fn plan_double_buffer(num_experts: usize) -> Vec<PrefetchStep> {
    let mut steps = Vec::new();
    if num_experts == 0 {
        return steps;
    }
    // Prime: start expert 0's transfer into buffer 0 before the loop.
    steps.push(PrefetchStep::Prefetch {
        expert: 0,
        buffer: 0,
    });
    for n in 0..num_experts {
        // Issue the next expert's prefetch first so its transfer overlaps the
        // compute of the current expert.
        if n + 1 < num_experts {
            steps.push(PrefetchStep::Prefetch {
                expert: n + 1,
                buffer: (n + 1) % 2,
            });
        }
        steps.push(PrefetchStep::Await { expert: n });
        steps.push(PrefetchStep::Compute {
            expert: n,
            buffer: n % 2,
        });
    }
    steps
}

/// Drive a double-buffered prefetch over `sources` against `ep`.
///
/// Executes the [`plan_double_buffer`] schedule for real: each expert's weights
/// are staged with [`ExecutionProvider::copy_async`] onto the EP's transfer
/// stream (overlapping the prior expert's compute), ordered before consumption
/// with [`ExecutionProvider::wait_fence`], and then handed to `compute`.
///
/// * `buffers` — exactly two device staging buffers (the double-buffer slots),
///   each at least as large as the biggest transfer in `sizes`.
/// * `sources` / `sizes` — per-expert source buffers and transfer byte counts
///   (must have equal length).
/// * `compute(expert, weights)` — invoked once per expert with the buffer that
///   now holds its fully-transferred weights.
///
/// The next expert's prefetch is always issued before the current expert's
/// `wait_fence`/`compute`, which is what produces the overlap. Before a reuse
/// copy overwrites a double-buffer slot, the driver makes the transfer stream
/// wait on the prior consumer of that slot ([`ExecutionProvider::copy_wait_fence`]
/// on the fence recorded by [`ExecutionProvider::record_compute_fence`]), so
/// buffer reuse is write-after-read safe on async EPs. On a synchronous EP this
/// still runs correctly — `copy_async` completes inline, and `wait_fence` /
/// `copy_wait_fence` are no-ops — it simply does not overlap.
pub fn drive_double_buffer<F>(
    ep: &dyn ExecutionProvider,
    buffers: &mut [DeviceBuffer; 2],
    sources: &[DeviceBuffer],
    sizes: &[usize],
    mut compute: F,
) -> Result<()>
where
    F: FnMut(usize, &DeviceBuffer) -> Result<()>,
{
    assert_eq!(
        sources.len(),
        sizes.len(),
        "double-buffer prefetch: sources and sizes must have equal length"
    );
    let num_experts = sources.len();
    if num_experts == 0 {
        return Ok(());
    }

    // WAR guard: the last compute that read each double-buffer slot. Before a
    // reuse copy overwrites a slot, the transfer stream must wait on the prior
    // consumer's completion (`copy_wait_fence`), or a still-running kernel's read
    // races the overwrite. Slots start already-signalled (no prior consumer).
    let mut last_compute_fence: [Fence; 2] = [Fence::signalled(), Fence::signalled()];

    // Prime expert 0's transfer into buffer 0.
    let mut current_fence: Fence = ep.copy_async(&sources[0], &mut buffers[0], sizes[0])?;

    for n in 0..num_experts {
        let current_slot = n % 2;

        // Overlap: enqueue expert n+1's transfer into the alternate buffer
        // before we wait on / consume expert n.
        let next_fence = if n + 1 < num_experts {
            let next_slot = (n + 1) % 2;
            // WAR: the alternate slot may still be under read by the consumer of
            // expert n-1 (which shares this slot). Hold the reuse copy on the
            // transfer stream until that prior consumer completes. This is
            // enforced by the driver over any `&dyn ExecutionProvider`; on a
            // synchronous EP the fence is already-signalled and this is a no-op.
            ep.copy_wait_fence(&last_compute_fence[next_slot])?;
            Some(ep.copy_async(&sources[n + 1], &mut buffers[next_slot], sizes[n + 1])?)
        } else {
            None
        };

        // RAW: the compute stream must observe expert n's fully-transferred
        // bytes before the kernel reads them.
        ep.wait_fence(&current_fence)?;
        compute(n, &buffers[current_slot])?;

        // Mark this slot busy until the just-issued consumer finishes, so a
        // future reuse prefetch into it waits (WAR).
        last_compute_fence[current_slot] = ep.record_compute_fence()?;

        if let Some(next) = next_fence {
            current_fence = next;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use onnx_runtime_ep_api::{EpConfig, KernelMatch};
    use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};

    #[test]
    fn plan_double_buffer_empty_is_empty() {
        assert!(plan_double_buffer(0).is_empty());
    }

    #[test]
    fn plan_double_buffer_single_expert_needs_no_second_prefetch() {
        assert_eq!(
            plan_double_buffer(1),
            vec![
                PrefetchStep::Prefetch {
                    expert: 0,
                    buffer: 0
                },
                PrefetchStep::Await { expert: 0 },
                PrefetchStep::Compute {
                    expert: 0,
                    buffer: 0
                },
            ]
        );
    }

    #[test]
    fn plan_double_buffer_interleaves_next_prefetch_before_current_compute() {
        let steps = plan_double_buffer(4);
        assert_eq!(
            steps,
            vec![
                PrefetchStep::Prefetch {
                    expert: 0,
                    buffer: 0
                },
                PrefetchStep::Prefetch {
                    expert: 1,
                    buffer: 1
                },
                PrefetchStep::Await { expert: 0 },
                PrefetchStep::Compute {
                    expert: 0,
                    buffer: 0
                },
                PrefetchStep::Prefetch {
                    expert: 2,
                    buffer: 0
                },
                PrefetchStep::Await { expert: 1 },
                PrefetchStep::Compute {
                    expert: 1,
                    buffer: 1
                },
                PrefetchStep::Prefetch {
                    expert: 3,
                    buffer: 1
                },
                PrefetchStep::Await { expert: 2 },
                PrefetchStep::Compute {
                    expert: 2,
                    buffer: 0
                },
                PrefetchStep::Await { expert: 3 },
                PrefetchStep::Compute {
                    expert: 3,
                    buffer: 1
                },
            ]
        );
    }

    /// For every expert `n`, the prefetch of `n+1` must be scheduled strictly
    /// before the compute of `n` (the overlap invariant), and the await of `n`
    /// must precede the compute of `n` (RAW). Buffers must alternate.
    #[test]
    fn plan_double_buffer_overlap_and_raw_invariants_hold() {
        for num_experts in 1..=8 {
            let steps = plan_double_buffer(num_experts);
            let pos = |pred: &dyn Fn(&PrefetchStep) -> bool| steps.iter().position(pred);
            for n in 0..num_experts {
                let compute_n =
                    pos(&|s| matches!(s, PrefetchStep::Compute { expert, .. } if *expert == n))
                        .expect("compute step present");
                let await_n = pos(&|s| matches!(s, PrefetchStep::Await { expert } if *expert == n))
                    .expect("await step present");
                assert!(await_n < compute_n, "await(n) must precede compute(n)");

                if n + 1 < num_experts {
                    let prefetch_next = pos(
                        &|s| matches!(s, PrefetchStep::Prefetch { expert, .. } if *expert == n + 1),
                    )
                    .expect("prefetch(n+1) present");
                    assert!(
                        prefetch_next < compute_n,
                        "prefetch(n+1) must be issued before compute(n) to overlap"
                    );
                }

                // Buffer for expert n is n % 2 in both prefetch and compute.
                let compute_slot = steps.iter().find_map(|s| match s {
                    PrefetchStep::Compute { expert, buffer } if *expert == n => Some(*buffer),
                    _ => None,
                });
                assert_eq!(compute_slot, Some(n % 2));
            }
        }
    }

    /// Records the exact order of transfer/consume operations so a test can
    /// assert the schedule actually overlaps at the `&dyn ExecutionProvider`
    /// boundary. `copy_async` returns a *real*, un-signalled fence and logs a
    /// `copy_async(id)`; `wait_fence` logs `wait_fence(id)`. If the driver ever
    /// consumed an expert without awaiting its transfer, the recorded log would
    /// not contain the matching `wait_fence` before the `compute`.
    struct RecordingEp {
        cpu_device: DeviceId,
        log: Mutex<Vec<String>>,
        next_fence_id: std::sync::atomic::AtomicU64,
    }

    impl RecordingEp {
        fn new() -> Self {
            Self {
                cpu_device: DeviceId::cpu(),
                log: Mutex::new(Vec::new()),
                next_fence_id: std::sync::atomic::AtomicU64::new(1),
            }
        }

        fn log(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    impl ExecutionProvider for RecordingEp {
        fn consume_route_residency_at_boundary(&self) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            "recording_ep"
        }
        fn device_type(&self) -> DeviceType {
            self.cpu_device.device_type
        }
        fn device_id(&self) -> DeviceId {
            self.cpu_device
        }
        fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
            Ok(())
        }
        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
        fn supports_op(
            &self,
            _op: &Node,
            _opset: u64,
            _shapes: &[Shape],
            _input_dtypes: &[DataType],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::unsupported("recording_ep runs no kernels")
        }
        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> Result<Box<dyn onnx_runtime_ep_api::Kernel>> {
            Err(onnx_runtime_ep_api::EpError::KernelFailed(
                "recording_ep runs no kernels".into(),
            ))
        }
        fn allocate(&self, size: usize, alignment: usize) -> Result<DeviceBuffer> {
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .map_err(|_| onnx_runtime_ep_api::EpError::AlignmentError)?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(onnx_runtime_ep_api::EpError::OutOfMemory {
                    requested: size,
                    available: 0,
                });
            }
            Ok(unsafe {
                DeviceBuffer::from_raw_parts(ptr.cast(), self.cpu_device, size, alignment)
            })
        }
        fn deallocate(&self, buffer: DeviceBuffer) -> Result<()> {
            let size = buffer.len();
            let alignment = buffer.alignment();
            let ptr = buffer.into_raw().cast::<u8>();
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .expect("recording_ep allocated this layout");
            unsafe { std::alloc::dealloc(ptr, layout) };
            Ok(())
        }
        fn copy(&self, src: &DeviceBuffer, dst: &mut DeviceBuffer, size: usize) -> Result<()> {
            if size != 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().cast::<u8>(),
                        dst.as_mut_ptr().cast::<u8>(),
                        size,
                    )
                };
            }
            Ok(())
        }
        fn copy_async(
            &self,
            src: &DeviceBuffer,
            dst: &mut DeviceBuffer,
            size: usize,
        ) -> Result<Fence> {
            // Perform the byte copy so `compute` can validate contents, but
            // return a *real* (un-signalled) fence and log the enqueue so the
            // test can prove the driver issues the next prefetch before it
            // waits on / consumes the current expert.
            self.copy(src, dst, size)?;
            let id = self
                .next_fence_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.log.lock().unwrap().push(format!("copy_async({id})"));
            Ok(Fence::new(id))
        }
        fn wait_fence(&self, fence: &Fence) -> Result<()> {
            self.log
                .lock()
                .unwrap()
                .push(format!("wait_fence({})", fence.id));
            Ok(())
        }
        fn sync(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn drive_double_buffer_overlaps_and_orders_transfers() {
        let mut ep = RecordingEp::new();
        ep.initialize(&EpConfig::default()).unwrap();

        // Three "experts", each a distinct 8-byte payload.
        let payloads: [[u8; 8]; 3] = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ];
        let sizes = [8usize, 8, 8];

        let mut sources: Vec<DeviceBuffer> = Vec::new();
        for p in &payloads {
            let mut b = ep.allocate(8, 8).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(p.as_ptr(), b.as_mut_ptr().cast::<u8>(), 8);
            }
            sources.push(b);
        }
        let mut buffers = [ep.allocate(8, 8).unwrap(), ep.allocate(8, 8).unwrap()];

        // Each compute reads its buffer and records what it observed, proving
        // the awaited transfer delivered the correct expert's bytes.
        let observed: Mutex<Vec<[u8; 8]>> = Mutex::new(Vec::new());
        drive_double_buffer(&ep, &mut buffers, &sources, &sizes, |expert, weights| {
            let mut got = [0u8; 8];
            unsafe {
                std::ptr::copy_nonoverlapping(weights.as_ptr().cast::<u8>(), got.as_mut_ptr(), 8);
            }
            let _ = expert;
            observed.lock().unwrap().push(got);
            Ok(())
        })
        .unwrap();

        // Every expert saw its own payload (double-buffering never crossed the
        // wires between the two slots).
        assert_eq!(
            observed.into_inner().unwrap(),
            vec![payloads[0], payloads[1], payloads[2]]
        );

        // The recorded op order proves overlap + RAW ordering:
        //   copy_async(1)  prime expert 0
        //   copy_async(2)  prefetch expert 1  <- issued before waiting on 0
        //   wait_fence(1)  await expert 0
        //   (compute 0)
        //   copy_async(3)  prefetch expert 2  <- issued before waiting on 1
        //   wait_fence(2)  await expert 1
        //   (compute 1)
        //   wait_fence(3)  await expert 2
        //   (compute 2)
        let log = ep.log();
        assert_eq!(
            log,
            vec![
                "copy_async(1)".to_string(),
                "copy_async(2)".to_string(),
                "wait_fence(1)".to_string(),
                "copy_async(3)".to_string(),
                "wait_fence(2)".to_string(),
                "wait_fence(3)".to_string(),
            ]
        );

        // Explicitly assert the overlap invariant: prefetch of expert n+1 is
        // enqueued before expert n's fence is awaited.
        let idx = |needle: &str| log.iter().position(|s| s == needle).unwrap();
        assert!(
            idx("copy_async(2)") < idx("wait_fence(1)"),
            "expert 1 prefetch must be issued before awaiting expert 0"
        );
        assert!(
            idx("copy_async(3)") < idx("wait_fence(2)"),
            "expert 2 prefetch must be issued before awaiting expert 1"
        );

        for b in sources {
            ep.deallocate(b).unwrap();
        }
        for b in buffers {
            ep.deallocate(b).unwrap();
        }
    }
}
