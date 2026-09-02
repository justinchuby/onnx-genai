#![cfg(feature = "gpu-tests")]

use anyhow::{Context, Result};
use cudarc::driver::sys::{
    CUdeviceptr, CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode,
};
use cudarc::driver::{CudaContext, CudaStream, result};
use onnx_genai_bench::freetoken_byte_ab::{
    ByteClass, CounterClass, FailureDisposition, LedgerAuthority, Phase, ScopeIdentity,
    ScopedLedger,
};

const EXPERT_BYTES: usize = 64 * 1024;
const BASELINE_COPIES: u64 = 4;
const OPTIMIZED_COPIES: u64 = 2;
const REPLAYS: u64 = 3;

struct CudaCompletion<'a> {
    context: &'a std::sync::Arc<CudaContext>,
    stream: &'a std::sync::Arc<CudaStream>,
    destination: CUdeviceptr,
}

impl CudaCompletion<'_> {
    fn completed_h2d(
        &self,
        source: &[u8],
        ledger: &mut ScopedLedger,
        authority: &LedgerAuthority,
        phase: Phase,
        feature: bool,
    ) -> Result<()> {
        let submission = ledger.begin_submission(authority, phase)?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::H2d,
            source.len() as u64,
            feature,
        )?;
        let completion = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("create H2D completion event")?;
        // SAFETY: destination covers source.len() bytes and remains live through
        // the completion event.
        unsafe { result::memcpy_htod_async(self.destination, source, self.stream.cu_stream()) }
            .context("enqueue H2D positive control")?;
        completion
            .record(self.stream)
            .context("record H2D completion event")?;
        completion
            .synchronize()
            .context("wait for H2D completion event")?;
        ledger.commit_submission(authority, submission)
    }

    fn completed_graph_replay(
        &self,
        graph: &cudarc::driver::CudaGraph,
        bytes: u64,
        ledger: &mut ScopedLedger,
        authority: &LedgerAuthority,
    ) -> Result<()> {
        let submission = ledger.begin_submission(authority, Phase::Replay)?;
        ledger.stage_bytes(authority, submission, ByteClass::D2d, bytes, true)?;
        ledger.stage_counter(authority, submission, CounterClass::Replays, 1)?;
        graph.launch().context("launch captured D2D replay")?;
        let completion = self
            .context
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .context("create replay completion event")?;
        completion
            .record(self.stream)
            .context("record replay completion event")?;
        completion
            .synchronize()
            .context("wait for replay completion event")?;
        ledger.commit_submission(authority, submission)
    }
}

#[test]
fn real_cuda_completion_and_capture_replay_match_exact_byte_receipts() -> Result<()> {
    let context = CudaContext::new(0).context(
        "FreeToken GPU test requires CUDA_VISIBLE_DEVICES to expose one serialized idle A100",
    )?;
    context.bind_to_thread().context("bind CUDA context")?;
    let stream = context
        .new_stream()
        .context("create non-default capture stream")?;
    let source = (0..EXPERT_BYTES)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(7))
        .collect::<Vec<_>>();
    // SAFETY: both allocations are freed exactly once after all events complete.
    let source_device =
        unsafe { result::malloc_sync(EXPERT_BYTES) }.context("allocate captured source buffer")?;
    // SAFETY: see above.
    let destination_device = unsafe { result::malloc_sync(EXPERT_BYTES) }
        .context("allocate captured destination buffer")?;

    let result = (|| -> Result<()> {
        let completion = CudaCompletion {
            context: &context,
            stream: &stream,
            destination: source_device,
        };
        let (mut baseline, baseline_authority) = ScopedLedger::new(ScopeIdentity {
            provider: 1,
            device: 0,
            executor: 11,
            generation: 1,
            logical_session: 101,
        });
        for _ in 0..BASELINE_COPIES {
            completion.completed_h2d(
                &source,
                &mut baseline,
                &baseline_authority,
                Phase::DecodeSteady,
                false,
            )?;
        }

        let (mut optimized, optimized_authority) = ScopedLedger::new(ScopeIdentity {
            provider: 2,
            device: 0,
            executor: 12,
            generation: 1,
            logical_session: 102,
        });
        for _ in 0..OPTIMIZED_COPIES {
            completion.completed_h2d(
                &source,
                &mut optimized,
                &optimized_authority,
                Phase::DirectWarmup,
                true,
            )?;
        }

        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .context("begin D2D graph capture")?;
        // SAFETY: both allocations cover EXPERT_BYTES and remain live through
        // graph destruction.
        unsafe {
            result::memcpy_dtod_async(
                destination_device,
                source_device,
                EXPERT_BYTES,
                stream.cu_stream(),
            )
        }
        .context("capture D2D state carry copy")?;
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .context("end D2D graph capture")?
            .context("capture produced no graph")?;
        for _ in 0..REPLAYS {
            completion.completed_graph_replay(
                &graph,
                EXPERT_BYTES as u64,
                &mut optimized,
                &optimized_authority,
            )?;
        }

        let mut copied = vec![0u8; EXPERT_BYTES];
        // SAFETY: destination_device covers copied.len() initialized bytes.
        unsafe { result::memcpy_dtoh_sync(&mut copied, destination_device) }
            .context("read captured D2D output")?;
        assert_eq!(copied, source, "captured replay changed copied bytes");

        let baseline_phases = baseline.snapshot(&baseline_authority)?;
        let baseline_decode = baseline_phases
            .iter()
            .find(|phase| phase.phase == Phase::DecodeSteady)
            .context("baseline decode phase")?;
        assert_eq!(
            baseline_decode.accounting.bytes.value(ByteClass::H2d),
            BASELINE_COPIES * EXPERT_BYTES as u64
        );
        assert_eq!(
            baseline_decode.accounting.feature_bytes,
            Default::default(),
            "default-off arm must record zero feature bytes"
        );

        let optimized_phases = optimized.snapshot(&optimized_authority)?;
        let warmup = optimized_phases
            .iter()
            .find(|phase| phase.phase == Phase::DirectWarmup)
            .context("optimized warmup phase")?;
        assert_eq!(
            warmup.accounting.bytes.value(ByteClass::H2d),
            OPTIMIZED_COPIES * EXPERT_BYTES as u64
        );
        let replay = optimized_phases
            .iter()
            .find(|phase| phase.phase == Phase::Replay)
            .context("optimized replay phase")?;
        assert_eq!(
            replay.accounting.bytes.value(ByteClass::D2d),
            REPLAYS * EXPERT_BYTES as u64
        );
        assert_eq!(replay.accounting.counter(CounterClass::Replays), REPLAYS);

        let pending = optimized.begin_submission(&optimized_authority, Phase::Failure)?;
        optimized.stage_bytes(
            &optimized_authority,
            pending,
            ByteClass::H2d,
            EXPERT_BYTES as u64,
            true,
        )?;
        optimized.fail_submission(
            &optimized_authority,
            pending,
            FailureDisposition::RolledBack,
        )?;
        let failure = optimized
            .snapshot(&optimized_authority)?
            .into_iter()
            .find(|phase| phase.phase == Phase::Failure)
            .context("failure phase")?;
        assert_eq!(failure.accounting.bytes.value(ByteClass::H2d), 0);
        assert_eq!(
            failure.accounting.bytes.rolled_back[&ByteClass::H2d],
            EXPERT_BYTES as u64
        );
        eprintln!(
            "freetoken_cuda_bytes: baseline_decode_h2d={} optimized_warmup_h2d={} \
             replay_d2d={} replays={} rolled_back_h2d={} useful_failure_h2d=0",
            BASELINE_COPIES * EXPERT_BYTES as u64,
            OPTIMIZED_COPIES * EXPERT_BYTES as u64,
            REPLAYS * EXPERT_BYTES as u64,
            REPLAYS,
            EXPERT_BYTES,
        );
        Ok(())
    })();

    // SAFETY: allocations are live and no work remains after completion events.
    let free_destination = unsafe { result::free_sync(destination_device) };
    // SAFETY: same.
    let free_source = unsafe { result::free_sync(source_device) };
    result?;
    free_destination.context("free captured destination buffer")?;
    free_source.context("free captured source buffer")?;
    Ok(())
}
