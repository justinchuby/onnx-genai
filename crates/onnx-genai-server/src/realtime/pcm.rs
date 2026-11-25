//! PCM16 message framing with one persistent sampling clock per audio turn.
use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_engine::pipeline::StreamingResampler;
use std::{
    collections::VecDeque,
    num::{NonZeroU32, NonZeroUsize},
};

pub(super) struct PcmInput {
    input_rate: NonZeroU32,
    native_rate: NonZeroU32,
    frame_samples: NonZeroUsize,
    byte_limit: NonZeroUsize,
    native_sample_limit: NonZeroUsize,
    source: VecDeque<i16>,
    pending: VecDeque<f32>,
    resampler: Option<StreamingResampler>,
    committed: bool,
}
impl PcmInput {
    pub(super) fn new(
        input_rate: NonZeroU32,
        native_rate: NonZeroU32,
        frame_samples: NonZeroUsize,
        byte_limit: NonZeroUsize,
        native_sample_limit: NonZeroUsize,
    ) -> anyhow::Result<Self> {
        ensure!(
            byte_limit.get() >= 2,
            "realtime PCM input byte limit must allow at least one two-byte sample"
        );
        ensure!(
            frame_samples <= native_sample_limit,
            "realtime model frame size {frame_samples} exceeds the configured native PCM buffer limit {native_sample_limit}; increase the audio memory budget or select a compatible package"
        );
        Ok(Self {
            input_rate,
            native_rate,
            frame_samples,
            byte_limit,
            native_sample_limit,
            source: VecDeque::new(),
            pending: VecDeque::new(),
            resampler: Some(StreamingResampler::new(input_rate, native_rate)),
            committed: false,
        })
    }
    pub(super) fn append(&mut self, encoded: &str) -> anyhow::Result<()> {
        ensure!(
            !self.committed,
            "input_audio_buffer.append cannot append to an already committed PCM turn; begin a new audio buffer"
        );
        let queued_bytes = self
            .source
            .len()
            .checked_mul(2)
            .context("realtime PCM queue byte count overflow")?;
        let remaining = self
            .byte_limit
            .get()
            .checked_sub(queued_bytes)
            .context("realtime PCM queue exceeds its configured byte limit")?;
        let encoded_limit = remaining
            .div_ceil(3)
            .checked_mul(4)
            .context("realtime base64 admission limit overflow")?;
        ensure!(
            encoded.len() <= encoded_limit,
            "input_audio_buffer.append exceeds the remaining {remaining}-byte PCM queue budget; send smaller audio messages and wait for the server to consume them"
        );
        let bytes = STANDARD.decode(encoded).context("input_audio_buffer.append.audio must be valid base64 containing little-endian mono PCM16 samples")?;
        ensure!(
            bytes.len() <= remaining,
            "input_audio_buffer.append exceeds the remaining {remaining}-byte PCM queue budget; send smaller audio messages"
        );
        ensure!(
            bytes.len().is_multiple_of(2),
            "input_audio_buffer.append.audio decoded to {} bytes; PCM16 requires an even byte count with two little-endian bytes per mono sample",
            bytes.len()
        );
        self.source.extend(
            bytes
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]])),
        );
        Ok(())
    }
    pub(super) fn change_input_rate(&mut self, rate: NonZeroU32) -> anyhow::Result<()> {
        ensure!(
            !self.committed && self.source.is_empty(),
            "drain queued PCM before changing input rate; already converted native samples remain in order"
        );
        self.admit_output(0)?;
        if let Some(resampler) = self.resampler.take() {
            self.pending.extend(resampler.finish()?);
        }
        self.resampler = Some(StreamingResampler::new(rate, self.native_rate));
        self.input_rate = rate;
        Ok(())
    }
    pub(super) fn commit(&mut self) -> anyhow::Result<()> {
        ensure!(
            !self.committed,
            "input_audio_buffer.commit was already received for this audio turn; begin a new audio buffer before committing again"
        );
        self.committed = true;
        Ok(())
    }
    fn admit_output(&self, additional: usize) -> anyhow::Result<()> {
        let resampler = self
            .resampler
            .as_ref()
            .context("realtime PCM resampler is already finalized")?;
        let remaining = resampler.remaining_output_samples(additional)?;
        let buffered = self
            .pending
            .len()
            .checked_add(remaining)
            .context("realtime resampled PCM buffer size overflow")?;
        ensure!(
            buffered <= self.native_sample_limit.get(),
            "realtime resampling expansion needs {buffered} buffered native samples, exceeding the {}-sample audio memory budget; reduce input/native rate ratio or raise the native PCM buffer limit",
            self.native_sample_limit
        );
        Ok(())
    }
    /// Feed bounded source chunks until one native frame is ready. Only commit
    /// flushes filter lookahead and permits a final partial model frame.
    pub(super) fn next_frame(&mut self) -> anyhow::Result<Option<Vec<f32>>> {
        while self.pending.len() < self.frame_samples.get() {
            if self.source.is_empty() {
                if self.committed && self.resampler.is_some() {
                    self.admit_output(0)?;
                    if let Some(resampler) = self.resampler.take() {
                        self.pending.extend(resampler.finish()?);
                    }
                }
                break;
            }
            let needed = self.frame_samples.get() - self.pending.len();
            let source_needed = (needed as u128 * u128::from(self.input_rate.get()))
                .div_ceil(u128::from(self.native_rate.get()));
            let take = usize::try_from(source_needed).context("realtime source frame length exceeds host address space; reduce the model's declared frame bound")?.max(1).min(self.source.len());
            self.admit_output(take)?;
            let input: Vec<f32> = self
                .source
                .drain(..take)
                .map(|sample| f32::from(sample) / 32768.0)
                .collect();
            let resampler = self.resampler.as_mut().context(
                "realtime PCM resampler was finalized before consuming its source samples",
            )?;
            self.pending.extend(resampler.push(&input)?);
        }
        let count = self.pending.len().min(self.frame_samples.get());
        if count == 0 || (count < self.frame_samples.get() && !self.committed) {
            return Ok(None);
        }
        Ok(Some(self.pending.drain(..count).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input(source: u32, target: u32, frame: usize, limit: usize) -> PcmInput {
        PcmInput::new(
            NonZeroU32::new(source).unwrap(),
            NonZeroU32::new(target).unwrap(),
            NonZeroUsize::new(frame).unwrap(),
            NonZeroUsize::new(limit).unwrap(),
            NonZeroUsize::new(8192).unwrap(),
        )
        .unwrap()
    }
    fn encoded(samples: &[i16]) -> String {
        STANDARD.encode(
            samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    #[test]
    fn pcm_messages_keep_one_clock_and_commit_the_exact_tail() {
        let source: Vec<i16> = (0..4410)
            .map(|i| ((i as f64 * 0.3).sin() * 20_000.0) as i16)
            .collect();
        let expected = onnx_genai_engine::pipeline::resample_planar(
            &source
                .iter()
                .map(|sample| f32::from(*sample) / 32768.0)
                .collect::<Vec<_>>(),
            1,
            source.len(),
            44_100,
            32_000,
        )
        .unwrap();
        for chunk in [1, 7, 257, 1024] {
            let mut input = input(44_100, 32_000, 319, 8192);
            let mut actual = Vec::new();
            for samples in source.chunks(chunk) {
                input.append(&encoded(samples)).unwrap();
                while let Some(frame) = input.next_frame().unwrap() {
                    assert_eq!(frame.len(), 319);
                    actual.extend(frame);
                }
                assert!(input.pending.len() < 319);
            }
            input.commit().unwrap();
            while let Some(frame) = input.next_frame().unwrap() {
                assert!(frame.len() <= 319);
                actual.extend(frame);
            }
            assert_eq!(actual, expected, "chunk size {chunk}");
            assert_eq!(actual.len(), 3200);
            assert!(input.next_frame().unwrap().is_none());
            assert!(
                input
                    .append("")
                    .unwrap_err()
                    .to_string()
                    .contains("committed")
            );
        }
    }
    #[test]
    fn extreme_rate_expansion_is_rejected_before_allocating_output() {
        let mut input = PcmInput::new(
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(u32::MAX).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(8).unwrap(),
            NonZeroUsize::new(64).unwrap(),
        )
        .unwrap();
        input.append(&encoded(&[1])).unwrap();
        let error = input.next_frame().unwrap_err().to_string();
        assert!(
            error.contains("resampling expansion") && error.contains("64-sample"),
            "{error}"
        );
        assert_eq!(
            input.source.len(),
            1,
            "failed expansion admission does not consume PCM"
        );
        assert!(input.pending.is_empty());
    }
    #[test]
    fn pcm_admission_is_bounded_and_failed_messages_do_not_mutate_the_buffer() {
        let mut input = input(16000, 16000, 4, 8);
        input.append(&encoded(&[i16::MIN, 0, i16::MAX])).unwrap();
        assert!(
            input
                .append(&encoded(&[1, 2]))
                .unwrap_err()
                .to_string()
                .contains("budget")
        );
        assert!(
            input
                .append("!")
                .unwrap_err()
                .to_string()
                .contains("base64")
        );
        assert!(
            input
                .append(&STANDARD.encode([0]))
                .unwrap_err()
                .to_string()
                .contains("even byte count")
        );
        assert!(input.next_frame().unwrap().is_none());
        input.commit().unwrap();
        let frame = input.next_frame().unwrap().unwrap();
        assert_eq!(frame, [-1.0, 0.0, 32767.0 / 32768.0]);
        assert!(input.next_frame().unwrap().is_none());
        assert!(input.commit().is_err());
    }
}
