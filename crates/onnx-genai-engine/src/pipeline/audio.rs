use anyhow::{Context, bail};
use onnx_genai_metadata::{
    MediaContainer, MediaDelivery, MediaEncoding, OutputStage, WorkflowOutput, WorkflowOutputRole,
    WorkflowSpec,
};

use super::{PipelineOutputs, WorkflowRuntime};

const RESAMPLE_RADIUS: isize = 16;

/// Whether a single workflow output declares a compatible buffered PCM16 WAV
/// audio contract: role `audio`, buffered delivery, WAV container, `pcm_s16_le`
/// encoding, and a positive sample rate and channel count.
fn is_buffered_pcm16_wav_output(output: &WorkflowOutput) -> bool {
    output.role == WorkflowOutputRole::Audio
        && output.media.as_ref().is_some_and(|media| {
            media.delivery == MediaDelivery::Buffered
                && media.container == MediaContainer::Wav
                && media.encoding == MediaEncoding::PcmS16Le
                && media.sample_rate_hz.is_some_and(|rate| rate > 0)
                && media.channels.is_some_and(|channels| channels > 0)
        })
}

/// Whether the workflow declares *any* compatible buffered PCM16 WAV audio
/// output. This is a capability probe only; serving admission must instead
/// resolve the *exact* output with [`buffered_pcm16_wav_output_names`] so the
/// output that is validated is the same one that is later encoded.
pub fn has_buffered_pcm16_wav_output(workflow: &WorkflowSpec) -> bool {
    workflow.outputs.values().any(is_buffered_pcm16_wav_output)
}

/// Names of every workflow output that declares a compatible buffered PCM16 WAV
/// audio contract, in deterministic map-key order.
///
/// Serving must resolve *exactly one* such output and bind it to the selected
/// text-assembly contract at load time, rather than admitting on the mere
/// existence of an audio output and then re-selecting the first output by role
/// at encode time. Returning every candidate lets the caller fail closed on the
/// zero and ambiguous cases with a precise error.
pub fn buffered_pcm16_wav_output_names(workflow: &WorkflowSpec) -> Vec<String> {
    workflow
        .outputs
        .iter()
        .filter(|(_, output)| is_buffered_pcm16_wav_output(output))
        .map(|(name, _)| name.clone())
        .collect()
}

/// Transport encoding of the selected declared audio output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputFormat {
    Wav,
    PcmS16Le,
    PcmF32Le,
}
impl AudioOutputFormat {
    fn content_type(self, sample_rate: u32, channels: u16) -> String {
        if self == Self::Wav {
            return "audio/wav".into();
        }
        let mut content = format!("audio/pcm;rate={sample_rate}");
        if channels > 1 {
            content.push_str(&format!(";channels={channels}"));
        }
        content.push_str(match self {
            Self::PcmF32Le => ";encoding=float;bits=32",
            _ => ";encoding=signed-int;bits=16",
        });
        content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedAudio {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl WorkflowRuntime {
    /// Encode the named workflow audio output into a buffered WAV.
    ///
    /// The caller passes the exact output name that admission resolved and
    /// stored, so the output that was validated as compatible at load time is
    /// the one that is served. This deliberately does not re-select "the first
    /// output with role: audio", which could differ from the validated output
    /// when a workflow declares more than one audio output.
    pub fn encode_audio_output(
        &self,
        outputs: &PipelineOutputs,
        output_name: &str,
        format: AudioOutputFormat,
    ) -> anyhow::Result<EncodedAudio> {
        let declaration = self
            .plan
            .workflow
            .outputs
            .get(output_name)
            .with_context(|| format!("workflow declares no output named '{output_name}'"))?;
        if declaration.role != WorkflowOutputRole::Audio {
            bail!("workflow output '{output_name}' does not have role: audio");
        }
        let media = declaration.media.as_ref().with_context(|| {
            format!("audio output '{output_name}' has no media delivery contract")
        })?;
        if media.delivery != MediaDelivery::Buffered {
            bail!("audio delivery currently supports buffered outputs only");
        }
        if media.container != MediaContainer::Wav || media.encoding != MediaEncoding::PcmS16Le {
            bail!("audio delivery currently supports WAV with pcm_s16_le encoding only");
        }
        let target_rate = media
            .sample_rate_hz
            .context("audio output has no sample_rate_hz")?;
        let channels = media.channels.context("audio output has no channels")?;
        anyhow::ensure!(
            target_rate > 0 && channels > 0,
            "audio output '{output_name}' must declare positive sample_rate_hz and channels"
        );
        let value = self
            .structured_output_for_name(outputs, output_name)
            .with_context(|| format!("workflow did not emit audio output '{output_name}'"))?;

        if declaration.stage == OutputStage::PostAdapter {
            let bytes = value.to_raw_bytes()?;
            let source = validate_wav_header(&bytes, target_rate, channels)?;
            let encoded = match format {
                AudioOutputFormat::Wav => bytes,
                AudioOutputFormat::PcmS16Le => source.payload.to_vec(),
                AudioOutputFormat::PcmF32Le => source.float_pcm()?,
            };
            return Ok(EncodedAudio {
                bytes: encoded,
                content_type: format.content_type(target_rate, channels),
                sample_rate_hz: target_rate,
                channels,
            });
        }

        let shape = value.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != i64::from(channels) {
            bail!(
                "pre-adapter audio must have shape [1, channels, samples], got {shape:?} for {channels} channels"
            );
        }
        let samples_per_channel =
            usize::try_from(shape[2]).context("audio sample length is negative or too large")?;
        let source_rate = media.source_sample_rate_hz.unwrap_or(target_rate);
        let planar = value.to_vec_f32_lossy()?;
        let resampled = resample_planar(
            &planar,
            usize::from(channels),
            samples_per_channel,
            source_rate,
            target_rate,
        )?;
        let bytes = match format {
            AudioOutputFormat::Wav => {
                encode_pcm16_wav(&resampled, usize::from(channels), target_rate)
            }
            _ => encode_planar_pcm(&resampled, usize::from(channels), format),
        }
        .with_context(|| format!("encode workflow audio output '{output_name}' as {format:?}"))?;
        Ok(EncodedAudio {
            bytes,
            content_type: format.content_type(target_rate, channels),
            sample_rate_hz: target_rate,
            channels,
        })
    }
}

/// Band-limited windowed-sinc resampling for planar audio.
///
/// For output sample `m`, `t = m * source_rate / target_rate` and
///
/// `y[m] = sum_n x[n] * c*sinc(c*(t-n))*hann((t-n)/R) / sum_n h(t-n)`,
///
/// where `c = min(1, target_rate/source_rate)`, `R=16`, and the Hann window is
/// zero outside `[-R, R]`. The cutoff term suppresses frequencies above the
/// destination Nyquist limit during downsampling.
pub fn resample_planar(
    input: &[f32],
    channels: usize,
    source_samples: usize,
    source_rate: u32,
    target_rate: u32,
) -> anyhow::Result<Vec<f32>> {
    if channels == 0 || source_rate == 0 || target_rate == 0 {
        bail!("audio channels and sample rates must be greater than zero");
    }
    if input.len() != channels.saturating_mul(source_samples) {
        bail!(
            "planar audio length {} does not match channels {channels} * samples {source_samples}",
            input.len()
        );
    }
    if source_rate == target_rate {
        return Ok(input.to_vec());
    }
    let mut output = Vec::new();
    for channel in 0..channels {
        let source = &input[channel * source_samples..(channel + 1) * source_samples];
        let mut resampler = StreamingResampler::new(
            std::num::NonZeroU32::new(source_rate).context("source rate must be positive")?,
            std::num::NonZeroU32::new(target_rate).context("target rate must be positive")?,
        );
        output.extend(resampler.push(source)?);
        output.extend(resampler.finish()?);
    }
    Ok(output)
}

/// A mono PCM stream resampler with a persistent rational sampling clock.
///
/// `push` retains only the filter's lookbehind/lookahead window after emitting
/// ready samples. `finish` consumes the stream and emits its rounded-length
/// tail. A new audio turn must use a new instance; message boundaries must not.
/// Rates are supplied by the admitted audio contract, never a model identity.
pub struct StreamingResampler {
    source_rate: std::num::NonZeroU32,
    target_rate: std::num::NonZeroU32,
    samples: std::collections::VecDeque<f32>,
    first_sample: usize,
    received: usize,
    emitted: usize,
}

impl StreamingResampler {
    pub fn new(source_rate: std::num::NonZeroU32, target_rate: std::num::NonZeroU32) -> Self {
        Self {
            source_rate,
            target_rate,
            samples: std::collections::VecDeque::new(),
            first_sample: 0,
            received: 0,
            emitted: 0,
        }
    }

    /// Upper bound on output still owed after accepting additional source
    /// samples, including filter lookahead. Callers can admit an expansion
    /// budget before `push` or `finish` allocates an output chunk.
    pub fn remaining_output_samples(&self, additional: usize) -> anyhow::Result<usize> {
        let received = self
            .received
            .checked_add(additional)
            .context("audio stream sample count overflow; start a new audio turn")?;
        self.output_len(received)?
            .checked_sub(self.emitted)
            .context("audio resampling clock emitted beyond its admitted source length")
    }

    pub fn push(&mut self, samples: &[f32]) -> anyhow::Result<Vec<f32>> {
        let received = self
            .received
            .checked_add(samples.len())
            .context("audio stream sample count overflow; start a new audio turn")?;
        // Validate the eventual output count before changing the stream.
        self.output_len(received)?;
        self.received = received;
        if self.source_rate == self.target_rate {
            self.emitted = received;
            self.first_sample = received;
            return Ok(samples.to_vec());
        }
        self.samples.extend(samples);
        self.drain(false)
    }

    pub fn finish(mut self) -> anyhow::Result<Vec<f32>> {
        self.drain(true)
    }

    fn output_len(&self, received: usize) -> anyhow::Result<usize> {
        usize::try_from(
            (received as u128 * u128::from(self.target_rate.get())
                + u128::from(self.source_rate.get()) / 2)
                / u128::from(self.source_rate.get()),
        )
        .context("resampled audio length exceeds host address space; use a shorter audio turn")
    }

    fn drain(&mut self, finishing: bool) -> anyhow::Result<Vec<f32>> {
        let limit = self.output_len(self.received)?;
        let cutoff =
            (f64::from(self.target_rate.get()) / f64::from(self.source_rate.get())).min(1.0);
        let mut output = Vec::new();
        while self.emitted < limit {
            let clock = self.emitted as u128 * u128::from(self.source_rate.get());
            let center = clock / u128::from(self.target_rate.get());
            if !finishing && center + RESAMPLE_RADIUS as u128 >= self.received as u128 {
                break;
            }
            let center = i128::try_from(center).context("audio sampling clock overflow")?;
            let fraction = (clock % u128::from(self.target_rate.get())) as f64
                / f64::from(self.target_rate.get());
            let mut weighted = 0.0;
            let mut weight_sum = 0.0;
            for index in center - RESAMPLE_RADIUS as i128 + 1..=center + RESAMPLE_RADIUS as i128 {
                if index < 0 || index >= self.received as i128 {
                    continue;
                }
                let distance = (center - index) as f64 + fraction;
                let normalized = distance / RESAMPLE_RADIUS as f64;
                if normalized.abs() >= 1.0 {
                    continue;
                }
                let window = 0.5 * (1.0 + (std::f64::consts::PI * normalized).cos());
                let argument = cutoff * distance;
                let sinc = if argument.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * argument).sin() / (std::f64::consts::PI * argument)
                };
                let weight = cutoff * sinc * window;
                let offset = usize::try_from(index)
                    .context("audio sample index overflow")?
                    .checked_sub(self.first_sample)
                    .context("audio resampling lookbehind was released before use")?;
                let sample = self
                    .samples
                    .get(offset)
                    .context("audio resampling window does not contain the required sample")?;
                weighted += f64::from(*sample) * weight;
                weight_sum += weight;
            }
            output.push(if weight_sum.abs() > 1e-12 {
                (weighted / weight_sum) as f32
            } else {
                0.0
            });
            self.emitted += 1;
        }
        let next_center = self.emitted as u128 * u128::from(self.source_rate.get())
            / u128::from(self.target_rate.get());
        let keep_from = next_center
            .saturating_sub(RESAMPLE_RADIUS as u128 - 1)
            .min(self.received as u128) as usize;
        let release = keep_from
            .saturating_sub(self.first_sample)
            .min(self.samples.len());
        self.samples.drain(..release);
        self.first_sample += release;
        // A large network message must not leave its allocation resident for
        // the lifetime of the connection after the filter window is drained.
        self.samples.shrink_to(2 * RESAMPLE_RADIUS as usize);
        Ok(output)
    }
}

fn encode_planar_pcm(
    planar: &[f32],
    channels: usize,
    format: AudioOutputFormat,
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        channels > 0 && channels <= u16::MAX as usize,
        "PCM channel count must be between 1 and {}",
        u16::MAX
    );
    anyhow::ensure!(
        planar.len().is_multiple_of(channels),
        "planar audio length {} is not divisible by channel count {channels}",
        planar.len()
    );
    if let Some((index, value)) = planar
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!(
            "audio sample {index} is not finite ({value}); repair the workflow waveform output before encoding"
        );
    }
    let width = if format == AudioOutputFormat::PcmF32Le {
        4
    } else {
        2
    };
    let length = planar
        .len()
        .checked_mul(width)
        .context("PCM output byte length overflows")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .context("reserve interleaved PCM output bytes")?;
    let samples = planar.len() / channels;
    for sample in 0..samples {
        for channel in 0..channels {
            let value = planar[channel * samples + sample];
            if format == AudioOutputFormat::PcmF32Le {
                bytes.extend_from_slice(&value.to_le_bytes());
            } else {
                // Match llama-server's rounded symmetric PCM16 conversion.
                let value = (value.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

pub fn encode_pcm16_wav(
    planar: &[f32],
    channels: usize,
    sample_rate: u32,
) -> anyhow::Result<Vec<u8>> {
    if channels == 0 || channels > u16::MAX as usize || sample_rate == 0 {
        bail!("WAV channels and sample rate must be greater than zero");
    }
    if !planar.len().is_multiple_of(channels) {
        bail!("planar audio length is not divisible by channel count");
    }
    let samples = planar.len() / channels;
    let data_len = samples
        .checked_mul(channels)
        .and_then(|count| count.checked_mul(2))
        .context("WAV data length overflow")?;
    let riff_len = 36usize
        .checked_add(data_len)
        .context("WAV RIFF length overflow")?;
    let data_len_u32 = u32::try_from(data_len).context("WAV data exceeds RIFF size limit")?;
    let riff_len_u32 = u32::try_from(riff_len).context("WAV file exceeds RIFF size limit")?;
    let channels_u16 = channels as u16;
    let byte_rate = sample_rate
        .checked_mul(u32::from(channels_u16))
        .and_then(|rate| rate.checked_mul(2))
        .context("WAV byte rate overflow")?;
    let block_align = channels_u16
        .checked_mul(2)
        .context("WAV block align overflow")?;

    let pcm = encode_planar_pcm(planar, channels, AudioOutputFormat::PcmS16Le)?;
    let mut wav = Vec::new();
    wav.try_reserve_exact(44 + data_len)
        .context("reserve PCM16 WAV output bytes")?;
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_len_u32.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len_u32.to_le_bytes());
    wav.extend(pcm);
    Ok(wav)
}

struct ValidatedPcm16Wav<'a> {
    payload: &'a [u8],
}
impl ValidatedPcm16Wav<'_> {
    fn float_pcm(&self) -> anyhow::Result<Vec<u8>> {
        let length = self
            .payload
            .len()
            .checked_mul(2)
            .context("float PCM output byte length overflows")?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .context("reserve float PCM bytes from the validated PCM16 audio output")?;
        for sample in self.payload.chunks_exact(2) {
            let sample = f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32768.0;
            output.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(output)
    }
}
fn validate_wav_header(
    bytes: &[u8],
    sample_rate: u32,
    channels: u16,
) -> anyhow::Result<ValidatedPcm16Wav<'_>> {
    anyhow::ensure!(
        channels > 0 && sample_rate > 0,
        "declared PCM16 WAV sample rate and channels must be positive"
    );
    if bytes.len() < 44
        || &bytes[0..4] != b"RIFF"
        || &bytes[8..12] != b"WAVE"
        || &bytes[12..16] != b"fmt "
        || &bytes[36..40] != b"data"
    {
        bail!("post-adapter audio output is not a valid WAV byte stream");
    }
    let riff_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let fmt_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let audio_format = u16::from_le_bytes([bytes[20], bytes[21]]);
    let actual_channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let actual_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let byte_rate = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
    let block_align = u16::from_le_bytes([bytes[32], bytes[33]]);
    let bits_per_sample = u16::from_le_bytes([bytes[34], bytes[35]]);
    let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    let expected_block_align = channels
        .checked_mul(2)
        .context("declared WAV channel count overflows block alignment")?;
    let expected_byte_rate = sample_rate
        .checked_mul(u32::from(expected_block_align))
        .context("declared WAV byte rate overflows")?;
    if fmt_size != 16
        || audio_format != 1
        || bits_per_sample != 16
        || block_align != expected_block_align
        || byte_rate != expected_byte_rate
        || usize::try_from(riff_size).ok() != bytes.len().checked_sub(8)
        || usize::try_from(data_size).ok() != bytes.len().checked_sub(44)
        || !data_size.is_multiple_of(u32::from(expected_block_align))
    {
        bail!("post-adapter WAV must contain canonical PCM16 fmt and data chunks");
    }
    if actual_channels != channels || actual_rate != sample_rate {
        bail!(
            "WAV header declares {actual_channels} channels at {actual_rate} Hz; metadata declares {channels} channels at {sample_rate} Hz"
        );
    }
    Ok(ValidatedPcm16Wav {
        payload: &bytes[44..],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_genai_metadata::InferenceMetadata;

    fn sine(rate: u32, frequency: f64, seconds: f64) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize)
            .map(|index| {
                (2.0 * std::f64::consts::PI * frequency * index as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    fn rms(values: &[f32]) -> f64 {
        (values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt()
    }

    #[test]
    fn streaming_resampling_is_independent_of_message_boundaries() {
        for (source, target) in [
            (44_100, 32_000),
            (8_000, 48_000),
            (48_000, 8_000),
            (24_000, 24_000),
        ] {
            let input = sine(source, 751.0, 0.13);
            let expected = resample_planar(&input, 1, input.len(), source, target).unwrap();
            for chunk_size in [1, 7, 31, 257, 1024] {
                let mut stream = StreamingResampler::new(
                    std::num::NonZeroU32::new(source).unwrap(),
                    std::num::NonZeroU32::new(target).unwrap(),
                );
                let mut actual = Vec::new();
                for chunk in input.chunks(chunk_size) {
                    actual.extend(stream.push(chunk).unwrap());
                    assert!(stream.push(&[]).unwrap().is_empty());
                    assert!(stream.samples.len() <= 2 * RESAMPLE_RADIUS as usize);
                }
                actual.extend(stream.finish().unwrap());
                assert_eq!(
                    actual, expected,
                    "{source}->{target}, chunk size {chunk_size}"
                );
                assert_eq!(
                    actual.len(),
                    (input.len() * target as usize + source as usize / 2) / source as usize
                );
            }
        }
    }

    #[test]
    fn streaming_resampling_emits_before_finish_and_preserves_constant_signal() {
        let mut stream = StreamingResampler::new(
            std::num::NonZeroU32::new(48_000).unwrap(),
            std::num::NonZeroU32::new(44_100).unwrap(),
        );
        let ready = stream.push(&[0.25; 1024]).unwrap();
        assert!(ready.len() > 900);
        let tail = stream.finish().unwrap();
        assert!(!tail.is_empty());
        assert!(
            ready
                .iter()
                .chain(&tail)
                .all(|sample| (*sample - 0.25).abs() < 1e-6)
        );
        assert_eq!(ready.len() + tail.len(), 941);
    }

    #[test]
    fn streaming_resampling_empty_and_sub_filter_turns_have_exact_lengths() {
        for size in 0..32 {
            for (source, target) in [(192_000, 8_000), (8_000, 192_000)] {
                let input = vec![1.0; size];
                let output = resample_planar(&input, 1, size, source, target).unwrap();
                assert_eq!(
                    output.len(),
                    (size * target as usize + source as usize / 2) / source as usize
                );
                assert!(output.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
            }
        }
    }

    #[test]
    fn resampling_44100_to_32000_has_exact_rounded_length_and_preserves_stereo() {
        let left = sine(44_100, 1_000.0, 0.1);
        let right = sine(44_100, 2_000.0, 0.1);
        let mut planar = left.clone();
        planar.extend_from_slice(&right);
        let output = resample_planar(&planar, 2, left.len(), 44_100, 32_000).expect("resampling");
        assert_eq!(output.len(), 2 * 3_200);
        assert!((rms(&output[..3_200]) / rms(&left) - 1.0).abs() < 0.01);
        assert!((rms(&output[3_200..]) / rms(&right) - 1.0).abs() < 0.01);
        assert_ne!(&output[..3_200], &output[3_200..]);
    }

    #[test]
    fn downsampling_suppresses_energy_above_destination_nyquist() {
        let input = sine(44_100, 18_000.0, 0.1);
        let output = resample_planar(&input, 1, input.len(), 44_100, 32_000).expect("resampling");
        assert!(rms(&output) < 0.08, "aliased RMS was {}", rms(&output));
    }

    #[test]
    fn wav_header_and_interleaved_pcm_data_are_exact() {
        let wav = encode_pcm16_wav(&[0.0, 1.0, -1.0, 0.5], 2, 32_000).expect("WAV");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            32_000
        );
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 8);
        assert_eq!(
            &wav[44..],
            &[
                0, 0, 1, 128, // L0=0, R0=-1 (symmetric PCM16)
                255, 127, 0, 64, // L1=1, R1=0.5
            ]
        );
    }

    #[test]
    fn raw_pcm_preserves_float_precision_and_interleaves_channels() {
        let planar = [0.1234567, -1.25, 0.75, 1.5];
        let bytes = encode_planar_pcm(&planar, 2, AudioOutputFormat::PcmF32Le).unwrap();
        let actual: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, [0.1234567, 0.75, -1.25, 1.5]);
        let bytes = encode_planar_pcm(&planar, 2, AudioOutputFormat::PcmS16Le).unwrap();
        let actual: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(actual, [4045, 24575, -32767, 32767]);
        assert_eq!(
            AudioOutputFormat::PcmF32Le.content_type(24000, 2),
            "audio/pcm;rate=24000;channels=2;encoding=float;bits=32"
        );
        assert_eq!(
            AudioOutputFormat::PcmS16Le.content_type(44100, 1),
            "audio/pcm;rate=44100;encoding=signed-int;bits=16"
        );
    }

    #[test]
    fn raw_pcm_rejects_invalid_shapes_and_nonfinite_samples() {
        for format in [AudioOutputFormat::PcmS16Le, AudioOutputFormat::PcmF32Le] {
            assert!(encode_planar_pcm(&[], 0, format).is_err());
            assert!(encode_planar_pcm(&[0.0], 2, format).is_err());
            assert!(encode_planar_pcm(&[], 2, format).unwrap().is_empty());
            for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let error = encode_planar_pcm(&[0.0, sample], 1, format)
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains("sample 1") && error.contains("not finite"),
                    "{error}"
                );
            }
        }
        assert!(encode_pcm16_wav(&[f32::NAN], 1, 24000).is_err());
    }

    #[test]
    fn validated_post_adapter_pcm_preserves_full_signed_range_and_rejects_partial_frames() {
        let mut wav = encode_pcm16_wav(&[0.0, 0.0, 0.0, 0.0], 2, 24000).unwrap();
        let source: Vec<u8> = [i16::MIN, -1, 0, i16::MAX]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        wav[44..].copy_from_slice(&source);
        let validated = validate_wav_header(&wav, 24000, 2).unwrap();
        assert_eq!(validated.payload, source);
        let floats: Vec<f32> = validated
            .float_pcm()
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(floats, [-1.0, -1.0 / 32768.0, 0.0, 32767.0 / 32768.0]);
        wav.pop();
        let riff_size = u32::try_from(wav.len() - 8).unwrap();
        let data_size = u32::try_from(wav.len() - 44).unwrap();
        wav[4..8].copy_from_slice(&riff_size.to_le_bytes());
        wav[40..44].copy_from_slice(&data_size.to_le_bytes());
        assert!(validate_wav_header(&wav, 24000, 2).is_err());
    }

    #[test]
    fn wav_validation_rejects_non_pcm16_and_inconsistent_data_lengths() {
        let wav = encode_pcm16_wav(&[0.0, 0.0], 2, 32_000).expect("WAV");
        validate_wav_header(&wav, 32_000, 2).expect("canonical WAV validates");

        let mut float_wav = wav.clone();
        float_wav[20..22].copy_from_slice(&3u16.to_le_bytes());
        assert!(validate_wav_header(&float_wav, 32_000, 2).is_err());

        let mut truncated = wav;
        truncated.pop();
        assert!(validate_wav_header(&truncated, 32_000, 2).is_err());
    }

    #[test]
    fn speech_delivery_capability_requires_buffered_pcm16_wav_audio() {
        let metadata: InferenceMetadata = serde_yaml::from_str(
            r#"
schema_version: "1"
model:
  kind: pipeline
  artifacts: []
pipeline:
  workflow:
    manifest: {}
    inputs: {}
    outputs:
      audio:
        role: audio
        stage: post_adapter
        contract: {dtype: uint8, shape: [Any]}
        media:
          delivery: buffered
          container: wav
          encoding: pcm_s16_le
          sample_rate_hz: 32000
          channels: 2
    components: {}
    state: {}
    steps: []
"#,
        )
        .expect("metadata");
        let workflow = &metadata.pipeline.expect("pipeline").workflow;
        assert!(has_buffered_pcm16_wav_output(workflow));
        assert_eq!(buffered_pcm16_wav_output_names(workflow), ["audio"]);

        let mut incompatible = workflow.clone();
        incompatible.outputs.get_mut("audio").expect("audio").media = None;
        assert!(!has_buffered_pcm16_wav_output(&incompatible));
        assert!(buffered_pcm16_wav_output_names(&incompatible).is_empty());
    }

    fn workflow_with_two_outputs(
        first_media: &str,
        second_media: &str,
    ) -> onnx_genai_metadata::WorkflowSpec {
        // `alpha` sorts before `omega`, so any "first output by role" selection
        // would pick `alpha`. The resolver instead reports every compatible
        // output by name so the caller can bind exactly one.
        let metadata: InferenceMetadata = serde_yaml::from_str(&format!(
            r#"
schema_version: "1"
model:
  kind: pipeline
  artifacts: []
pipeline:
  workflow:
    manifest: {{}}
    inputs: {{}}
    outputs:
      alpha:
        role: audio
        stage: pre_adapter
        contract: {{dtype: float32, shape: [Any, Any, Any]}}
        media:
{first_media}
      omega:
        role: audio
        stage: pre_adapter
        contract: {{dtype: float32, shape: [Any, Any, Any]}}
        media:
{second_media}
    components: {{}}
    state: {{}}
    steps: []
"#
        ))
        .expect("metadata");
        metadata.pipeline.expect("pipeline").workflow
    }

    const COMPATIBLE_MEDIA: &str = "          delivery: buffered\n          container: wav\n          encoding: pcm_s16_le\n          sample_rate_hz: 24000\n          channels: 2";
    const INCOMPATIBLE_MEDIA: &str = "          delivery: buffered\n          container: raw\n          encoding: pcm_f32_le\n          sample_rate_hz: 24000\n          channels: 2";

    #[test]
    fn resolver_reports_only_the_compatible_output_when_one_is_incompatible() {
        // `alpha` (the map-first audio output) is incompatible; only the
        // compatible `omega` is reported, so admission binds exactly that one.
        let workflow = workflow_with_two_outputs(INCOMPATIBLE_MEDIA, COMPATIBLE_MEDIA);
        assert_eq!(buffered_pcm16_wav_output_names(&workflow), ["omega"]);
        assert!(has_buffered_pcm16_wav_output(&workflow));
    }

    #[test]
    fn resolver_reports_every_output_when_multiple_are_compatible() {
        // Two compatible audio outputs are ambiguous; the resolver reports both
        // so the caller fails closed instead of silently picking one.
        let workflow = workflow_with_two_outputs(COMPATIBLE_MEDIA, COMPATIBLE_MEDIA);
        assert_eq!(
            buffered_pcm16_wav_output_names(&workflow),
            ["alpha", "omega"]
        );
    }

    #[test]
    fn resolver_reports_nothing_when_no_output_is_compatible() {
        let workflow = workflow_with_two_outputs(INCOMPATIBLE_MEDIA, INCOMPATIBLE_MEDIA);
        assert!(buffered_pcm16_wav_output_names(&workflow).is_empty());
        assert!(!has_buffered_pcm16_wav_output(&workflow));
    }
}
