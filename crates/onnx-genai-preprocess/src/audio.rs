//! Metadata-driven waveform preprocessing and configurable log-mel analysis.

use std::f32::consts::PI;
use std::fmt;
use std::io::Cursor;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// Whisper's required waveform sample rate.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// Whisper's FFT size (25 ms at 16 kHz).
pub const WHISPER_N_FFT: usize = 400;
/// Whisper's STFT hop (10 ms at 16 kHz).
pub const WHISPER_HOP_LENGTH: usize = 160;
/// Samples in Whisper's fixed 30-second input.
pub const WHISPER_N_SAMPLES: usize = 30 * WHISPER_SAMPLE_RATE as usize;
/// Frames in Whisper's fixed encoder input.
pub const WHISPER_N_FRAMES: usize = 3_000;

/// Errors returned by audio decoding and feature extraction.
#[derive(Debug)]
pub enum AudioPreprocessError {
    InvalidConfig(String),
    InvalidSampleRate,
    UnsupportedWav(String),
    Wav(hound::Error),
}

impl fmt::Display for AudioPreprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid audio configuration: {message}"),
            Self::InvalidSampleRate => write!(f, "sample rate must be greater than zero"),
            Self::UnsupportedWav(message) => write!(f, "unsupported WAV input: {message}"),
            Self::Wav(error) => write!(f, "failed to decode WAV: {error}"),
        }
    }
}

impl std::error::Error for AudioPreprocessError {}

impl From<hound::Error> for AudioPreprocessError {
    fn from(value: hound::Error) -> Self {
        Self::Wav(value)
    }
}

/// A contiguous `[1, n_mels, n_frames]` model input in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct LogMelFeatures {
    pub data: Vec<f32>,
    pub n_mels: usize,
    pub n_frames: usize,
}

impl LogMelFeatures {
    pub fn shape(&self) -> [usize; 3] {
        [1, self.n_mels, self.n_frames]
    }
}

/// Mono PCM decoded from a WAV file.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Inspectable periodic-Hann / Slaney analysis geometry, supplied by a package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogMelConfig {
    pub n_mels: usize,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
}

/// Reusable configurable log-mel extractor.
///
/// The extractor uses a periodic Hann window, centered/reflected STFT,
/// Slaney-normalized mel filters, power spectra, and Whisper's dynamic-range
/// normalization.
pub struct LogMelExtractor {
    n_mels: usize,
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for LogMelExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogMelExtractor")
            .field("n_mels", &self.n_mels)
            .field("sample_rate", &self.sample_rate)
            .field("n_fft", &self.n_fft)
            .field("hop_length", &self.hop_length)
            .finish_non_exhaustive()
    }
}

impl LogMelExtractor {
    /// Creates an extractor using the standard 400-point / 160-hop Hann recipe.
    /// Mel bins and target sample rate remain caller supplied.
    pub fn new(n_mels: usize, sample_rate: u32) -> Result<Self, AudioPreprocessError> {
        Self::from_config(LogMelConfig {
            n_mels,
            sample_rate,
            n_fft: WHISPER_N_FFT,
            hop_length: WHISPER_HOP_LENGTH,
            win_length: WHISPER_N_FFT,
        })
    }

    /// Creates an extractor with explicit analysis geometry.
    pub fn from_config(config: LogMelConfig) -> Result<Self, AudioPreprocessError> {
        let LogMelConfig {
            n_mels,
            sample_rate,
            n_fft,
            hop_length,
            win_length,
        } = config;
        if n_mels == 0
            || sample_rate == 0
            || n_fft < 2
            || hop_length == 0
            || win_length == 0
            || win_length > n_fft
            || n_fft > isize::MAX as usize
        {
            return Err(AudioPreprocessError::InvalidConfig(format!(
                "log-mel requires positive mel bins/sample rate/hop and 1 <= win_length <= n_fft (n_fft >= 2); got bins={n_mels}, sample_rate={sample_rate}, n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}; correct the declared analysis geometry"
            )));
        }
        let mut window = Vec::new();
        window.try_reserve_exact(n_fft).map_err(|error| {
            AudioPreprocessError::InvalidConfig(format!(
                "allocate {n_fft}-point analysis window: {error}; reduce n_fft or free host memory"
            ))
        })?;
        window.resize(n_fft, 0.0);
        let offset = (n_fft - win_length) / 2;
        for index in 0..win_length {
            window[offset + index] =
                0.5 - 0.5 * (2.0 * PI * index as f32 / win_length as f32).cos();
        }
        let mel_filters = create_mel_filterbank(n_mels, sample_rate, n_fft)?;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);
        Ok(Self {
            n_mels,
            sample_rate,
            n_fft,
            hop_length,
            window,
            mel_filters,
            fft,
        })
    }

    /// Extracts a dynamically sized `[1, n_mels, n_frames]` tensor.
    ///
    /// Frame count is `floor(resampled_samples / hop_length)`, using centered
    /// reflected windows centered at each `frame * hop_length`.
    pub fn extract(
        &self,
        samples: &[f32],
        input_sample_rate: u32,
    ) -> Result<LogMelFeatures, AudioPreprocessError> {
        let resampled = resample(samples, input_sample_rate, self.sample_rate)?;
        self.extract_resampled(&resampled)
    }

    /// Pads with silence or truncates to 30 seconds at the configured rate.
    /// The standard 16 kHz / 160-hop recipe produces 3000 frames.
    pub fn extract_padded(
        &self,
        samples: &[f32],
        input_sample_rate: u32,
    ) -> Result<LogMelFeatures, AudioPreprocessError> {
        let mut resampled = resample(samples, input_sample_rate, self.sample_rate)?;
        let target = usize::try_from(self.sample_rate)?
            .checked_mul(30)
            .ok_or_else(|| {
                AudioPreprocessError::InvalidConfig(
                    "30-second sample count overflows; reduce sample_rate".into(),
                )
            })?;
        resampled.resize(target, 0.0);
        self.extract_resampled(&resampled)
    }

    fn extract_resampled(&self, samples: &[f32]) -> Result<LogMelFeatures, AudioPreprocessError> {
        let mut features = self.extract_log_resampled(samples)?;
        normalize_log_mel(&mut features.data);
        Ok(features)
    }
    fn extract_log_resampled(
        &self,
        samples: &[f32],
    ) -> Result<LogMelFeatures, AudioPreprocessError> {
        let n_frames = samples.len() / self.hop_length;
        let count = self.n_mels.checked_mul(n_frames).ok_or_else(|| {
            AudioPreprocessError::InvalidConfig(
                "log-mel feature dimensions overflow; shorten audio or reduce num_mel_bins".into(),
            )
        })?;
        let mut features = Vec::new();
        features.try_reserve_exact(count).map_err(|error|AudioPreprocessError::InvalidConfig(format!("allocate {count} log-mel feature values: {error}; shorten audio or free host memory")))?;
        features.resize(count, 0.0);
        let mut fft_buffer = vec![Complex32::default(); self.n_fft];
        let mut power = vec![0.0; self.n_fft / 2 + 1];

        for frame in 0..n_frames {
            let frame_start = frame * self.hop_length;
            for (index, value) in fft_buffer.iter_mut().enumerate() {
                let sample_index =
                    frame_start as isize + index as isize - (self.n_fft / 2) as isize;
                value.re = reflected_sample(samples, sample_index) * self.window[index];
                value.im = 0.0;
            }
            self.fft.process(&mut fft_buffer);
            for (bin, magnitude) in power.iter_mut().enumerate() {
                *magnitude = fft_buffer[bin].norm_sqr();
            }

            for mel in 0..self.n_mels {
                let filter = &self.mel_filters[mel * power.len()..(mel + 1) * power.len()];
                let energy = filter
                    .iter()
                    .zip(&power)
                    .map(|(weight, magnitude)| weight * magnitude)
                    .sum::<f32>()
                    .max(1e-10);
                features[mel * n_frames + frame] = energy.log10();
            }
        }

        if features.iter().any(|value| !value.is_finite()) {
            return Err(AudioPreprocessError::InvalidConfig("log-mel power/log values are nonfinite; reduce waveform amplitude or rescale before analysis".into()));
        }
        Ok(LogMelFeatures {
            data: features,
            n_mels: self.n_mels,
            n_frames,
        })
    }
}

fn normalize_log_mel(features: &mut [f32]) {
    if let Some(maximum) = features.iter().copied().reduce(f32::max) {
        let floor = maximum - 8.0;
        for value in features {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
    }
}

/// Decodes integer 16-bit PCM WAV bytes and mixes all channels to mono.
pub fn decode_wav_pcm16(bytes: &[u8]) -> Result<DecodedAudio, AudioPreprocessError> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Err(AudioPreprocessError::InvalidSampleRate);
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(AudioPreprocessError::UnsupportedWav(format!(
            "expected 16-bit integer PCM, got {:?} with {} bits",
            spec.sample_format, spec.bits_per_sample
        )));
    }
    if spec.channels == 0 {
        return Err(AudioPreprocessError::UnsupportedWav(
            "channel count is zero".to_owned(),
        ));
    }

    let channels = usize::from(spec.channels);
    let interleaved = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    let mut samples = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        let sum = frame.iter().map(|&sample| f32::from(sample)).sum::<f32>();
        samples.push(sum / channels as f32 / 32768.0);
    }

    Ok(DecodedAudio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// How a waveform is cut into transcribable segments.
///
/// A speech encoder consumes a bounded window, so a long recording — or a live
/// stream that never ends — must be split before it can be transcribed. Cutting
/// at a silence keeps words intact; the window bound is the hard limit.
#[derive(Clone, Copy, Debug)]
pub struct SegmentConfig {
    /// Hard cap on one segment, in samples: the model's own input window.
    pub max_samples: usize,
    /// Consecutive silent samples that end a segment early. Zero disables
    /// silence cutting, leaving fixed-size windows.
    pub silence_samples: usize,
    /// Mean-square amplitude at or below which a window counts as silence.
    /// Compared against squared amplitude to avoid a square root per window.
    pub silence_mean_square: f32,
    /// Segments shorter than this are held back rather than transcribed, so a
    /// lone click between two silences is not sent to the model.
    pub min_samples: usize,
}

impl SegmentConfig {
    /// Build a configuration from durations in seconds.
    pub fn from_seconds(
        sample_rate: u32,
        max_seconds: f32,
        silence_seconds: f32,
        silence_rms: f32,
        min_seconds: f32,
    ) -> Result<Self, AudioPreprocessError> {
        if sample_rate == 0 {
            return Err(AudioPreprocessError::InvalidSampleRate);
        }
        let samples = |seconds: f32| (seconds.max(0.0) * sample_rate as f32).round() as usize;
        let max_samples = samples(max_seconds);
        if max_samples == 0 {
            return Err(AudioPreprocessError::InvalidConfig(
                "maximum segment length rounds to zero samples".to_owned(),
            ));
        }
        Ok(Self {
            max_samples,
            silence_samples: samples(silence_seconds),
            silence_mean_square: silence_rms.max(0.0) * silence_rms.max(0.0),
            min_samples: samples(min_seconds).min(max_samples),
        })
    }
}

/// One transcribable span of audio.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioSegment {
    pub samples: Vec<f32>,
    /// Index of this segment's first sample within the whole stream, for timestamps.
    pub start_sample: usize,
}

/// Cuts a growing waveform into segments, incrementally.
///
/// Feed it whatever arrives with [`push`](Self::push) — a whole file or one
/// read from a pipe — and it returns the segments that are final. Nothing is
/// buffered beyond the current segment, so a stream of any length runs in
/// bounded memory. [`flush`](Self::flush) emits the trailing partial segment
/// once the input ends.
#[derive(Debug)]
pub struct StreamSegmenter {
    config: SegmentConfig,
    pending: Vec<f32>,
    /// Stream position of `pending[0]`.
    pending_start: usize,
    /// Consecutive silent samples at the tail of `pending`.
    trailing_silence: usize,
    /// True while sitting in the gap that ended the previous segment, so the
    /// gap is discarded as it arrives instead of becoming the next segment's
    /// leading silence (which would report a timestamp before the speech).
    in_gap: bool,
}

impl StreamSegmenter {
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            pending_start: 0,
            trailing_silence: 0,
            in_gap: false,
        }
    }

    /// Absorb `samples` and return every segment that became final.
    pub fn push(&mut self, samples: &[f32]) -> Vec<AudioSegment> {
        let mut finalized = Vec::new();
        for &sample in samples {
            let silent = sample * sample <= self.config.silence_mean_square;
            if self.in_gap {
                if silent {
                    // Still in the gap: drop it, but keep the stream position.
                    self.pending_start += 1;
                    continue;
                }
                self.in_gap = false;
            }

            self.pending.push(sample);
            if silent {
                self.trailing_silence += 1;
            } else {
                self.trailing_silence = 0;
            }

            let hit_window = self.pending.len() >= self.config.max_samples;
            let hit_silence = self.config.silence_samples > 0
                && self.trailing_silence >= self.config.silence_samples;
            if hit_window || hit_silence {
                finalized.extend(self.take(hit_window));
                // Once the pending buffer is drained, any further silence is a
                // gap rather than part of a segment.
                self.in_gap = self.config.silence_samples > 0 && self.pending.is_empty();
            }
        }
        finalized
    }

    /// Emit the trailing audio, if it is long enough to be worth transcribing.
    pub fn flush(&mut self) -> Option<AudioSegment> {
        self.take(false)
    }

    /// Split off the pending audio.
    ///
    /// `forced` marks a cut made by the window bound rather than by a silence:
    /// that audio is emitted whole, because there is no boundary to trim and no
    /// later chance to emit it. Otherwise the trailing silence is dropped — it
    /// is a boundary, not content — and the stream position advances past it so
    /// the next segment's timestamp points at speech.
    fn take(&mut self, forced: bool) -> Option<AudioSegment> {
        let voiced = self.pending.len().saturating_sub(self.trailing_silence);
        let keep = if forced { self.pending.len() } else { voiced };
        if keep == 0 || (!forced && keep < self.config.min_samples) {
            // Nothing worth transcribing. A silence boundary means the burst is
            // over, so a sub-minimum one is dropped here rather than held:
            // keeping it would let a click merge with unrelated speech that
            // arrives later, and would report a start time from before the gap.
            self.pending_start += self.pending.len();
            self.pending.clear();
            self.trailing_silence = 0;
            return None;
        }
        let start_sample = self.pending_start;
        let mut samples = std::mem::take(&mut self.pending);
        let dropped = samples.len() - keep;
        samples.truncate(keep);
        self.pending_start = start_sample + keep + dropped;
        self.trailing_silence = 0;
        Some(AudioSegment {
            samples,
            start_sample,
        })
    }
}

/// Encodes `[-1, 1]` float samples as 16-bit integer PCM WAV bytes.
///
/// The inverse of [`decode_wav_pcm16`], for handing a synthesized waveform back
/// to a caller. `channels` interleaved frames are written as-is; values outside
/// `[-1, 1]` are clamped rather than allowed to wrap.
pub fn encode_wav_pcm16(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>, AudioPreprocessError> {
    if channels == 0 {
        return Err(AudioPreprocessError::UnsupportedWav(
            "channel count is zero".to_owned(),
        ));
    }
    if sample_rate == 0 {
        return Err(AudioPreprocessError::UnsupportedWav(
            "sample rate is zero".to_owned(),
        ));
    }
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
        for &sample in samples {
            // i16::MIN..=i16::MAX is asymmetric, so scale by 32767 and clamp;
            // scaling by 32768 would wrap a full-scale -1.0..1.0 signal.
            let scaled = (sample.clamp(-1.0, 1.0) * 32767.0).round();
            writer.write_sample(scaled as i16)?;
        }
        writer.finalize()?;
    }
    Ok(buffer.into_inner())
}

fn resample(
    samples: &[f32],
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f32>, AudioPreprocessError> {
    if input_rate == 0 || output_rate == 0 {
        return Err(AudioPreprocessError::InvalidSampleRate);
    }
    if input_rate == output_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let output_len =
        (samples.len() as f64 * f64::from(output_rate) / f64::from(input_rate)).round() as usize;
    let ratio = f64::from(input_rate) / f64::from(output_rate);
    let cutoff = (f64::from(output_rate) / f64::from(input_rate)).min(1.0);
    const RADIUS: isize = 16;
    let mut output = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let source_position = output_index as f64 * ratio;
        let center = source_position.floor() as isize;
        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for source_index in center - RADIUS + 1..=center + RADIUS {
            if !(0..samples.len() as isize).contains(&source_index) {
                continue;
            }
            let distance = source_position - source_index as f64;
            let weight = cutoff * sinc(cutoff * distance) * sinc(distance / RADIUS as f64);
            weighted_sum += f64::from(samples[source_index as usize]) * weight;
            weight_sum += weight;
        }
        output.push(if weight_sum.abs() > f64::EPSILON {
            (weighted_sum / weight_sum) as f32
        } else {
            0.0
        });
    }
    Ok(output)
}

fn sinc(value: f64) -> f64 {
    if value.abs() < f64::EPSILON {
        1.0
    } else {
        let angle = std::f64::consts::PI * value;
        angle.sin() / angle
    }
}

fn reflected_sample(samples: &[f32], index: isize) -> f32 {
    match samples.len() {
        0 => 0.0,
        1 => samples[0],
        len => {
            let period = 2 * (len - 1) as isize;
            let folded = index.rem_euclid(period);
            let reflected = if folded < len as isize {
                folded
            } else {
                period - folded
            };
            samples[reflected as usize]
        }
    }
}

fn create_mel_filterbank(
    n_mels: usize,
    sample_rate: u32,
    n_fft: usize,
) -> Result<Vec<f32>, AudioPreprocessError> {
    let n_freqs = n_fft / 2 + 1;
    let filter_count = n_mels.checked_mul(n_freqs).ok_or_else(|| {
        AudioPreprocessError::InvalidConfig(
            "log-mel filter dimensions overflow; reduce num_mel_bins or n_fft".into(),
        )
    })?;
    let point_count = n_mels.checked_add(2).ok_or_else(|| {
        AudioPreprocessError::InvalidConfig(
            "log-mel point count overflows; reduce num_mel_bins".into(),
        )
    })?;
    let mut filters = Vec::new();
    filters.try_reserve_exact(filter_count).map_err(|error|AudioPreprocessError::InvalidConfig(format!("allocate {filter_count} log-mel filter values: {error}; reduce analysis geometry or free host memory")))?;
    filters.resize(filter_count, 0.0);
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(f64::from(sample_rate) / 2.0);
    let mel_points = (0..point_count)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let fft_frequencies = (0..n_freqs)
        .map(|bin| bin as f64 * f64::from(sample_rate) / n_fft as f64)
        .collect::<Vec<_>>();

    for mel in 0..n_mels {
        let lower = mel_points[mel];
        let center = mel_points[mel + 1];
        let upper = mel_points[mel + 2];
        let normalization = 2.0 / (upper - lower);
        for (bin, &frequency) in fft_frequencies.iter().enumerate() {
            let lower_slope = (frequency - lower) / (center - lower);
            let upper_slope = (upper - frequency) / (upper - center);
            filters[mel * n_freqs + bin] =
                (lower_slope.min(upper_slope).max(0.0) * normalization) as f32;
        }
    }
    Ok(filters)
}

fn hz_to_mel(frequency: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOG_STEP: f64 = 0.068_751_777_420_949_12;

    if frequency < MIN_LOG_HZ {
        frequency / F_SP
    } else {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOG_STEP: f64 = 0.068_751_777_420_949_12;

    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * (LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_wave_produces_finite_whisper_features() {
        let extractor = LogMelExtractor::new(80, WHISPER_SAMPLE_RATE).unwrap();
        let samples = (0..WHISPER_SAMPLE_RATE)
            .map(|index| (2.0 * PI * 440.0 * index as f32 / WHISPER_SAMPLE_RATE as f32).sin())
            .collect::<Vec<_>>();

        let features = extractor.extract(&samples, WHISPER_SAMPLE_RATE).unwrap();

        assert_eq!(features.shape(), [1, 80, 100]);
        assert!(features.data.iter().all(|value| value.is_finite()));
        assert!(features.data.iter().copied().fold(f32::MIN, f32::max) > 1.0);
    }

    #[test]
    fn padded_large_v3_features_have_fixed_shape() {
        let extractor = LogMelExtractor::new(128, WHISPER_SAMPLE_RATE).unwrap();
        let features = extractor.extract_padded(&[], 44_100).unwrap();

        assert_eq!(features.shape(), [1, 128, WHISPER_N_FRAMES]);
        assert!(
            features
                .data
                .iter()
                .all(|value| (*value + 1.5).abs() < 1e-6)
        );
    }

    #[test]
    fn slaney_mel_filterbank_matches_reference_spots() {
        let filters = create_mel_filterbank(80, WHISPER_SAMPLE_RATE, WHISPER_N_FFT).unwrap();
        let n_freqs = WHISPER_N_FFT / 2 + 1;

        assert!((filters[1] - 0.024_862_594).abs() < 1e-7);
        assert!((filters[n_freqs + 2] - 0.022_871_772).abs() < 1e-7);
        assert!((filters[79 * n_freqs + 198] - 0.000_897_518_07).abs() < 1e-9);
    }

    #[test]
    fn wav_pcm16_decodes_and_mixes_to_mono() {
        let expected = [-32768_i16, -16384, 0, 16384, 32767];
        let mut bytes = Vec::new();
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 8_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let cursor = Cursor::new(&mut bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for sample in expected {
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        let decoded = decode_wav_pcm16(&bytes).unwrap();

        assert_eq!(decoded.sample_rate, 8_000);
        assert_eq!(decoded.samples.len(), expected.len());
        for (actual, expected) in decoded.samples.iter().zip(expected) {
            assert!((actual - f32::from(expected) / 32768.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn resampling_changes_rate_and_preserves_finite_values() {
        let input = (0..480)
            .map(|index| (2.0 * PI * 1_000.0 * index as f32 / 48_000.0).sin())
            .collect::<Vec<_>>();
        let output = resample(&input, 48_000, WHISPER_SAMPLE_RATE).unwrap();

        assert_eq!(output.len(), 160);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }
}

#[cfg(test)]
mod segmenter_tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn config(max_seconds: f32, silence_seconds: f32) -> SegmentConfig {
        SegmentConfig::from_seconds(RATE, max_seconds, silence_seconds, 0.01, 0.0)
            .expect("a valid segment configuration")
    }

    fn tone(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| if index % 2 == 0 { 0.5 } else { -0.5 })
            .collect()
    }

    fn silence(samples: usize) -> Vec<f32> {
        vec![0.0; samples]
    }

    #[test]
    fn a_stream_is_cut_at_the_model_window() {
        // No silence cutting: segments are exactly one window long.
        let mut segmenter = StreamSegmenter::new(config(1.0, 0.0));

        let segments = segmenter.push(&tone(RATE as usize * 2 + 5));

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].samples.len(), RATE as usize);
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[1].start_sample, RATE as usize);
        // The remainder stays pending until the stream ends.
        assert_eq!(segmenter.flush().expect("trailing audio").samples.len(), 5);
    }

    #[test]
    fn silence_cuts_a_segment_early_and_is_not_transcribed() {
        let mut segmenter = StreamSegmenter::new(config(10.0, 0.1));

        let mut samples = tone(4_000);
        samples.extend(silence(3_000));
        let segments = segmenter.push(&samples);

        assert_eq!(segments.len(), 1);
        // The trailing silence is a boundary, not content.
        assert_eq!(segments[0].samples.len(), 4_000);
        assert_eq!(segments[0].start_sample, 0);
    }

    #[test]
    fn segments_report_their_position_in_the_stream() {
        let mut segmenter = StreamSegmenter::new(config(10.0, 0.1));

        let mut samples = tone(2_000);
        samples.extend(silence(2_000));
        samples.extend(tone(3_000));
        samples.extend(silence(2_000));
        let segments = segmenter.push(&samples);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments[0].samples.len(), 2_000);
        // The second segment starts after the first plus its silence.
        assert_eq!(segments[1].start_sample, 4_000);
        assert_eq!(segments[1].samples.len(), 3_000);
    }

    #[test]
    fn pushing_in_arbitrary_chunks_gives_the_same_result() {
        let mut samples = tone(2_000);
        samples.extend(silence(2_000));
        samples.extend(tone(3_000));

        let mut whole = StreamSegmenter::new(config(10.0, 0.1));
        let mut expected = whole.push(&samples);
        expected.extend(whole.flush());

        // A live stream arrives in whatever sizes the pipe hands over.
        let mut chunked = StreamSegmenter::new(config(10.0, 0.1));
        let mut actual = Vec::new();
        for chunk in samples.chunks(37) {
            actual.extend(chunked.push(chunk));
        }
        actual.extend(chunked.flush());

        assert_eq!(actual, expected);
    }

    #[test]
    fn unbroken_silence_is_discarded_rather_than_buffered() {
        let mut segmenter = StreamSegmenter::new(config(10.0, 0.1));

        for _ in 0..5 {
            assert!(segmenter.push(&silence(RATE as usize)).is_empty());
            assert!(
                segmenter.pending.len() <= segmenter.config.silence_samples,
                "an idle stream must not buffer without bound: {} samples pending",
                segmenter.pending.len()
            );
        }
        assert!(segmenter.flush().is_none());
    }

    #[test]
    fn segments_shorter_than_the_minimum_are_held_back() {
        let config = SegmentConfig::from_seconds(RATE, 10.0, 0.05, 0.01, 0.5)
            .expect("a valid segment configuration");
        let mut segmenter = StreamSegmenter::new(config);

        // A click far shorter than the 0.5s minimum, between two silences.
        let mut samples = silence(1_000);
        samples.extend(tone(100));
        samples.extend(silence(2_000));
        assert!(segmenter.push(&samples).is_empty());
    }

    #[test]
    fn a_sub_minimum_burst_is_dropped_at_its_silence_boundary() {
        let config = SegmentConfig::from_seconds(RATE, 10.0, 0.05, 0.01, 0.5)
            .expect("a valid segment configuration");
        let mut segmenter = StreamSegmenter::new(config);

        // A click, then a long silence: the click is discarded at the boundary
        // rather than buffered until the window bound flushes it with silence.
        let mut samples = tone(100);
        samples.extend(silence(RATE as usize * 3));
        assert!(segmenter.push(&samples).is_empty());
        assert!(
            segmenter.pending.len() <= segmenter.config.silence_samples,
            "a dropped burst must not keep accumulating: {} pending",
            segmenter.pending.len()
        );

        // Real speech afterwards must not be merged with the discarded click,
        // and must report its own start time.
        let speech_start = samples.len();
        let mut later = tone(RATE as usize);
        later.extend(silence(2_000));
        let segments = segmenter.push(&later);

        assert_eq!(segments.len(), 1, "{segments:?}");
        assert_eq!(segments[0].samples.len(), RATE as usize);
        assert_eq!(
            segments[0].start_sample, speech_start,
            "the timestamp must point at the speech, not before the discarded click"
        );
    }

    #[test]
    fn a_zero_length_window_is_rejected() {
        let error = SegmentConfig::from_seconds(RATE, 0.0, 0.1, 0.01, 0.0)
            .expect_err("a zero-length window must fail closed");

        assert!(error.to_string().contains("zero samples"), "{error}");
    }
}

/// Scalar payload of a preprocessing tensor produced by an audio program.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioTensorData {
    /// Floating-point payload (features, waveforms).
    Fp32(Vec<f32>),
    /// Integer payload (valid frame or sample counts).
    Int64(Vec<i64>),
}

/// One named tensor produced by an audio preprocessing program.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedAudioTensor {
    /// Workflow SSA value this tensor binds to.
    pub name: String,
    /// Tensor shape, batch-leading.
    pub shape: Vec<i64>,
    /// Tensor payload.
    pub data: AudioTensorData,
}

mod program;
pub use program::AudioProgram;

impl From<std::num::TryFromIntError> for AudioPreprocessError {
    fn from(error: std::num::TryFromIntError) -> Self {
        Self::InvalidConfig(error.to_string())
    }
}

#[cfg(test)]
mod audio_program_tests {
    use super::*;

    /// A whisper-shaped program with a one-second analysis window.
    ///
    /// The window is deliberately short so a test can pad and trim around it
    /// without synthesizing 30 seconds of audio.
    const PROGRAM: &str = "
transforms:
  - op: decode
    outputs: [samples]
  - op: resample
    inputs: [samples]
    outputs: [resampled]
    sample_rate: 16000
  - op: pad
    inputs: [resampled]
    outputs: [windowed]
    mode: fixed_window
    target_length: 16000
    pad_value: 0.0
  - op: log_mel
    inputs: [windowed]
    outputs: [mel]
    num_mel_bins: 80
    n_fft: 400
    hop_length: 160
    window: hann
    mel_scale: slaney
    sample_rate: 16000
  - op: normalize
    inputs: [mel]
    outputs: [features]
    mode: whisper_log_mel
  - op: emit_valid_frames
    hop_length: 160
    inputs: [windowed]
    outputs: [valid_frames]
outputs:
  - source: features
    name: audio.input_features
    content: audio_features
    dtype: float32
  - source: valid_frames
    name: audio.valid_frames
    content: valid_frames
    dtype: int64
";

    fn program(text: &str) -> Result<AudioProgram, AudioPreprocessError> {
        let declared: onnx_genai_metadata::AudioPreprocessingProgram =
            serde_yaml::from_str(text).expect("the test program must deserialize");
        AudioProgram::from_program(&declared)
    }

    fn clip(seconds: f32) -> Vec<u8> {
        let count = (WHISPER_SAMPLE_RATE as f32 * seconds) as usize;
        let samples = (0..count)
            .map(|index| (2.0 * PI * 440.0 * index as f32 / WHISPER_SAMPLE_RATE as f32).sin() * 0.5)
            .collect::<Vec<_>>();
        encode_wav_pcm16(&samples, WHISPER_SAMPLE_RATE, 1).expect("wav encoding must succeed")
    }

    fn features_of(tensors: &[NamedAudioTensor]) -> (&[i64], &[f32]) {
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name == "audio.input_features")
            .expect("the program declares a feature output");
        match &tensor.data {
            AudioTensorData::Fp32(data) => (&tensor.shape, data),
            other => panic!("features must be fp32, got {other:?}"),
        }
    }

    fn valid_frames_of(tensors: &[NamedAudioTensor]) -> &[i64] {
        let tensor = tensors
            .iter()
            .find(|tensor| tensor.name == "audio.valid_frames")
            .expect("the program declares a validity output");
        match &tensor.data {
            AudioTensorData::Int64(data) => data,
            other => panic!("valid frames must be int64, got {other:?}"),
        }
    }

    #[test]
    fn a_short_clip_is_padded_to_the_declared_window() {
        let tensors = program(PROGRAM).unwrap().run(&[&clip(0.25)]).unwrap();

        let (shape, data) = features_of(&tensors);
        // The window, not the clip, sets the row length: 16000 / 160 = 100 frames.
        assert_eq!(shape, [1, 80, 100]);
        assert_eq!(data.len(), 80 * 100);
        assert!(data.iter().all(|value| value.is_finite()));
        // The true duration stays recoverable: 0.25 s is 25 hops of the window.
        assert_eq!(valid_frames_of(&tensors), [25]);
    }

    #[test]
    fn a_long_clip_is_trimmed_to_the_declared_window() {
        let tensors = program(PROGRAM).unwrap().run(&[&clip(2.0)]).unwrap();

        let (shape, _) = features_of(&tensors);
        assert_eq!(shape, [1, 80, 100]);
        // Validity saturates at the window; it never claims frames that were trimmed.
        assert_eq!(valid_frames_of(&tensors), [100]);
    }

    #[test]
    fn batched_rows_match_their_standalone_runs() {
        let short = clip(0.25);
        let long = clip(0.75);
        let resolved = program(PROGRAM).unwrap();

        let batched = resolved.run(&[&short, &long]).unwrap();
        let alone_short = resolved.run(&[&short]).unwrap();
        let alone_long = resolved.run(&[&long]).unwrap();

        let (shape, data) = features_of(&batched);
        assert_eq!(shape, [2, 80, 100]);
        // Row order is request order, so encoder states stay aligned with the
        // decoder rows that consume them.
        let row = 80 * 100;
        assert_eq!(&data[..row], features_of(&alone_short).1);
        assert_eq!(&data[row..], features_of(&alone_long).1);
        assert_eq!(valid_frames_of(&batched), [25, 75]);
    }

    #[test]
    fn batching_without_a_declared_window_is_rejected() {
        let text = PROGRAM.replace(
            "  - op: pad\n    inputs: [resampled]\n    outputs: [windowed]\n    mode: \
             fixed_window\n    target_length: 16000\n    pad_value: 0.0\n",
            "",
        );
        let text = text.replace("inputs: [windowed]", "inputs: [resampled]");
        let resolved = program(&text).unwrap();

        // One clip per call is still fine without a window.
        assert!(resolved.run(&[&clip(0.25)]).is_ok());

        let error = resolved
            .run(&[&clip(0.25), &clip(0.75)])
            .expect_err("ragged rows cannot form a rectangular feature tensor");
        assert!(
            error.to_string().contains("fixed analysis window"),
            "{error}"
        );
    }

    #[test]
    fn a_zero_hop_length_is_rejected() {
        let text = PROGRAM.replace("hop_length: 160", "hop_length: 0");

        let error = program(&text).expect_err("the runtime must not silently retune the STFT");

        assert!(error.to_string().contains("hop_length"), "{error}");
    }

    #[test]
    fn an_unsupported_mel_scale_is_rejected() {
        let text = PROGRAM.replace("mel_scale: slaney", "mel_scale: htk");

        let error = program(&text).expect_err("an unimplemented mel scale must fail closed");

        assert!(error.to_string().contains("mel scale"), "{error}");
    }

    #[test]
    fn a_program_without_decode_is_rejected() {
        let text = PROGRAM.replace("  - op: decode\n    outputs: [samples]\n", "");

        let error = program(&text).expect_err("encoded bytes must be decoded first");

        assert!(error.to_string().contains("decode"), "{error}");
    }

    #[test]
    fn an_output_with_no_producing_transform_is_rejected() {
        let text = PROGRAM.replace("source: valid_frames", "source: never_produced");

        let error = program(&text).expect_err("dangling output sources must fail closed");

        assert!(error.to_string().contains("never_produced"), "{error}");
    }

    #[test]
    fn an_empty_batch_is_rejected() {
        let error = program(PROGRAM)
            .unwrap()
            .run(&[])
            .expect_err("an empty submission has no rows to align");

        assert!(error.to_string().contains("at least one clip"), "{error}");
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    #[test]
    fn declared_geometry_matches_independent_direct_dft() {
        // Odd/even FFTs, shortened/full windows, nonstandard rates/mel counts.
        for (fft, hop, window, rate, mels) in [
            (16, 3, 10, 8000, 7),
            (15, 4, 15, 22050, 5),
            (32, 7, 17, 44100, 9),
        ] {
            let extractor = LogMelExtractor::from_config(LogMelConfig {
                n_mels: mels,
                sample_rate: rate,
                n_fft: fft,
                hop_length: hop,
                win_length: window,
            })
            .unwrap();
            let samples = (0..91)
                .map(|i| (i as f32 * 0.31).sin() * 0.3)
                .collect::<Vec<_>>();
            let actual = extractor.extract(&samples, rate).unwrap();
            assert_eq!(actual.shape(), [1, mels, samples.len() / hop]);
            let bins = fft / 2 + 1;
            let filters = create_mel_filterbank(mels, rate, fft).unwrap();
            let frames = samples.len() / hop;
            let mut expected = vec![0.0f64; mels * frames];
            for frame in 0..frames {
                let mut power = vec![0.0f64; bins];
                for (bin, power) in power.iter_mut().enumerate() {
                    let (mut re, mut im) = (0.0, 0.0);
                    for j in 0..window {
                        let offset = (fft - window) / 2 + j;
                        let index =
                            frame as isize * hop as isize + offset as isize - (fft / 2) as isize;
                        let x = f64::from(reflected_sample(&samples, index))
                            * (0.5
                                - 0.5
                                    * (2.0 * std::f64::consts::PI * j as f64 / window as f64)
                                        .cos());
                        let phase =
                            2.0 * std::f64::consts::PI * bin as f64 * offset as f64 / fft as f64;
                        re += x * phase.cos();
                        im -= x * phase.sin();
                    }
                    *power = re * re + im * im;
                }
                for mel in 0..mels {
                    expected[mel * frames + frame] = filters[mel * bins..(mel + 1) * bins]
                        .iter()
                        .zip(&power)
                        .map(|(w, p)| f64::from(*w) * p)
                        .sum::<f64>()
                        .max(1e-10)
                        .log10();
                }
            }
            let maximum = expected.iter().copied().reduce(f64::max).unwrap();
            for (actual, expected) in actual.data.iter().zip(expected) {
                let expected = (expected.max(maximum - 8.0) + 4.0) / 4.0;
                assert!(
                    (f64::from(*actual) - expected).abs() < 2e-5,
                    "geometry {fft}/{hop}/{window}/{rate}/{mels}: {actual} != {expected}"
                );
            }
        }
    }
    #[test]
    fn invalid_geometry_fails_with_parameters_and_recovery() {
        for (mels, rate, fft, hop, window) in [
            (0, 8000, 16, 4, 16),
            (4, 0, 16, 4, 16),
            (4, 8000, 1, 4, 1),
            (4, 8000, 16, 0, 16),
            (4, 8000, 16, 4, 17),
            (4, 8000, 16, 4, 0),
        ] {
            let error = LogMelExtractor::from_config(LogMelConfig {
                n_mels: mels,
                sample_rate: rate,
                n_fft: fft,
                hop_length: hop,
                win_length: window,
            })
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("correct the declared analysis geometry"),
                "{error}"
            );
        }
    }
    #[test]
    fn excessive_mel_dimensions_fail_without_allocation_or_panic() {
        let error = LogMelExtractor::from_config(LogMelConfig {
            n_mels: usize::MAX,
            sample_rate: 8000,
            n_fft: 16,
            hop_length: 4,
            win_length: 16,
        })
        .unwrap_err();
        assert!(
            error.to_string().contains("filter dimensions overflow"),
            "{error}"
        );
    }
    #[test]
    fn thirty_second_padding_uses_the_declared_target_rate() {
        let extractor = LogMelExtractor::from_config(LogMelConfig {
            n_mels: 4,
            sample_rate: 8000,
            n_fft: 128,
            hop_length: 1024,
            win_length: 96,
        })
        .unwrap();
        let features = extractor.extract_padded(&[], 8000).unwrap();
        assert_eq!(features.shape(), [1, 4, 240000 / 1024]);
        assert!(features.data.iter().all(|x| (*x + 1.5).abs() < 1e-6));
    }
    #[test]
    fn audio_program_uses_authored_geometry_and_frame_usage() {
        let declared: onnx_genai_metadata::AudioPreprocessingProgram = serde_yaml::from_str(
            r#"
transforms:
  - op: decode
    outputs: [samples]
  - op: resample
    inputs: [samples]
    outputs: [resampled]
    sample_rate: 8000
  - op: log_mel
    inputs: [resampled]
    outputs: [features]
    sample_rate: 8000
    num_mel_bins: 7
    n_fft: 128
    hop_length: 64
    win_length: 96
  - op: emit_valid_frames
    hop_length: 64
    inputs: [resampled]
    outputs: [frames]
outputs:
  - {source: features, name: features, content: audio_features, dtype: float32}
  - {source: frames, name: frames, content: valid_frames, dtype: int64}
"#,
        )
        .unwrap();
        let program = AudioProgram::from_program(&declared).unwrap();
        let wav = encode_wav_pcm16(&vec![0.0; 1600], 16000, 1).unwrap();
        let tensors = program.run(&[&wav]).unwrap();
        assert_eq!(tensors[0].shape, [1, 7, 12]);
        assert_eq!(tensors[1].data, AudioTensorData::Int64(vec![12]));
    }
}
