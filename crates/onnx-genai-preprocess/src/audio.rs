//! Whisper-compatible audio preprocessing.

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

/// Reusable Whisper log-mel extractor.
///
/// The extractor uses a periodic Hann window, centered/reflected STFT,
/// Slaney-normalized mel filters, power spectra, and Whisper's dynamic-range
/// normalization.
pub struct LogMelExtractor {
    n_mels: usize,
    sample_rate: u32,
    window: Vec<f32>,
    mel_filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for LogMelExtractor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogMelExtractor")
            .field("n_mels", &self.n_mels)
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

impl LogMelExtractor {
    /// Creates an extractor. Whisper models use 80 or 128 mel bins and 16 kHz.
    pub fn new(n_mels: usize, sample_rate: u32) -> Result<Self, AudioPreprocessError> {
        if !matches!(n_mels, 80 | 128) {
            return Err(AudioPreprocessError::InvalidConfig(format!(
                "Whisper supports 80 or 128 mel bins, got {n_mels}"
            )));
        }
        if sample_rate != WHISPER_SAMPLE_RATE {
            return Err(AudioPreprocessError::InvalidConfig(format!(
                "Whisper requires a {WHISPER_SAMPLE_RATE} Hz target sample rate"
            )));
        }

        let window = (0..WHISPER_N_FFT)
            .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / WHISPER_N_FFT as f32).cos())
            .collect();
        let mel_filters = create_mel_filterbank(n_mels, sample_rate);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(WHISPER_N_FFT);

        Ok(Self {
            n_mels,
            sample_rate,
            window,
            mel_filters,
            fft,
        })
    }

    /// Extracts a dynamically sized `[1, n_mels, n_frames]` tensor.
    ///
    /// Frame count is `floor(resampled_samples / 160)`, matching Whisper's
    /// centered STFT after its final frame is discarded.
    pub fn extract(
        &self,
        samples: &[f32],
        input_sample_rate: u32,
    ) -> Result<LogMelFeatures, AudioPreprocessError> {
        let resampled = resample(samples, input_sample_rate, self.sample_rate)?;
        Ok(self.extract_resampled(&resampled))
    }

    /// Pads with silence or truncates to 30 seconds before producing the fixed
    /// `[1, n_mels, 3000]` Whisper encoder input.
    pub fn extract_padded(
        &self,
        samples: &[f32],
        input_sample_rate: u32,
    ) -> Result<LogMelFeatures, AudioPreprocessError> {
        let mut resampled = resample(samples, input_sample_rate, self.sample_rate)?;
        resampled.resize(WHISPER_N_SAMPLES, 0.0);
        resampled.truncate(WHISPER_N_SAMPLES);
        Ok(self.extract_resampled(&resampled))
    }

    fn extract_resampled(&self, samples: &[f32]) -> LogMelFeatures {
        let n_frames = samples.len() / WHISPER_HOP_LENGTH;
        let mut features = vec![0.0; self.n_mels * n_frames];
        let mut fft_buffer = vec![Complex32::default(); WHISPER_N_FFT];
        let mut power = vec![0.0; WHISPER_N_FFT / 2 + 1];

        for frame in 0..n_frames {
            let frame_start = frame * WHISPER_HOP_LENGTH;
            for (index, value) in fft_buffer.iter_mut().enumerate() {
                let sample_index =
                    frame_start as isize + index as isize - (WHISPER_N_FFT / 2) as isize;
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

        if let Some(maximum) = features.iter().copied().reduce(f32::max) {
            let floor = maximum - 8.0;
            for value in &mut features {
                *value = (value.max(floor) + 4.0) / 4.0;
            }
        }

        LogMelFeatures {
            data: features,
            n_mels: self.n_mels,
            n_frames,
        }
    }
}

/// Decodes integer 16-bit PCM WAV bytes and mixes all channels to mono.
pub fn decode_wav_pcm16(bytes: &[u8]) -> Result<DecodedAudio, AudioPreprocessError> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
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

fn create_mel_filterbank(n_mels: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = WHISPER_N_FFT / 2 + 1;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(f64::from(sample_rate) / 2.0);
    let mel_points = (0..n_mels + 2)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let fft_frequencies = (0..n_freqs)
        .map(|bin| bin as f64 * f64::from(sample_rate) / WHISPER_N_FFT as f64)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0; n_mels * n_freqs];

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
    filters
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
        let filters = create_mel_filterbank(80, WHISPER_SAMPLE_RATE);
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
