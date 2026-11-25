//! Typed, processor-local audio dataflow. Output roles never choose their data.
use super::*;
use onnx_genai_metadata::{AudioPreprocessingProgram, AudioTransform};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Range,
};

fn invalid(message: impl Into<String>) -> AudioPreprocessError {
    AudioPreprocessError::InvalidConfig(message.into())
}
#[derive(Debug, Clone, Copy)]
struct ValueId(usize);
#[derive(Debug, Clone, Copy)]
enum Kind {
    Wave(Option<u32>),
    Mel,
    Length(LengthUnit),
    Mask,
}
#[derive(Debug, Clone, Copy)]
enum LengthUnit {
    Samples,
    Frames,
}
#[derive(Debug)]
enum Op {
    Decode,
    Resample(u32),
    Downmix,
    Rescale(f32),
    UnitVariance(f64),
    Pad {
        target: Option<usize>,
        value: f32,
        left: bool,
        trim: bool,
    },
    Trim(usize),
    Mel(Box<LogMelExtractor>),
    NormalizeMel,
    Samples,
    Frames(Option<NonZeroUsize>),
    Mask,
}
#[derive(Debug)]
struct Step {
    input: Option<ValueId>,
    op: Op,
    context: String,
}
#[derive(Debug)]
struct Output {
    name: String,
    source: ValueId,
}
/// Compiled input/source/type proof for a package-authored audio program.
#[derive(Debug)]
pub struct AudioProgram {
    steps: Vec<Step>,
    outputs: Vec<Output>,
}
#[derive(Clone, Debug)]
struct Wave {
    samples: Vec<f32>,
    rate: u32,
    valid: Range<usize>,
}
#[derive(Clone, Debug)]
struct Mel {
    features: LogMelFeatures,
    valid: Range<usize>,
}
#[derive(Clone, Debug)]
enum Data {
    Wave(Vec<Wave>),
    Mel(Vec<Mel>),
    Length(Vec<usize>),
    Mask(Vec<Vec<i64>>),
}
impl Data {
    fn waves(&self) -> Result<&[Wave], AudioPreprocessError> {
        match self {
            Self::Wave(rows) => Ok(rows),
            _ => Err(invalid(
                "compiled audio input is not waveform; report this runtime invariant failure",
            )),
        }
    }
    fn mels(&self) -> Result<&[Mel], AudioPreprocessError> {
        match self {
            Self::Mel(rows) => Ok(rows),
            _ => Err(invalid(
                "compiled audio input is not log-mel; report this runtime invariant failure",
            )),
        }
    }
}
impl AudioProgram {
    pub fn from_program(program: &AudioPreprocessingProgram) -> Result<Self, AudioPreprocessError> {
        if !program.transforms.iter().any(|t| t.op == "decode") {
            return Err(invalid(
                "audio preprocessing must declare a decode transform before consuming encoded clips",
            ));
        }
        let mut steps = Vec::new();
        let mut kinds = Vec::<Kind>::new();
        let mut names = BTreeMap::<String, ValueId>::new();
        for (index, t) in program.transforms.iter().enumerate() {
            let context = format!("audio transform[{index}] '{}'", t.op);
            let compile = || -> Result<(Option<ValueId>, Op, Kind), AudioPreprocessError> {
                if t.op == "decode" {
                    if t.inputs.is_some() {
                        return Err(invalid(
                            "decode consumes the adapter's encoded bytes; omit transform.inputs",
                        ));
                    }
                    return Ok((None, Op::Decode, Kind::Wave(None)));
                }
                let input=match t.inputs.as_deref() {
                    Some([name])=>*names.get(name).ok_or_else(||invalid(format!("input '{name}' has no preceding producer; correct inputs or move its producer before this transform")))?,
                    None=>ValueId(index.checked_sub(1).ok_or_else(||invalid("no preceding audio value; declare decode first"))?),
                    _=>return Err(invalid("this unary transform requires exactly one input; use a declared preceding value")),
                };
                let kind = kinds[input.0];
                let wave_rate = || match kind {
                    Kind::Wave(rate) => Ok(rate),
                    _ => Err(invalid(
                        "requires waveform input; bind a decode/resample/waveform transform instead",
                    )),
                };
                let positive = |value: Option<usize>, field: &str| {
                    value.filter(|v| *v > 0).ok_or_else(|| {
                        invalid(format!(
                            "requires positive {field}; author it in this transform"
                        ))
                    })
                };
                let (op, kind) = match t.op.as_str() {
                    "resample" => {
                        wave_rate()?;
                        let rate = t
                            .sample_rate
                            .filter(|v| *v > 0)
                            .ok_or_else(|| invalid("resample requires positive sample_rate"))?;
                        (Op::Resample(rate), Kind::Wave(Some(rate)))
                    }
                    "downmix" => {
                        let rate = wave_rate()?;
                        if t.channels != Some(1) {
                            return Err(invalid(
                                "decoded audio is mono; downmix requires channels: 1; multichannel outputs are not implemented",
                            ));
                        }
                        (Op::Downmix, Kind::Wave(rate))
                    }
                    "rescale" => {
                        let rate = wave_rate()?;
                        let scale = finite(t.scale, "scale")?;
                        (Op::Rescale(scale), Kind::Wave(rate))
                    }
                    "zero_mean_unit_variance" => {
                        let rate = wave_rate()?;
                        (Op::UnitVariance(epsilon(t)?), Kind::Wave(rate))
                    }
                    "normalize" => match t.mode.as_deref().unwrap_or("whisper_log_mel") {
                        "zero_mean_unit_variance" => {
                            let rate = wave_rate()?;
                            (Op::UnitVariance(epsilon(t)?), Kind::Wave(rate))
                        }
                        "whisper_log_mel" => {
                            if !matches!(kind, Kind::Mel) {
                                return Err(invalid(
                                    "log-mel normalization requires a preceding spectrogram; bind its named output",
                                ));
                            }
                            (Op::NormalizeMel, Kind::Mel)
                        }
                        mode => {
                            return Err(invalid(format!(
                                "normalization mode '{mode}' is unsupported; use zero_mean_unit_variance for waveform or whisper_log_mel for log-mel"
                            )));
                        }
                    },
                    "pad" => {
                        let rate = wave_rate()?;
                        let mode = t.mode.as_deref().unwrap_or("right");
                        if !matches!(mode, "right" | "left" | "fixed_window") {
                            return Err(invalid(format!(
                                "padding mode '{mode}' is unsupported; use right, left or fixed_window"
                            )));
                        }
                        if t.target_length == Some(0)
                            || (mode == "fixed_window" && t.target_length.is_none())
                        {
                            return Err(invalid(
                                "padding needs positive target_length; fixed_window requires it, right/left may omit it to pad to the batch maximum",
                            ));
                        }
                        let value = finite(Some(t.pad_value.unwrap_or(0.0)), "pad_value")?;
                        (
                            Op::Pad {
                                target: t.target_length,
                                value,
                                left: mode == "left",
                                trim: mode == "fixed_window",
                            },
                            Kind::Wave(rate),
                        )
                    }
                    "trim" => {
                        let rate = wave_rate()?;
                        if !matches!(t.mode.as_deref(), None | Some("right")) {
                            return Err(invalid(
                                "trim currently supports mode: right; select the right boundary explicitly",
                            ));
                        }
                        (
                            Op::Trim(positive(t.target_length, "target_length")?),
                            Kind::Wave(rate),
                        )
                    }
                    "log_mel" => {
                        let input_rate = wave_rate()?;
                        if t.window.as_deref().is_some_and(|v| v != "hann") {
                            return Err(invalid("unsupported analysis window; use window: hann"));
                        }
                        if t.mel_scale.as_deref().is_some_and(|v| v != "slaney") {
                            return Err(invalid("unsupported mel scale; use mel_scale: slaney"));
                        }
                        let rate=t.sample_rate.or(input_rate).ok_or_else(||invalid("log_mel needs sample_rate or a preceding declared resample; do not infer an analysis rate from model identity"))?;
                        if input_rate.is_some_and(|input| input != rate) {
                            return Err(invalid(format!(
                                "log_mel sample_rate {rate} differs from its input rate {input_rate:?}; insert an explicit resample"
                            )));
                        }
                        let fft = t.n_fft.unwrap_or(WHISPER_N_FFT);
                        let config = LogMelConfig {
                            n_mels: positive(t.num_mel_bins, "num_mel_bins")?,
                            sample_rate: rate,
                            n_fft: fft,
                            hop_length: t.hop_length.unwrap_or(WHISPER_HOP_LENGTH),
                            win_length: t.win_length.unwrap_or(fft),
                        };
                        (
                            Op::Mel(Box::new(LogMelExtractor::from_config(config)?)),
                            Kind::Mel,
                        )
                    }
                    "emit_valid_samples" | "emit_sample_lengths" => {
                        wave_rate()?;
                        (Op::Samples, Kind::Length(LengthUnit::Samples))
                    }
                    "emit_valid_frames" => {
                        let hop = match kind {
                            Kind::Wave(_) => Some(NonZeroUsize::new(t.hop_length.unwrap_or(0)).ok_or_else(||invalid("waveform frame counts require positive hop_length; declare the frame geometry explicitly"))?),
                            Kind::Mel => None,
                            _ => {
                                return Err(invalid(
                                    "frame counts require waveform or log-mel input",
                                ));
                            }
                        };
                        (Op::Frames(hop), Kind::Length(LengthUnit::Frames))
                    }
                    "emit_validity_mask" => {
                        if !matches!(kind, Kind::Wave(_) | Kind::Mel) {
                            return Err(invalid(
                                "validity masks require waveform or log-mel input",
                            ));
                        }
                        (Op::Mask, Kind::Mask)
                    }
                    other => {
                        return Err(invalid(format!(
                            "unsupported audio transform '{other}'; select an executable audio operation instead of silently treating it as a no-op"
                        )));
                    }
                };
                Ok((Some(input), op, kind))
            };
            let (input, op, kind) = compile().map_err(|e| invalid(format!("{context}: {e}")))?;
            if let Some(outputs) = &t.outputs {
                if outputs.is_empty() {
                    return Err(invalid(format!(
                        "{context}.outputs is empty; omit it or name the produced value"
                    )));
                }
                for name in outputs {
                    if name.trim().is_empty()
                        || names.insert(name.clone(), ValueId(index)).is_some()
                    {
                        return Err(invalid(format!(
                            "{context}.outputs contains empty/duplicate value '{name}'; each processor-local name needs a unique producer"
                        )));
                    }
                }
            }
            kinds.push(kind);
            steps.push(Step { input, op, context });
        }
        let mut output_names = BTreeSet::new();
        let outputs = program.outputs.iter().map(|binding| {
            if binding.name.trim().is_empty() { return Err(invalid("audio output.name is empty; declare a nonempty component output port name")); }
            let source = *names.get(&binding.source).ok_or_else(|| invalid(format!(
                "audio output '{}' has no transform producing '{}'; bind a declared processor-local source",
                binding.name, binding.source
            )))?;
            if !output_names.insert(&binding.name) {
                return Err(invalid(format!("audio output '{}' is duplicated; bind each output port once", binding.name)));
            }
            let (dtype, content) = match kinds[source.0] {
                Kind::Wave(_) => ("float32", binding.content == "waveform"),
                Kind::Mel => ("float32", matches!(binding.content.as_str(), "features" | "audio_features")),
                Kind::Length(unit) => {
                    let correct_unit = match unit {
                        LengthUnit::Samples => matches!(binding.content.as_str(), "valid_samples" | "sample_lengths"),
                        LengthUnit::Frames => matches!(binding.content.as_str(), "valid_frames" | "frame_lengths"),
                    };
                    ("int64", correct_unit || binding.content == "valid_lengths")
                }
                Kind::Mask => ("int64", binding.content == "validity_mask"),
            };
            if binding.dtype != dtype || !content {
                return Err(invalid(format!(
                    "audio output '{}' source '{}' produces {:?} ({dtype}), incompatible with content '{}' / dtype '{}'; bind the intended transform and correct its declared role/type",
                    binding.name, binding.source, kinds[source.0], binding.content, binding.dtype
                )));
            }
            Ok(Output { name: binding.name.clone(), source })
        }).collect::<Result<Vec<_>, AudioPreprocessError>>()?;
        if outputs.is_empty() {
            return Err(invalid(
                "audio program needs at least one named output; declare outputs",
            ));
        }
        Ok(Self { steps, outputs })
    }
    pub fn run(&self, encoded: &[&[u8]]) -> Result<Vec<NamedAudioTensor>, AudioPreprocessError> {
        if encoded.is_empty() {
            return Err(invalid("audio preprocessing requires at least one clip"));
        }
        let mut values = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let input = step.input.map(|id| &values[id.0]);
            let result = execute(&step.op, input, encoded).map_err(|e| {
                invalid(format!(
                    "{}: {e}; check the encoded clip and declared input/parameters",
                    step.context
                ))
            })?;
            values.push(result);
        }
        self.outputs
            .iter()
            .map(|output| tensor(&output.name, &values[output.source.0]))
            .collect()
    }
}
fn finite(value: Option<f64>, field: &str) -> Result<f32, AudioPreprocessError> {
    let value = value
        .ok_or_else(|| invalid(format!("requires {field}; declare a finite float32 value")))?;
    if !value.is_finite() || !(value as f32).is_finite() {
        return Err(invalid(format!(
            "{field} {value} is not finite float32; correct the transform parameter"
        )));
    }
    Ok(value as f32)
}
fn epsilon(t: &AudioTransform) -> Result<f64, AudioPreprocessError> {
    let v = t.epsilon.unwrap_or(1e-7);
    if !v.is_finite() || v <= 0.0 {
        return Err(invalid(format!("epsilon {v} must be positive and finite")));
    }
    Ok(v)
}
fn execute(op: &Op, input: Option<&Data>, encoded: &[&[u8]]) -> Result<Data, AudioPreprocessError> {
    if matches!(op, Op::Decode) {
        return encoded
            .iter()
            .map(|bytes| {
                let audio = decode_audio(bytes)?;
                let len = audio.samples.len();
                Ok(Wave {
                    samples: audio.samples,
                    rate: audio.sample_rate,
                    valid: 0..len,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Data::Wave);
    }
    let input = input.ok_or_else(|| {
        invalid("compiled unary transform has no input; report the invalid runtime plan")
    })?;
    match op {
        Op::Resample(rate) => {
            let rows = input
                .waves()?
                .iter()
                .map(|row| -> Result<Wave, AudioPreprocessError> {
                    let samples = resample(&row.samples, row.rate, *rate)?;
                    // The product fits u128 for every usize/u32 input. Clamp only
                    // after division so large positions never silently saturate.
                    let start = (row.valid.start as u128 * u128::from(*rate) / u128::from(row.rate))
                        .min(samples.len() as u128) as usize;
                    let end = if row.valid.is_empty() {
                        start
                    } else {
                        (row.valid.end as u128 * u128::from(*rate))
                            .div_ceil(u128::from(row.rate))
                            .min(samples.len() as u128) as usize
                    };
                    Ok(Wave {
                        samples,
                        rate: *rate,
                        valid: start..end,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Data::Wave(rows))
        }
        Op::Downmix => Ok(input.clone()),
        Op::Rescale(scale) => {
            let mut rows = input.waves()?.to_vec();
            for row in &mut rows {
                for value in &mut row.samples {
                    *value *= scale;
                    if !value.is_finite() {
                        return Err(invalid(
                            "rescale produced nonfinite samples; reduce scale or waveform amplitude",
                        ));
                    }
                }
            }
            Ok(Data::Wave(rows))
        }
        Op::UnitVariance(epsilon) => {
            let mut rows = input.waves()?.to_vec();
            for row in &mut rows {
                let values = &row.samples[row.valid.clone()];
                if values.is_empty() {
                    continue;
                }
                let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64;
                let variance = values
                    .iter()
                    .map(|v| (f64::from(*v) - mean).powi(2))
                    .sum::<f64>()
                    / values.len() as f64;
                for value in &mut row.samples[row.valid.clone()] {
                    *value = ((f64::from(*value) - mean) / (variance + epsilon).sqrt()) as f32;
                }
            }
            Ok(Data::Wave(rows))
        }
        Op::Pad {
            target,
            value,
            left,
            trim,
        } => {
            let source = input.waves()?;
            let target =
                target.unwrap_or_else(|| source.iter().map(|r| r.samples.len()).max().unwrap_or(0));
            let mut rows = source.to_vec();
            for row in &mut rows {
                if *trim {
                    row.samples.truncate(target);
                    row.valid.start = row.valid.start.min(target);
                    row.valid.end = row.valid.end.min(target);
                }
                let old_length = row.samples.len();
                if old_length >= target {
                    continue;
                }
                let pad = target - old_length;
                row.samples.try_reserve_exact(pad).map_err(|e| invalid(format!("allocate {target}-sample padding: {e}; shorten the target or free host memory")))?;
                row.samples.resize(target, *value);
                if *left {
                    row.samples.copy_within(0..old_length, pad);
                    row.samples[..pad].fill(*value);
                    row.valid.start += pad;
                    row.valid.end += pad;
                }
            }
            Ok(Data::Wave(rows))
        }
        Op::Trim(target) => {
            let mut rows = input.waves()?.to_vec();
            for row in &mut rows {
                row.samples.truncate(*target);
                row.valid.start = row.valid.start.min(*target);
                row.valid.end = row.valid.end.min(*target);
            }
            Ok(Data::Wave(rows))
        }
        Op::Mel(extractor) => {
            let rows = input.waves()?.iter().map(|row| {
                if row.rate != extractor.sample_rate {
                    return Err(invalid(format!("log_mel analysis rate {} does not match input audio rate {}; declare resample before this transform", extractor.sample_rate, row.rate)));
                }
                let features = extractor.extract_log_resampled(&row.samples)?;
                let valid = frame_range(&row.valid, features.n_frames, extractor.hop_length);
                Ok(Mel { features, valid })
            }).collect::<Result<Vec<_>, _>>()?;
            Ok(Data::Mel(rows))
        }
        Op::NormalizeMel => {
            let mut rows = input.mels()?.to_vec();
            for row in &mut rows {
                normalize_log_mel(&mut row.features.data);
            }
            Ok(Data::Mel(rows))
        }
        Op::Samples => Ok(Data::Length(
            input.waves()?.iter().map(|row| row.valid.len()).collect(),
        )),
        Op::Frames(hop) => match input {
            Data::Mel(rows) => Ok(Data::Length(
                rows.iter().map(|row| row.valid.len()).collect(),
            )),
            Data::Wave(rows) => {
                let hop = hop
                    .ok_or_else(|| invalid("waveform frame-count plan lacks hop_length"))?
                    .get();
                Ok(Data::Length(
                    rows.iter()
                        .map(|row| frame_range(&row.valid, row.samples.len() / hop, hop).len())
                        .collect(),
                ))
            }
            _ => Err(invalid("invalid frame-count input type")),
        },
        Op::Mask => match input {
            Data::Wave(rows) => Ok(Data::Mask(
                rows.iter()
                    .map(|row| {
                        (0..row.samples.len())
                            .map(|i| i64::from(row.valid.contains(&i)))
                            .collect()
                    })
                    .collect(),
            )),
            Data::Mel(rows) => Ok(Data::Mask(
                rows.iter()
                    .map(|row| {
                        (0..row.features.n_frames)
                            .map(|i| i64::from(row.valid.contains(&i)))
                            .collect()
                    })
                    .collect(),
            )),
            _ => Err(invalid("invalid validity-mask input type")),
        },
        Op::Decode => unreachable!("decode handled before unary input admission"),
    }
}
fn frame_range(valid: &Range<usize>, frames: usize, hop: usize) -> Range<usize> {
    let start = (valid.start / hop).min(frames);
    let end = if valid.is_empty() {
        start
    } else {
        valid.end.div_ceil(hop).min(frames)
    };
    start..end
}
fn tensor(name: &str, data: &Data) -> Result<NamedAudioTensor, AudioPreprocessError> {
    let rectangular = |sizes: Vec<usize>| -> Result<usize, AudioPreprocessError> {
        let n = sizes[0];
        if sizes.iter().any(|v| *v != n) {
            return Err(invalid(format!(
                "audio output '{name}' has ragged rows; declare pad with a fixed analysis window or batch-maximum target before this output"
            )));
        }
        Ok(n)
    };
    let (shape, data) = match data {
        Data::Wave(rows) => (
            vec![
                rows.len(),
                rectangular(rows.iter().map(|r| r.samples.len()).collect())?,
            ],
            AudioTensorData::Fp32(
                rows.iter()
                    .flat_map(|r| r.samples.iter().copied())
                    .collect(),
            ),
        ),
        Data::Mel(rows) => {
            let n = rectangular(rows.iter().map(|r| r.features.n_frames).collect())?;
            (
                vec![rows.len(), rows[0].features.n_mels, n],
                AudioTensorData::Fp32(
                    rows.iter()
                        .flat_map(|r| r.features.data.iter().copied())
                        .collect(),
                ),
            )
        }
        Data::Length(rows) => (
            vec![rows.len()],
            AudioTensorData::Int64(
                rows.iter()
                    .map(|v| i64::try_from(*v))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        ),
        Data::Mask(rows) => (
            vec![
                rows.len(),
                rectangular(rows.iter().map(Vec::len).collect())?,
            ],
            AudioTensorData::Int64(rows.iter().flatten().copied().collect()),
        ),
    };
    Ok(NamedAudioTensor {
        name: name.into(),
        shape: shape
            .into_iter()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()?,
        data,
    })
}
