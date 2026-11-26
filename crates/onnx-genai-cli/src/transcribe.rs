use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use anyhow::Context as _;
use onnx_genai::Engine;
use onnx_genai::engine::PipelineGenerateRequest;
use onnx_genai::ort::Tokenizer;
use onnx_genai::preprocess::audio::{AudioSegment, SegmentConfig, StreamSegmenter, decode_audio};
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest, GenerateToken};
use onnx_genai_server::multimodal;

use super::execution::{GENERATING, INTERRUPT_REQUESTED, Interrupted, install_ctrlc_handler};
use super::profile::RunProfile;
use super::{
    ProfileArgs, TranscribeArgs, TranscriptFormat, decode_backend_name, resolve_model_dir,
};

struct Transcript {
    index: usize,
    start: f32,
    end: f32,
    text: String,
}

/// Prints transcripts in the requested shape as they are recognized.
///
/// Emitting per segment rather than at the end is what makes a live stream
/// usable: output appears while the speaker is still talking.
struct TranscriptWriter {
    format: TranscriptFormat,
}

impl TranscriptWriter {
    fn emit(&self, transcript: &Transcript) -> anyhow::Result<()> {
        let Transcript {
            index,
            start,
            end,
            text,
        } = transcript;
        match self.format {
            TranscriptFormat::Text => println!("{text}"),
            TranscriptFormat::Json => println!(
                "{{\"index\":{index},\"start\":{start:.3},\"end\":{end:.3},\"text\":{}}}",
                json_string(text)
            ),
            TranscriptFormat::Srt => println!(
                "{}\n{} --> {}\n{text}\n",
                index + 1,
                srt_timestamp(*start),
                srt_timestamp(*end)
            ),
        }
        // A live stream is usually piped, so flush rather than let the
        // transcript sit in a block buffer until the process exits.
        io::stdout().flush()?;
        Ok(())
    }
}

/// Escape a string as a JSON scalar.
fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control < ' ' => escaped.push_str(&format!("\\u{:04x}", control as u32)),
            other => escaped.push(other),
        }
    }
    escaped.push('"');
    escaped
}

/// `HH:MM:SS,mmm`, the SubRip timestamp format.
fn srt_timestamp(seconds: f32) -> String {
    let total_millis = (seconds.max(0.0) * 1000.0).round() as u64;
    let millis = total_millis % 1000;
    let total_seconds = total_millis / 1000;
    format!(
        "{:02}:{:02}:{:02},{millis:03}",
        total_seconds / 3600,
        (total_seconds / 60) % 60,
        total_seconds % 60
    )
}

/// A loaded speech package plus everything needed to transcribe segments.
enum SpeechExecution {
    Frames {
        contract: std::sync::Arc<onnx_genai::metadata::transcription::TranscriptionContract>,
        program: Option<onnx_genai::preprocess::audio::AudioProgram>,
    },
    Generative {
        tokenizer: Box<Tokenizer>,
        spec: multimodal::AudioInputSpec,
    },
}
struct Transcriber {
    engine: Engine,
    audio_decode_limits: onnx_genai::preprocess::audio::AudioDecodeLimits,
    execution: SpeechExecution,
    language: Option<String>,
    prompt: Option<String>,
    prompt_limits: onnx_genai::engine::transcription::generative::PromptLimits,
    max_new_tokens: Option<std::num::NonZeroUsize>,
}
impl Transcriber {
    fn load(model_dir: &Path, args: &TranscribeArgs) -> anyhow::Result<Self> {
        let max_new_tokens=args.max_new_tokens.map(|value|std::num::NonZeroUsize::new(value).context("--max-new-tokens must be positive for transcription; use at least one committed output token")).transpose()?;
        let mut config = args.engine.to_config();
        config.audio_decode_limits = onnx_genai::preprocess::audio::AudioDecodeLimits {
            max_decoded_bytes: args.max_decoded_audio_bytes,
        };
        let audio_decode_limits = config.audio_decode_limits;
        let engine = Engine::from_dir(model_dir, config)?;
        let execution = if let Some(contract) =
            onnx_genai::metadata::transcription::transcription_contract(engine.metadata())
                .map_err(anyhow::Error::msg)?
        {
            anyhow::ensure!(
                args.language.as_ref().is_none_or(|value| value.is_empty())
                    && args.prompt.as_ref().is_none_or(|value| value.is_empty()),
                "--language/--prompt conditioning is not declared by this frame-transcription profile; omit it or select a conditioned speech package"
            );
            let program = engine
                .metadata()
                .preprocessing
                .as_ref()
                .and_then(|p| p.audio.as_ref())
                .map(onnx_genai::preprocess::audio::AudioProgram::from_program)
                .transpose()?;
            SpeechExecution::Frames {
                contract: std::sync::Arc::new(contract),
                program,
            }
        } else {
            let setup=multimodal::load(model_dir)?.with_context(||format!("{} declares no composed speech workflow; select a frame-transcription profile or a generative package with an explicit audio input",model_dir.display()))?;
            let spec=setup.multimodal.audio.with_context(||format!("{} declares neither a frame-transcription profile nor an admitted audio workflow input; select a speech-to-text package",model_dir.display()))?;
            anyhow::ensure!(
                spec.transcription().is_some(),
                "{} has no transcription_config.json; declare its decoder prompt and response boundaries",
                model_dir.display()
            );
            let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).with_context(|| {
                format!(
                    "load speech tokenizer {}; verify the package's tokenizer artifact",
                    setup.tokenizer_path.display()
                )
            })?;
            SpeechExecution::Generative {
                tokenizer: Box::new(tokenizer),
                spec,
            }
        };
        Ok(Self {
            engine,
            audio_decode_limits,
            execution,
            language: args.language.clone(),
            prompt: args.prompt.clone(),
            prompt_limits: onnx_genai::engine::transcription::generative::PromptLimits {
                max_input_tokens: args.max_input_tokens,
                template: onnx_genai::ort::TemplateLimits {
                    max_output_bytes: args.max_prompt_bytes,
                    max_operations: args.max_template_operations,
                },
                max_text_bytes: args.max_text_bytes,
            },
            max_new_tokens,
        })
    }
    fn maximum_input_samples(
        &self,
        source_rate: u32,
    ) -> anyhow::Result<Option<std::num::NonZeroUsize>> {
        match &self.execution {
            SpeechExecution::Frames {
                program: Some(program),
                ..
            } => Ok(program.maximum_input_samples(source_rate)?),
            SpeechExecution::Frames { program: None, .. } => Ok(None),
            SpeechExecution::Generative { spec, .. } => spec.maximum_input_samples(source_rate),
        }
    }

    /// Transcribe one segment of `[-1, 1]` mono samples.
    fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
        GENERATING.store(true, Ordering::SeqCst);
        let result = (|| -> anyhow::Result<String> {
            let audio_limits = self.audio_decode_limits.within_host(self.engine.resource_snapshot().host_ram.headroom).context("admit transcription segment under configured PCM policy and current host headroom")?;
            match &self.execution {
                SpeechExecution::Frames { contract, .. } => {
                    use onnx_genai::engine::transcription::{
                        PreparedTranscription, TranscriptionLimits,
                    };
                    let bytes =
                        onnx_genai::preprocess::audio::encode_wav_f32(samples, sample_rate, 1)?;
                    let mut limits = TranscriptionLimits {
                        max_text_bytes: self.prompt_limits.max_text_bytes,
                        ..TranscriptionLimits::default()
                    };
                    if let Some(maximum) = self.max_new_tokens {
                        limits.max_tokens = maximum;
                    }
                    let request = PreparedTranscription::prepare(
                        std::sync::Arc::clone(contract),
                        bytes.into(),
                        limits,
                    )?
                    .with_output_limit(limits.max_tokens)?;
                    let result = self.engine.transcribe(request)?;
                    if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
                        return Err(anyhow::Error::new(Interrupted));
                    }
                    Ok(result.text)
                }
                SpeechExecution::Generative { tokenizer, spec } => {
                    let input = multimodal::MultimodalInput::from_samples(
                        spec,
                        samples,
                        sample_rate,
                        audio_limits,
                    )?;
                    let prepared=spec.transcription().context("generative transcription contract is absent; declare transcription_config.json")?
                        .prepare(tokenizer,self.language.as_deref(),self.prompt.as_deref(),self.prompt_limits)?;
                    let mut output = prepared.output;
                    let mut options = GenerateOptions {
                        temperature: 0.0,
                        response_projection: prepared.projection,
                        ..Default::default()
                    };
                    if let Some(maximum) = self.max_new_tokens {
                        options.max_new_tokens = maximum.get();
                    }
                    let request = input.bind(PipelineGenerateRequest::new(GenerateRequest {
                        prompt: GeneratePrompt::TokenIds(prepared.token_ids),
                        options,
                    }))?;
                    let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
                        if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
                            return Err(anyhow::Error::new(Interrupted));
                        }
                        output.push(&token.text)?;
                        Ok(())
                    };
                    let result = self.engine.generate_with_pipeline_callbacks(
                        request,
                        None,
                        Some(&mut callback),
                    )?;
                    Ok(output.finish(&result.text)?.text)
                }
            }
        })();
        GENERATING.store(false, Ordering::SeqCst);
        crate::flush_deferred_tracing()?;
        result
    }
}

pub(super) fn transcribe(args: TranscribeArgs, profiling: &ProfileArgs) -> anyhow::Result<()> {
    args.cpu.apply().map_err(anyhow::Error::msg)?;
    install_ctrlc_handler();
    let model_dir = resolve_model_dir(&args.model);
    let mut profile = RunProfile::new(model_dir.display().to_string());
    let load_started = std::time::Instant::now();
    let mut transcriber = Transcriber::load(&model_dir, &args)?;
    profile.execution_provider = transcriber.engine.execution_provider_status();
    profile.decode_backend =
        Some(decode_backend_name(transcriber.engine.decode_backend()).to_string());
    profile.phase("model load", load_started.elapsed());

    let writer = TranscriptWriter {
        format: args.format,
    };

    let sources = if args.audio.is_empty() {
        vec![PathBuf::from("-")]
    } else {
        args.audio.clone()
    };
    let mut index = 0;
    let mut totals = TranscriptionTotals::default();
    for source in &sources {
        if source.as_os_str() == "-" {
            index = transcribe_stream(&mut transcriber, &args, &writer, index, &mut totals)?;
        } else {
            index = transcribe_file(&mut transcriber, source, &args, &writer, index, &mut totals)?;
        }
    }
    totals.record(&mut profile);
    profiling.emit(&mut profile)?;
    Ok(())
}

/// Running totals across every transcribed source, for the profile report.
#[derive(Debug, Default)]
struct TranscriptionTotals {
    audio_seconds: f32,
    compute: std::time::Duration,
    segments: usize,
    /// Wall time spent transcribing each segment, for the latency tail.
    segment_latencies: Vec<std::time::Duration>,
}

impl TranscriptionTotals {
    fn record(&self, profile: &mut RunProfile) {
        if self.segments == 0 {
            return;
        }
        profile.phase("transcription", self.compute);
        profile.counter("audio transcribed", self.audio_seconds as f64, "s");
        profile.counter("segments", self.segments as f64, "segments");
        if self.audio_seconds > 0.0 {
            // Below 1.0 means the model keeps up with a live stream.
            profile.counter(
                "real-time factor",
                self.compute.as_secs_f64() / self.audio_seconds as f64,
                "x",
            );
        }
        let mut latencies: Vec<f64> = self
            .segment_latencies
            .iter()
            .map(|latency| latency.as_secs_f64() * 1000.0)
            .collect();
        latencies.sort_by(|left, right| left.total_cmp(right));
        if let Some(worst) = latencies.last() {
            profile.counter("slowest segment", *worst, "ms");
        }
    }
}

/// Build the segmenter for a stream at `sample_rate`.
fn build_segmenter(
    args: &TranscribeArgs,
    sample_rate: u32,
    limit: Option<std::num::NonZeroUsize>,
) -> anyhow::Result<(StreamSegmenter, f32)> {
    for (field, value) in [
        ("--silence-seconds", args.silence_seconds),
        ("--silence-threshold", args.silence_threshold),
        ("--min-segment-seconds", args.min_segment_seconds),
    ] {
        anyhow::ensure!(
            value.is_finite() && value >= 0.0,
            "{field} {value} must be finite and nonnegative; choose a finite segmentation setting"
        );
    }
    let seconds = match (args.segment_seconds, limit) {
        (Some(seconds), _) => {
            anyhow::ensure!(
                seconds.is_finite() && seconds > 0.0,
                "--segment-seconds {seconds} must be finite and positive; choose a bounded audio segment duration"
            );
            seconds
        }
        (None, Some(maximum)) => maximum.get() as f32 / sample_rate as f32,
        (None, None) => anyhow::bail!(
            "audio preprocessing declares no bounded trimming window; pass --segment-seconds explicitly or author a fixed_window/trim on the published audio path"
        ),
    };
    let mut config=SegmentConfig::from_seconds(sample_rate,seconds,args.silence_seconds,args.silence_threshold,args.min_segment_seconds)
        .context("build audio segmenter; use positive finite segmentation settings and the recording's actual sample rate")?;
    if let Some(maximum) = limit {
        if args.segment_seconds.is_some() {
            anyhow::ensure!(
                config.max_samples <= maximum.get(),
                "--segment-seconds {seconds} uses {} samples at {sample_rate} Hz, beyond this program's {}-sample input window; lower the duration to at most {} seconds",
                config.max_samples,
                maximum,
                maximum.get() as f64 / f64::from(sample_rate)
            );
        } else {
            config.max_samples = maximum.get();
            config.min_samples = config.min_samples.min(config.max_samples);
        }
    }
    let seconds = config.max_samples as f32 / sample_rate as f32;
    Ok((StreamSegmenter::new(config), seconds))
}

/// Transcribe every segment a segmenter yields, reporting progress.
fn drain(
    transcriber: &mut Transcriber,
    segments: impl IntoIterator<Item = AudioSegment>,
    sample_rate: u32,
    writer: &TranscriptWriter,
    index: &mut usize,
    totals: &mut TranscriptionTotals,
) -> anyhow::Result<f32> {
    let mut audio_seconds = 0.0;
    for segment in segments {
        let start = segment.start_sample as f32 / sample_rate as f32;
        let duration = segment.samples.len() as f32 / sample_rate as f32;
        audio_seconds += duration;
        let segment_started = std::time::Instant::now();
        let text = transcriber.transcribe(&segment.samples, sample_rate)?;
        let latency = segment_started.elapsed();
        totals.audio_seconds += duration;
        totals.compute += latency;
        totals.segments += 1;
        totals.segment_latencies.push(latency);
        if text.is_empty() {
            continue;
        }
        writer.emit(&Transcript {
            index: *index,
            start,
            end: start + duration,
            text,
        })?;
        *index += 1;
    }
    Ok(audio_seconds)
}

fn transcribe_file(
    transcriber: &mut Transcriber,
    path: &Path,
    args: &TranscribeArgs,
    writer: &TranscriptWriter,
    mut index: usize,
    totals: &mut TranscriptionTotals,
) -> anyhow::Result<usize> {
    let bytes = std::fs::read(path).map_err(|error| {
        anyhow::anyhow!(
            "What: the audio file {} could not be read. \
             Why: {error}. \
             How: check the path and that the file is readable.",
            path.display()
        )
    })?;
    let audio = decode_audio(
        &bytes,
        onnx_genai::preprocess::audio::AudioDecodeLimits {
            max_decoded_bytes: args.max_decoded_audio_bytes,
        }
        .within_host(transcriber.engine.resource_snapshot().host_ram.headroom)?,
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "What: the audio file {} could not be decoded. \
             Why: {error}. \
             How: provide a valid supported audio file (e.g. WAV, FLAC, MP3 or AAC/M4A).",
            path.display()
        )
    })?;

    let started = std::time::Instant::now();
    let (mut segmenter, _) = build_segmenter(
        args,
        audio.sample_rate,
        transcriber.maximum_input_samples(audio.sample_rate)?,
    )?;
    let mut audio_seconds = drain(
        transcriber,
        segmenter.push(&audio.samples),
        audio.sample_rate,
        writer,
        &mut index,
        totals,
    )?;
    audio_seconds += drain(
        transcriber,
        segmenter.flush(),
        audio.sample_rate,
        writer,
        &mut index,
        totals,
    )?;
    report_realtime_factor(started.elapsed().as_secs_f32(), audio_seconds);
    Ok(index)
}

fn transcribe_stream(
    transcriber: &mut Transcriber,
    args: &TranscribeArgs,
    writer: &TranscriptWriter,
    mut index: usize,
    totals: &mut TranscriptionTotals,
) -> anyhow::Result<usize> {
    let stdin = io::stdin();
    let mut reader = PcmStreamReader::new(stdin.lock(), args.sample_rate, args.channels)?;
    let sample_rate = reader.sample_rate();
    let (mut segmenter, segment_seconds) = build_segmenter(
        args,
        sample_rate,
        transcriber.maximum_input_samples(sample_rate)?,
    )?;

    eprintln!(
        "listening on stdin: {sample_rate} Hz, {} channel{}, up to {segment_seconds:.2}s per segment (Ctrl-C to stop)",
        reader.channels(),
        if reader.channels() == 1 { "" } else { "s" }
    );

    let started = std::time::Instant::now();
    let mut audio_seconds = 0.0;
    loop {
        let samples = reader.read_chunk()?;
        if samples.is_empty() {
            break;
        }
        audio_seconds += drain(
            transcriber,
            segmenter.push(&samples),
            sample_rate,
            writer,
            &mut index,
            totals,
        )?;
    }
    audio_seconds += drain(
        transcriber,
        segmenter.flush(),
        sample_rate,
        writer,
        &mut index,
        totals,
    )?;
    report_realtime_factor(started.elapsed().as_secs_f32(), audio_seconds);
    Ok(index)
}

/// Report how far ahead of (or behind) real time the transcription ran.
///
/// A factor below 1.0 means the model keeps up with a live stream; above 1.0
/// means audio arrives faster than it can be transcribed.
fn report_realtime_factor(elapsed_seconds: f32, audio_seconds: f32) {
    if audio_seconds <= 0.0 {
        return;
    }
    eprintln!(
        "[transcribe] {audio_seconds:.2}s audio in {elapsed_seconds:.2}s (real-time factor {:.2}x)",
        elapsed_seconds / audio_seconds
    );
}

/// Reads mono `[-1, 1]` samples from a byte stream of PCM16.
///
/// Accepts either a WAV stream — whose header declares the rate and channels,
/// overriding the flags — or headerless PCM16 as `ffmpeg -f s16le` and
/// `arecord` emit. Reading proceeds in chunks so a live stream is transcribed
/// as it arrives rather than after it ends.
struct PcmStreamReader<R: BufRead> {
    reader: R,
    sample_rate: u32,
    channels: u16,
    buffer: Vec<u8>,
}

/// Outcome of sniffing the first bytes of a stream.
enum StreamHeader {
    /// A WAV header was consumed; it declares this rate and channel count.
    Wav { sample_rate: u32, channels: u16 },
    /// Not a WAV stream. These bytes were read while sniffing and are audio.
    Raw(Vec<u8>),
}

/// Bytes pulled per read: a quarter second of 16 kHz mono PCM16, small enough
/// to keep latency low and large enough to avoid a syscall per sample.
const STREAM_CHUNK_BYTES: usize = 8_192;

/// Largest `fmt ` chunk this reader will hold. A PCM format chunk is 16-40
/// bytes; the cap keeps an untrusted declared size from becoming an allocation.
const MAX_WAV_FORMAT_CHUNK_BYTES: usize = 4_096;

impl<R: BufRead> PcmStreamReader<R> {
    fn new(mut reader: R, sample_rate: u32, channels: u16) -> anyhow::Result<Self> {
        if channels == 0 {
            anyhow::bail!(
                "What: --channels 0 was rejected. \
                 Why: a stream needs at least one channel. \
                 How: pass --channels 1 for mono."
            );
        }
        let mut buffer = Vec::new();
        let (sample_rate, channels) = match read_wav_stream_header(&mut reader)? {
            StreamHeader::Wav {
                sample_rate,
                channels,
            } => (sample_rate, channels),
            StreamHeader::Raw(sniffed) => {
                // Those bytes are samples, not a header: put them back.
                buffer = sniffed;
                (sample_rate, channels)
            }
        };
        // Re-validated after the header override: a WAV `fmt ` chunk supplies
        // its own values, and a zero channel count would make the frame size
        // zero and panic every later modulo.
        if sample_rate == 0 {
            anyhow::bail!(
                "What: a sample rate of zero was rejected. \
                 Why: the stream declared it, or --sample-rate 0 was passed, and audio needs a positive rate. \
                 How: pass the rate the source produces, e.g. --sample-rate 16000."
            );
        }
        if channels == 0 {
            anyhow::bail!(
                "What: a channel count of zero was rejected. \
                 Why: the WAV header declared it, but audio needs at least one channel. \
                 How: re-encode the stream with a valid channel count, e.g. `ffmpeg ... -ac 1`."
            );
        }
        Ok(Self {
            reader,
            sample_rate,
            channels,
            buffer,
        })
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    /// Read the next chunk, mixed to mono. An empty result means end of stream.
    fn read_chunk(&mut self) -> anyhow::Result<Vec<f32>> {
        let frame_bytes = usize::from(self.channels) * 2;
        let mut chunk = vec![0_u8; STREAM_CHUNK_BYTES];
        loop {
            let read = self.reader.read(&mut chunk).map_err(|error| {
                anyhow::anyhow!(
                    "What: the audio stream could not be read. \
                     Why: {error}. \
                     How: check the process feeding standard input."
                )
            })?;
            if read == 0 {
                // End of stream. Emit whatever whole frames remain; a trailing
                // partial frame is not usable audio.
                let usable = self.buffer.len() - self.buffer.len() % frame_bytes;
                if usable == 0 {
                    self.buffer.clear();
                    return Ok(Vec::new());
                }
                let samples = self.mix_to_mono(usable);
                self.buffer.clear();
                return Ok(samples);
            }
            self.buffer.extend_from_slice(&chunk[..read]);
            let usable = self.buffer.len() - self.buffer.len() % frame_bytes;
            if usable == 0 {
                continue;
            }
            let samples = self.mix_to_mono(usable);
            self.buffer.drain(..usable);
            return Ok(samples);
        }
    }

    /// Convert the first `usable` buffered bytes to mono `[-1, 1]` samples.
    fn mix_to_mono(&self, usable: usize) -> Vec<f32> {
        let frame_bytes = usize::from(self.channels) * 2;
        self.buffer[..usable]
            .chunks_exact(frame_bytes)
            .map(|frame| {
                let sum: f32 = frame
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])))
                    .sum();
                sum / f32::from(self.channels) / 32768.0
            })
            .collect()
    }
}

/// Consume a WAV header if the stream starts with one.
///
/// A pipe cannot seek, so the header is parsed by hand and the sniffed bytes are
/// handed back when the stream turns out to be headerless PCM — otherwise the
/// first samples of a raw stream would be swallowed.
fn read_wav_stream_header<R: BufRead>(reader: &mut R) -> anyhow::Result<StreamHeader> {
    let read_error = |error: io::Error| {
        anyhow::anyhow!(
            "What: the audio stream could not be read. \
             Why: {error}. \
             How: check the process feeding standard input."
        )
    };
    let mut magic = [0_u8; 4];
    let mut filled = 0;
    while filled < magic.len() {
        let read = reader.read(&mut magic[filled..]).map_err(read_error)?;
        if read == 0 {
            break;
        }
        filled += read;
    }
    if filled < magic.len() || &magic != b"RIFF" {
        return Ok(StreamHeader::Raw(magic[..filled].to_vec()));
    }

    // "RIFF" is committed to now: this must be a WAVE stream.
    let mut rest = [0_u8; 8];
    reader.read_exact(&mut rest).map_err(|_| {
        anyhow::anyhow!(
            "What: the stream on standard input was rejected. \
             Why: it begins with `RIFF` but ended before declaring a `WAVE` form. \
             How: pipe a complete WAV stream, or headerless PCM16 with --sample-rate."
        )
    })?;
    if &rest[4..8] != b"WAVE" {
        anyhow::bail!(
            "What: the stream on standard input was rejected. \
             Why: it begins with a RIFF container that is not WAVE audio. \
             How: pipe a WAV stream (`ffmpeg ... -f wav -`) or headerless PCM16 (`ffmpeg ... -f s16le -`) with --sample-rate."
        );
    }

    let mut format = None;
    loop {
        let mut header = [0_u8; 8];
        reader.read_exact(&mut header).map_err(|_| {
            anyhow::anyhow!(
                "What: the WAV stream ended inside its header. \
                 Why: no `data` chunk arrived before the stream ended. \
                 How: pipe a complete WAV stream."
            )
        })?;
        let id = [header[0], header[1], header[2], header[3]];
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if &id == b"fmt " {
            // The size is attacker-controlled on a pipe: a 12-byte header must
            // not be able to request a multi-gigabyte zeroed allocation.
            if size > MAX_WAV_FORMAT_CHUNK_BYTES {
                anyhow::bail!(
                    "What: the WAV format chunk was rejected. \
                     Why: it declares {size} bytes, far beyond the {MAX_WAV_FORMAT_CHUNK_BYTES} a PCM format chunk needs. \
                     How: pipe a standard PCM WAV stream."
                );
            }
            let mut chunk = vec![0_u8; size];
            reader.read_exact(&mut chunk).map_err(|_| {
                anyhow::anyhow!(
                    "What: the WAV stream ended inside its format chunk. \
                     Why: it was truncated. \
                     How: pipe a complete WAV stream."
                )
            })?;
            if chunk.len() < 16 {
                anyhow::bail!(
                    "What: the WAV format chunk was rejected. \
                     Why: it is {} bytes, too short to declare a channel count and sample rate. \
                     How: pipe a standard PCM WAV stream.",
                    chunk.len()
                );
            }
            // Format tag 1 is uncompressed PCM; 0xFFFE is extensible, whose
            // sub-format this reader does not inspect. Anything else would be
            // decoded as PCM16 and produce noise.
            let format_tag = u16::from_le_bytes([chunk[0], chunk[1]]);
            if !matches!(format_tag, 1 | 0xFFFE) {
                anyhow::bail!(
                    "What: the WAV stream was rejected. \
                     Why: its format tag {format_tag} is not uncompressed PCM, so its bytes are not samples. \
                     How: convert with `ffmpeg ... -acodec pcm_s16le -f wav -`."
                );
            }
            let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
            let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            let bits = u16::from_le_bytes([chunk[14], chunk[15]]);
            if bits != 16 {
                anyhow::bail!(
                    "What: the WAV stream was rejected. \
                     Why: it declares {bits}-bit samples, but this reader handles 16-bit PCM. \
                     How: convert with `ffmpeg ... -acodec pcm_s16le -f wav -`."
                );
            }
            format = Some((sample_rate, channels));
        } else if &id == b"data" {
            let (sample_rate, channels) = format.context(
                "What: the WAV stream was rejected. \
                 Why: its `data` chunk arrived before any `fmt ` chunk, so the sample rate is unknown. \
                 How: pipe a standard WAV stream.",
            )?;
            return Ok(StreamHeader::Wav {
                sample_rate,
                channels,
            });
        } else {
            // Skip any other chunk (LIST, fact, ...), padded to even length.
            // Discarded in bounded blocks: the declared size is untrusted, so it
            // must never become an allocation.
            let padded = size as u64 + (size as u64 % 2);
            let mut remaining = padded;
            let mut sink = [0_u8; 8192];
            while remaining > 0 {
                let want = remaining.min(sink.len() as u64) as usize;
                reader.read_exact(&mut sink[..want]).map_err(|_| {
                    anyhow::anyhow!(
                        "What: the WAV stream ended inside a header chunk. \
                         Why: it was truncated. \
                         How: pipe a complete WAV stream."
                    )
                })?;
                remaining -= want as u64;
            }
        }
    }
}

#[cfg(test)]
mod native_transcription_tests {
    use super::*;
    use clap::Parser;
    fn args(model: &str, extra: &[&str]) -> TranscribeArgs {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows")
            .join(model);
        let mut argv = vec!["nxrt", "transcribe", root.to_str().unwrap()];
        argv.extend_from_slice(extra);
        let crate::Commands::Transcribe(args) = crate::Cli::try_parse_from(argv).unwrap().command
        else {
            panic!("transcribe command")
        };
        *args
    }
    #[test]
    fn native_transcription_cli_runs_ctc_and_enforces_output_and_conditioning_budgets() {
        let args = args("transcription_ctc", &[]);
        let mut transcriber = Transcriber::load(&args.model, &args).unwrap();
        let samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5];
        assert_eq!(transcriber.transcribe(&samples, 8000).unwrap(), "bb aa");
        assert_eq!(
            transcriber.transcribe(&samples.map(|v| -v), 8000).unwrap(),
            "aa bb"
        );
        transcriber.max_new_tokens = std::num::NonZeroUsize::new(1);
        assert_eq!(transcriber.transcribe(&samples, 8000).unwrap(), "b");
        for extra in [["--language", "en"], ["--max-new-tokens", "0"]] {
            let args = self::args("transcription_ctc", &extra);
            let error = Transcriber::load(&args.model, &args)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains(extra[0]), "{error}");
        }
    }
    #[test]
    fn native_transcription_cli_windows_follow_actual_program_and_source_rate() {
        for (model, count) in [("transcription_ctc", 12), ("transcription_tokens", 256)] {
            let args = args(model, &["--min-segment-seconds", "0"]);
            let transcriber = Transcriber::load(&args.model, &args).unwrap();
            for (rate, expected) in [(8000, count), (16000, count * 2)] {
                assert_eq!(
                    transcriber
                        .maximum_input_samples(rate)
                        .unwrap()
                        .unwrap()
                        .get(),
                    expected
                );
                let (mut segmenter, _) = build_segmenter(
                    &args,
                    rate,
                    transcriber.maximum_input_samples(rate).unwrap(),
                )
                .unwrap();
                let segments = segmenter.push(&vec![0.5; expected * 2]);
                assert_eq!(segments.len(), 2);
                assert!(
                    segments
                        .iter()
                        .all(|segment| segment.samples.len() == expected)
                );
            }
        }
    }
    #[test]
    fn native_transcription_cli_keeps_generative_speech_audio_dependent() {
        let args = args("transcription_tokens", &[]);
        let mut transcriber = Transcriber::load(&args.model, &args).unwrap();
        assert_eq!(transcriber.transcribe(&[0.0; 32], 8000).unwrap(), "world");
        assert_eq!(transcriber.transcribe(&[0.5; 32], 8000).unwrap(), "hello");
    }
    #[test]
    fn native_transcription_cli_requires_explicit_duration_when_no_window_is_authored() {
        let mut args = args("transcription_ctc", &[]);
        let error = build_segmenter(&args, 8000, None)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("--segment-seconds") && error.contains("no bounded"),
            "{error}"
        );
        args.segment_seconds = Some(0.25);
        let (mut segmenter, _) = build_segmenter(&args, 8000, None).unwrap();
        assert_eq!(segmenter.push(&[0.5; 2000])[0].samples.len(), 2000);
        args.silence_seconds = f32::NAN;
        assert!(
            build_segmenter(&args, 8000, None)
                .err()
                .unwrap()
                .to_string()
                .contains("--silence-seconds")
        );
        args.silence_seconds = 0.5;
        args.min_segment_seconds = f32::INFINITY;
        assert!(
            build_segmenter(&args, 8000, None)
                .err()
                .unwrap()
                .to_string()
                .contains("--min-segment-seconds")
        );
        args.min_segment_seconds = 0.1;
        args.silence_threshold = -0.1;
        assert!(
            build_segmenter(&args, 8000, None)
                .err()
                .unwrap()
                .to_string()
                .contains("--silence-threshold")
        );
    }
    #[test]
    fn native_transcription_cli_rejects_oversized_or_nonfinite_segment_requests() {
        let mut args = args("transcription_ctc", &[]);
        let transcriber = Transcriber::load(&args.model, &args).unwrap();
        for seconds in [0.002, 0.0, -1.0, f32::INFINITY, f32::NAN] {
            args.segment_seconds = Some(seconds);
            let error = build_segmenter(
                &args,
                8000,
                transcriber.maximum_input_samples(8000).unwrap(),
            )
            .err()
            .unwrap()
            .to_string();
            assert!(
                error.contains("--segment-seconds")
                    && (error.contains("window") || error.contains("finite")),
                "{error}"
            );
        }
        args.segment_seconds = Some(0.001);
        assert!(
            build_segmenter(
                &args,
                8000,
                transcriber.maximum_input_samples(8000).unwrap()
            )
            .is_ok()
        );
    }
}
