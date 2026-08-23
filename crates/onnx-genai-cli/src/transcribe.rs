use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use anyhow::Context as _;
use onnx_genai::Engine;
use onnx_genai::engine::PipelineGenerateRequest;
use onnx_genai::ort::Tokenizer;
use onnx_genai::preprocess::audio::{
    AudioSegment, SegmentConfig, StreamSegmenter, decode_wav_pcm16,
};
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest, GenerateToken};
use onnx_genai_server::multimodal;

use super::interactive::{GENERATING, INTERRUPT_REQUESTED, Interrupted, install_ctrlc_handler};
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
struct Transcriber {
    engine: Engine,
    tokenizer: Tokenizer,
    spec: multimodal::AudioInputSpec,
    language: Option<String>,
    max_new_tokens: Option<usize>,
}

impl Transcriber {
    fn load(model_dir: &Path, args: &TranscribeArgs) -> anyhow::Result<Self> {
        let setup = multimodal::load(model_dir)?.with_context(|| {
            format!(
                "What: {} could not be loaded as a speech package. \
                 Why: it declares no `pipeline`, so it has no audio encoder to run. \
                 How: point `transcribe` at a speech-to-text package.",
                model_dir.display()
            )
        })?;
        let spec = setup.multimodal.audio.clone().with_context(|| {
            format!(
                "What: {} cannot transcribe audio. \
                 Why: no component of its package declares an `input_features` audio input. \
                 How: point `transcribe` at a speech-to-text package.",
                model_dir.display()
            )
        })?;
        let tokenizer = Tokenizer::from_file(&setup.tokenizer_path).map_err(|error| {
            anyhow::anyhow!(
                "What: the package's tokenizer could not be loaded from {}. \
                 Why: {error}. \
                 How: verify the package ships a valid tokenizer.json.",
                setup.tokenizer_path.display()
            )
        })?;
        let engine = Engine::from_dir(model_dir, args.engine.to_config())?;
        Ok(Self {
            engine,
            tokenizer,
            spec,
            language: args.language.clone(),
            max_new_tokens: args.max_new_tokens,
        })
    }

    /// The model's declared input window, in seconds.
    fn window_seconds(&self) -> f32 {
        multimodal::audio_window_seconds(&self.spec)
    }

    /// Transcribe one segment of `[-1, 1]` mono samples.
    fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> anyhow::Result<String> {
        let input = multimodal::MultimodalInput::from_samples(&self.spec, samples, sample_rate)?;
        let token_ids =
            multimodal::audio_decoder_prompt(&self.tokenizer, self.language.as_deref())?;
        let mut options = GenerateOptions {
            temperature: 0.0,
            ..GenerateOptions::default()
        };
        if let Some(max_new_tokens) = self.max_new_tokens.or(self.spec.max_tokens) {
            options.max_new_tokens = max_new_tokens;
        }
        let request = input.bind(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        }))?;

        INTERRUPT_REQUESTED.store(false, Ordering::SeqCst);
        GENERATING.store(true, Ordering::SeqCst);
        let mut text = String::new();
        let mut callback = |token: GenerateToken| -> anyhow::Result<()> {
            if INTERRUPT_REQUESTED.load(Ordering::SeqCst) {
                return Err(anyhow::Error::new(Interrupted));
            }
            text.push_str(&token.text);
            Ok(())
        };
        let result =
            self.engine
                .generate_with_pipeline_callbacks(request, None, Some(&mut callback));
        GENERATING.store(false, Ordering::SeqCst);
        crate::flush_deferred_tracing()?;
        result?;
        Ok(text.trim().to_string())
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

    let window = transcriber.window_seconds();
    let segment_seconds = args.segment_seconds.unwrap_or(window);
    if segment_seconds > window {
        anyhow::bail!(
            "What: --segment-seconds {segment_seconds} was rejected. \
             Why: this model's declared audio input holds at most {window:.2}s, so a longer segment could not be encoded. \
             How: request at most {window:.2}s per segment."
        );
    }
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
            index = transcribe_stream(
                &mut transcriber,
                &args,
                segment_seconds,
                &writer,
                index,
                &mut totals,
            )?;
        } else {
            index = transcribe_file(
                &mut transcriber,
                source,
                &args,
                segment_seconds,
                &writer,
                index,
                &mut totals,
            )?;
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
    segment_seconds: f32,
    sample_rate: u32,
) -> anyhow::Result<StreamSegmenter> {
    let config = SegmentConfig::from_seconds(
        sample_rate,
        segment_seconds,
        args.silence_seconds,
        args.silence_threshold,
        args.min_segment_seconds,
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "What: the segmentation settings were rejected. \
             Why: {error}. \
             How: give positive --segment-seconds and a sample rate above zero."
        )
    })?;
    Ok(StreamSegmenter::new(config))
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
    segment_seconds: f32,
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
    let audio = decode_wav_pcm16(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "What: the audio file {} could not be decoded. \
             Why: {error}. \
             How: provide a PCM16 WAV file (convert with `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`).",
            path.display()
        )
    })?;

    let started = std::time::Instant::now();
    let mut segmenter = build_segmenter(args, segment_seconds, audio.sample_rate)?;
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
    segment_seconds: f32,
    writer: &TranscriptWriter,
    mut index: usize,
    totals: &mut TranscriptionTotals,
) -> anyhow::Result<usize> {
    let stdin = io::stdin();
    let mut reader = PcmStreamReader::new(stdin.lock(), args.sample_rate, args.channels)?;
    let sample_rate = reader.sample_rate();
    let mut segmenter = build_segmenter(args, segment_seconds, sample_rate)?;

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
