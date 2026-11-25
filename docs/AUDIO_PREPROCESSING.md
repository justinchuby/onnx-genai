# Executable audio preprocessing

`audio-preprocess@2` executes the package's ordered transform program. Every
transform produces a processor-local value. `inputs: [name]` selects an earlier
value; omitted `inputs` selects the immediately preceding transform. `decode`
reads the adapter's encoded bytes and omits `inputs`. Named outputs select their
exact `source`; their `content` and `dtype` validate that source rather than
choosing another value at runtime. Duplicate producers, dangling inputs and
incompatible output roles/types fail during compilation.

For example, a pure-waveform program can normalize each decoded clip and then
pad the batch to its longest row:

```yaml
transforms:
  - op: decode
    outputs: [decoded]
  - op: resample
    sample_rate: 16000
    outputs: [samples]
  - op: zero_mean_unit_variance
    epsilon: 1.0e-7
    outputs: [normalized]
  - op: pad
    mode: right
    pad_value: 0.0
    outputs: [padded]
  - op: emit_validity_mask
    inputs: [padded]
    outputs: [mask]
  - op: emit_valid_samples
    inputs: [padded]
    outputs: [length]
outputs:
  - {name: waveform, source: padded, content: waveform, dtype: float32}
  - {name: mask, source: mask, content: validity_mask, dtype: int64}
  - {name: length, source: length, content: valid_samples, dtype: int64}
```

Workflow packages additionally declare the tensor contracts and component ports
for these outputs. Version 2 binds component-local output names and permits
repeated invocations; version 1 binds workflow values and admits one invocation.
The same executable program semantics apply to both.

## Waveforms and validity

`decode` accepts [supported encoded audio](#encoded-audio-formats) by content and averages channels to mono.
Decoding runs in portable Rust without external binaries, preserving the
source sample rate and applying declared codec delay/padding metadata. Invalid
packets and changing rate/channel layouts fail with context. An explicit
`downmix` therefore supports `channels: 1`; multichannel waveform outputs and
other codecs are not yet implemented. `resample` changes the rate at its declared position in the program.
`rescale` multiplies samples by the declared finite scale.
`zero_mean_unit_variance` (also selected by `normalize` with that mode) uses the
valid samples' population variance plus a positive finite epsilon. Existing
padding values are retained instead of being treated as real samples.

`pad` supports `right` and `left`, with any finite fill value. Its optional target
pads short rows without truncating long rows; omitting the target pads to the
batch maximum. `fixed_window` requires a target and both pads and truncates.
`trim` truncates at a declared right boundary without extending short rows.
Sample validity follows these operations. `emit_valid_samples` and
`emit_sample_lengths` publish the surviving real sample count, and
`emit_validity_mask` publishes a batch-aligned int64 mask. Ragged published
waveform/feature/mask rows fail with padding guidance.

## Input windows

`AudioProgram::maximum_input_samples(source_rate)` derives the largest source
clip whose valid samples survive every published path's `trim`/`fixed_window`.
It uses execution's integer resampling lengths and valid intervals, including
multiple resamplers and left/right padding. Trimming padding does not count as
losing real audio. An unpublished branch does not constrain the window. `None`
means no live clipping bound was declared; a caller must choose its own segment
allowance instead of assuming a model-specific duration. Zero source rate and
windows too small to preserve one source sample fail with guidance.

The CLI uses this proof for both file and live transcription after resolving the
recording's actual source rate. Serving encoded audio binds the compiled program,
not a guessed rank-3 mel output or a fixed hop/rate formula.

## Log-mel and normalization

`log_mel` consumes its named waveform, using the declared mel count, sample rate,
FFT size, hop and periodic-Hann window length. It does **not** implicitly resample:
its analysis rate must match the input; insert `resample` explicitly when needed.
A preceding `resample` supplies the rate if the analysis transform omits it.
Centered reflected analysis uses `floor(samples / hop_length)` frames and Slaney
filters. Its output is raw base-10 log-mel energy. To apply dynamic-range clipping
and the established `(log_energy + 4) / 4` scaling, declare a separate
`normalize` transform with `mode: whisper_log_mel`. Earlier raw outputs remain
available even after normalization, so branches retain their own values.

`emit_valid_frames` reads the selected mel value's frame validity. When its input
is a waveform it requires an explicit `hop_length`, avoiding an inferred hop from
another branch. Frame validity maps the surviving sample interval to hop bins,
clamped to the emitted frame count; empty real intervals have zero valid frames.
Sample and frame length output roles are checked as different units.

The public `LogMelExtractor` convenience API still returns normalized features;
the executable program applies only the transforms it actually declares.
Unsupported operations such as standalone `spectrogram` fail explicitly instead
of being accepted as no-ops. More DSP operations, windows/scales, codecs and the
server's remaining legacy feature-input path still require compatibility work.
Declared pure-waveform CTC/greedy packages execute through the
[file transcription API](TRANSCRIPTION.md).

The maintained `audio_waveform` fixture executes actual Native CPU preprocessing
and an ONNX head, checking independently normalized/rescaled branches, padding,
masks and lengths. Direct processor tests cover branching, batching, explicit
rates and failure cases. These establish dataflow and numerical behavior, not
speech-recognition quality or device performance.

### Encoded audio formats

Audio inputs use one content-probed Symphonia decoder, including WAV (integer
and floating point), MP3/MP2, FLAC, Vorbis/Ogg, AAC (ADTS/MP4), ALAC, AIFF, CAF
and supported audio tracks in Matroska. The codec inside the container must also
be supported. Chat `input_audio.format` is a nonempty descriptive label; bytes
determine the actual format. Decoding preserves the source sample rate and uses
the arithmetic channel mean; internal WAV transport uses float32.

Opus is unsupported in Symphonia 0.6.1. MP4 edit lists are not applied, so priming
that depends on an edit list remains audible to preprocessing. Midstream rate or
channel changes are rejected. Convert such inputs to a supported WAV file first.
Live stdin framing continues to use its documented WAV/PCM16 protocol.
