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
`downmix` therefore supports `channels: 1`; multichannel waveform outputs
are not yet implemented. `resample` changes the rate at its declared position in the program.
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

## Linear spectrograms

`spectrogram` consumes its named waveform and publishes a float32 tensor with
shape `[batch, n_fft / 2 + 1, floor(samples / hop_length)]`, in frequency-major
order. Declare `mode: power` or `mode: magnitude`, and `window: hann`,
`window: hamming` or `window: rectangular`. Both selectors are required; unknown
or missing selectors fail at compilation. Also author positive `n_fft` (at least
2), `hop_length` and `win_length` (at most `n_fft`).

All three windows occupy the centered `win_length` positions of the FFT input,
with zero coefficients outside that interval. For `0 <= j < win_length`, their
explicit coefficient contracts are:

| Selector | Coefficient |
| --- | --- |
| `hann` | `0.5 - 0.5 cos(2πj / win_length)` |
| `hamming` | `0.54 - 0.46 cos(2πj / win_length)` |
| `rectangular` | `1` |

Hann and Hamming are periodic, using `win_length` rather than `win_length - 1`
in the denominator, including a length-one window. Sample boundaries use
centered reflection, the forward FFT is unnormalized, and only its nonnegative
frequency bins are emitted without doubling. Power is `re² + im²`; magnitude is
`hypot(re, im)` directly, preserving representable finite magnitudes when squaring
would overflow. A nonfinite projected value still fails with rescaling guidance.
Complex and uncentered analyses have no operation/selector in this contract.
Odd FFT lengths and windows shorter than the FFT are supported.

The analysis rate comes from `sample_rate` or a preceding declared `resample`.
It must match the input rate; the operation does not implicitly resample.
`audio_features`/`features` outputs select the resulting matrix. The spectrum
retains its own valid-frame interval for `emit_valid_frames` and
`emit_validity_mask`, including padding and clipping on the selected branch.
An unpadded ragged batch fails with padding guidance. `whisper_log_mel`
normalization requires a `log_mel` source and rejects a linear spectrum.
The FFT implementation is shared with log-mel, which retains its existing
periodic-Hann/power projection and normalization recipe without materializing a full intermediate
spectrum.

```yaml
- {op: decode, outputs: [decoded]}
- {op: resample, sample_rate: 16000, outputs: [samples]}
- op: spectrogram
  mode: power
  window: hann
  n_fft: 512
  hop_length: 160
  win_length: 400
  outputs: [power]
- {op: emit_valid_frames, inputs: [power], outputs: [frames]}
```

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
Unsupported operations fail explicitly instead of being accepted as no-ops.
More DSP operations, windows/scales, codecs and the
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

Ogg Opus uses the published pure-Rust `symphonia-opus` 0.1.3 decoder with the
same Symphonia content probe. This decoder is constructed only after the probe
identifies Ogg and Opus; it is never globally registered for other containers.
Its raw 48 kHz PCM is presented by one clip-level cursor: `OpusHead` pre-skip
can span packets, the exact typed track final granule removes end padding, and
the decoder applies Q7.8 output gain once. Family 0 mono/stereo and fully routed
family 1 layouts with 1–8 output channels are admitted. Family 1's Vorbis plane
order is converted to canonical positioned-channel order before arithmetic
mono averaging. The original recording-rate header does not change the 48 kHz
decoding clock; authored `resample` still controls subsequent conversion.

This is a bounded Ogg Opus capability, not every Opus presentation convention.
It requires a complete seekable recording with a zero-origin `1/48000` track
clock and matching exact duration/frame-count metadata. Cropped/nonzero-origin
streams, missing final granules, unreferenced elementary channels and other
mapping families fail explicitly. Ogg code 3 packets with 32–48 frames also fail:
Symphonia 0.6.1 masks their frame count to five bits, damaging page timestamps.
The runtime independently checks the six-bit RFC count and refuses those packets
until a published demuxer repair is available. It never trusts the damaged packet
duration to cut PCM. WebM/Matroska and MP4 Opus remain explicitly unsupported;
their delay, discard-padding and identification-header conventions need separate
complete repairs. Existing non-Opus codecs in those containers remain available.

`decode_audio(bytes, limits)` and `AudioProgram::run(encoded, limits)` require
an explicit `AudioDecodeLimits`. `max_decoded_bytes` bounds cumulative raw
multichannel float32 PCM work (including samples later skipped/trimmed) and the
simultaneously retained mono waveform, interleaved staging and codec PCM packet
capacity. It also bounds packet progress at one packet per four budget bytes;
zero-frame packets cannot evade this bound. The maximum 120 ms Opus packet PCM
allocation is checked before decoder construction, and each Opus packet's exact
TOC duration is checked before decode/growth. Codec-private fixed-size state and
container metadata are not exact heap accounting under this PCM policy.

The CLI file decoder and server expose `--max-decoded-audio-bytes` with a finite
64 MiB default. Engine-backed admission additionally intersects the policy with
live host-memory headroom; zero headroom fails. Workflow audio uses the same
configured `EngineConfig.audio_decode_limits` and its existing shared host
authority. A program divides its caller budget across its declared decode
invocations and encoded batch rows. These bounds do not cover every later
authored DSP tensor or all request allocations. The API provides no unbounded default or compressed-byte expansion
guess. Encoding/generation tools are only used to prepare independent test
fixtures, never during runtime decoding. Dependency license/source information
is recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

MP4 edit lists are not applied, so priming
that depends on an edit list remains audible to preprocessing. Midstream rate or
channel changes are rejected. Convert such inputs to a supported WAV file first.
Live stdin framing continues to use its documented WAV/PCM16 protocol.

## Feature tensor inputs

Schema v1.16 admits externally executed audio feature programs.
A rank-3 float32 runtime media input also requires `preprocessing.audio`.
The program output whose `name` exactly matches that workflow input must declare
`content: audio_features` and `dtype: float32`. Serving executes that output's
program before binding the tensor and verifies the result against the workflow
shape. The same program determines source-rate clip limits for CLI segmentation.
There is no implicit sample rate, FFT, hop, window or normalization recipe.
Other program outputs do not impose a clip limit on the selected feature branch.

For example, `transcription_features` binds `request.features` after its authored
8 kHz / 128 FFT / 64 hop / 7 mel / 4 frame program. Its Native head and the
encoded-input `transcription_tokens` package produce matching buffered/SSE
transcripts and usage from the same waveform. Missing programs, output-name or
dtype mismatches, and actual feature-shape mismatches fail with repair guidance.
