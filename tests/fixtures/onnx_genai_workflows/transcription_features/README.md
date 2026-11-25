# Authored feature-input transcription

The Native token head receives float32 features rather than encoded bytes.
Serving executes the declared 8 kHz, 128 FFT, 64 hop, 7 mel / 4 frame audio
program and binds its explicitly named `request.features` output. The head
is shared with `transcription_tokens`, allowing encoded and feature-bound
workflows to exercise the same DSP and actual inference. This synthetic head
checks data plumbing, not speech-recognition quality.
