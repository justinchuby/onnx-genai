Original generated audio for codec conformance (no speech or third-party content).
`stereo-tone.mp3`: 100 ms, 16 kHz stereo, 440 Hz sine, left amplitude 0.5,
right amplitude 0.25, encoded with libmp3lame at 64 kbit/s including gapless tags.
`stereo-tone-reference.f32le`: independently decoded arithmetic channel mean,
1600 little-endian float32 samples. Tests compare waveform samples, sample rate,
channel averaging and removal of codec delay/padding. No recognition-quality claim.

Reproduce with ffmpeg (generation/oracle only, never a runtime dependency):

```sh
ffmpeg -f lavfi -i 'aevalsrc=0.5*sin(2*PI*440*t)|0.25*sin(2*PI*440*t):s=16000:d=0.1' -c:a libmp3lame -b:a 64k -map_metadata -1 stereo-tone.mp3
ffmpeg -i stereo-tone.mp3 -af 'pan=mono|c0=0.5*c0+0.5*c1' -f f32le stereo-tone-reference.f32le
```
