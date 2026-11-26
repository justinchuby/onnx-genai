//! Container-specific Opus presentation. The codec returns raw gain-adjusted PCM;
//! Ogg owns its exact clip range, while Matroska consumes normalized packet trims.
//! Both use the same validated canonical channel routing.
use super::AudioPreprocessError;
use symphonia::core::codecs::audio::AudioCodecParameters;

fn invalid(message: impl Into<String>) -> AudioPreprocessError {
    AudioPreprocessError::Encoded(message.into())
}

pub(super) struct OggOpus {
    remaining_skip: usize,
    cursor: usize,
    end: usize,
    // Source Vorbis-order plane -> canonical positioned-channel index.
    canonical: Vec<usize>,
}

impl OggOpus {
    pub(super) fn packet_frames(
        packet: &symphonia::core::packet::Packet,
    ) -> Result<usize, AudioPreprocessError> {
        let info = onnx_genai_symphonia_mkv::opus::packet_info(&packet.data).map_err(|source| {
            AudioPreprocessError::Codec {
                context: "read Opus RFC packet geometry".into(),
                source,
            }
        })?;
        // This restriction belongs to the upstream Ogg demuxer, not the codec.
        if info.frame_count() >= 32 {
            return Err(invalid(
                "Ogg Opus code 3 packets with 32..=48 frames are not supported by this build's Ogg demuxer timing; re-encode with fewer frames per packet or use WAV/FLAC",
            ));
        }
        Ok(info.raw_frames())
    }

    pub(super) fn new(
        params: &AudioCodecParameters,
        track: &symphonia::core::formats::Track,
    ) -> Result<Self, AudioPreprocessError> {
        let info = header(params, "Ogg")?;
        let skip = info.pre_skip;
        if track.delay != Some(u32::from(skip)) {
            return Err(invalid(format!(
                "Ogg Opus track delay {:?} disagrees with OpusHead pre-skip {skip}; repair the container metadata",
                track.delay
            )));
        }
        require_clock(track, "Ogg")?;
        let end=track.num_frames.filter(|frames|track.duration.is_some_and(|duration|duration.get()==*frames)).and_then(|frames|usize::try_from(frames).ok()).ok_or_else(||invalid("Ogg Opus requires an exact bounded final granule from the container track duration/frame count; provide a complete seekable recording"))?;
        if end < usize::from(skip) {
            return Err(invalid(
                "Ogg Opus final granule precedes its pre-skip; repair the container duration",
            ));
        }
        Ok(Self {
            remaining_skip: usize::from(skip),
            cursor: 0,
            end,
            canonical: info.canonical_order().to_vec(),
        })
    }

    pub(super) fn present(&mut self, raw: std::ops::Range<usize>) -> std::ops::Range<usize> {
        let available = raw.len().min(self.end.saturating_sub(self.cursor));
        self.cursor += raw.len();
        let discard = self.remaining_skip.min(available);
        self.remaining_skip -= discard;
        raw.start + discard..raw.start + available
    }

    /// Check the demuxer's packet timing against the independent raw TOC clock.
    /// Timing proves consistency only; this layer still owns every PCM cut.
    pub(super) fn validate_packet(
        &self,
        packet: &symphonia::core::packet::Packet,
        raw: usize,
    ) -> Result<(), AudioPreprocessError> {
        let cursor = i64::try_from(self.cursor).map_err(|_| {
            invalid("Ogg Opus raw presentation cursor exceeds the signed timestamp range")
        })?;
        if packet.pts.get() != cursor || packet.trim_start.get() != 0 {
            return Err(invalid(format!(
                "Ogg Opus packet clock {:?} disagrees with raw TOC cursor {}; discontinuous/cropped streams or duplicated start trimming are unsupported",
                packet.pts, self.cursor
            )));
        }
        let declared = packet
            .dur
            .get()
            .checked_add(packet.trim_end.get())
            .ok_or_else(|| invalid("Ogg Opus packet valid duration plus end trim overflows"))?;
        if declared != raw as u64 {
            return Err(invalid(format!(
                "Ogg Opus packet timing declares {declared} raw frames, but its TOC declares {raw}; repair the container duration metadata"
            )));
        }
        // A later page's final granule cannot precede frames already declared
        // valid on an earlier page. Symphonia's saturating end trim otherwise
        // hides a backwards EOS granule; do not silently truncate prior audio.
        if packet.dur.get() != 0 {
            let valid_end = self
                .cursor
                .checked_add(usize::try_from(packet.dur.get()).map_err(|_| {
                    invalid("Ogg Opus packet duration exceeds the host frame range")
                })?)
                .ok_or_else(|| invalid("Ogg Opus packet end timestamp overflows"))?;
            if valid_end > self.end {
                return Err(invalid(format!(
                    "Ogg Opus final granule {} precedes a container-declared valid packet end {valid_end}; granule positions must not move backwards, so repair the recording",
                    self.end
                )));
            }
        }
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<(), AudioPreprocessError> {
        if self.remaining_skip != 0 || self.cursor < self.end {
            return Err(invalid(format!(
                "Ogg Opus ends with {} unconsumed pre-skip frames and {} decoded raw frames before final granule {}; supply a complete recording",
                self.remaining_skip, self.cursor, self.end
            )));
        }
        Ok(())
    }
}

fn header(
    params: &AudioCodecParameters,
    container: &str,
) -> Result<onnx_genai_symphonia_mkv::opus::HeaderInfo, AudioPreprocessError> {
    let info = onnx_genai_symphonia_mkv::opus::HeaderInfo::parse(
        params
            .extra_data
            .as_deref()
            .ok_or_else(|| invalid(format!("{container} Opus track has no OpusHead")))?,
    )
    .map_err(|source| AudioPreprocessError::Codec {
        context: format!("validate {container} OpusHead"),
        source,
    })?;
    if params.sample_rate != Some(48000) || params.channels.as_ref() != Some(&info.channels) {
        return Err(invalid(format!(
            "{container} Opus requires its canonical 48000 Hz output clock and positioned OpusHead channel layout"
        )));
    }
    Ok(info)
}
fn require_clock(
    track: &symphonia::core::formats::Track,
    container: &str,
) -> Result<(), AudioPreprocessError> {
    let clock = track
        .time_base
        .ok_or_else(|| invalid(format!("{container} Opus track has no presentation clock")))?;
    if clock.numer.get() != 1 || clock.denom.get() != 48000 || track.start_ts.get() != 0 {
        return Err(invalid(format!(
            "{container} Opus requires a zero-origin 1/48000 presentation clock; cropped/nonzero-origin streams are unsupported, so normalize to WAV/FLAC"
        )));
    }
    Ok(())
}

pub(super) struct PacketPlan {
    pub raw_frames: usize,
    pub range: std::ops::Range<usize>,
}
pub(super) struct MatroskaOpus {
    cursor: u64,
    delay: u32,
    canonical: Vec<usize>,
}
/// Codec output is raw in both modes, but each container supplies one distinct
/// presentation contract; packet trims must never also apply Ogg's header skip.
pub(super) enum Presentation {
    Ogg(OggOpus),
    Matroska(MatroskaOpus),
}
impl Presentation {
    pub fn new(
        params: &AudioCodecParameters,
        track: &symphonia::core::formats::Track,
        container: &str,
    ) -> Result<Self, AudioPreprocessError> {
        match container {
            "ogg" => Ok(Self::Ogg(OggOpus::new(params, track)?)),
            "matroska" => {
                let info = header(params, "Matroska")?;
                require_clock(track, "Matroska")?;
                if track.delay != Some(u32::from(info.pre_skip)) {
                    return Err(invalid(
                        "Matroska Opus track has no validated matching CodecDelay; use the maintained default-audio packet-concatenation profile",
                    ));
                }
                Ok(Self::Matroska(MatroskaOpus {
                    cursor: 0,
                    delay: u32::from(info.pre_skip),
                    canonical: info.canonical_order().to_vec(),
                }))
            }
            _ => Err(invalid(format!(
                "Opus presentation in {container} is unsupported by this build; use admitted Ogg/WebM Opus or WAV/FLAC (MP4 Opus edit lists are not admitted)"
            ))),
        }
    }
    pub fn plan(
        &self,
        packet: &symphonia::core::packet::Packet,
    ) -> Result<PacketPlan, AudioPreprocessError> {
        match self {
            Self::Ogg(owner) => {
                let raw = OggOpus::packet_frames(packet)?;
                owner.validate_packet(packet, raw)?;
                Ok(PacketPlan {
                    raw_frames: raw,
                    range: 0..raw,
                })
            }
            Self::Matroska(owner) => {
                let raw = packet
                    .dur
                    .get()
                    .checked_add(packet.trim_start.get())
                    .and_then(|n| n.checked_add(packet.trim_end.get()))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        invalid("Matroska Opus normalized raw duration/trim count overflows")
                    })?;
                if raw == 0 || raw > onnx_genai_symphonia_mkv::opus::MAX_PACKET_FRAMES {
                    return Err(invalid(
                        "Matroska Opus packet lacks a bounded nonzero normalized duration; use the maintained packet-trim reader",
                    ));
                }
                let priming = u64::from(owner.delay)
                    .saturating_sub(owner.cursor)
                    .min(raw as u64);
                if packet.dts != packet.pts || packet.trim_start.get() < priming {
                    return Err(invalid(
                        "Matroska Opus normalized packet priming disagrees with its raw decode cursor; repair the container timing",
                    ));
                }
                let start = usize::try_from(packet.trim_start.get())
                    .map_err(|_| invalid("Matroska Opus start trim exceeds host range"))?;
                let valid = usize::try_from(packet.dur.get())
                    .map_err(|_| invalid("Matroska Opus valid duration exceeds host range"))?;
                Ok(PacketPlan {
                    raw_frames: raw,
                    range: start..start + valid,
                })
            }
        }
    }
    pub fn present(&mut self, raw: usize, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        match self {
            Self::Ogg(owner) => owner.present(range),
            Self::Matroska(owner) => {
                owner.cursor += raw as u64;
                range
            }
        }
    }
    pub fn canonicalize(&self, samples: &mut [f32]) {
        let order = match self {
            Self::Ogg(owner) => &owner.canonical,
            Self::Matroska(owner) => &owner.canonical,
        };
        canonicalize(samples, order);
    }
    pub fn finish(&self) -> Result<(), AudioPreprocessError> {
        match self {
            Self::Ogg(owner) => owner.finish(),
            Self::Matroska(owner) => {
                if owner.cursor >= u64::from(owner.delay) {
                    Ok(())
                } else {
                    Err(invalid(
                        "Matroska Opus ends before declared CodecDelay; supply a complete recording",
                    ))
                }
            }
        }
    }
}
fn canonicalize(samples: &mut [f32], order: &[usize]) {
    for frame in samples.chunks_exact_mut(order.len()) {
        let mut canonical = [0.0; 8];
        for (source, &destination) in order.iter().enumerate() {
            canonical[destination] = frame[source];
        }
        frame.copy_from_slice(&canonical[..order.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::{
        codecs::audio::{AudioDecoder, AudioDecoderOptions},
        formats::{FormatOptions, TrackType, probe::Hint},
        io::MediaSourceStream,
        meta::MetadataOptions,
    };

    #[test]
    fn surround_canonical_planes_match_independent_libopus_not_only_the_mono_mean() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/audio/opus");
        for (name, count) in [("mapping1-3", 3), ("mapping1-6", 6)] {
            for suffix in ["ogg", "webm"] {
                let bytes = std::fs::read(root.join(format!("{name}.{suffix}"))).unwrap();
                let mut format = super::super::audio_probe()
                    .probe(
                        &Hint::new(),
                        MediaSourceStream::new(
                            Box::new(std::io::Cursor::new(bytes)),
                            Default::default(),
                        ),
                        FormatOptions::default(),
                        MetadataOptions::default(),
                    )
                    .unwrap();
                let track = format.default_track(TrackType::Audio).unwrap();
                let id = track.id;
                let params = track.codec_params.as_ref().unwrap().audio().unwrap();
                let mut presentation =
                    Presentation::new(params, track, format.format_info().short_name).unwrap();
                let mut decoder = symphonia_opus::OpusDecoder::try_new(
                    params,
                    &AudioDecoderOptions {
                        gapless: false,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut actual = Vec::new();
                while let Some(packet) = format.next_packet().unwrap() {
                    if packet.track_id != id {
                        continue;
                    }
                    let plan = presentation.plan(&packet).unwrap();
                    let expected = plan.raw_frames;
                    let decoded = decoder.decode(&packet).unwrap();
                    assert_eq!(decoded.frames(), expected);
                    let mut samples = vec![0.0; decoded.samples_interleaved()];
                    decoded.copy_to_slice_interleaved(&mut samples);
                    presentation.canonicalize(&mut samples);
                    let range = presentation.present(expected, plan.range);
                    actual.extend_from_slice(&samples[range.start * count..range.end * count]);
                }
                presentation.finish().unwrap();
                let reference = std::fs::read(root.join(format!("{name}.reference.f32")))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(actual.len(), 11520 * count);
                assert_eq!(actual.len(), reference.len());
                for channel in 0..count {
                    let maximum = actual[channel..]
                        .iter()
                        .step_by(count)
                        .zip(reference[channel..].iter().step_by(count))
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        maximum <= 1.0 / 32768.0,
                        "{name}/{suffix} canonical channel {channel}: max error {maximum}"
                    );
                }
            }
        }
    }

    #[test]
    fn code_three_frame_count_bounds_are_explicit() {
        use symphonia::core::{
            packet::Packet,
            units::{Duration, Timestamp},
        };
        // CELT 2.5 ms, code 3. No decoder invocation is needed to prove the
        // exact metadata boundary that the published Ogg mapper mishandles.
        for count in [0_u8, 31, 32, 48, 49, 63] {
            let packet = Packet::new(
                0,
                Timestamp::ZERO,
                Duration::ZERO,
                vec![0x83, count].into_boxed_slice(),
            );
            match count {
                31 => assert_eq!(OggOpus::packet_frames(&packet).unwrap(), 31 * 120),
                32 | 48 => assert!(
                    OggOpus::packet_frames(&packet)
                        .unwrap_err()
                        .to_string()
                        .contains("not supported by this build's Ogg demuxer timing")
                ),
                _ => assert!(OggOpus::packet_frames(&packet).is_err()),
            }
        }
    }
}
