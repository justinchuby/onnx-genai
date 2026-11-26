//! Ogg Opus presentation contract. The codec returns raw, gain-adjusted PCM;
//! this layer alone owns pre-skip and canonical channel order (RFC 7845).
use super::AudioPreprocessError;
use symphonia::core::{
    audio::{Channels, Position},
    codecs::audio::AudioCodecParameters,
};

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
        let toc = *packet
            .data
            .first()
            .ok_or_else(|| invalid("Ogg Opus packet has no TOC byte"))?;
        let config = toc >> 3;
        let frame = if config >= 16 {
            120usize << (config & 3)
        } else if config >= 12 {
            480usize << (config & 1)
        } else {
            [480, 960, 1920, 2880][usize::from(config & 3)]
        };
        let count = match toc & 3 {
            0 => 1,
            1 | 2 => 2,
            _ => {
                let count =
                    usize::from(
                        packet.data.get(1).ok_or_else(|| {
                            invalid("Ogg Opus code 3 packet has no frame-count byte")
                        })? & 63,
                    );
                // Symphonia 0.6.1's Ogg mapper masks five rather than six bits,
                // corrupting page timestamp origins for these valid packets.
                // Reject this capability until the published demuxer is fixed.
                if count > 48 {
                    return Err(invalid(
                        "Ogg Opus code 3 frame count exceeds the RFC 6716 maximum of 48 frames",
                    ));
                }
                if count >= 32 {
                    return Err(invalid(
                        "Ogg Opus code 3 packets with 32..=48 frames are not supported by this build's Ogg demuxer timing; re-encode with fewer frames per packet or use WAV/FLAC",
                    ));
                }
                count
            }
        };
        let total = frame * count;
        if total == 0 || total > 5760 {
            return Err(invalid(format!(
                "Ogg Opus packet declares {total} raw frames; require a nonzero duration no longer than the RFC 6716 120 ms maximum (5760 frames)"
            )));
        }
        Ok(total)
    }

    pub(super) fn new(
        params: &AudioCodecParameters,
        track: &symphonia::core::formats::Track,
    ) -> Result<Self, AudioPreprocessError> {
        let head = params
            .extra_data
            .as_deref()
            .ok_or_else(|| invalid("Ogg Opus track has no OpusHead identification header"))?;
        if head.len() < 19 || &head[..8] != b"OpusHead" || head[8] > 15 {
            return Err(invalid(
                "Ogg Opus identification header is truncated or has an unsupported version; provide a complete RFC 7845 stream",
            ));
        }
        let count = usize::from(head[9]);
        let channels = params
            .channels
            .as_ref()
            .ok_or_else(|| invalid("Ogg Opus track has no positioned channel layout"))?;
        if params.sample_rate != Some(48_000)
            || count == 0
            || count > 8
            || channels.count() != count
        {
            return Err(invalid(
                "Ogg Opus requires its 48000 Hz decoding clock and a consistent 1..=8 channel layout; the original recording rate is not the decoding rate",
            ));
        }
        let skip = u16::from_le_bytes([head[10], head[11]]);
        if track.delay != Some(u32::from(skip)) {
            return Err(invalid(format!(
                "Ogg Opus track delay {:?} disagrees with OpusHead pre-skip {skip}; repair the container metadata",
                track.delay
            )));
        }
        let clock = track
            .time_base
            .ok_or_else(|| invalid("Ogg Opus track has no presentation clock"))?;
        if clock.numer.get() != 1 || clock.denom.get() != 48_000 || track.start_ts.get() != 0 {
            return Err(invalid(
                "Ogg Opus requires a zero-origin 1/48000 presentation clock; cropped/nonzero-origin Ogg streams are not yet supported, so normalize the recording to WAV/FLAC",
            ));
        }
        let end = track.num_frames.filter(|frames|track.duration.is_some_and(|duration|duration.get()==*frames)).and_then(|frames|usize::try_from(frames).ok()).ok_or_else(||invalid("Ogg Opus requires an exact bounded final granule from the container track duration/frame count; provide a complete seekable recording"))?;
        if end < usize::from(skip) {
            return Err(invalid(
                "Ogg Opus final granule precedes its pre-skip; repair the container duration",
            ));
        }
        if head[18] != 0 && head[18] != 1 || head[18] == 0 && count > 2 {
            return Err(invalid(format!(
                "Ogg Opus mapping family {} with {count} channels is unsupported; use family 0 mono/stereo or family 1 surround",
                head[18]
            )));
        }
        if head[18] == 1 && head.len() < 21 + count {
            return Err(invalid(
                "Ogg Opus family 1 identification header has a truncated stream/channel mapping table",
            ));
        }
        if head[18] == 1 {
            let streams = usize::from(head[19]);
            let coupled = usize::from(head[20]);
            let decoded_channels = streams + coupled;
            let map = &head[21..21 + count];
            if streams == 0
                || coupled > streams
                || decoded_channels > 255
                || map
                    .iter()
                    .any(|channel| *channel != 255 && usize::from(*channel) >= decoded_channels)
            {
                return Err(invalid(
                    "Ogg Opus family 1 stream/coupled counts or channel mapping indices are invalid; repair OpusHead",
                ));
            }
            // Every admitted elementary channel has a real output destination.
            // This bounds decoder state by declared output channels and avoids
            // the dependency's hidden-stream scratch allocation before factory.
            if (0..decoded_channels)
                .any(|channel| !map.iter().any(|mapped| usize::from(*mapped) == channel))
            {
                return Err(invalid(
                    "Ogg Opus family 1 contains unreferenced elementary channels; this build admits fully routed streams only, so remove unused encoded streams or convert to WAV/FLAC",
                ));
            }
        }
        let canonical = canonical_order(channels, count)?;
        Ok(Self {
            remaining_skip: usize::from(skip),
            cursor: 0,
            end,
            canonical,
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

    pub(super) fn canonicalize(&self, samples: &mut [f32]) {
        let channels = self.canonical.len();
        for frame in samples.chunks_exact_mut(channels) {
            let mut canonical = [0.0; 8];
            for (source, &destination) in self.canonical.iter().enumerate() {
                canonical[destination] = frame[source];
            }
            frame.copy_from_slice(&canonical[..channels]);
        }
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

fn canonical_order(channels: &Channels, count: usize) -> Result<Vec<usize>, AudioPreprocessError> {
    use Position as P;
    // RFC 7845 §5.1.1.2: family 1 uses Vorbis channel ordering, independently
    // of its stream mapping bytes. Symphonia positioned layouts use bit order.
    let positions: &[Position] = match count {
        1 => &[P::FRONT_LEFT],
        2 => &[P::FRONT_LEFT, P::FRONT_RIGHT],
        3 => &[P::FRONT_LEFT, P::FRONT_CENTER, P::FRONT_RIGHT],
        4 => &[P::FRONT_LEFT, P::FRONT_RIGHT, P::REAR_LEFT, P::REAR_RIGHT],
        5 => &[
            P::FRONT_LEFT,
            P::FRONT_CENTER,
            P::FRONT_RIGHT,
            P::REAR_LEFT,
            P::REAR_RIGHT,
        ],
        6 => &[
            P::FRONT_LEFT,
            P::FRONT_CENTER,
            P::FRONT_RIGHT,
            P::REAR_LEFT,
            P::REAR_RIGHT,
            P::LFE1,
        ],
        7 => &[
            P::FRONT_LEFT,
            P::FRONT_CENTER,
            P::FRONT_RIGHT,
            P::SIDE_LEFT,
            P::SIDE_RIGHT,
            P::REAR_CENTER,
            P::LFE1,
        ],
        8 => &[
            P::FRONT_LEFT,
            P::FRONT_CENTER,
            P::FRONT_RIGHT,
            P::SIDE_LEFT,
            P::SIDE_RIGHT,
            P::REAR_LEFT,
            P::REAR_RIGHT,
            P::LFE1,
        ],
        _ => return Err(invalid("Ogg Opus family 1 channel count must be 1..=8")),
    };
    positions.iter().map(|&position| channels.get_canonical_index_for_positioned_channel(position).ok_or_else(|| invalid(format!("Ogg Opus channel layout {channels:?} omits required position {position:?}; repair its identification header")))).collect()
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
            let bytes = std::fs::read(root.join(format!("{name}.ogg"))).unwrap();
            let mut format = symphonia::default::get_probe()
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
            let mut presentation = OggOpus::new(params, track).unwrap();
            let mut decoder =
                symphonia_opus::OpusDecoder::try_new(params, &AudioDecoderOptions::default())
                    .unwrap();
            let mut actual = Vec::new();
            while let Some(packet) = format.next_packet().unwrap() {
                if packet.track_id != id {
                    continue;
                }
                let expected = OggOpus::packet_frames(&packet).unwrap();
                let decoded = decoder.decode(&packet).unwrap();
                assert_eq!(decoded.frames(), expected);
                let mut samples = vec![0.0; decoded.samples_interleaved()];
                decoded.copy_to_slice_interleaved(&mut samples);
                presentation.canonicalize(&mut samples);
                let range = presentation.present(0..expected);
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
                    "{name} canonical channel {channel}: max error {maximum}"
                );
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
