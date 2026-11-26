// Copyright (c) 2026 Justin Chu.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Canonical RFC 6716 packet geometry and RFC 7845 header admission.
//! This module does not impose a particular container's presentation clock.
use std::num::{NonZeroU8, NonZeroU16};
use symphonia_core::{
    audio::{Channels, Position},
    errors::{Error, Result},
    io::BufReader,
    units::TimeBase,
};

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::IoError(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

/// Maximum decoded frames in one Opus packet at the canonical 48 kHz output rate.
pub const MAX_PACKET_FRAMES: usize = 5760;
/// Opus's canonical full-band decoding clock, independently of original recording rate.
pub const DECODE_RATE: u32 = 48000;

/// Nonzero RFC framing facts; the codec still validates each compressed frame.
#[derive(Debug, Clone, Copy)]
pub struct PacketInfo {
    frames: NonZeroU16,
    count: NonZeroU8,
}
impl PacketInfo {
    pub fn raw_frames(self) -> usize {
        usize::from(self.frames.get())
    }
    pub fn frame_count(self) -> u8 {
        self.count.get()
    }
}

/// Read the RFC TOC once using the six-bit code-3 frame count, not a demuxer mask.
pub fn packet_info(data: &[u8]) -> Result<PacketInfo> {
    let toc = *data
        .first()
        .ok_or_else(|| invalid("Opus packet has no TOC byte; supply a complete encoded packet"))?;
    let config = toc >> 3;
    let frame = if config >= 16 {
        120_u16 << (config & 3)
    } else if config >= 12 {
        480_u16 << (config & 1)
    } else {
        [480_u16, 960, 1920, 2880][usize::from(config & 3)]
    };
    let count = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => {
            *data.get(1).ok_or_else(|| {
                invalid("Opus code 3 packet has no frame-count byte; supply complete framing")
            })? & 63
        }
    };
    if count == 0 || count > 48 {
        return Err(invalid(format!(
            "Opus code 3 frame count {count} is outside RFC 6716's 1..=48 range; repair the encoded packet"
        )));
    }
    let frames = u32::from(frame) * u32::from(count);
    if frames > MAX_PACKET_FRAMES as u32 {
        return Err(invalid(format!(
            "Opus packet duration {frames} frames exceeds the RFC 6716 120 ms maximum ({MAX_PACKET_FRAMES}); repair its TOC"
        )));
    }
    Ok(PacketInfo {
        frames: NonZeroU16::new(
            u16::try_from(frames)
                .map_err(|_| invalid("Opus packet duration exceeds bounded frame range"))?,
        )
        .ok_or_else(|| invalid("Opus packet duration is zero"))?,
        count: NonZeroU8::new(count).ok_or_else(|| invalid("Opus packet frame count is zero"))?,
    })
}

/// Header facts validated before constructing an Opus decoder's stream buffers.
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub channels: Channels,
    pub pre_skip: u16,
    pub original_sample_rate: u32,
    canonical: Vec<usize>,
}
impl HeaderInfo {
    pub fn parse(data: &[u8]) -> Result<Self> {
        let head =
            symphonia_common::xiph::audio::opus::OpusHead::read(&mut BufReader::new(data), 15)?;
        let count = head.channels.count();
        if !(1..=8).contains(&count) || data.len() < 19 {
            return Err(invalid(
                "OpusHead must declare 1..=8 output channels and a complete identification header",
            ));
        }
        let family = data[18];
        if family != 0 && family != 1 || family == 0 && count > 2 {
            return Err(invalid(format!(
                "OpusHead mapping family {family} with {count} channels is unsupported; use family 0 mono/stereo or fully routed family 1"
            )));
        }
        if family == 1 {
            if data.len() < 21 + count {
                return Err(invalid(
                    "OpusHead family 1 stream/channel mapping table is truncated",
                ));
            }
            let streams = usize::from(data[19]);
            let coupled = usize::from(data[20]);
            let decoded = streams + coupled;
            let map = &data[21..21 + count];
            if streams == 0
                || coupled > streams
                || decoded > 255
                || map
                    .iter()
                    .any(|value| *value != 255 && usize::from(*value) >= decoded)
            {
                return Err(invalid(
                    "OpusHead family 1 stream/coupled counts or mapping indices are invalid; repair the identification header",
                ));
            }
            if (0..decoded).any(|channel| !map.iter().any(|mapped| usize::from(*mapped) == channel))
            {
                return Err(invalid(
                    "OpusHead contains unreferenced elementary channels; this build admits fully routed streams only to bound decoder state",
                ));
            }
        }
        let canonical = canonical_order(&head.channels, count)?;
        Ok(Self {
            channels: head.channels,
            pre_skip: head.pre_skip,
            original_sample_rate: head.original_sample_rate,
            canonical,
        })
    }
    /// Source Vorbis plane index to canonical positioned-channel destination.
    pub fn canonical_order(&self) -> &[usize] {
        &self.canonical
    }
}

fn canonical_order(channels: &Channels, count: usize) -> Result<Vec<usize>> {
    use Position as P;
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
        _ => return Err(invalid("Opus channel count must be 1..=8")),
    };
    positions
        .iter()
        .map(|position| {
            channels
                .get_canonical_index_for_positioned_channel(*position)
                .ok_or_else(|| {
                    invalid("Opus positioned channel layout omits a required RFC 7845 position")
                })
        })
        .collect()
}

pub(crate) fn nanoseconds_to_frames(value: i128, field: &str) -> Result<i64> {
    let numerator = value.checked_mul(i128::from(DECODE_RATE)).ok_or_else(|| {
        invalid(format!(
            "Matroska Opus {field} nanosecond conversion overflows; shorten the recording"
        ))
    })?;
    if numerator % 1_000_000_000 != 0 {
        return Err(invalid(format!(
            "Matroska Opus {field}={value} ns is not integral in the 48000 Hz clock; fractional-frame presentation is unsupported, so normalize to WAV/FLAC"
        )));
    }
    i64::try_from(numerator / 1_000_000_000).map_err(|_| {
        invalid(format!(
            "Matroska Opus {field} exceeds signed frame range; repair its timing"
        ))
    })
}

/// A normalized raw-buffer interval. Delay is already represented in these trims.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PacketTiming {
    pub pts: i64,
    pub valid: u64,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug)]
pub(crate) struct TrackClock {
    delay: u32,
    raw_cursor: u64,
    source_clock: TimeBase,
    terminal_padding: bool,
    previous_coded: Option<i128>,
}
impl TrackClock {
    pub fn new(
        head: &HeaderInfo,
        delay_ns: Option<u64>,
        source_clock: TimeBase,
        track_scale: f64,
    ) -> Result<Self> {
        if track_scale != 1.0 {
            return Err(invalid(format!(
                "Matroska Opus TrackTimestampScale={track_scale} is unsupported; use exact unit track scale"
            )));
        }
        let delay_ns = delay_ns.ok_or_else(|| {
            invalid("Matroska Opus requires an explicit CodecDelay; repair the TrackEntry")
        })?;
        let delay = nanoseconds_to_frames(i128::from(delay_ns), "CodecDelay")?;
        if delay != i64::from(head.pre_skip) {
            return Err(invalid(format!(
                "Matroska Opus CodecDelay={delay} frames disagrees with OpusHead pre-skip {}; repair TrackEntry/CodecPrivate",
                head.pre_skip
            )));
        }
        Ok(Self {
            delay: u32::from(head.pre_skip),
            raw_cursor: 0,
            source_clock,
            terminal_padding: false,
            previous_coded: None,
        })
    }
    pub fn normalize(
        &mut self,
        coded_ticks: i128,
        info: PacketInfo,
        padding_ns: Option<i64>,
        block_duration: Option<u64>,
        default_ns: Option<u64>,
    ) -> Result<PacketTiming> {
        if self.terminal_padding {
            return Err(invalid(
                "Matroska Opus contains audio after end DiscardPadding; nonterminal end padding is unsupported in full-file presentation",
            ));
        }
        let numerator = coded_ticks
            .checked_mul(i128::from(self.source_clock.numer.get()))
            .and_then(|n| n.checked_mul(i128::from(DECODE_RATE)))
            .ok_or_else(|| invalid("Matroska Opus coded timestamp conversion overflows"))?;
        let denominator = i128::from(self.source_clock.denom.get());
        if numerator % denominator != 0 {
            return Err(invalid(
                "Matroska Opus coded timestamp is fractional in the 48 kHz clock; normalize its TimestampScale",
            ));
        }
        let coded = numerator / denominator;
        if self.previous_coded.is_none() && coded != 0 {
            return Err(invalid(
                "Matroska Opus nonzero coded origin is unsupported by this full-file profile; normalize the container origin without changing encoded audio",
            ));
        }
        if self.previous_coded.is_some_and(|previous| coded < previous) {
            return Err(invalid(
                "Matroska Opus coded timestamps move backwards; reordered audio is unsupported",
            ));
        }
        let raw = info.raw_frames() as u64;
        // BlockDuration/DefaultDuration describe container presentation timing,
        // not decoder output geometry. A coarse clock may legitimately round
        // them. Retain those original facts in diagnostics, but never use them
        // to crop PCM, inject silence, or override exact DiscardPadding.
        log::debug!(
            "Matroska Opus coded_ticks={coded_ticks}, source_clock={:?}, BlockDuration={block_duration:?}, DefaultDuration_ns={default_ns:?}, raw_frames={raw}",
            self.source_clock
        );
        let padding = nanoseconds_to_frames(i128::from(padding_ns.unwrap_or(0)), "DiscardPadding")?;
        if padding.unsigned_abs() > raw {
            return Err(invalid(
                "Matroska Opus DiscardPadding exceeds its raw block duration; repair the BlockGroup",
            ));
        }
        // Bound the first profile to independently testable prefix/suffix cuts.
        let priming = u64::from(self.delay)
            .saturating_sub(self.raw_cursor)
            .min(raw);
        let prefix = priming.max(if padding < 0 {
            padding.unsigned_abs()
        } else {
            0
        });
        let suffix = if padding > 0 { padding as u64 } else { 0 };
        let valid_end = raw.saturating_sub(suffix).max(prefix);
        let pts = i64::try_from(coded)
            .ok()
            .and_then(|n| n.checked_sub(i64::from(self.delay)))
            .ok_or_else(|| invalid("Matroska Opus raw packet PTS overflows"))?;
        self.raw_cursor = self
            .raw_cursor
            .checked_add(raw)
            .ok_or_else(|| invalid("Matroska Opus cumulative raw frames overflow"))?;
        self.terminal_padding = padding > 0;
        self.previous_coded = Some(coded);
        Ok(PacketTiming {
            pts,
            valid: valid_end - prefix,
            start: prefix,
            end: raw - valid_end,
        })
    }
    pub fn finish(&self) -> Result<()> {
        if self.raw_cursor < u64::from(self.delay) {
            return Err(invalid(
                "Matroska Opus ends before its declared CodecDelay; supply a complete recording",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn header(delay: u16) -> HeaderInfo {
        let mut data = b"OpusHead\x01\x02".to_vec();
        data.extend_from_slice(&delay.to_le_bytes());
        data.extend_from_slice(&48000_u32.to_le_bytes());
        data.extend_from_slice(&[0, 0, 0]);
        HeaderInfo::parse(&data).unwrap()
    }
    fn make_clock(delay: u16) -> TrackClock {
        TrackClock::new(
            &header(delay),
            Some(u64::from(delay) * 1_000_000_000 / 48000),
            TimeBase::new(NonZeroU32::new(1).unwrap(), NonZeroU32::new(1000).unwrap()),
            1.0,
        )
        .unwrap()
    }
    use std::num::NonZeroU32;
    #[test]
    fn rfc_geometry_includes_full_six_bit_count_and_bounds() {
        for count in 1..=48 {
            let info = packet_info(&[0x83, count]).unwrap();
            assert_eq!(info.raw_frames(), usize::from(count) * 120);
            assert_eq!(info.frame_count(), count);
        }
        for bytes in [
            &[][..],
            &[0x83][..],
            &[0x83, 0][..],
            &[0x83, 49][..],
            &[0x1b, 3][..],
        ] {
            assert!(packet_info(bytes).is_err(), "{bytes:?}");
        }
        assert_eq!(packet_info(&[0x98]).unwrap().raw_frames(), 960);
    }
    #[test]
    fn exact_submillisecond_delay_and_end_padding_have_one_owner() {
        let mut clock = make_clock(120);
        let packet = packet_info(&[0x98]).unwrap();
        let mut presented = 0;
        for index in 0..13 {
            let t = clock
                .normalize(
                    index * 20,
                    packet,
                    (index == 12).then_some(17_500_000),
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(t.pts, index as i64 * 960 - 120);
            assert_eq!(t.valid + t.start + t.end, 960);
            presented += t.valid;
            if index == 0 {
                assert_eq!((t.start, t.valid, t.end), (120, 840, 0));
            }
            if index == 12 {
                assert_eq!((t.start, t.valid, t.end), (0, 120, 840));
            }
        }
        assert_eq!(presented, 11520);
        clock.finish().unwrap();
        assert!(clock.normalize(260, packet, None, None, None).is_err());
    }
    #[test]
    fn priming_spans_packets_and_overlapping_padding_is_not_subtracted_twice() {
        let mut clock = make_clock(1200);
        let packet = packet_info(&[0x98]).unwrap();
        let first = clock
            .normalize(0, packet, Some(-10_000_000), None, None)
            .unwrap();
        assert_eq!((first.start, first.valid, first.end), (960, 0, 0));
        let second = clock
            .normalize(20, packet, Some(17_500_000), None, None)
            .unwrap();
        assert_eq!((second.start, second.valid, second.end), (240, 0, 720));
        clock.finish().unwrap();
        let mut clock = make_clock(120);
        let first = clock
            .normalize(0, packet, Some(-5_000_000), None, None)
            .unwrap();
        assert_eq!((first.start, first.valid, first.end), (240, 720, 0));
    }
    #[test]
    fn ambiguous_timing_metadata_is_rejected_instead_of_rounded() {
        let packet = packet_info(&[0x98]).unwrap();
        for (ticks, padding, duration, default) in [
            (1, None, None, None),
            (0, Some(1), None, None),
            (0, Some(30_000_000), None, None),
        ] {
            assert!(
                make_clock(120)
                    .normalize(ticks, packet, padding, duration, default)
                    .is_err()
            );
        }
        assert!(
            TrackClock::new(
                &header(120),
                None,
                TimeBase::new(NonZeroU32::new(1).unwrap(), NonZeroU32::new(1000).unwrap()),
                1.0
            )
            .is_err()
        );
        assert!(
            TrackClock::new(
                &header(120),
                Some(2_500_001),
                TimeBase::new(NonZeroU32::new(1).unwrap(), NonZeroU32::new(1000).unwrap()),
                1.0
            )
            .is_err()
        );
        assert!(make_clock(120).finish().is_err());
        let mut valid = make_clock(120);
        valid
            .normalize(0, packet, None, None, Some(20_000_000))
            .unwrap();
        let next = valid
            .normalize(21, packet, None, Some(3), Some(19_000_000))
            .unwrap();
        assert_eq!(next.pts, 888);
        assert_eq!(next.valid, 960);
        assert!(valid.normalize(20, packet, None, None, None).is_err());
    }
}
