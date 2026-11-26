//! Authored EBML fixtures exercise the actual reader, not a parallel production parser.
use onnx_genai_symphonia_mkv::MkvReader;
use symphonia_core::{
    formats::{FormatOptions, FormatReader, SeekMode, SeekTo, TrackType},
    io::MediaSourceStream,
    units::Timestamp,
};
fn element(id: u32, body: impl AsRef<[u8]>) -> Vec<u8> {
    let body = body.as_ref();
    let id = id.to_be_bytes();
    let first = id.iter().position(|b| *b != 0).unwrap();
    let mut out = id[first..].to_vec();
    let length = body.len() as u64;
    let width = (1..=8).find(|n| length < (1_u64 << (7 * n)) - 1).unwrap();
    let size = (length | (1_u64 << (7 * width))).to_be_bytes();
    out.extend_from_slice(&size[8 - width..]);
    out.extend_from_slice(body);
    out
}
fn unsigned(id: u32, value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|v| *v != 0).unwrap_or(7);
    element(id, &bytes[first..])
}
fn head() -> Vec<u8> {
    let mut h = b"OpusHead\x01\x02".to_vec();
    h.extend_from_slice(&120_u16.to_le_bytes());
    h.extend_from_slice(&48000_u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]);
    h
}
fn track(id: u64, opus: bool, default: bool, delay: Option<u64>, enabled: bool) -> Vec<u8> {
    let mut body = unsigned(0xd7, id);
    body.extend(unsigned(0x73c5, id));
    body.extend(unsigned(0x83, 2));
    body.extend(unsigned(0x88, u64::from(default)));
    body.extend(unsigned(0xb9, u64::from(enabled)));
    body.extend(element(
        0x86,
        if opus {
            b"A_OPUS".as_slice()
        } else {
            b"A_PCM/INT/LIT".as_slice()
        },
    ));
    if opus {
        body.extend(element(0x63a2, head()));
        if let Some(delay) = delay {
            body.extend(unsigned(0x56aa, delay));
        }
    }
    let mut audio = element(0xb5, 48000_f64.to_be_bytes());
    audio.extend(unsigned(0x9f, 2));
    audio.extend(unsigned(0x6264, 16));
    body.extend(element(0xe1, audio));
    element(0xae, body)
}
fn block(
    id: u8,
    ticks: i16,
    data: &[u8],
    padding: Option<i64>,
    flags: u8,
    codec_state: bool,
) -> Vec<u8> {
    let mut body = vec![0x80 | id];
    body.extend_from_slice(&ticks.to_be_bytes());
    body.push(flags);
    body.extend_from_slice(data);
    if padding.is_none() && !codec_state {
        return element(0xa3, body);
    }
    let mut group = element(0xa1, body);
    if let Some(pad) = padding {
        group.extend(element(0x75a2, pad.to_be_bytes()));
    }
    if codec_state {
        group.extend(element(0xa4, [1]));
    }
    element(0xa0, group)
}
fn file(tracks: Vec<Vec<u8>>, blocks: Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = element(0x1a45dfa3, element(0x4282, b"matroska"));
    let mut info = element(0x4d80, b"nxrt test");
    info.extend(element(0x5741, b"nxrt test"));
    info.extend(unsigned(0x2ad7b1, 1_000_000));
    info.extend(element(0x4489, 260_f64.to_be_bytes()));
    let mut segment = element(0x1549a966, info);
    segment.extend(element(0x1654ae6b, tracks.concat()));
    let mut cluster = unsigned(0xe7, 0);
    cluster.extend(blocks.concat());
    segment.extend(element(0x1f43b675, cluster));
    out.extend(element(0x18538067, segment));
    out
}
fn reader(bytes: Vec<u8>) -> symphonia_core::errors::Result<MkvReader<'static>> {
    MkvReader::try_new(
        MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default()),
        FormatOptions::default(),
    )
}
#[test]
fn real_reader_normalizes_trim_identity_without_forging_segment_duration() {
    let bytes = file(
        vec![track(1, true, true, Some(2_500_000), true)],
        vec![
            block(1, 0, &[0x98], None, 0x80, false),
            block(1, 20, &[0x98], Some(17_500_000), 0, false),
        ],
    );
    let mut reader = reader(bytes).unwrap();
    let track = reader.default_track(TrackType::Audio).unwrap();
    assert_eq!(track.time_base.unwrap().denom.get(), 48000);
    assert_eq!(track.delay, Some(120));
    assert_eq!(track.duration, None);
    assert_eq!(track.num_frames, None);
    assert_eq!(track.padding, None);
    assert_eq!(track.start_ts, Timestamp::new(0));
    for (pts, start, duration, end) in [(-120, 120, 840, 0), (840, 0, 120, 840)] {
        let p = reader.next_packet().unwrap().unwrap();
        assert_eq!(p.pts.get(), pts);
        assert_eq!(p.dts, p.pts);
        assert_eq!(
            (p.trim_start.get(), p.dur.get(), p.trim_end.get()),
            (start, duration, end)
        );
        assert_eq!(start + duration + end, 960);
    }
    assert!(reader.next_packet().unwrap().is_none());
    assert!(
        reader
            .seek(
                SeekMode::Accurate,
                SeekTo::Timestamp {
                    ts: Timestamp::new(0),
                    track_id: 1
                }
            )
            .unwrap_err()
            .to_string()
            .contains("full-file")
    );
}
#[test]
fn unused_unsupported_opus_track_preserves_legacy_selection_and_packets() {
    for enabled in [false, true] {
        let bytes = file(
            vec![
                track(1, false, true, None, true),
                track(2, true, false, None, enabled),
            ],
            vec![
                block(1, 0, &[1, 2, 3, 4], None, 0x80, false),
                block(2, 0, &[0x98], None, 0x80, false),
            ],
        );
        let mut reader = reader(bytes).unwrap();
        assert_eq!(reader.tracks().len(), 2);
        assert_eq!(reader.default_track(TrackType::Audio).unwrap().id, 1);
        let opus = &reader.tracks()[1];
        assert_eq!(opus.delay, None);
        assert_eq!(opus.time_base.unwrap().denom.get(), 1000);
        let first = reader.next_packet().unwrap().unwrap();
        assert_eq!(first.track_id, 1);
        assert_eq!(first.data.as_ref(), [1, 2, 3, 4]);
        let unused = reader.next_packet().unwrap().unwrap();
        assert_eq!(unused.track_id, 2);
        assert_eq!(unused.trim_start.get(), 0);
        assert_eq!(unused.trim_end.get(), 0);
        assert!(reader.next_packet().unwrap().is_none());
    }
    let bytes = file(
        vec![
            track(1, true, true, None, true),
            track(2, false, false, None, true),
        ],
        vec![],
    );
    assert!(
        reader(bytes)
            .err()
            .unwrap()
            .to_string()
            .contains("explicit CodecDelay"),
        "selected unsupported Opus cannot fall back to the legacy second track"
    );
}
#[test]
fn real_reader_rejects_unsupported_timeline_lacing_and_codec_state() {
    for (ticks, padding, flags, state, context) in [
        (1, None, 0x80, false, "nonzero coded origin"),
        (0, Some(1), 0, false, "fractional-frame"),
        (0, None, 0x82, false, "lacing"),
        (0, None, 0, true, "CodecState"),
    ] {
        let mut reader = reader(file(
            vec![track(1, true, true, Some(2_500_000), true)],
            vec![block(1, ticks, &[0x98], padding, flags, state)],
        ))
        .unwrap();
        assert!(
            reader
                .next_packet()
                .err()
                .expect("unsupported packet must fail")
                .to_string()
                .contains(context),
            "{context}"
        );
    }
}

#[test]
fn actual_ffmpeg_quantized_pts_and_block_duration_do_not_choose_pcm_cuts() {
    let mut reader = reader(include_bytes!("fixtures/celt-2.webm").to_vec()).unwrap();
    let mut sum = 0;
    let mut packets = 0;
    while let Some(packet) = reader.next_packet().unwrap() {
        if packets == 0 {
            assert_eq!(packet.pts.get(), -120);
            assert_eq!(packet.trim_start.get(), 120);
            assert_eq!(packet.dur.get(), 840);
        }
        if packets == 1 {
            assert_eq!(
                packet.pts.get(),
                888,
                "preserve actual21ms coded PTS, not forced raw cursor840"
            );
        }
        if packets == 12 {
            assert_eq!(packet.pts.get(), 11448);
            assert_eq!(packet.dur.get(), 120);
            assert_eq!(packet.trim_end.get(), 840);
        }
        assert_eq!(
            packet.dur.get() + packet.trim_start.get() + packet.trim_end.get(),
            960
        );
        sum += packet.dur.get();
        packets += 1;
    }
    assert_eq!(packets, 13);
    assert_eq!(sum, 11520);
}
