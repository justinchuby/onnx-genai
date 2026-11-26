//! Checked-in independent libopus PCM references; no executable runtime dependency.
use onnx_genai_preprocess::audio::{
    AudioDecodeLimits, AudioProgram, AudioTensorData, decode_audio,
};
use serde_json::json;
use std::{num::NonZeroUsize, path::PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/opus")
}
fn limit(bytes: usize) -> AudioDecodeLimits {
    AudioDecodeLimits {
        max_decoded_bytes: NonZeroUsize::new(bytes).unwrap(),
    }
}

// Test-only mutation of a known fixture page, including its Ogg checksum.
// Container parsing remains exclusively Symphonia's responsibility in production.
fn rewrite_page(bytes: &mut [u8], offset: usize, change: impl FnOnce(&mut [u8], usize)) {
    assert_eq!(&bytes[offset..offset + 4], b"OggS");
    let segments = usize::from(bytes[offset + 26]);
    let body = offset + 27 + segments;
    let end = body
        + bytes[offset + 27..body]
            .iter()
            .map(|n| usize::from(*n))
            .sum::<usize>();
    change(&mut bytes[offset..end], body - offset);
    bytes[offset + 22..offset + 26].fill(0);
    let mut crc = 0_u32;
    for byte in &bytes[offset..end] {
        crc ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    bytes[offset + 22..offset + 26].copy_from_slice(&crc.to_le_bytes());
}

// Re-page the real encoded CELT fixture without modifying any Opus packets.
// The small fixture has no continued packets; production has no Ogg parser here.
fn two_audio_page_fixture(final_granule: u64) -> Vec<u8> {
    let source = std::fs::read(root().join("celt-1.ogg")).unwrap();
    let mut pages = Vec::new();
    let mut offset = 0;
    while offset < source.len() {
        assert_eq!(&source[offset..offset + 4], b"OggS");
        let count = usize::from(source[offset + 26]);
        let body = offset + 27 + count;
        let laces = &source[offset + 27..body];
        assert!(
            laces.iter().all(|length| *length < 255),
            "fixture has complete packets per lace"
        );
        let end = body
            + laces
                .iter()
                .map(|length| usize::from(*length))
                .sum::<usize>();
        let mut position = body;
        let packets = laces
            .iter()
            .map(|length| {
                let end = position + usize::from(*length);
                let packet = source[position..end].to_vec();
                position = end;
                packet
            })
            .collect::<Vec<_>>();
        pages.push((source[offset..end].to_vec(), packets));
        offset = end;
    }
    assert_eq!(pages.len(), 3, "OpusHead, OpusTags, one EOS audio page");
    let audio = &pages[2].1;
    assert_eq!(audio.len(), 13);
    let serial = &pages[0].0[14..18];
    let mut output = Vec::new();
    output.extend_from_slice(&pages[0].0);
    output.extend_from_slice(&pages[1].0);
    for (index, (packets, granule, flags)) in [
        (&audio[..6], 5760_u64, 0_u8),
        (&audio[6..], final_granule, 4_u8),
    ]
    .into_iter()
    .enumerate()
    {
        let mut page = vec![0_u8; 27 + packets.len()];
        page[..4].copy_from_slice(b"OggS");
        page[5] = flags;
        page[6..14].copy_from_slice(&granule.to_le_bytes());
        page[14..18].copy_from_slice(serial);
        page[18..22].copy_from_slice(&(index as u32 + 2).to_le_bytes());
        page[26] = packets.len() as u8;
        for (index, packet) in packets.iter().enumerate() {
            page[27 + index] = packet.len() as u8;
            page.extend_from_slice(packet);
        }
        rewrite_page(&mut page, 0, |_, _| {});
        // Independently recompute the serialized page's stored CRC, including
        // its final granule/sequence/lacing. Both cases are actual valid Ogg pages.
        let stored = u32::from_le_bytes(page[22..26].try_into().unwrap());
        let mut checksum = 0_u32;
        for (index, byte) in page.iter().enumerate() {
            let byte = if (22..26).contains(&index) { 0 } else { *byte };
            checksum ^= u32::from(byte) << 24;
            for _ in 0..8 {
                checksum = if checksum & 0x8000_0000 != 0 {
                    (checksum << 1) ^ 0x04c1_1db7
                } else {
                    checksum << 1
                };
            }
        }
        assert_eq!(stored, checksum);
        output.extend_from_slice(&page);
    }
    output
}

#[test]
fn ogg_opus_rejects_eos_granule_before_a_previous_audio_page() {
    let valid = two_audio_page_fixture(11640);
    let actual = decode_audio(&valid, limit(1024 * 1024)).unwrap();
    let original = decode_audio(
        &std::fs::read(root().join("celt-1.ogg")).unwrap(),
        limit(1024 * 1024),
    )
    .unwrap();
    assert_eq!(
        actual.samples, original.samples,
        "re-paging unchanged packets preserves all presented PCM"
    );
    assert_eq!(actual.samples.len(), 11520);
    // Earlier page granule=5760; final=5000 > pre-skip120. Merely cropping the
    // raw buffer to final-pre-skip would silently accept 4880 frames.
    let backward = two_audio_page_fixture(5000);
    let error = decode_audio(&backward, limit(1024 * 1024))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("final granule 5000 precedes") && error.contains("must not move backwards"),
        "{error}"
    );
}

#[test]
fn opus_preskip_spans_packets_gain_applies_once_and_end_granule_is_exact() {
    let original = std::fs::read(root().join("celt-1.ogg")).unwrap();
    let reference = std::fs::read(root().join("celt-1.reference.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut delayed = original.clone();
    rewrite_page(&mut delayed, 0, |page, body| {
        page[body + 10..body + 12].copy_from_slice(&2500_u16.to_le_bytes())
    });
    let actual = decode_audio(&delayed, limit(1024 * 1024)).unwrap();
    let expected = &reference[2500 - 120..];
    assert_eq!(actual.samples.len(), expected.len());
    assert!(
        actual
            .samples
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1.0 / 32768.0)
    );

    let mut attenuated = original.clone();
    rewrite_page(&mut attenuated, 0, |page, body| {
        page[body + 16..body + 18].copy_from_slice(&(-1536_i16).to_le_bytes())
    });
    let actual = decode_audio(&attenuated, limit(1024 * 1024)).unwrap();
    let gain = 10.0_f32.powf(-6.0 / 20.0);
    assert_eq!(actual.samples.len(), reference.len());
    assert!(
        actual
            .samples
            .iter()
            .zip(&reference)
            .all(|(a, b)| (a - b * gain).abs() <= 1.0 / 32768.0)
    );

    let mut trimmed = original.clone();
    let final_page = trimmed
        .windows(4)
        .rposition(|chunk| chunk == b"OggS")
        .unwrap();
    rewrite_page(&mut trimmed, final_page, |page, _| {
        assert_ne!(page[5] & 4, 0, "fixture EOS page");
        let end = u64::from_le_bytes(page[6..14].try_into().unwrap());
        page[6..14].copy_from_slice(&(end - 200).to_le_bytes());
    });
    let actual = decode_audio(&trimmed, limit(1024 * 1024)).unwrap();
    assert_eq!(actual.samples.len(), reference.len() - 200);
    assert!(
        actual
            .samples
            .iter()
            .zip(&reference)
            .all(|(a, b)| (a - b).abs() <= 1.0 / 32768.0)
    );

    let mut invalid = original;
    rewrite_page(&mut invalid, 0, |page, body| {
        page[body + 10..body + 12].copy_from_slice(&u16::MAX.to_le_bytes())
    });
    assert!(
        decode_audio(&invalid, limit(1024 * 1024))
            .unwrap_err()
            .to_string()
            .contains("final granule precedes its pre-skip")
    );
}

#[test]
fn opus_silk_hybrid_celt_and_surround_match_independent_presented_pcm() {
    let root = root();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let channels = case["channels"].as_u64().unwrap() as usize;
        for suffix in ["ogg", "webm"] {
            let bytes = std::fs::read(root.join(format!("{name}.{suffix}"))).unwrap();
            let reference = std::fs::read(root.join(format!("{name}.reference.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let actual = decode_audio(&bytes, limit(1024 * 1024))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(actual.sample_rate, 48_000, "{name}");
            assert_eq!(
                actual.samples.len(),
                11_520,
                "{name}: pre-skip/end trim must each apply once"
            );
            assert_eq!(reference.len(), actual.samples.len() * channels, "{name}");
            let maximum = actual
                .samples
                .iter()
                .zip(reference.chunks_exact(channels))
                .map(|(actual, frame)| {
                    let expected = (frame.iter().map(|sample| f64::from(*sample)).sum::<f64>()
                        / channels as f64) as f32;
                    assert!(actual.is_finite());
                    (actual - expected).abs()
                })
                .fold(0.0_f32, f32::max);
            assert!(
                maximum <= 1.0 / 32768.0,
                "{name}: independent PCM maximum error {maximum}"
            );
        }
    }
}

#[test]
fn ogg_opus_program_emits_presented_waveform_and_count() {
    let bytes = std::fs::read(root().join("hybrid-2.ogg")).unwrap();
    let declaration = serde_json::from_value(json!({
        "transforms":[{"op":"decode","outputs":["wave"]},{"op":"emit_valid_samples","outputs":["count"]}],
        "outputs":[{"name":"wave","source":"wave","content":"waveform","dtype":"float32"},{"name":"count","source":"count","content":"valid_samples","dtype":"int64"}]
    })).unwrap();
    let program = AudioProgram::from_program(&declaration).unwrap();
    let tensors = program.run(&[&bytes], limit(1024 * 1024)).unwrap();
    assert_eq!(tensors[0].shape, [1, 11520]);
    let AudioTensorData::Int64(count) = &tensors[1].data else {
        panic!("count dtype")
    };
    assert_eq!(count, &[11520]);
    assert!(program.run(&[&bytes], limit(200_000)).is_ok());
    let error = program
        .run(&[&bytes, &bytes], limit(200_000))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("max_decoded_bytes=100000"),
        "batch invocations must share the caller's raw/resident PCM budget: {error}"
    );
}

#[test]
fn ogg_opus_budget_bounds_initial_packet_raw_work_and_host_headroom() {
    let bytes = std::fs::read(root().join("celt-2.ogg")).unwrap();
    let initial = decode_audio(&bytes, limit(5760 * 2 * 4 - 1))
        .unwrap_err()
        .to_string();
    assert!(initial.contains("max_decoded_bytes=46079"), "{initial}");
    // 13 packets * 960 frames * 2ch * 4 exceeds this budget even though
    // presented mono output is only 46080 bytes. Trim never refunds raw work.
    let accumulated = decode_audio(&bytes, limit(90_000)).unwrap_err().to_string();
    assert!(
        accumulated.contains("max_decoded_bytes=90000"),
        "{accumulated}"
    );
    assert!(
        limit(1024)
            .within_host(0)
            .unwrap_err()
            .to_string()
            .contains("zero host-memory headroom")
    );
    assert_eq!(
        limit(1024)
            .within_host(u64::MAX)
            .unwrap()
            .max_decoded_bytes
            .get(),
        1024
    );
}

#[test]
fn opus_other_containers_chaining_and_truncated_end_fail_explicitly() {
    {
        let bytes = std::fs::read(root().join("unsupported-opus.mp4")).unwrap();
        let error = decode_audio(&bytes, limit(1024 * 1024))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Opus presentation in") && error.contains("unsupported by this build"),
            "mp4: {error}"
        );
    }
    let quantized = std::fs::read(root().join("unsupported-opus.webm")).unwrap();
    let actual = decode_audio(&quantized, limit(1024 * 1024)).unwrap();
    let reference = decode_audio(
        &std::fs::read(root().join("celt-2.ogg")).unwrap(),
        limit(1024 * 1024),
    )
    .unwrap();
    assert_eq!(
        actual.samples, reference.samples,
        "container PTS quantization must not crop or fill PCM"
    );
    for name in ["celt-1", "mapping1-6"] {
        let bytes = std::fs::read(root().join(format!("{name}.webm"))).unwrap();
        assert!(
            decode_audio(&bytes, limit(1024))
                .unwrap_err()
                .to_string()
                .contains("max_decoded_bytes=1024")
        );
    }
    let bytes = std::fs::read(root().join("celt-1.ogg")).unwrap();
    let mut chained = bytes.clone();
    chained.extend_from_slice(&bytes);
    assert!(
        decode_audio(&chained, limit(1024 * 1024)).is_err(),
        "chained logical stream must not silently lose its second recording"
    );
    assert!(
        decode_audio(&bytes[..bytes.len() - 100], limit(1024 * 1024)).is_err(),
        "missing EOS/granule must not invent trailing presentation"
    );
}
