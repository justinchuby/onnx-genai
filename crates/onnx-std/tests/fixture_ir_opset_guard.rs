//! Repo-wide guard: every maintained checked-in ONNX textproto fixture must
//! declare ONNX IR version >= 11 and a default (`ai.onnx`) opset >= 24.
//!
//! Fixtures are committed as protobuf TextFormat (`*.onnx.textproto`) and loaded
//! by the runtime via `onnx_std::textproto::to_binary`. Keeping them on a modern
//! IR/opset floor ensures the exported models exercise the same schema surface
//! the runtime targets in production, and prevents new fixtures from silently
//! regressing to legacy IR 8 / opset 13.
//!
//! Fixtures that *intentionally* exercise legacy IR/opset compatibility, or that
//! are non-executable package-structure placeholders, are exempt via the
//! explicit [`ALLOWLIST`] below. Every exemption carries a reason. Adding a new
//! fixture below the floor without an allowlist entry fails this test.

use std::path::{Path, PathBuf};
use std::process::Command;

use onnx_runtime_loader::proto::decode_model;

/// Minimum ONNX IR version for maintained fixtures (IR 11, ONNX 1.18 / 2025-05).
const MIN_IR_VERSION: i64 = 11;
/// Minimum default-domain ONNX opset for maintained fixtures (opset 24).
const MIN_DEFAULT_OPSET: i64 = 24;

/// Fixtures deliberately kept below the modern IR/opset floor. Each entry is a
/// repo-root-relative path paired with the reason its legacy state is intended.
///
/// The names are already descriptive of their legacy/placeholder intent; the
/// reason string documents *why* an upgrade is intentionally withheld so the
/// exemption is reviewable and does not silently mask a regression.
const ALLOWLIST: &[(&str, &str)] = &[
    (
        "crates/onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto",
        "legacy imported BERT model (IR 4 / opset 12): the onnx-runtime-session \
         executor dispatches hand-written kernels by opset and the committed ORT \
         reference outputs were computed against this exact model; upgrading would \
         desync kernel dispatch and require regenerating the golden references.",
    ),
    (
        "crates/onnx-model-package/tests/fixtures/valid-package/cpu-fp16/model.onnx.textproto",
        "non-executable package-structure placeholder (ir_version: 0); model bytes \
         are never parsed — only ModelPackage manifest/variant selection is tested.",
    ),
    (
        "crates/onnx-model-package/tests/fixtures/valid-package/cpu-fp32/model.onnx.textproto",
        "non-executable package-structure placeholder (ir_version: 0); model bytes \
         are never parsed — only ModelPackage manifest/variant selection is tested.",
    ),
    (
        "crates/onnx-model-package/tests/fixtures/valid-package/cuda-fp16/model.onnx.textproto",
        "non-executable package-structure placeholder (ir_version: 0); model bytes \
         are never parsed — only ModelPackage manifest/variant selection is tested.",
    ),
    // ── legacy opset node form: `axes` attribute (pre-migration) ──────────────
    // These models use ops whose `axes` moved from an attribute to an input in a
    // later opset (ReduceMean/ReduceSum: opset 18/13; Unsqueeze: opset 13). A
    // *valid* opset-24 model must pass `axes` as an input, so a version bump
    // alone yields an invalid graph (ORT: "Unrecognized attribute: axes"). They
    // are pinned to their original opset until the node form is migrated.
    (
        "tests/fixtures/onnx_genai_workflows/speech_wav/vocoder/model.onnx.textproto",
        "opset-13 TTS vocoder using ReduceMean with the legacy `axes` attribute \
         (moved to an input in opset 18); a valid opset-24 upgrade requires \
         migrating `axes` to an input.",
    ),
    (
        "tests/fixtures/onnx_genai_workflows/speech_wav_mixed_audio/vocoder/model.onnx.textproto",
        "opset-13 TTS vocoder using ReduceMean with the legacy `axes` attribute \
         (moved to an input in opset 18); a valid opset-24 upgrade requires \
         migrating `axes` to an input.",
    ),
    (
        "tests/fixtures/onnx_genai_workflows/speech_wav_two_adapters/vocoder/model.onnx.textproto",
        "opset-13 TTS vocoder using ReduceMean with the legacy `axes` attribute \
         (moved to an input in opset 18); a valid opset-24 upgrade requires \
         migrating `axes` to an input.",
    ),
    (
        "tests/fixtures/onnx_genai_workflows/speech_wav_two_audio/vocoder/model.onnx.textproto",
        "opset-13 TTS vocoder using ReduceMean with the legacy `axes` attribute \
         (moved to an input in opset 18); a valid opset-24 upgrade requires \
         migrating `axes` to an input.",
    ),
    (
        "tests/fixtures/tiny-multiaxis-state-decoder/decoder.onnx.textproto",
        "opset-12 hybrid-recurrent decoder using ReduceSum with the legacy `axes` \
         attribute (moved to an input in opset 13); a valid opset-24 upgrade \
         requires migrating `axes` to an input.",
    ),
    (
        "tests/fixtures/tiny-native-engine/model.onnx.textproto",
        "opset-11 native-runtime decoder using Unsqueeze with the legacy `axes` \
         attribute (moved to an input in opset 13); consumed only by the native \
         engine, which handles this node form directly. A standards-compliant \
         opset-24 upgrade requires attr->input migration the native decoder path \
         does not consume.",
    ),
    (
        "tests/fixtures/tiny-native-scalar-gqa/model.onnx.textproto",
        "opset-11 native-runtime scalar-GQA fixture using ReduceSum with the legacy \
         `axes` attribute (moved to an input in opset 13); consumed only by the \
         native engine. A standards-compliant opset-24 upgrade requires attr->input \
         migration the native decoder path does not consume.",
    ),
    (
        "tests/fixtures/tiny-native-sub4-engine/model.onnx.textproto",
        "opset-11 native-runtime sub-4-bit fixture using Unsqueeze with the legacy \
         `axes` attribute (moved to an input in opset 13); consumed only by the \
         native engine. A standards-compliant opset-24 upgrade requires attr->input \
         migration the native decoder path does not consume.",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tracked_textproto_fixtures(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .args(["ls-files", "*.textproto"])
        .current_dir(root)
        .output()
        .expect("git must be available to enumerate tracked fixtures");
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("git paths are UTF-8")
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Parse only the model header (`ir_version` + `opset_import`) from a committed
/// textproto. Uses the loader-compatible textproto->binary path so it never
/// resolves external weights or builds the runtime graph, and therefore works
/// for every fixture (including external-weight and custom-domain models).
fn header(path: &Path) -> (i64, Option<i64>) {
    let text = std::fs::read_to_string(path).expect("fixture is readable");
    let bytes = onnx_std::textproto::to_binary(&text)
        .unwrap_or_else(|error| panic!("{}: textproto parse failed: {error}", path.display()));
    let proto = decode_model(&bytes)
        .unwrap_or_else(|error| panic!("{}: ModelProto decode failed: {error}", path.display()));
    let default_opset = proto
        .opset_import
        .iter()
        .find(|import| import.domain.is_empty() || import.domain == "ai.onnx")
        .map(|import| import.version);
    (proto.ir_version, default_opset)
}

#[test]
fn maintained_fixtures_meet_ir_and_opset_floor() {
    let root = repo_root();

    // Guard against a stale allowlist: every exempted path must still exist.
    let mut stale = Vec::new();
    for (path, _reason) in ALLOWLIST {
        if !root.join(path).is_file() {
            stale.push(*path);
        }
    }
    assert!(
        stale.is_empty(),
        "stale IR/opset allowlist entries (file no longer tracked — remove the \
         exemption): {}",
        stale.join(", ")
    );

    let allow: std::collections::HashSet<&str> = ALLOWLIST.iter().map(|(path, _)| *path).collect();

    let mut violations = Vec::new();
    for rel in tracked_textproto_fixtures(&root) {
        if allow.contains(rel.as_str()) {
            continue;
        }
        let (ir_version, default_opset) = header(&root.join(&rel));
        let mut problems = Vec::new();
        if ir_version < MIN_IR_VERSION {
            problems.push(format!("ir_version {ir_version} < {MIN_IR_VERSION}"));
        }
        match default_opset {
            Some(opset) if opset < MIN_DEFAULT_OPSET => {
                problems.push(format!("default opset {opset} < {MIN_DEFAULT_OPSET}"));
            }
            None => problems.push("no default (ai.onnx) opset import".to_string()),
            Some(_) => {}
        }
        if !problems.is_empty() {
            violations.push(format!("  {rel}: {}", problems.join(", ")));
        }
    }

    assert!(
        violations.is_empty(),
        "maintained ONNX textproto fixtures must declare IR >= {MIN_IR_VERSION} and \
         default opset >= {MIN_DEFAULT_OPSET}. Regenerate them (see the fixture \
         generators / scripts/upgrade path) or, if a fixture intentionally exercises \
         legacy IR/opset compatibility, add it to ALLOWLIST with a reason.\n{}",
        violations.join("\n")
    );
}
