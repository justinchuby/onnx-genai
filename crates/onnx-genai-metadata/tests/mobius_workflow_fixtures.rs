use std::path::{Path, PathBuf};

use onnx_genai_metadata::{parse_metadata, validate_metadata};

const EXPECTED_PACKAGES: [&str; 17] = [
    "adapter",
    "codec",
    "decoder",
    "diffusion",
    "diffusion_guided",
    "gemma4_chained",
    "gemma4_chained_mixed",
    "masked",
    "speculative",
    "speech_wav",
    "speech_wav_mixed_audio",
    "speech_wav_two_adapters",
    "speech_wav_two_audio",
    "static_cache",
    "tts",
    "video",
    "vlm",
];

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/onnx_genai_workflows")
}

#[test]
fn checked_in_mobius_workflow_packages_use_current_metadata_contracts() {
    let root = fixture_root();
    let mut actual = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("failed to enumerate {}: {error}", root.display()))
        .filter_map(|entry| {
            let entry = entry.expect("Mobius fixture directory entry is readable");
            let path = entry.path();
            path.join("inference_metadata.yaml")
                .is_file()
                .then(|| entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    actual.sort();
    assert_eq!(
        actual, EXPECTED_PACKAGES,
        "checked-in Mobius workflow package inventory changed; update the explicit inventory and \
         ensure every package is validated before the execution lane"
    );

    for package in EXPECTED_PACKAGES {
        let path = root.join(package).join("inference_metadata.yaml");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let metadata = parse_metadata(&source, Some("yaml")).unwrap_or_else(|error| {
            panic!(
                "{} does not match the current metadata schema: {error}. Remove retired tensor \
                 fields such as `rank`, emit explicit `shape` (`[]` for scalars and `Any` for \
                 independently unconstrained dimensions), then regenerate the fixture.",
                path.display()
            )
        });
        validate_metadata(&metadata).unwrap_or_else(|errors| {
            panic!(
                "{} fails current metadata semantics:\n{}",
                path.display(),
                errors.join("\n")
            )
        });
    }
}

#[test]
fn checked_in_package_guard_rejects_retired_tensor_rank() {
    let path = fixture_root()
        .join("adapter")
        .join("inference_metadata.yaml");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let stale = source.replacen(
        "          dtype: bool\n",
        "          dtype: bool\n          rank: 1\n",
        1,
    );
    let error = parse_metadata(&stale, Some("yaml"))
        .expect_err("the package guard must reject retired tensor rank");
    let message = error.to_string();
    assert!(
        message.contains("retired field `rank`")
            && message.contains("pipeline.workflow.inputs.request.active.contract"),
        "retired-rank rejection must identify the stale tensor contract: {message}"
    );
}
