use std::path::{Path, PathBuf};

use onnx_genai_genai_config::{
    GraphTensorInfo, ModelGraphInfo, PipelineGraphInfo, pipeline_inference_metadata_from_dir,
};
use onnx_genai_metadata::{PhaseRunOn, PipelineStrategyKind};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn tensor(name: impl Into<String>, dtype: &str, rank: usize) -> GraphTensorInfo {
    GraphTensorInfo {
        name: name.into(),
        dtype: dtype.to_owned(),
        dimensions: vec![None; rank],
    }
}

fn graph_inventory() -> PipelineGraphInfo {
    let sparse_layers = [3, 7, 11, 15, 19, 23, 27, 31];
    let fixed_layers = (0..32)
        .filter(|layer| !sparse_layers.contains(layer))
        .collect::<Vec<_>>();
    let mut decoder_inputs = vec![
        tensor("attention_mask", "int64", 2),
        tensor("inputs_embeds", "float32", 3),
        tensor("position_ids", "int64", 3),
    ];
    let mut decoder_outputs = vec![tensor("logits", "float32", 3)];
    for layer in sparse_layers {
        decoder_inputs.push(tensor(format!("past_key_values.{layer}.key"), "float32", 4));
        decoder_inputs.push(tensor(
            format!("past_key_values.{layer}.value"),
            "float32",
            4,
        ));
        decoder_outputs.push(tensor(format!("present.{layer}.key"), "float32", 4));
        decoder_outputs.push(tensor(format!("present.{layer}.value"), "float32", 4));
    }
    for layer in fixed_layers {
        for state in ["conv_state", "recurrent_state"] {
            let dimensions = if state == "conv_state" {
                vec![None, Some(16), Some(3)]
            } else {
                vec![None, Some(4), Some(8), Some(8)]
            };
            decoder_inputs.push(GraphTensorInfo {
                name: format!("past_key_values.{layer}.{state}"),
                dtype: "float32".to_owned(),
                dimensions: dimensions.clone(),
            });
            decoder_outputs.push(GraphTensorInfo {
                name: format!("present.{layer}.{state}"),
                dtype: "float32".to_owned(),
                dimensions,
            });
        }
    }

    PipelineGraphInfo {
        vision: ModelGraphInfo {
            inputs: vec![
                tensor("pixel_values", "float32", 2),
                tensor("image_grid_thw", "int64", 2),
            ],
            outputs: vec![tensor("image_features", "float32", 2)],
        },
        embedding: ModelGraphInfo {
            inputs: vec![
                tensor("input_ids", "int64", 2),
                tensor("image_features", "float32", 2),
            ],
            outputs: vec![tensor("inputs_embeds", "float32", 3)],
        },
        decoder: ModelGraphInfo {
            inputs: decoder_inputs,
            outputs: decoder_outputs,
        },
    }
}

#[test]
fn complete_config_synthesizes_typed_vlm_pipeline() {
    let metadata =
        pipeline_inference_metadata_from_dir(&fixture("vlm-complete"), &graph_inventory())
            .expect("complete compatibility package converts")
            .expect("compatibility metadata exists");
    let pipeline = metadata.pipeline.expect("pipeline");
    let preprocessing = metadata
        .preprocessing
        .and_then(|preprocessing| preprocessing.image)
        .expect("typed image preprocessing");
    let decoder_io = pipeline.models["decoder"].io.as_ref().expect("decoder io");

    assert_eq!(preprocessing.outputs.len(), 2);
    assert_eq!(preprocessing.outputs[0].name, "pixel_values");
    assert_eq!(preprocessing.outputs[0].dtype, "float32");
    assert_eq!(preprocessing.outputs[1].name, "image_grid_thw");
    assert_eq!(preprocessing.outputs[1].dtype, "int64");
    assert_eq!(pipeline.phases["embedding"].run_on, PhaseRunOn::EveryStep);
    assert_eq!(pipeline.dataflow.len(), 2);
    assert_eq!(pipeline.strategy.kind, PipelineStrategyKind::Composite);
    assert_eq!(pipeline.strategy.stages[1].name, "embed_tokens");
    assert_eq!(
        pipeline.strategy.stages[1].run_on,
        Some(PhaseRunOn::EveryStep)
    );
    let positions = pipeline.positions.as_ref().expect("position program");
    assert_eq!(positions.rank, 3);
    assert_eq!(positions.dtype.as_deref(), Some("int64"));
    assert_eq!(positions.sections.as_deref(), Some([11, 11, 10].as_slice()));
    assert_eq!(decoder_io.kv_inputs.as_ref().map(Vec::len), Some(16));
    assert_eq!(decoder_io.kv_outputs.as_ref().map(Vec::len), Some(16));
    assert_eq!(decoder_io.state_pairs.as_ref().map(Vec::len), Some(48));
}

#[test]
fn incomplete_config_fails_with_actionable_no_guess_error() {
    let error =
        pipeline_inference_metadata_from_dir(&fixture("vlm-incomplete"), &graph_inventory())
            .expect_err("missing position semantics must fail")
            .to_string();

    assert!(error.contains("missing required semantics"));
    assert!(error.contains("mrope_section"));
    assert!(error.contains("Why:"));
    assert!(error.contains("never guesses from model.type"));
    assert!(error.contains("How to fix:"));
    assert!(error.contains("native inference_metadata.json"));
}

#[test]
fn mixed_sparse_kv_dtypes_fail_loudly() {
    let mut graphs = graph_inventory();
    for tensor in graphs
        .decoder
        .inputs
        .iter_mut()
        .chain(graphs.decoder.outputs.iter_mut())
        .filter(|tensor| {
            matches!(
                tensor.name.as_str(),
                "past_key_values.31.key"
                    | "past_key_values.31.value"
                    | "present.31.key"
                    | "present.31.value"
            )
        })
    {
        tensor.dtype = "float16".to_owned();
    }

    let error = pipeline_inference_metadata_from_dir(&fixture("vlm-complete"), &graphs)
        .expect_err("mixed KV dtypes must fail")
        .to_string();

    assert!(error.contains("missing required semantics"));
    assert!(error.contains("one dtype"));
    assert!(error.contains("past_key_values.31.key"));
    assert!(error.contains("float16"));
}
