use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use onnx_genai_server::{AppState, app};
use serde_json::{Value, json};
use tower::ServiceExt;

static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new(max_context: usize) -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/vlm-image-bundle-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&root).expect("create fixture directory");
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm");
        for name in [
            "embedding.onnx.textproto",
            "decoder.onnx.textproto",
            "tokenizer.json",
        ] {
            fs::copy(source.join(name), root.join(name)).expect("copy shared fixture asset");
        }
        fs::write(
            root.join("vision_encoder.onnx.textproto"),
            VISION_MODEL_TEXTPROTO,
        )
        .expect("write generated vision model");
        fs::write(root.join("inference_metadata.yaml"), metadata(max_context))
            .expect("write fixture metadata");
        fs::write(
            root.join("chat_template.jinja"),
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .expect("write fixture chat template");
        Self(root)
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn chat(fixture: &FixtureDir, content: Value, max_tokens: usize) -> (StatusCode, Value) {
    let state =
        AppState::load(&fixture.0, Some("tiny-packed-vlm".to_string())).expect("load fixture");
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-packed-vlm",
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "logprobs": true,
                        "top_logprobs": 0
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("server response");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    (
        status,
        serde_json::from_slice(&body).expect("JSON response body"),
    )
}

fn image_part(uri: String) -> Value {
    json!({"type": "image_url", "image_url": {"url": uri}})
}

fn data_uri(color: [u8; 3]) -> String {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb(color)));
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("encode PNG");
    format!(
        "data:image/png;base64,{}",
        STANDARD.encode(bytes.into_inner())
    )
}

fn patterned_data_uri(red_pixel_index: usize) -> String {
    let image = RgbImage::from_fn(2, 2, |x, y| {
        let index = (y * 2 + x) as usize;
        Rgb(if index == red_pixel_index {
            [255, 0, 0]
        } else {
            [0, 0, 0]
        })
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("encode patterned PNG");
    format!(
        "data:image/png;base64,{}",
        STANDARD.encode(bytes.into_inner())
    )
}

#[tokio::test]
async fn public_chat_path_injects_both_vision_inputs_and_expands_placeholder() {
    let fixture = FixtureDir::new(64);
    let (status, body) = chat(
        &fixture,
        json!([
            {"type": "text", "text": "three <image>"},
            image_part(patterned_data_uri(1))
        ]),
        4,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body:#}");
    assert_eq!(body["choices"][0]["message"]["content"], "five six five");
    assert_eq!(body["usage"]["prompt_tokens"], 4);
}

#[tokio::test]
async fn two_images_follow_prompt_order() {
    let fixture = FixtureDir::new(64);
    let first_slot = patterned_data_uri(1);
    let last_slot = patterned_data_uri(2);
    let ordered = json!([
        {"type": "text", "text": "three <image> "},
        image_part(first_slot.clone()),
        {"type": "text", "text": "<image>"},
        image_part(last_slot.clone())
    ]);
    let reversed = json!([
        {"type": "text", "text": "three <image> "},
        image_part(last_slot),
        {"type": "text", "text": "<image>"},
        image_part(first_slot)
    ]);

    let (ordered_status, ordered_body) = chat(&fixture, ordered.clone(), 1).await;
    let (repeat_status, repeat_body) = chat(&fixture, ordered, 1).await;
    let (reversed_status, reversed_body) = chat(&fixture, reversed, 1).await;

    assert_eq!(ordered_status, StatusCode::OK, "{ordered_body:#}");
    assert_eq!(repeat_status, StatusCode::OK, "{repeat_body:#}");
    assert_eq!(reversed_status, StatusCode::OK, "{reversed_body:#}");
    assert_eq!(
        ordered_body["choices"][0]["message"]["content"],
        repeat_body["choices"][0]["message"]["content"]
    );
    assert_ne!(
        ordered_body["choices"][0]["message"]["content"],
        reversed_body["choices"][0]["message"]["content"],
        "swapping image content parts must change the order-sensitive vision result"
    );
}

#[tokio::test]
async fn missing_placeholder_has_what_why_how_error() {
    let fixture = FixtureDir::new(64);
    let (status, body) = chat(
        &fixture,
        json!([
            {"type": "text", "text": "three"},
            image_part(data_uri([255, 0, 0]))
        ]),
        1,
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_structured_error(&body, "placeholder");
}

#[tokio::test]
async fn wrong_image_count_has_what_why_how_error() {
    let fixture = FixtureDir::new(64);
    let (status, body) = chat(
        &fixture,
        json!([
            {"type": "text", "text": "three <image>"},
            image_part(data_uri([255, 0, 0])),
            image_part(data_uri([0, 0, 255]))
        ]),
        1,
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_structured_error(&body, "count mismatch");
}

#[tokio::test]
async fn expanded_context_overflow_is_rejected_before_driver_admission() {
    let fixture = FixtureDir::new(4);
    let (status, body) = chat(
        &fixture,
        json!([
            {"type": "text", "text": "three <image>"},
            image_part(data_uri([255, 0, 0]))
        ]),
        1,
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_structured_error(&body, "final prefill length");
}

fn assert_structured_error(body: &Value, expected: &str) {
    let message = body["error"]["message"]
        .as_str()
        .expect("error message string");
    assert!(message.contains("What:"), "{message}");
    assert!(message.contains("Why:"), "{message}");
    assert!(message.contains("How:"), "{message}");
    assert!(message.contains(expected), "{message}");
}

fn metadata(max_context: usize) -> String {
    format!(
        r#"schema_version: v1
model:
  max_sequence_length: {max_context}
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: 2
        mode: stretch
        interpolation: bicubic
      - op: rescale
        scale: 0.00392156862745098
      - op: patchify
        patch_size: 2
        flatten: true
    outputs:
      - name: vision_encoder.packed_pixels
        content: pixels
        dtype: float32
      - name: vision_encoder.patch_positions
        content: patch_coordinates
        dtype: int64
pipeline:
  models:
    vision_encoder:
      filename: vision_encoder.onnx.textproto
      type: vision_encoder
    embedding:
      filename: embedding.onnx.textproto
      type: encoder
      io:
        token_input: input_ids
    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
    - from: vision_encoder.image_features
      to: embedding.image_features
      dtype: fp32
      device_transfer: false
    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
      dtype: fp32
      device_transfer: false
  strategy:
    kind: composite
    stages:
      - name: encode_vision
        strategy:
          kind: single_pass
          model: vision_encoder
        run_on: prompt_only
      - name: fuse_embeddings
        strategy:
          kind: single_pass
          model: embedding
        run_on: every_step
      - name: decode
        strategy:
          kind: autoregressive
          decoder: decoder
          max_tokens: 4
        run_on: every_step
  phases:
    vision_encoder:
      run_on: prompt_only
    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
  vision:
    image_placeholder_token_id: 7
    image_token_id: 7
    token_count_source: per_patch
    tokens_per_patch: 3
    placeholder_per_image: true
    thumbnail_order: none
"#
    )
}

// Generated and ONNX-checked with onnxscript.ir (ir.Value/Node/Graph/Model + ir.to_proto).
const VISION_MODEL_TEXTPROTO: &str = r#"ir_version: 8
producer_name: "onnx-genai VLM WP5 fixture"
graph {
  node {
    output: "first_idx"
    name: "node_Constant_0"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        data_type: 7
        name: "first_idx_value"
        raw_data: "\000\000\000\000\000\000\000\000"
      }
      type: TENSOR
    }
  }
  node {
    output: "feature_idx"
    name: "node_Constant_1"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 4
        data_type: 7
        name: "feature_idx_value"
        raw_data: "\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000"
      }
      type: TENSOR
    }
  }
  node {
    output: "reduce_axes"
    name: "node_Constant_2"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 2
        data_type: 7
        name: "reduce_axes_value"
        raw_data: "\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000"
      }
      type: TENSOR
    }
  }
  node {
    output: "unsqueeze_axis"
    name: "node_Constant_3"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        data_type: 7
        name: "unsqueeze_axis_value"
        raw_data: "\001\000\000\000\000\000\000\000"
      }
      type: TENSOR
    }
  }
  node {
    output: "feature_scale"
    name: "node_Constant_4"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        data_type: 1
        name: "feature_scale_value"
        raw_data: "\000\000\000@"
      }
      type: TENSOR
    }
  }
  node {
    input: "packed_pixels"
    input: "first_idx"
    output: "first_patch"
    name: "node_Gather_5"
    op_type: "Gather"
    attribute {
      name: "axis"
      i: 0
      type: INT
    }
  }
  node {
    input: "first_patch"
    input: "feature_idx"
    output: "first_four"
    name: "node_Gather_6"
    op_type: "Gather"
    attribute {
      name: "axis"
      i: 1
      type: INT
    }
  }
  node {
    input: "patch_positions"
    output: "positions_f32"
    name: "node_Cast_7"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 1
      type: INT
    }
  }
  node {
    input: "positions_f32"
    input: "reduce_axes"
    output: "position_sum"
    name: "node_ReduceSum_8"
    op_type: "ReduceSum"
    attribute {
      name: "keepdims"
      i: 0
      type: INT
    }
  }
  node {
    input: "first_four"
    input: "feature_scale"
    output: "scaled_features"
    name: "node_Mul_9"
    op_type: "Mul"
  }
  node {
    input: "scaled_features"
    input: "position_sum"
    output: "combined"
    name: "node_Add_10"
    op_type: "Add"
  }
  node {
    input: "combined"
    input: "unsqueeze_axis"
    output: "image_features"
    name: "node_Unsqueeze_11"
    op_type: "Unsqueeze"
  }
  name: "tiny_packed_two_input_vision"
  input {
    name: "packed_pixels"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "patches"
          }
          dim {
            dim_value: 12
          }
        }
      }
    }
  }
  input {
    name: "patch_positions"
    type {
      tensor_type {
        elem_type: 7
        shape {
          dim {
            dim_param: "patches"
          }
          dim {
            dim_value: 2
          }
        }
      }
    }
  }
  output {
    name: "image_features"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 13
}
"#;
