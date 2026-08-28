use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};

#[derive(Clone, Copy)]
struct Geometry {
    name: &'static str,
    vocab_size: u64,
    ngram_size: usize,
    heads_per_ngram: usize,
    hc_count: usize,
    hidden_size: usize,
    conv_kernel: usize,
    conv_dilation: usize,
    table_base: usize,
    seed: u64,
    eos_token_id: i64,
}

impl Geometry {
    fn context_len(self) -> usize {
        self.ngram_size - 1
    }

    fn ngram_heads(self) -> usize {
        self.context_len() * self.heads_per_ngram
    }

    fn channels(self) -> usize {
        self.hc_count * self.hidden_size
    }

    fn conv_history_len(self) -> usize {
        (self.conv_kernel - 1) * self.conv_dilation
    }
}

// Qwen/Qwen3.8-Flash-Next's published Qwen4Exp config uses vocabulary 248320,
// ngram_size 3, eight heads per n-gram order, four hyper-connection streams,
// PLE at layer id 2, and a four-tap convolution dilated by ngram_size. Only the
// learned table sizes and hidden width are reduced here; indexing, projection,
// gating, convolution, recurrence, and residual injection stay structural.
const QWEN_REDUCED: Geometry = Geometry {
    name: "qwen-reduced",
    vocab_size: 248_320,
    ngram_size: 3,
    heads_per_ngram: 8,
    hc_count: 4,
    hidden_size: 2,
    conv_kernel: 4,
    conv_dilation: 3,
    table_base: 31,
    seed: 1234,
    eos_token_id: 248_044,
};

const ALTERNATE: Geometry = Geometry {
    name: "alternate-geometry",
    vocab_size: 101,
    ngram_size: 4,
    heads_per_ngram: 2,
    hc_count: 2,
    hidden_size: 3,
    conv_kernel: 3,
    conv_dilation: 2,
    table_base: 17,
    seed: 29,
    eos_token_id: 100,
};

fn package(geometry: Geometry) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/token-context")
        .join(geometry.name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata(geometry))?;
    fs::write(root.join("token-context.onnx.textproto"), model(geometry))?;
    Ok(root)
}

fn forged_position_package() -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/token-context")
        .join("forged-position-source");
    fs::create_dir_all(&root)?;
    let mut metadata = metadata(QWEN_REDUCED);
    metadata = metadata.replace(
        "    components:\n      token_context:",
        r#"    components:
      position_source:
        implementation: { kind: onnx, artifact: position-source.onnx.textproto }
        ports:
          inputs:
            value:
              dtype: int64
              shape: [batch, sequence]
              batch_layout: { kind: request_aligned, axis: 0 }
          outputs:
            position_ids:
              dtype: int64
              shape: [batch, sequence]
              batch_layout: { kind: request_aligned, axis: 0 }
          roles: { position_ids: position_ids }
      token_context:"#,
    );
    metadata = metadata.replace(
        "    steps:\n      - kind: invoke\n        component: token_context",
        r#"    steps:
      - kind: invoke
        component: position_source
        inputs: { value: token_ids }
        outputs: { position_ids: derived.position_ids }
      - kind: invoke
        component: token_context"#,
    );
    metadata = metadata.replace(
        "          token_ids: token_ids\n          inputs_embeds:",
        "          token_ids: derived.position_ids\n          inputs_embeds:",
    );
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    fs::write(
        root.join("token-context.onnx.textproto"),
        model(QWEN_REDUCED),
    )?;
    fs::write(
        root.join("position-source.onnx.textproto"),
        r#"
ir_version: 8
graph {
  name: "position_source"
  node { input: "value" output: "position_ids" op_type: "Identity" }
  input {
    name: "value"
    type { tensor_type { elem_type: 7 shape {
      dim { dim_param: "batch" } dim { dim_param: "sequence" }
    } } }
  }
  output {
    name: "position_ids"
    type { tensor_type { elem_type: 7 shape {
      dim { dim_param: "batch" } dim { dim_param: "sequence" }
    } } }
  }
}
opset_import { version: 18 }
"#,
    )?;
    Ok(root)
}

fn metadata(geometry: Geometry) -> String {
    format!(
        r#"
schema_version: v1.4
pipeline:
  workflow:
    manifest:
      capabilities:
        [workflow_ssa, typed_emit, serving_service_contract, input_presence,
         session_state_lease, token_context]
    inputs:
      token_ids:
        contract:
          dtype: int64
          shape: [batch, sequence]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: application, name: token_ids }}
        required: true
      inputs_embeds:
        contract:
          dtype: float32
          shape: [batch, sequence, {channels}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: inputs_embeds }}
        required: true
      initial_token_history:
        contract:
          dtype: int64
          shape: [batch, {context_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: initial_token_history }}
        required: true
      initial_conv_history:
        contract:
          dtype: float32
          shape: [batch, {channels}, {conv_history_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: initial_conv_history }}
        required: true
      active:
        contract:
          dtype: bool
          shape: [batch]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: active }}
        required: true
      done:
        contract:
          dtype: bool
          shape: [batch]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: done }}
        required: true
      accepted_len:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: accepted_len }}
        required: true
      row_selection:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: {{ kind: runtime, version: v1, role: row_selection }}
        source: {{ kind: application, name: row_selection }}
        required: false
        present_as: has_row_selection
    outputs:
      hidden_states:
        contract:
          dtype: float32
          shape: [batch, sequence, {channels}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: tensor
        stage: pre_adapter
      token_history:
        contract:
          dtype: int64
          shape: [batch, {context_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: tensor
        stage: pre_adapter
      conv_history:
        contract:
          dtype: float32
          shape: [batch, {channels}, {conv_history_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        role: tensor
        stage: pre_adapter
    components:
      token_context:
        implementation: {{ kind: onnx, artifact: token-context.onnx.textproto }}
        ports:
          inputs:
            token_ids:
              dtype: int64
              shape: [batch, sequence]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            inputs_embeds:
              dtype: float32
              shape: [batch, sequence, {channels}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            token_history:
              dtype: int64
              shape: [batch, {context_len}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            conv_history:
              dtype: float32
              shape: [batch, {channels}, {conv_history_len}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
          outputs:
            output:
              dtype: float32
              shape: [batch, sequence, {channels}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            next_token_history:
              dtype: int64
              shape: [batch, {context_len}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            next_conv_history:
              dtype: float32
              shape: [batch, {channels}, {conv_history_len}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
          roles:
            token_ids: token_ids
            inputs_embeds: inputs_embeds
        contract: {{ id: onnx-genai.token-context, version: "1" }}
        batch_capacity: {{ uniform_dimensions: [sequence] }}
    state:
      token_history:
        contract:
          dtype: int64
          shape: [batch, {context_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        class: semantic
        scope: session
        initializer: initial_token_history
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: session
        service_group: token_history
        session: {{ policy: exclusive }}
      conv_history:
        contract:
          dtype: float32
          shape: [batch, {channels}, {conv_history_len}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
        class: semantic
        scope: session
        initializer: initial_conv_history
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: session
        service_group: conv_history
        session: {{ policy: exclusive }}
    steps:
      - kind: invoke
        component: token_context
        inputs:
          token_ids: token_ids
          inputs_embeds: inputs_embeds
          token_history: initial_token_history
          conv_history: initial_conv_history
        outputs:
          output: context.output
          next_token_history: context.next_tokens
          next_conv_history: context.next_conv
      - {{ kind: emit, value: context.output, output: hidden_states, mode: replace }}
      - {{ kind: emit, value: context.next_tokens, output: token_history, mode: replace }}
      - {{ kind: emit, value: context.next_conv, output: conv_history, mode: replace }}
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          token_history:
            kind: recurrent
            layout: bt
            update: {{ kind: replace }}
            capabilities: {{ snapshot: true, fork: true }}
            checkpoint: {{ adapter: onnx-genai.tensor-checkpoint, version: "1" }}
            ports:
              token_context:
                token_history:
                  input: token_history
                  output: next_token_history
          conv_history:
            kind: recurrent
            layout: bch
            update: {{ kind: replace }}
            capabilities: {{ snapshot: true, fork: true }}
            checkpoint: {{ adapter: onnx-genai.tensor-checkpoint, version: "1" }}
            ports:
              token_context:
                conv_history:
                  input: conv_history
                  output: next_conv_history
"#,
        channels = geometry.channels(),
        context_len = geometry.context_len(),
        conv_history_len = geometry.conv_history_len(),
    )
}

fn initializer_i64(name: &str, dims: &[usize], values: &[i64]) -> String {
    let mut output = format!("  initializer {{ name: \"{name}\" data_type: 7");
    for dim in dims {
        write!(output, " dims: {dim}").unwrap();
    }
    for value in values {
        write!(output, " int64_data: {value}").unwrap();
    }
    output.push_str(" }\n");
    output
}

fn initializer_f32(name: &str, dims: &[usize], values: &[f32]) -> String {
    let mut output = format!("  initializer {{ name: \"{name}\" data_type: 1");
    for dim in dims {
        write!(output, " dims: {dim}").unwrap();
    }
    for value in values {
        write!(output, " float_data: {value:.9}").unwrap();
    }
    output.push_str(" }\n");
    output
}

fn value_info(name: &str, elem_type: i32, dimensions: &[String], output: bool) -> String {
    let field = if output { "output" } else { "input" };
    let mut value = format!(
        "  {field} {{ name: \"{name}\" type {{ tensor_type {{ elem_type: {elem_type} shape {{"
    );
    for dimension in dimensions {
        if dimension.parse::<usize>().is_ok() {
            write!(value, " dim {{ dim_value: {dimension} }}").unwrap();
        } else {
            write!(value, " dim {{ dim_param: \"{dimension}\" }}").unwrap();
        }
    }
    value.push_str(" } } } }\n");
    value
}

fn node(op: &str, inputs: &[&str], output: &str, attributes: &str) -> String {
    let mut value = String::from("  node {");
    for input in inputs {
        write!(value, " input: \"{input}\"").unwrap();
    }
    writeln!(
        value,
        " output: \"{output}\" op_type: \"{op}\"{attributes} }}"
    )
    .unwrap();
    value
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn multipliers(geometry: Geometry) -> Vec<i64> {
    let multiplier_max = i64::MAX as u64 / geometry.vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    (0..geometry.ngram_size)
        .map(|index| {
            let value = geometry
                .seed
                .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1));
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

fn is_prime(value: usize) -> bool {
    value >= 2
        && (2..=((value as f64).sqrt() as usize)).all(|divisor| !value.is_multiple_of(divisor))
}

fn head_tables(geometry: Geometry) -> (Vec<i64>, Vec<i64>, usize) {
    let mut sizes = Vec::new();
    let mut offsets = Vec::new();
    let mut total = 0;
    let mut candidate = geometry.table_base.saturating_sub(1);
    for _ in 0..geometry.ngram_heads() {
        candidate += 1;
        while !is_prime(candidate) {
            candidate += 1;
        }
        offsets.push(total as i64);
        sizes.push(candidate as i64);
        total += candidate;
    }
    (sizes, offsets, total)
}

fn model(geometry: Geometry) -> String {
    let channels = geometry.channels();
    let heads = geometry.ngram_heads();
    let context = geometry.context_len();
    let conv_history = geometry.conv_history_len();
    let (table_sizes, offsets, table_rows) = head_tables(geometry);
    let embedding = (0..table_rows)
        .map(|index| ((index % 23) as f32 - 11.0) / 16.0)
        .collect::<Vec<_>>();
    let key_weights = (0..heads * channels)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 32.0)
        .collect::<Vec<_>>();
    let value_weights = (0..heads * geometry.hidden_size)
        .map(|index| ((index * 11 % 17) as f32 - 8.0) / 24.0)
        .collect::<Vec<_>>();
    let conv_weights = (0..channels * geometry.conv_kernel)
        .map(|index| {
            let tap = index % geometry.conv_kernel;
            (0.5f32).powi(tap as i32 + 1) * if tap.is_multiple_of(2) { 1.0 } else { -1.0 }
        })
        .collect::<Vec<_>>();

    let mut graph = String::from("ir_version: 8\ngraph {\n  name: \"token_context\"\n");
    graph.push_str(&value_info(
        "token_ids",
        7,
        &["batch".into(), "sequence".into()],
        false,
    ));
    graph.push_str(&value_info(
        "inputs_embeds",
        1,
        &["batch".into(), "sequence".into(), channels.to_string()],
        false,
    ));
    graph.push_str(&value_info(
        "token_history",
        7,
        &["batch".into(), context.to_string()],
        false,
    ));
    graph.push_str(&value_info(
        "conv_history",
        1,
        &[
            "batch".into(),
            channels.to_string(),
            conv_history.to_string(),
        ],
        false,
    ));
    graph.push_str(&value_info(
        "output",
        1,
        &["batch".into(), "sequence".into(), channels.to_string()],
        true,
    ));
    graph.push_str(&value_info(
        "next_token_history",
        7,
        &["batch".into(), context.to_string()],
        true,
    ));
    graph.push_str(&value_info(
        "next_conv_history",
        1,
        &[
            "batch".into(),
            channels.to_string(),
            conv_history.to_string(),
        ],
        true,
    ));

    graph.push_str(&initializer_i64("slice_axis_1", &[1], &[1]));
    graph.push_str(&initializer_i64("slice_axis_0", &[1], &[0]));
    graph.push_str(&initializer_i64("slice_step_1", &[1], &[1]));
    graph.push_str(&initializer_i64("slice_end", &[1], &[i64::MAX]));
    graph.push_str(&initializer_i64(
        "next_token_start",
        &[1],
        &[-(context as i64)],
    ));
    graph.push_str(&initializer_i64(
        "next_conv_start",
        &[1],
        &[-(conv_history as i64)],
    ));
    graph.push_str(&initializer_i64("conv_axis_2", &[1], &[2]));
    graph.push_str(&initializer_i64("axes_last", &[1], &[-1]));
    graph.push_str(&initializer_i64("axes_minus_two", &[1], &[-2]));
    graph.push_str(&initializer_i64("rms_axes", &[1], &[-1]));
    graph.push_str(&initializer_i64(
        "embedding_shape",
        &[3],
        &[0, 0, heads as i64],
    ));
    graph.push_str(&initializer_i64(
        "group_shape",
        &[4],
        &[0, 0, geometry.hc_count as i64, geometry.hidden_size as i64],
    ));
    graph.push_str(&initializer_i64(
        "flat_shape",
        &[3],
        &[0, 0, channels as i64],
    ));
    graph.push_str(&initializer_f32("epsilon", &[], &[1.0e-6]));
    graph.push_str(&initializer_i64(
        "eos_token_id",
        &[],
        &[geometry.eos_token_id],
    ));
    graph.push_str(&initializer_f32(
        "sqrt_hidden",
        &[],
        &[(geometry.hidden_size as f32).sqrt()],
    ));
    graph.push_str(&initializer_i64("table_sizes", &[heads], &table_sizes));
    graph.push_str(&initializer_i64("table_offsets", &[heads], &offsets));
    graph.push_str(&initializer_f32(
        "ngram_embedding",
        &[table_rows, 1],
        &embedding,
    ));
    graph.push_str(&initializer_f32(
        "key_weights",
        &[heads, channels],
        &key_weights,
    ));
    graph.push_str(&initializer_f32(
        "value_weights",
        &[heads, geometry.hidden_size],
        &value_weights,
    ));
    graph.push_str(&initializer_f32(
        "conv_weights",
        &[channels, 1, geometry.conv_kernel],
        &conv_weights,
    ));

    graph.push_str(&node(
        "Concat",
        &["token_history", "token_ids"],
        "all_tokens",
        " attribute { name: \"axis\" i: 1 type: 2 }",
    ));
    for shift in 1..geometry.ngram_size {
        graph.push_str(&initializer_i64(
            &format!("shift_{shift}_start"),
            &[1],
            &[(context - shift) as i64],
        ));
        graph.push_str(&initializer_i64(
            &format!("shift_{shift}_end"),
            &[1],
            &[-(shift as i64)],
        ));
        graph.push_str(&node(
            "Slice",
            &[
                "all_tokens",
                &format!("shift_{shift}_start"),
                &format!("shift_{shift}_end"),
                "slice_axis_1",
                "slice_step_1",
            ],
            &format!("shifted_raw_{shift}"),
            "",
        ));
        for previous in 1..=shift {
            graph.push_str(&node(
                "Equal",
                &[&format!("shifted_raw_{previous}"), "eos_token_id"],
                &format!("shift_{shift}_eos_{previous}"),
                "",
            ));
        }
        let mut invalid = format!("shift_{shift}_eos_1");
        for previous in 2..=shift {
            graph.push_str(&node(
                "Or",
                &[&invalid, &format!("shift_{shift}_eos_{previous}")],
                &format!("shift_{shift}_invalid_{previous}"),
                "",
            ));
            invalid = format!("shift_{shift}_invalid_{previous}");
        }
        graph.push_str(&node(
            "Where",
            &[&invalid, "eos_token_id", &format!("shifted_raw_{shift}")],
            &format!("shifted_{shift}"),
            "",
        ));
    }
    let multiplier_values = multipliers(geometry);
    for (index, multiplier) in multiplier_values.iter().enumerate() {
        graph.push_str(&initializer_i64(
            &format!("multiplier_{index}"),
            &[],
            &[*multiplier],
        ));
    }
    graph.push_str(&node("Mul", &["token_ids", "multiplier_0"], "mixed_1", ""));
    for order in 2..=geometry.ngram_size {
        let previous = if order == 2 {
            "mixed_1".to_string()
        } else {
            format!("mixed_{}", order - 1)
        };
        graph.push_str(&node(
            "Mul",
            &[
                &format!("shifted_{}", order - 1),
                &format!("multiplier_{}", order - 1),
            ],
            &format!("weighted_{}", order - 1),
            "",
        ));
        graph.push_str(&node(
            "BitwiseXor",
            &[&previous, &format!("weighted_{}", order - 1)],
            &format!("mixed_{order}"),
            "",
        ));
        graph.push_str(&node(
            "Unsqueeze",
            &[&format!("mixed_{order}"), "axes_last"],
            &format!("mixed_{order}_heads"),
            "",
        ));
        let start = (order - 2) * geometry.heads_per_ngram;
        graph.push_str(&initializer_i64(
            &format!("head_{order}_start"),
            &[1],
            &[start as i64],
        ));
        graph.push_str(&initializer_i64(
            &format!("head_{order}_end"),
            &[1],
            &[(start + geometry.heads_per_ngram) as i64],
        ));
        for source in ["table_sizes", "table_offsets"] {
            graph.push_str(&node(
                "Slice",
                &[
                    source,
                    &format!("head_{order}_start"),
                    &format!("head_{order}_end"),
                    "slice_axis_0",
                    "slice_step_1",
                ],
                &format!("{source}_{order}"),
                "",
            ));
        }
        graph.push_str(&node(
            "Mod",
            &[
                &format!("mixed_{order}_heads"),
                &format!("table_sizes_{order}"),
            ],
            &format!("local_ids_{order}"),
            "",
        ));
        graph.push_str(&node(
            "Add",
            &[
                &format!("local_ids_{order}"),
                &format!("table_offsets_{order}"),
            ],
            &format!("ngram_ids_{order}"),
            "",
        ));
    }
    let id_blocks = (2..=geometry.ngram_size)
        .map(|order| format!("ngram_ids_{order}"))
        .collect::<Vec<_>>();
    graph.push_str(&node(
        "Concat",
        &id_blocks.iter().map(String::as_str).collect::<Vec<_>>(),
        "ngram_ids",
        " attribute { name: \"axis\" i: 2 type: 2 }",
    ));
    graph.push_str(&node(
        "Gather",
        &["ngram_embedding", "ngram_ids"],
        "looked_up",
        " attribute { name: \"axis\" i: 0 type: 2 }",
    ));
    graph.push_str(&node(
        "Reshape",
        &["looked_up", "embedding_shape"],
        "embeddings",
        "",
    ));
    graph.push_str(&node("MatMul", &["embeddings", "key_weights"], "key", ""));
    graph.push_str(&node(
        "MatMul",
        &["embeddings", "value_weights"],
        "value",
        "",
    ));
    for (source, prefix) in [("key", "key"), ("inputs_embeds", "query")] {
        graph.push_str(&node(
            "Reshape",
            &[source, "group_shape"],
            &format!("{prefix}_grouped"),
            "",
        ));
        graph.push_str(&node(
            "Mul",
            &[&format!("{prefix}_grouped"), &format!("{prefix}_grouped")],
            &format!("{prefix}_squared"),
            "",
        ));
        graph.push_str(&node(
            "ReduceMean",
            &[&format!("{prefix}_squared"), "rms_axes"],
            &format!("{prefix}_mean"),
            " attribute { name: \"keepdims\" i: 1 type: 2 }",
        ));
        graph.push_str(&node(
            "Add",
            &[&format!("{prefix}_mean"), "epsilon"],
            &format!("{prefix}_variance"),
            "",
        ));
        graph.push_str(&node(
            "Sqrt",
            &[&format!("{prefix}_variance")],
            &format!("{prefix}_scale"),
            "",
        ));
        graph.push_str(&node(
            "Div",
            &[&format!("{prefix}_grouped"), &format!("{prefix}_scale")],
            &format!("{prefix}_normed"),
            "",
        ));
    }
    graph.push_str(&node(
        "Mul",
        &["key_normed", "query_normed"],
        "gate_products",
        "",
    ));
    graph.push_str(&node(
        "ReduceSum",
        &["gate_products", "rms_axes"],
        "gate_sum",
        " attribute { name: \"keepdims\" i: 1 type: 2 }",
    ));
    graph.push_str(&node(
        "Div",
        &["gate_sum", "sqrt_hidden"],
        "gate_scaled",
        "",
    ));
    graph.push_str(&node("Abs", &["gate_scaled"], "gate_abs", ""));
    graph.push_str(&node("Clip", &["gate_abs", "epsilon"], "gate_clamped", ""));
    graph.push_str(&node("Sqrt", &["gate_clamped"], "gate_root", ""));
    graph.push_str(&node("Sign", &["gate_scaled"], "gate_sign", ""));
    graph.push_str(&node(
        "Mul",
        &["gate_root", "gate_sign"],
        "gate_signed_root",
        "",
    ));
    graph.push_str(&node("Sigmoid", &["gate_signed_root"], "gate", ""));
    graph.push_str(&node(
        "Unsqueeze",
        &["value", "axes_minus_two"],
        "value_grouped",
        "",
    ));
    graph.push_str(&node(
        "Mul",
        &["gate", "value_grouped"],
        "gated_value_grouped",
        "",
    ));
    graph.push_str(&node(
        "Reshape",
        &["gated_value_grouped", "flat_shape"],
        "gated_value",
        "",
    ));
    graph.push_str(&node(
        "Reshape",
        &["gated_value", "group_shape"],
        "conv_grouped",
        "",
    ));
    graph.push_str(&node(
        "Mul",
        &["conv_grouped", "conv_grouped"],
        "conv_squared",
        "",
    ));
    graph.push_str(&node(
        "ReduceMean",
        &["conv_squared", "rms_axes"],
        "conv_mean",
        " attribute { name: \"keepdims\" i: 1 type: 2 }",
    ));
    graph.push_str(&node("Add", &["conv_mean", "epsilon"], "conv_variance", ""));
    graph.push_str(&node("Sqrt", &["conv_variance"], "conv_scale", ""));
    graph.push_str(&node(
        "Div",
        &["conv_grouped", "conv_scale"],
        "conv_normed_grouped",
        "",
    ));
    graph.push_str(&node(
        "Reshape",
        &["conv_normed_grouped", "flat_shape"],
        "conv_normed",
        "",
    ));
    graph.push_str(&node(
        "Transpose",
        &["conv_normed"],
        "conv_current",
        " attribute { name: \"perm\" ints: 0 ints: 2 ints: 1 type: 7 }",
    ));
    graph.push_str(&node(
        "Concat",
        &["conv_history", "conv_current"],
        "conv_all",
        " attribute { name: \"axis\" i: 2 type: 2 }",
    ));
    graph.push_str(&node(
        "Conv",
        &["conv_all", "conv_weights"],
        "convolved",
        &format!(
            " attribute {{ name: \"dilations\" ints: {} type: 7 }} attribute {{ name: \"group\" i: {channels} type: 2 }}",
            geometry.conv_dilation
        ),
    ));
    graph.push_str(&node("Sigmoid", &["convolved"], "conv_sigmoid", ""));
    graph.push_str(&node(
        "Mul",
        &["convolved", "conv_sigmoid"],
        "conv_silu",
        "",
    ));
    graph.push_str(&node(
        "Transpose",
        &["conv_silu"],
        "conv_output",
        " attribute { name: \"perm\" ints: 0 ints: 2 ints: 1 type: 7 }",
    ));
    graph.push_str(&node(
        "Add",
        &["gated_value", "conv_output"],
        "ple_output",
        "",
    ));
    graph.push_str(&node("Add", &["inputs_embeds", "ple_output"], "output", ""));
    graph.push_str(&node(
        "Slice",
        &[
            "all_tokens",
            "next_token_start",
            "slice_end",
            "slice_axis_1",
            "slice_step_1",
        ],
        "next_token_history",
        "",
    ));
    graph.push_str(&node(
        "Slice",
        &[
            "conv_all",
            "next_conv_start",
            "slice_end",
            "conv_axis_2",
            "slice_step_1",
        ],
        "next_conv_history",
        "",
    ));
    graph.push_str("}\nopset_import { version: 18 }\n");
    graph
}

fn request(
    geometry: Geometry,
    session: &str,
    tokens: &[i64],
    base_offset: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    request_rows(geometry, session, &[tokens], base_offset, None)
}

fn request_rows(
    geometry: Geometry,
    session: &str,
    rows: &[&[i64]],
    base_offset: usize,
    selection: Option<&[i64]>,
) -> anyhow::Result<PipelineGenerateRequest> {
    anyhow::ensure!(!rows.is_empty(), "at least one row is required");
    let sequence = rows[0].len();
    anyhow::ensure!(
        rows.iter().all(|row| row.len() == sequence),
        "test rows must have equal sequence lengths"
    );
    let batch = rows.len();
    let channels = geometry.channels();
    let mut embeddings = Vec::with_capacity(batch * sequence * channels);
    for _row in 0..batch {
        for position in 0..sequence {
            for lane in 0..channels {
                embeddings.push(((base_offset + position) * channels + lane) as f32 / 32.0);
            }
        }
    }
    let tokens = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let options = GenerateOptions::default();
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenRows(
            rows.iter()
                .map(|row| row.iter().map(|&token| token as u32).collect())
                .collect(),
        ),
        options,
    })
    .with_session_id(session)
    .with_input(
        "token_ids",
        Value::from_slice_i64(&tokens, &[batch as i64, sequence as i64])?,
    )
    .with_input(
        "inputs_embeds",
        Value::from_slice_f32(
            &embeddings,
            &[batch as i64, sequence as i64, channels as i64],
        )?,
    )
    .with_input(
        "initial_token_history",
        Value::from_slice_i64(
            &vec![geometry.eos_token_id; batch * geometry.context_len()],
            &[batch as i64, geometry.context_len() as i64],
        )?,
    )
    .with_input(
        "initial_conv_history",
        Value::from_slice_f32(
            &vec![0.0; batch * channels * geometry.conv_history_len()],
            &[
                batch as i64,
                channels as i64,
                geometry.conv_history_len() as i64,
            ],
        )?,
    )
    .with_input(
        "active",
        Value::from_raw_bytes(vec![1; batch], &[batch as i64], DataType::Bool)?,
    )
    .with_input(
        "done",
        Value::from_raw_bytes(vec![0; batch], &[batch as i64], DataType::Bool)?,
    )
    .with_input(
        "accepted_len",
        Value::from_slice_i64(&vec![sequence as i64; batch], &[batch as i64])?,
    );
    if let Some(selection) = selection {
        request = request.with_input(
            "row_selection",
            Value::from_slice_i64(selection, &[selection.len() as i64])?,
        );
    }
    Ok(request)
}

fn outputs(
    engine: &mut Engine,
    geometry: Geometry,
    session: &str,
    chunks: &[&[i64]],
) -> anyhow::Result<(Vec<f32>, Vec<i64>, Vec<f32>)> {
    let mut hidden = Vec::new();
    let mut position = 0;
    let mut final_tokens = Vec::new();
    let mut final_conv = Vec::new();
    for chunk in chunks {
        let run = engine.run_pipeline(request(geometry, session, chunk, position)?)?;
        hidden.extend(run["hidden_states"].to_vec_f32()?);
        final_tokens = run["token_history"].to_vec_i64()?;
        final_conv = run["conv_history"].to_vec_f32()?;
        position += chunk.len();
    }
    Ok((hidden, final_tokens, final_conv))
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= 1.0e-5,
            "value {index}: {left} != {right}"
        );
    }
}

#[test]
fn executable_token_context_preserves_chunk_decode_checkpoint_and_release_semantics()
-> anyhow::Result<()> {
    for geometry in [QWEN_REDUCED, ALTERNATE] {
        let root = package(geometry)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let tokens = [5, 7, geometry.eos_token_id, 13];

        let full = outputs(&mut engine, geometry, "full", &[&tokens])?;
        let chunked = outputs(
            &mut engine,
            geometry,
            "chunked",
            &[&tokens[..2], &tokens[2..]],
        )?;
        let decoded = outputs(
            &mut engine,
            geometry,
            "decoded",
            &tokens.iter().map(std::slice::from_ref).collect::<Vec<_>>(),
        )?;
        assert_close(&full.0, &chunked.0);
        assert_close(&full.0, &decoded.0);
        assert_eq!(full.1, chunked.1);
        assert_eq!(full.1, decoded.1);
        assert_close(&full.2, &chunked.2);
        assert_close(&full.2, &decoded.2);
        assert_eq!(
            full.1,
            tokens[tokens.len() - geometry.context_len()..],
            "the committed token history is the exact final lexical context"
        );
        let first_channel = &full.2[..geometry.conv_history_len()];
        if geometry.name == QWEN_REDUCED.name {
            assert_close(
                first_channel,
                &[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.378_414_42,
                    -0.392_121_05,
                    0.793_429_3,
                    0.616_719_54,
                ],
            );
        } else {
            assert_close(
                first_channel,
                &[1.301_843_5, 1.053_156_6, 1.222_303_5, 0.553_421],
            );
        }

        let first = outputs(&mut engine, geometry, "checkpoint", &[&tokens[..2]])?;
        let checkpoint = engine.checkpoint_workflow_session("checkpoint")?;
        let advanced = outputs(&mut engine, geometry, "checkpoint", &[&tokens[2..]])?;
        engine.restore_workflow_session_checkpoint("checkpoint", &checkpoint)?;
        let invalid = request(geometry, "checkpoint", &tokens[2..], 2)?.with_input(
            "inputs_embeds",
            Value::from_slice_f32(
                &vec![0.0; geometry.channels()],
                &[1, 1, geometry.channels() as i64],
            )?,
        );
        let error = match engine.run_pipeline(invalid) {
            Ok(_) => panic!("a mismatched sequence shape must abort before state commit"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("sequence"),
            "the failure names the conflicting logical position: {error:#}"
        );
        let replayed = outputs(&mut engine, geometry, "checkpoint", &[&tokens[2..]])?;
        assert_close(&advanced.0, &replayed.0);
        assert_eq!(advanced.1, replayed.1);
        assert_close(&advanced.2, &replayed.2);
        assert_eq!(first.1.len(), geometry.context_len());
        assert_eq!(
            first.2.len(),
            geometry.channels() * geometry.conv_history_len()
        );

        let left_prefix = [3, 5];
        let right_prefix = [17, 19];
        engine.run_pipeline(request_rows(
            geometry,
            "batched",
            &[&left_prefix, &right_prefix],
            0,
            None,
        )?)?;
        let invalid_selection = request_rows(geometry, "batched", &[&[23]], 2, Some(&[2]))?;
        let error = match engine.run_pipeline(invalid_selection) {
            Ok(_) => panic!("an out-of-range row selection must fail before state commit"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("failed to compact session state")
                && message.contains("row index 2")
                && message.contains("shape [2"),
            "{message}"
        );
        let compacted =
            engine.run_pipeline(request_rows(geometry, "batched", &[&[23]], 2, Some(&[1]))?)?;
        outputs(
            &mut engine,
            geometry,
            "compaction-reference",
            &[&right_prefix],
        )?;
        let reference =
            engine.run_pipeline(request(geometry, "compaction-reference", &[23], 2)?)?;
        assert_close(
            &compacted["hidden_states"].to_vec_f32()?,
            &reference["hidden_states"].to_vec_f32()?,
        );
        assert_eq!(
            compacted["token_history"].to_vec_i64()?,
            reference["token_history"].to_vec_i64()?,
            "row selection keeps the selected row's token context and releases the other row"
        );
        assert_close(
            &compacted["conv_history"].to_vec_f32()?,
            &reference["conv_history"].to_vec_f32()?,
        );

        let session = engine.create_session()?;
        let session_name = session.to_string();
        let before_reset = outputs(
            &mut engine,
            geometry,
            &session_name,
            &[&tokens[..2], &tokens[2..]],
        )?;
        engine.reset_session(session)?;
        let after_reset = outputs(&mut engine, geometry, &session_name, &[&tokens])?;
        assert_close(&before_reset.0, &after_reset.0);
        assert_eq!(before_reset.1, after_reset.1);
        assert_close(&before_reset.2, &after_reset.2);
        engine.close_session(session)?;
    }
    Ok(())
}

#[test]
fn production_admission_rejects_position_ids_bound_as_token_identity() -> anyhow::Result<()> {
    let root = forged_position_package()?;
    let error = match Engine::from_dir(&root, EngineConfig::default()) {
        Ok(_) => panic!("a position-id source must be rejected before workflow execution"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("token-context component 'token_context'")
            && message.contains("token_ids port 'token_ids'")
            && message.contains("derived.position_ids")
            && message.contains("declared position_ids"),
        "{message}"
    );
    Ok(())
}
