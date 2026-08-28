use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest, SessionForkParticipantKind, SessionPosition,
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
    static NEXT_PACKAGE: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/token-context")
        .join(format!(
            "{}-{}",
            geometry.name,
            NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata(geometry))?;
    fs::write(root.join("token-context.onnx.textproto"), model(geometry))?;
    Ok(root)
}

fn package_with_generic_feature_state(geometry: Geometry) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/token-context")
        .join(format!("{}-generic-feature-fork", geometry.name));
    fs::create_dir_all(&root)?;
    let mut document = metadata(geometry);
    document = document.replacen(
        "    inputs:\n      token_ids:",
        r#"    inputs:
      initial_generic_feature:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        role: { kind: opaque }
        source: { kind: application, name: initial_generic_feature }
        required: true
      token_ids:"#,
        1,
    );
    document = document.replacen(
        "    components:\n      token_context:",
        r#"    components:
      generic_feature:
        implementation: { kind: onnx, artifact: generic-feature.onnx.textproto }
        ports:
          inputs:
            feature_state:
              dtype: int64
              shape: [batch]
              batch_layout: { kind: request_aligned, axis: 0 }
            delta:
              dtype: int64
              shape: [batch]
              batch_layout: { kind: request_aligned, axis: 0 }
          outputs:
            next_feature:
              dtype: int64
              shape: [batch]
              batch_layout: { kind: request_aligned, axis: 0 }
      token_context:"#,
        1,
    );
    document = document.replacen(
        "    state:\n      token_history:",
        r#"    state:
      generic_feature:
        contract:
          dtype: int64
          shape: [batch]
          batch_layout: { kind: request_aligned, axis: 0 }
        class: semantic
        scope: session
        initializer: initial_generic_feature
        recurrence: { kind: invariant }
        management: runtime
        release_boundary: session
        service_group: generic_feature
      token_history:"#,
        1,
    );
    document = document.replacen(
        "      - { kind: emit, value: context.output, output: hidden_states, mode: replace }",
        r#"      - kind: invoke
        component: generic_feature
        inputs: { feature_state: initial_generic_feature, delta: accepted_len }
        outputs: { next_feature: generic_feature }
      - { kind: emit, value: context.output, output: hidden_states, mode: replace }"#,
        1,
    );
    document = document.replacen(
        "        groups:\n          token_history:",
        r#"        groups:
          generic_feature:
            kind: encoder
            layout: b
            update: { kind: replace }
            capabilities: { snapshot: true, fork: true }
            ports:
              generic_feature:
                generic_feature:
                  input: feature_state
                  output: next_feature
          token_history:"#,
        1,
    );
    fs::write(root.join("inference_metadata.yaml"), document)?;
    fs::write(root.join("token-context.onnx.textproto"), model(geometry))?;
    fs::write(
        root.join("generic-feature.onnx.textproto"),
        r#"
ir_version: 8
graph {
  name: "generic_feature"
  node {
    input: "feature_state"
    input: "delta"
    output: "next_feature"
    op_type: "Add"
  }
  input {
    name: "feature_state"
    type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } } } }
  }
  input {
    name: "delta"
    type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } } } }
  }
  output {
    name: "next_feature"
    type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } } } }
  }
}
opset_import { version: 13 }
"#,
    )?;
    Ok(root)
}

fn failing_after_publication_package(geometry: Geometry) -> anyhow::Result<PathBuf> {
    static NEXT_FAILURE_PACKAGE: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/token-context")
        .join(format!(
            "{}-transaction-failure-{}",
            geometry.name,
            NEXT_FAILURE_PACKAGE.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root)?;
    let mut document = metadata(geometry);
    document = document.replacen(
        "    inputs:\n      token_ids:",
        r#"    inputs:
      failure_index:
        contract:
          dtype: int64
          shape: []
          batch_layout: { kind: shared }
        role: { kind: opaque }
        source: { kind: application, name: failure_index }
        required: false
        default: 0
      token_ids:"#,
        1,
    );
    document = document.replacen(
        "        batch_capacity: { uniform_dimensions: [sequence] }\n    state:",
        &format!(
            r#"        batch_capacity: {{ uniform_dimensions: [sequence] }}
      fail_after_publication:
        implementation: {{ kind: onnx, artifact: fail-after-publication.onnx.textproto }}
        ports:
          inputs:
            value:
              dtype: float32
              shape: [batch, sequence, {channels}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
            index:
              dtype: int64
              shape: []
              batch_layout: {{ kind: shared }}
          outputs:
            ignored:
              dtype: float32
              shape: [batch, {channels}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
        batch_capacity: {{ uniform_dimensions: [sequence] }}
    state:"#,
            channels = geometry.channels()
        ),
        1,
    );
    document = document.replacen(
        "      - { kind: emit, value: accepted_len, output: valid_lengths, mode: replace }\n    serving:",
        r#"      - { kind: emit, value: accepted_len, output: valid_lengths, mode: replace }
      - kind: invoke
        component: fail_after_publication
        inputs: { value: context.output, index: failure_index }
        outputs: { ignored: failure.ignored }
    serving:"#,
        1,
    );
    fs::write(root.join("inference_metadata.yaml"), document)?;
    fs::write(root.join("token-context.onnx.textproto"), model(geometry))?;
    fs::write(
        root.join("fail-after-publication.onnx.textproto"),
        failing_gather_model(geometry),
    )?;
    Ok(root)
}

fn failing_gather_model(geometry: Geometry) -> String {
    format!(
        r#"
ir_version: 8
graph {{
  name: "fail_after_publication"
  node {{
    input: "value"
    input: "index"
    output: "ignored"
    op_type: "Gather"
    attribute {{ name: "axis" i: 1 type: INT }}
  }}
  input {{
    name: "value"
    type {{ tensor_type {{ elem_type: 1 shape {{
      dim {{ dim_param: "batch" }}
      dim {{ dim_param: "sequence" }}
      dim {{ dim_value: {channels} }}
    }} }} }}
  }}
  input {{
    name: "index"
    type {{ tensor_type {{ elem_type: 7 shape {{ }} }} }}
  }}
  output {{
    name: "ignored"
    type {{ tensor_type {{ elem_type: 1 shape {{
      dim {{ dim_param: "batch" }}
      dim {{ dim_value: {channels} }}
    }} }} }}
  }}
}}
opset_import {{ version: 18 }}
"#,
        channels = geometry.channels()
    )
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
    manifest: {{}}
    inputs:
      token_ids:
        contract:
          dtype: int64
          shape: [batch, sequence]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
          padding: [{{ dimension: sequence, valid_lengths: accepted_len }}]
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: application, name: token_ids }}
        required: true
      inputs_embeds:
        contract:
          dtype: float32
          shape: [batch, sequence, {channels}]
          batch_layout: {{ kind: request_aligned, axis: 0 }}
          padding: [{{ dimension: sequence, valid_lengths: accepted_len }}]
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
          shape: [selected_batch]
          batch_layout: {{ kind: shared }}
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
          padding: [{{ dimension: sequence, valid_lengths: valid_lengths }}]
        role: tensor
        stage: pre_adapter
      valid_lengths:
        contract:
          dtype: int64
          shape: [batch]
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
              padding: [{{ dimension: sequence, valid_lengths: valid_lengths }}]
            inputs_embeds:
              dtype: float32
              shape: [batch, sequence, {channels}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}
              padding: [{{ dimension: sequence, valid_lengths: valid_lengths }}]
            valid_lengths:
              dtype: int64
              shape: [batch]
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
              padding: [{{ dimension: sequence, valid_lengths: valid_lengths }}]
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
          valid_lengths: accepted_len
          token_history: initial_token_history
          conv_history: initial_conv_history
        outputs:
          output: context.output
          next_token_history: token_history
          next_conv_history: conv_history
      - {{ kind: emit, value: context.output, output: hidden_states, mode: replace }}
      - {{ kind: emit, value: accepted_len, output: valid_lengths, mode: replace }}
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
            capabilities: {{ snapshot: true, fork: true, cascade: [conv_history] }}
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
    graph.push_str(&value_info("valid_lengths", 7, &["batch".into()], false));
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
    graph.push_str(&initializer_i64("axes_last", &[1], &[-1]));
    graph.push_str(&initializer_i64("axes_one", &[1], &[1]));
    graph.push_str(&initializer_i64("zero_i64", &[], &[0]));
    graph.push_str(&initializer_i64("one_i64", &[], &[1]));
    graph.push_str(&initializer_i64("shape_index_sequence", &[1], &[1]));
    graph.push_str(&initializer_i64(
        "token_history_offsets",
        &[1, context],
        &(0..context as i64).collect::<Vec<_>>(),
    ));
    graph.push_str(&initializer_i64(
        "conv_history_offsets",
        &[1, conv_history],
        &(0..conv_history as i64).collect::<Vec<_>>(),
    ));
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

    graph.push_str(&node("Shape", &["token_ids"], "token_ids_shape", ""));
    graph.push_str(&node(
        "Gather",
        &["token_ids_shape", "shape_index_sequence"],
        "sequence_extent",
        " attribute { name: \"axis\" i: 0 type: 2 }",
    ));
    graph.push_str(&node(
        "Squeeze",
        &["sequence_extent", "slice_axis_0"],
        "sequence_extent_scalar",
        "",
    ));
    graph.push_str(&node(
        "Range",
        &["zero_i64", "sequence_extent_scalar", "one_i64"],
        "sequence_positions",
        "",
    ));
    graph.push_str(&node(
        "Unsqueeze",
        &["valid_lengths", "axes_one"],
        "valid_lengths_column",
        "",
    ));
    graph.push_str(&node(
        "Less",
        &["sequence_positions", "valid_lengths_column"],
        "valid_tokens",
        "",
    ));
    graph.push_str(&node(
        "Where",
        &["valid_tokens", "token_ids", "zero_i64"],
        "valid_token_ids",
        "",
    ));
    graph.push_str(&node(
        "Concat",
        &["token_history", "valid_token_ids"],
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
    graph.push_str(&node(
        "Mul",
        &["valid_token_ids", "multiplier_0"],
        "mixed_1",
        "",
    ));
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
    graph.push_str(&node(
        "Cast",
        &["valid_tokens"],
        "valid_tokens_f32",
        " attribute { name: \"to\" i: 1 type: 2 }",
    ));
    graph.push_str(&node(
        "Unsqueeze",
        &["valid_tokens_f32", "axes_last"],
        "valid_tokens_f32_column",
        "",
    ));
    graph.push_str(&node(
        "Mul",
        &["ple_output", "valid_tokens_f32_column"],
        "valid_ple_output",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["inputs_embeds", "valid_ple_output"],
        "output",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["valid_lengths_column", "token_history_offsets"],
        "next_token_indices",
        "",
    ));
    graph.push_str(&node(
        "GatherElements",
        &["all_tokens", "next_token_indices"],
        "next_token_history",
        " attribute { name: \"axis\" i: 1 type: 2 }",
    ));
    graph.push_str(&node(
        "Add",
        &["valid_lengths_column", "conv_history_offsets"],
        "next_conv_indices_flat",
        "",
    ));
    graph.push_str(&node(
        "Unsqueeze",
        &["next_conv_indices_flat", "axes_one"],
        "next_conv_indices_column",
        "",
    ));
    graph.push_str(&node("Shape", &["conv_history"], "conv_history_shape", ""));
    graph.push_str(&node(
        "Expand",
        &["next_conv_indices_column", "conv_history_shape"],
        "next_conv_indices",
        "",
    ));
    graph.push_str(&node(
        "GatherElements",
        &["conv_all", "next_conv_indices"],
        "next_conv_history",
        " attribute { name: \"axis\" i: 2 type: 2 }",
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
    let sequence = rows.first().map_or(0, |row| row.len());
    request_padded_rows(
        geometry,
        session,
        rows,
        &vec![sequence; rows.len()],
        base_offset,
        selection,
    )
}

fn request_padded_rows(
    geometry: Geometry,
    session: &str,
    rows: &[&[i64]],
    valid_lengths: &[usize],
    base_offset: usize,
    selection: Option<&[i64]>,
) -> anyhow::Result<PipelineGenerateRequest> {
    anyhow::ensure!(!rows.is_empty(), "at least one row is required");
    let sequence = rows[0].len();
    anyhow::ensure!(
        rows.iter().all(|row| row.len() == sequence),
        "test rows must have equal sequence lengths"
    );
    anyhow::ensure!(
        valid_lengths.len() == rows.len() && valid_lengths.iter().all(|&length| length <= sequence),
        "test valid lengths must provide one in-range extent per row"
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
        Value::from_slice_i64(
            &valid_lengths
                .iter()
                .map(|&length| length as i64)
                .collect::<Vec<_>>(),
            &[batch as i64],
        )?,
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
        let run = engine.run_pipeline_retained(request(geometry, session, chunk, position)?)?;
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

fn assert_token_context_outputs_equal(
    left: &std::collections::HashMap<String, Value>,
    right: &std::collections::HashMap<String, Value>,
) -> anyhow::Result<()> {
    assert_close(
        &left["hidden_states"].to_vec_f32()?,
        &right["hidden_states"].to_vec_f32()?,
    );
    assert_eq!(
        left["token_history"].to_vec_i64()?,
        right["token_history"].to_vec_i64()?
    );
    assert_close(
        &left["conv_history"].to_vec_f32()?,
        &right["conv_history"].to_vec_f32()?,
    );
    assert_eq!(
        left["valid_lengths"].to_vec_i64()?,
        right["valid_lengths"].to_vec_i64()?
    );
    Ok(())
}

#[test]
fn failed_selected_turn_restores_both_histories_output_and_sibling_rows() -> anyhow::Result<()> {
    // The structurally different fixture proves transaction participation is
    // derived from generic state groups rather than a model-family identity.
    let geometry = ALTERNATE;
    let root = failing_after_publication_package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let prefixes = [&[3, 5][..], &[17, 19][..], &[29, 31][..]];
    for session in [
        "failed-retry",
        "expected-retry",
        "failed-sibling",
        "expected-sibling",
    ] {
        engine.run_pipeline_retained(request_rows(geometry, session, &prefixes, 0, None)?)?;
    }

    let source_tokens = [&[23][..], &[37][..], &[41][..]];
    for session in ["failed-retry", "failed-sibling"] {
        let failed = request_rows(geometry, session, &source_tokens, 2, Some(&[2, 0]))?
            .with_input("failure_index", Value::from_slice_i64(&[999], &[])?);
        let error = match engine.run_pipeline_retained(failed) {
            Ok(_) => panic!("the post-publication gather must fail the admitted turn"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("Gather"),
            "the injected failure must occur after the generalized history component: {error:#}"
        );
    }

    let expected_retry = engine.run_pipeline_retained(request_rows(
        geometry,
        "expected-retry",
        &source_tokens,
        2,
        Some(&[2, 0]),
    )?)?;
    let actual_retry = engine.run_pipeline_retained(request_rows(
        geometry,
        "failed-retry",
        &source_tokens,
        2,
        Some(&[2, 0]),
    )?)?;
    assert_token_context_outputs_equal(&actual_retry, &expected_retry)?;

    let expected_sibling = engine.run_pipeline_retained(request_rows(
        geometry,
        "expected-sibling",
        &source_tokens,
        2,
        Some(&[1]),
    )?)?;
    let actual_sibling = engine.run_pipeline_retained(request_rows(
        geometry,
        "failed-sibling",
        &source_tokens,
        2,
        Some(&[1]),
    )?)?;
    assert_token_context_outputs_equal(&actual_sibling, &expected_sibling)?;
    Ok(())
}

#[test]
fn semantic_session_fork_clones_token_and_convolution_histories_for_both_geometries()
-> anyhow::Result<()> {
    for geometry in [QWEN_REDUCED, ALTERNATE] {
        let root = package(geometry)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let parent = engine.create_session()?;
        let parent_name = parent.to_string();
        let prefix = [5, 7];
        engine.run_pipeline_retained(request(geometry, &parent_name, &prefix, 0)?)?;

        let plan = engine.prepare_session_fork(parent, SessionPosition::new(1))?;
        for state in ["token_history", "conv_history"] {
            assert!(
                plan.participants().iter().any(|participant| {
                    participant.kind == SessionForkParticipantKind::TokenContextHistory
                        && participant.name == state
                }),
                "fork plan omitted {state}: {:?}",
                plan.participants()
            );
        }
        assert!(
            plan.participants().iter().any(|participant| {
                participant.kind == SessionForkParticipantKind::OutputPublication
                    && participant.name == "hidden_states"
            }),
            "fork plan omitted output head/lineage state"
        );
        assert!(plan.participants().iter().any(|participant| {
            participant.kind == SessionForkParticipantKind::SpeculativeCascade
                && participant.name == "token_history->conv_history"
        }));
        let child = engine.fork_session(plan)?;
        let child_name = child.to_string();

        let parent_run =
            engine.run_pipeline_retained(request(geometry, &parent_name, &[11], prefix.len())?)?;
        let child_run =
            engine.run_pipeline_retained(request(geometry, &child_name, &[13], prefix.len())?)?;
        let mut parent_control = Engine::from_dir(&root, EngineConfig::default())?;
        let expected_parent = outputs(
            &mut parent_control,
            geometry,
            "expected-parent",
            &[&prefix, &[11]],
        )?;
        let mut child_control = Engine::from_dir(&root, EngineConfig::default())?;
        let expected_child = outputs(
            &mut child_control,
            geometry,
            "expected-child",
            &[&prefix, &[13]],
        )?;

        assert_eq!(parent_run["token_history"].to_vec_i64()?, expected_parent.1);
        assert_close(
            &parent_run["conv_history"].to_vec_f32()?,
            &expected_parent.2,
        );
        assert_eq!(child_run["token_history"].to_vec_i64()?, expected_child.1);
        assert_close(&child_run["conv_history"].to_vec_f32()?, &expected_child.2);
        assert_ne!(
            parent_run["token_history"].to_vec_i64()?,
            child_run["token_history"].to_vec_i64()?,
            "parent and child histories must diverge independently"
        );

        engine.close_session(parent)?;
        let continued_child = engine.run_pipeline_retained(request(
            geometry,
            &child_name,
            &[17],
            prefix.len() + 1,
        )?)?;
        let mut continued_control = Engine::from_dir(&root, EngineConfig::default())?;
        let expected_continued = outputs(
            &mut continued_control,
            geometry,
            "expected-continued",
            &[&prefix, &[13], &[17]],
        )?;
        assert_eq!(
            continued_child["token_history"].to_vec_i64()?,
            expected_continued.1
        );
        assert_close(
            &continued_child["conv_history"].to_vec_f32()?,
            &expected_continued.2,
        );
        engine.close_session(child)?;
    }
    Ok(())
}

#[test]
fn forked_token_context_sessions_abort_retry_and_commit_independently() -> anyhow::Result<()> {
    let geometry = ALTERNATE;
    let root = failing_after_publication_package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let parent = engine.create_session()?;
    let parent_name = parent.to_string();
    let prefix = [&[3, 5][..], &[17, 19][..], &[29, 31][..]];
    engine.run_pipeline_retained(request_rows(geometry, &parent_name, &prefix, 0, None)?)?;
    let child =
        engine.fork_session(engine.prepare_session_fork(parent, SessionPosition::new(1))?)?;
    let child_name = child.to_string();
    let source_tokens = [&[23][..], &[37][..], &[41][..]];

    let failed = request_rows(geometry, &parent_name, &source_tokens, 2, Some(&[2, 0]))?
        .with_input("failure_index", Value::from_slice_i64(&[999], &[])?);
    let error = match engine.run_pipeline_retained(failed) {
        Ok(_) => panic!("parent turn must fail after staging state and output"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("Gather"), "{error:#}");

    let child_result = engine.run_pipeline_retained(request_rows(
        geometry,
        &child_name,
        &source_tokens,
        2,
        Some(&[1]),
    )?)?;
    let parent_retry = engine.run_pipeline_retained(request_rows(
        geometry,
        &parent_name,
        &source_tokens,
        2,
        Some(&[2, 0]),
    )?)?;

    let mut parent_control = Engine::from_dir(&root, EngineConfig::default())?;
    parent_control.run_pipeline_retained(request_rows(
        geometry,
        "parent-control",
        &prefix,
        0,
        None,
    )?)?;
    let expected_parent = parent_control.run_pipeline_retained(request_rows(
        geometry,
        "parent-control",
        &source_tokens,
        2,
        Some(&[2, 0]),
    )?)?;
    assert_token_context_outputs_equal(&parent_retry, &expected_parent)?;

    let mut child_control = Engine::from_dir(&root, EngineConfig::default())?;
    child_control.run_pipeline_retained(request_rows(
        geometry,
        "child-control",
        &prefix,
        0,
        None,
    )?)?;
    let expected_child = child_control.run_pipeline_retained(request_rows(
        geometry,
        "child-control",
        &source_tokens,
        2,
        Some(&[1]),
    )?)?;
    assert_token_context_outputs_equal(&child_result, &expected_child)?;
    engine.close_session(parent)?;
    engine.close_session(child)?;
    Ok(())
}

#[test]
fn semantic_fork_clones_alternate_generic_feature_state() -> anyhow::Result<()> {
    let geometry = ALTERNATE;
    let root = package_with_generic_feature_state(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let parent = engine.create_session()?;
    let parent_name = parent.to_string();
    let generic_request = |session: &str, tokens: &[i64], position: usize| {
        Ok::<_, anyhow::Error>(request(geometry, session, tokens, position)?.with_input(
            "initial_generic_feature",
            Value::from_slice_i64(&[0], &[1])?,
        ))
    };
    engine.run_pipeline_retained(generic_request(&parent_name, &[3, 5], 0)?)?;
    let plan = engine.prepare_session_fork(parent, SessionPosition::new(1))?;
    assert!(plan.participants().iter().any(|participant| {
        participant.kind == SessionForkParticipantKind::GenericFeatureState
            && participant.name == "generic_feature"
    }));
    let child = engine.fork_session(plan)?;

    let parent_run = engine.run_pipeline_retained(generic_request(&parent_name, &[7], 2)?)?;
    let child_request = generic_request(&child.to_string(), &[11], 2)?
        .with_input("accepted_len", Value::from_slice_i64(&[0], &[1])?);
    let child_run = engine.run_pipeline_retained(child_request)?;
    assert_eq!(parent_run["generic_feature"].to_vec_i64()?, [3]);
    assert_eq!(child_run["generic_feature"].to_vec_i64()?, [2]);
    engine.close_session(parent)?;
    engine.close_session(child)?;
    Ok(())
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
        let error = match engine.run_pipeline_retained(invalid) {
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

        let row_zero_prefix = [3, 5];
        let row_one_prefix = [17, 19];
        let row_two_prefix = [29, 31];
        engine.run_pipeline_retained(request_rows(
            geometry,
            "batched",
            &[&row_zero_prefix, &row_one_prefix, &row_two_prefix],
            0,
            None,
        )?)?;
        let source_tokens = [&[23][..], &[37][..], &[41][..]];
        let invalid_selection = request_rows(geometry, "batched", &source_tokens, 2, Some(&[3]))?;
        let error = match engine.run_pipeline_retained(invalid_selection) {
            Ok(_) => panic!("an out-of-range row selection must fail before state commit"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("destination 0")
                && message.contains("source row 3")
                && message.contains("only 3 rows"),
            "{message}"
        );

        outputs(
            &mut engine,
            geometry,
            "row-zero-reference",
            &[&row_zero_prefix],
        )?;
        outputs(
            &mut engine,
            geometry,
            "row-two-reference",
            &[&row_two_prefix],
        )?;
        let row_zero = engine.run_pipeline_retained(request(
            geometry,
            "row-zero-reference",
            source_tokens[0],
            2,
        )?)?;
        let row_two = engine.run_pipeline_retained(request(
            geometry,
            "row-two-reference",
            source_tokens[2],
            2,
        )?)?;
        let cloned = engine.run_pipeline_retained(request_rows(
            geometry,
            "batched",
            &source_tokens,
            2,
            Some(&[2, 2, 0]),
        )?)?;
        let mut expected_hidden = row_two["hidden_states"].to_vec_f32()?;
        expected_hidden.extend(row_two["hidden_states"].to_vec_f32()?);
        expected_hidden.extend(row_zero["hidden_states"].to_vec_f32()?);
        assert_close(&cloned["hidden_states"].to_vec_f32()?, &expected_hidden);
        let mut expected_tokens = row_two["token_history"].to_vec_i64()?;
        expected_tokens.extend(row_two["token_history"].to_vec_i64()?);
        expected_tokens.extend(row_zero["token_history"].to_vec_i64()?);
        assert_eq!(cloned["token_history"].to_vec_i64()?, expected_tokens);
        let mut expected_conv = row_two["conv_history"].to_vec_f32()?;
        expected_conv.extend(row_two["conv_history"].to_vec_f32()?);
        expected_conv.extend(row_zero["conv_history"].to_vec_f32()?);
        assert_close(&cloned["conv_history"].to_vec_f32()?, &expected_conv);

        let selected_checkpoint = engine.checkpoint_workflow_session("batched")?;
        let continuation_rows = [&[43][..], &[47][..], &[53][..]];
        let shrunk = engine.run_pipeline_retained(request_rows(
            geometry,
            "batched",
            &continuation_rows,
            3,
            Some(&[2, 0]),
        )?)?;
        engine.restore_workflow_session_checkpoint("batched", &selected_checkpoint)?;
        let replayed_shrink = engine.run_pipeline_retained(request_rows(
            geometry,
            "batched",
            &continuation_rows,
            3,
            Some(&[2, 0]),
        )?)?;
        assert_close(
            &shrunk["hidden_states"].to_vec_f32()?,
            &replayed_shrink["hidden_states"].to_vec_f32()?,
        );
        assert_eq!(
            shrunk["token_history"].to_vec_i64()?,
            replayed_shrink["token_history"].to_vec_i64()?
        );
        assert_close(
            &shrunk["conv_history"].to_vec_f32()?,
            &replayed_shrink["conv_history"].to_vec_f32()?,
        );

        let row_zero_continued = engine.run_pipeline_retained(request(
            geometry,
            "row-zero-reference",
            continuation_rows[2],
            3,
        )?)?;
        let row_two_continued = engine.run_pipeline_retained(request(
            geometry,
            "row-two-reference",
            continuation_rows[0],
            3,
        )?)?;
        let mut expected_shrunk_hidden = row_zero_continued["hidden_states"].to_vec_f32()?;
        expected_shrunk_hidden.extend(row_two_continued["hidden_states"].to_vec_f32()?);
        assert_close(
            &shrunk["hidden_states"].to_vec_f32()?,
            &expected_shrunk_hidden,
        );
        let mut expected_shrunk_tokens = row_zero_continued["token_history"].to_vec_i64()?;
        expected_shrunk_tokens.extend(row_two_continued["token_history"].to_vec_i64()?);
        assert_eq!(
            shrunk["token_history"].to_vec_i64()?,
            expected_shrunk_tokens,
            "the same shrink plan selects request tokens and both restored histories"
        );
        let mut expected_shrunk_conv = row_zero_continued["conv_history"].to_vec_f32()?;
        expected_shrunk_conv.extend(row_two_continued["conv_history"].to_vec_f32()?);
        assert_close(&shrunk["conv_history"].to_vec_f32()?, &expected_shrunk_conv);

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
fn padded_rows_ignore_invalid_tokens_through_selection_restore_and_decode() -> anyhow::Result<()> {
    for geometry in [QWEN_REDUCED, ALTERNATE] {
        let root = package(geometry)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let original = [&[5, 7, 11][..], &[13, 88, 89][..], &[17, 19, 77][..]];
        let changed_padding = [&[5, 7, 11][..], &[13, 44, 45][..], &[17, 19, 33][..]];
        let lengths = [3, 1, 2];

        let padded = engine.run_pipeline_retained(request_padded_rows(
            geometry, "padded", &original, &lengths, 0, None,
        )?)?;
        let changed = engine.run_pipeline_retained(request_padded_rows(
            geometry,
            "changed-padding",
            &changed_padding,
            &lengths,
            0,
            None,
        )?)?;
        assert_close(
            &padded["hidden_states"].to_vec_f32()?,
            &changed["hidden_states"].to_vec_f32()?,
        );
        assert_eq!(
            padded["token_history"].to_vec_i64()?,
            changed["token_history"].to_vec_i64()?
        );
        assert_close(
            &padded["conv_history"].to_vec_f32()?,
            &changed["conv_history"].to_vec_f32()?,
        );
        assert_eq!(padded["valid_lengths"].to_vec_i64()?, [3, 1, 2]);

        for (row_index, (row, &valid_length)) in original.iter().zip(&lengths).enumerate() {
            let valid = &row[..valid_length];
            let reference_session = format!("padded-reference-{row_index}");
            let reference =
                engine.run_pipeline_retained(request(geometry, &reference_session, valid, 0)?)?;
            let token_start = row_index * geometry.context_len();
            let token_end = token_start + geometry.context_len();
            assert_eq!(
                &padded["token_history"].to_vec_i64()?[token_start..token_end],
                reference["token_history"].to_vec_i64()?
            );
            let conv_row = geometry.channels() * geometry.conv_history_len();
            let conv_start = row_index * conv_row;
            assert_close(
                &padded["conv_history"].to_vec_f32()?[conv_start..conv_start + conv_row],
                &reference["conv_history"].to_vec_f32()?,
            );
            let padded_hidden = padded["hidden_states"].to_vec_f32()?;
            let valid_hidden = valid_length * geometry.channels();
            let hidden_start = row_index * original[0].len() * geometry.channels();
            assert_close(
                &padded_hidden[hidden_start..hidden_start + valid_hidden],
                &reference["hidden_states"].to_vec_f32()?,
            );
        }

        let checkpoint = engine.checkpoint_workflow_session("padded")?;
        let continuation = [&[23, 90][..], &[29, 31][..], &[37, 91][..]];
        let changed_continuation = [&[23, 42][..], &[29, 31][..], &[37, 66][..]];
        let continuation_lengths = [1, 2, 1];
        let selected = engine.run_pipeline_retained(request_padded_rows(
            geometry,
            "padded",
            &continuation,
            &continuation_lengths,
            3,
            Some(&[2, 0]),
        )?)?;
        engine.restore_workflow_session_checkpoint("padded", &checkpoint)?;
        let selected_with_changed_padding = engine.run_pipeline_retained(request_padded_rows(
            geometry,
            "padded",
            &changed_continuation,
            &continuation_lengths,
            3,
            Some(&[2, 0]),
        )?)?;
        assert_close(
            &selected["hidden_states"].to_vec_f32()?,
            &selected_with_changed_padding["hidden_states"].to_vec_f32()?,
        );
        assert_eq!(
            selected["token_history"].to_vec_i64()?,
            selected_with_changed_padding["token_history"].to_vec_i64()?
        );
        assert_close(
            &selected["conv_history"].to_vec_f32()?,
            &selected_with_changed_padding["conv_history"].to_vec_f32()?,
        );
        assert_eq!(
            selected["valid_lengths"].to_vec_i64()?,
            [1, 1],
            "the row plan must compact the logical committed extents with token and state rows"
        );

        let decoded = engine.run_pipeline_retained(request_rows(
            geometry,
            "padded",
            &[&[43], &[47]],
            4,
            None,
        )?)?;
        engine.run_pipeline_retained(request(geometry, "padded-reference-2", &[37], 2)?)?;
        let row_two_reference =
            engine.run_pipeline_retained(request(geometry, "padded-reference-2", &[43], 3)?)?;
        engine.run_pipeline_retained(request(geometry, "padded-reference-0", &[23], 3)?)?;
        let row_zero_reference =
            engine.run_pipeline_retained(request(geometry, "padded-reference-0", &[47], 4)?)?;
        let mut expected_tokens = row_two_reference["token_history"].to_vec_i64()?;
        expected_tokens.extend(row_zero_reference["token_history"].to_vec_i64()?);
        assert_eq!(decoded["token_history"].to_vec_i64()?, expected_tokens);
        let mut expected_conv = row_two_reference["conv_history"].to_vec_f32()?;
        expected_conv.extend(row_zero_reference["conv_history"].to_vec_f32()?);
        assert_close(&decoded["conv_history"].to_vec_f32()?, &expected_conv);
    }
    Ok(())
}

#[test]
fn padded_rows_require_in_range_structural_valid_lengths() -> anyhow::Result<()> {
    let geometry = QWEN_REDUCED;
    let root = package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let request = request_padded_rows(
        geometry,
        "invalid-padding",
        &[&[5, 7, 11], &[13, 17, 19]],
        &[3, 3],
        0,
        None,
    )?
    .with_input("accepted_len", Value::from_slice_i64(&[3, 4], &[2])?);

    let error = match engine.run_pipeline_retained(request) {
        Ok(_) => panic!("an over-extent valid length must fail before component execution"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(message.contains("padding.valid_lengths"), "{message}");
    assert!(message.contains("accepted_len"), "{message}");
    assert!(message.contains("0..=3"), "{message}");
    assert!(message.contains("row 1"), "{message}");
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
