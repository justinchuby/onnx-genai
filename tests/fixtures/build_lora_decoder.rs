// Hermetic text-generation package with a real non-square adapted projection.
use std::path::Path;

pub fn build(root: &Path, fixtures: &Path) -> anyhow::Result<()> {
    fn copy_tree(from: &Path, to: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(to)?;
        for entry in std::fs::read_dir(from)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                copy_tree(&entry.path(), &to.join(entry.file_name()))?;
            } else {
                std::fs::copy(entry.path(), to.join(entry.file_name()))?;
            }
        }
        Ok(())
    }
    fn yaml(value: serde_json::Value) -> serde_yaml::Value {
        serde_yaml::to_value(value).unwrap()
    }
    copy_tree(&fixtures.join("onnx_genai_workflows/decoder"), root)?;
    std::fs::copy(
        fixtures.join("tiny-llm/tokenizer.json"),
        root.join("tokenizer.json"),
    )?;
    std::fs::write(
        root.join("chat_template.jinja"),
        "{% for message in messages %}{{ message['content'] }} {% endfor %}",
    )?;
    let mut doc: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(
        root.join("inference_metadata.yaml"),
    )?)?;
    let adapter: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(
        fixtures.join("onnx_genai_workflows/adapter/inference_metadata.yaml"),
    )?)?;
    let logits = doc["pipeline"]["workflow"]["components"]["last_token_logits"]["ports"]["outputs"]
        ["last_logits"]
        .clone();
    let width = usize::try_from(
        doc["pipeline"]["workflow"]["components"]["model"]["ports"]["outputs"]["logits"]["shape"]
            [2]
        .as_u64()
        .unwrap(),
    )?;
    let features = width + 1;
    let mut activations = logits.clone();
    activations["shape"][1] = yaml(serde_json::json!(features));
    let mut service = adapter["adapters"].clone();
    service["selection"]["active"] = yaml(serde_json::json!("request.adapter_active"));
    service["artifacts"]
        .as_mapping_mut()
        .unwrap()
        .retain(|k, _| matches!(k.as_str(), Some("blue" | "red")));
    service["artifacts"]["red"]["index"] = yaml(serde_json::json!(1));
    let target = &mut service["target_manifest"]["targets"][0];
    target["component"] = yaml(serde_json::json!("adapter_projection"));
    target["node_name"] = yaml(serde_json::json!("project"));
    target["output_name"] = yaml(serde_json::json!("base"));
    target["input_features"] = yaml(serde_json::json!(features));
    target["output_features"] = yaml(serde_json::json!(width));
    target
        .as_mapping_mut()
        .unwrap()
        .remove(yaml(serde_json::json!("graph_inputs")));
    doc["adapters"] = service;
    let workflow = &mut doc["pipeline"]["workflow"];
    workflow["manifest"]["adapter_abis"] =
        yaml(serde_json::json!({"onnx-genai.parameter-overlay":"1"}));
    for name in [
        "request.adapter_segments",
        "request.adapter_counts",
        "request.adapter_scales",
    ] {
        workflow["inputs"][name] = adapter["pipeline"]["workflow"]["inputs"][name].clone();
    }
    workflow["inputs"]["request.adapter_active"] =
        adapter["pipeline"]["workflow"]["inputs"]["request.active"].clone();
    workflow["components"]["adapter_projection"] = yaml(serde_json::json!({
        "implementation":{"kind":"onnx","artifact":"adapter_projection.onnx.textproto"},
        "batch_capacity":{"uniform_dimensions":["vocabulary"]},
        "ports":{"inputs":{"logits":logits},"outputs":{"activations":activations,"base":logits}}
    }));
    let mut overlay = adapter["pipeline"]["workflow"]["components"]["overlay"].clone();
    overlay["ports"]["inputs"]["input"] = activations;
    overlay["ports"]["inputs"]["base"] = logits.clone();
    overlay["ports"]["outputs"]["output"] = logits;
    overlay["contract"]["parameters"]["component"] = yaml(serde_json::json!("adapter_projection"));
    overlay["batch_capacity"] = yaml(serde_json::json!({"uniform_dimensions":["vocabulary"]}));
    workflow["components"]["overlay"] = overlay;
    for (section, prefix) in [("setup", "decoder.setup"), ("steps", "decoder.body")] {
        let steps = workflow["steps"][0][section].as_sequence_mut().unwrap();
        let original = format!("{prefix}.last_logits");
        let unadapted = format!("{prefix}.unadapted_logits");
        for step in steps.iter_mut() {
            if step["component"].as_str() == Some("last_token_logits") {
                step["outputs"]["last_logits"] = yaml(serde_json::json!(unadapted));
            }
        }
        steps.push(yaml(serde_json::json!({"kind":"invoke","component":"adapter_projection","inputs":{"logits":unadapted},"outputs":{"activations":format!("{prefix}.adapter_features"),"base":format!("{prefix}.base")}})));
        steps.push(yaml(serde_json::json!({"kind":"invoke","component":"overlay","inputs":{"input":format!("{prefix}.adapter_features"),"base":format!("{prefix}.base")},"outputs":{"output":original}})));
    }
    for (alias, token) in [("red", 4), ("blue", 5)] {
        let mut a = vec![0_f32; features];
        a[width] = 1.0;
        let mut b = vec![0_f32; width];
        b[token] = 10_000.0;
        std::fs::create_dir_all(root.join("adapters").join(alias))?;
        std::fs::write(
            root.join("adapters").join(alias).join("adapter.json"),
            serde_json::to_vec(&serde_json::json!({"targets":{"projection":{"a":a,"b":b}}}))?,
        )?;
    }
    std::fs::write(
        root.join("inference_metadata.yaml"),
        serde_yaml::to_string(&doc)?,
    )?;
    let projection = (0..features)
        .flat_map(|i| (0..width).map(move |j| if i == j { 1_f32 } else { 0_f32 }))
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    std::fs::write(root.join("projection.data"), &projection)?;
    let tensor_type = |name: &str, columns: usize| {
        r#"name: "@NAME@" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: @COLUMNS@ } } } }"#
            .replace("@NAME@", name).replace("@COLUMNS@", &columns.to_string())
    };
    let graph = format!(
        r#"ir_version: 11
opset_import{{domain:"" version:24}}
graph{{name:"projection"
node{{input:"logits" output:"dims" op_type:"Shape"}}
node{{input:"dims" input:"zero" output:"batch" op_type:"Gather"}}
node{{input:"batch" input:"one" output:"ones_shape" op_type:"Concat" attribute{{name:"axis" i:0 type:INT}}}}
node{{input:"ones_shape" output:"ones" op_type:"ConstantOfShape" attribute{{name:"value" type:TENSOR t{{dims:1 data_type:1 float_data:1}}}}}}
node{{input:"logits" input:"ones" output:"activations" op_type:"Concat" attribute{{name:"axis" i:1 type:INT}}}}
node{{input:"activations" input:"projection" output:"base" name:"project" op_type:"MatMul"}}
initializer{{name:"zero" dims:1 data_type:7 int64_data:0}}
initializer{{name:"one" dims:1 data_type:7 int64_data:1}}
initializer{{name:"projection" dims:{features} dims:{width} data_type:1 data_location:EXTERNAL external_data{{key:"location" value:"projection.data"}} external_data{{key:"length" value:"{bytes}"}}}}
input{{{input}}} output{{{base}}} output{{{activation}}}
}}"#,
        bytes = projection.len(),
        input = tensor_type("logits", width),
        base = tensor_type("base", width),
        activation = tensor_type("activations", features)
    );
    std::fs::write(root.join("adapter_projection.onnx.textproto"), graph)?;
    Ok(())
}

// The same projection package, exposing its adapted vector without sampling.
pub fn embedding(root: &Path, fixtures: &Path) -> anyhow::Result<()> {
    build(root, fixtures)?;
    let file = root.join("inference_metadata.yaml");
    let mut doc: serde_yaml::Value = serde_yaml::from_str(&std::fs::read_to_string(&file)?)?;
    let w = &mut doc["pipeline"]["workflow"];
    let mut steps = w["steps"][0]["setup"].as_sequence().unwrap().clone();
    steps.push(serde_yaml::to_value(serde_json::json!({"kind":"emit","value":"decoder.setup.last_logits","output":"embedding","mode":"replace"}))?);
    w["steps"] = serde_yaml::Value::Sequence(steps);
    w.as_mapping_mut()
        .unwrap()
        .remove(serde_yaml::Value::String("state".into()));
    w.as_mapping_mut()
        .unwrap()
        .remove(serde_yaml::Value::String("serving".into()));
    let contract = w["components"]["overlay"]["ports"]["outputs"]["output"].clone();
    w["outputs"] = serde_yaml::to_value(
        serde_json::json!({"embedding":{"contract":serde_json::to_value(contract)?,"role":"tensor","stage":"pre_adapter"}}),
    )?;
    doc["profiles"] = serde_yaml::to_value(
        serde_json::json!({"vectors":{"kind":"embedding","version":"1","outputs":{"embedding":"embedding"}}}),
    )?;
    std::fs::write(file, serde_yaml::to_string(&doc)?)?;
    Ok(())
}
