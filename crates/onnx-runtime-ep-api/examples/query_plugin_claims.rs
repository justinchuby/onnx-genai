use std::ffi::CString;
use std::path::PathBuf;

use onnx_runtime_ep_api::abi::OrtGraphView;
use onnx_runtime_ir::FrozenGraph;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(
        args.next()
            .ok_or("usage: query_plugin_claims <model.onnx> <plugin.dylib> [registration-name]")?,
    );
    let plugin = PathBuf::from(
        args.next()
            .ok_or("usage: query_plugin_claims <model.onnx> <plugin.dylib> [registration-name]")?,
    );
    let registration_name = args
        .next()
        .map(|s| CString::new(s.to_string_lossy().as_bytes()))
        .transpose()?;

    let graph = FrozenGraph::build(onnx_runtime_loader::load_model(&model)?)?;
    let graph_view = graph.view();
    let view = OrtGraphView::new(&graph_view);
    let claims = view.query_plugin_capabilities(&plugin, registration_name.as_deref())?;

    println!("claims={}", claims.len());
    for (i, claim) in claims.iter().enumerate() {
        let names: Vec<_> = claim
            .node_ids
            .iter()
            .map(|&id| {
                let node = graph_view.graph().node(id);
                if node.name.is_empty() {
                    format!("#{}:{}", id.0, node.op_type)
                } else {
                    format!("#{}:{}:{}", id.0, node.op_type, node.name)
                }
            })
            .collect();
        println!(
            "claim[{i}] nodes={} inputs={} outputs={}",
            claim.node_ids.len(),
            claim.input_values.len(),
            claim.output_values.len()
        );
        println!("  {}", names.join(", "));
    }
    Ok(())
}
