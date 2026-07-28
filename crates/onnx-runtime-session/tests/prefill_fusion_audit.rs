//! Prefill fusion instrumentation: loads the real Qwen2.5-0.5b-f16 model graph,
//! runs optimizer passes with counters, and reports exactly which fusions fire,
//! how many nodes each collapses, and the final op count.
//!
//! This is a measurement test — it prints to stdout (use `cargo test -- --nocapture`).
//! It does NOT assert thresholds; it exists to diagnose whether implemented fusions
//! are actually matching the real graph topology.

use std::collections::HashMap;
use std::path::PathBuf;

use onnx_runtime_ir::{Graph, NodeId, ValueId};
use onnx_runtime_optimizer::{
    ConstantFolding, DeadNodeElimination, FusionPattern, OpFusion, OptimizationPass, PassContext,
};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models/qwen2.5-0.5b-f16/model.onnx")
}

fn op_histogram(graph: &Graph) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for node in graph.nodes.values() {
        let key = if node.domain.is_empty() {
            node.op_type.clone()
        } else {
            format!("{}::{}", node.domain, node.op_type)
        };
        *counts.entry(key).or_default() += 1;
    }
    counts
}

fn print_histogram(hist: &HashMap<String, usize>, label: &str) {
    let mut items: Vec<_> = hist.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    println!("\n=== Op histogram ({label}) ===");
    println!("{:<50} {:>5}", "Op", "Count");
    println!("{}", "-".repeat(56));
    let mut total = 0;
    for (op, count) in &items {
        println!("{op:<50} {count:>5}");
        total += *count;
    }
    println!("{}", "-".repeat(56));
    println!("{:<50} {:>5}", "TOTAL", total);
}

#[test]
fn prefill_fusion_audit() {
    let path = model_path();
    if !path.exists() {
        eprintln!("SKIP: model not found at {}", path.display());
        return;
    }

    // Load the model graph (with weight store for initializer resolution)
    let (mut graph, weights) =
        onnx_runtime_loader::load_model_with_weights(&path).expect("load qwen2.5-0.5b-f16");
    let nodes_raw = graph.num_nodes();
    println!("\n{}", "=".repeat(70));
    println!("PREFILL FUSION AUDIT — Qwen2.5-0.5b-f16");
    println!("{}", "=".repeat(70));
    println!("Nodes in raw graph: {nodes_raw}");

    // Print raw histogram
    let hist_raw = op_histogram(&graph);
    print_histogram(&hist_raw, "raw model");

    // --- Phase 1: Run each pass individually on clones to measure individual deltas ---
    let ctx = PassContext::new();
    // For EP passes that need weight access, create a resolver-equipped context.
    struct TestResolver(std::sync::Arc<onnx_runtime_loader::WeightStore>);
    impl onnx_runtime_optimizer::InitializerResolver for TestResolver {
        fn bytes<'a>(&'a self, weight: &'a onnx_runtime_ir::WeightRef) -> Option<&'a [u8]> {
            self.0.bytes(weight)
        }
    }
    let ep_ctx =
        PassContext::new().with_initializer_resolver(std::sync::Arc::new(TestResolver(weights)));

    println!("\n\n=== Per-pass node deltas (run on fresh clone each time) ===");
    println!(
        "{:<35} {:>8} {:>8} {:>8}",
        "Pass", "Before", "After", "Delta"
    );
    println!("{}", "-".repeat(62));

    // ConstantFolding
    {
        let mut g = graph.clone();
        let before = g.num_nodes();
        ConstantFolding.run(&mut g, &ctx).unwrap();
        let after = g.num_nodes();
        println!(
            "{:<35} {:>8} {:>8} {:>8}",
            "ConstantFolding",
            before,
            after,
            after as i64 - before as i64
        );
    }

    // DeadNodeElimination
    {
        let mut g = graph.clone();
        let before = g.num_nodes();
        DeadNodeElimination.run(&mut g, &ctx).unwrap();
        let after = g.num_nodes();
        println!(
            "{:<35} {:>8} {:>8} {:>8}",
            "DeadNodeElimination",
            before,
            after,
            after as i64 - before as i64
        );
    }

    // --- Phase 2: Each fusion pattern individually ---
    println!("\n\n=== Fusion pattern deltas (each run individually on raw graph clone) ===");
    println!(
        "{:<35} {:>8} {:>8} {:>8}",
        "Pattern", "Before", "After", "Delta"
    );
    println!("{}", "-".repeat(62));

    let patterns = vec![
        ("Attention (SDPA)", FusionPattern::attention()),
        ("LayerNorm", FusionPattern::layernorm()),
        ("Gelu (Erf)", FusionPattern::gelu()),
        (
            "MatMul+Bias+Relu",
            FusionPattern::new("MatMul+Bias+Relu", &["MatMul", "Add", "Relu"], "FusedGemm"),
        ),
        (
            "MatMul+Bias",
            FusionPattern::new("MatMul+Bias", &["MatMul", "Add"], "FusedMatMulBias"),
        ),
    ];

    for (label, pattern) in &patterns {
        let mut g = graph.clone();
        let before = g.num_nodes();
        let fusion = OpFusion::with_patterns(vec![pattern.clone()]);
        fusion.run(&mut g, &ctx).unwrap();
        let after = g.num_nodes();
        let delta = after as i64 - before as i64;
        let fired = delta != 0;
        println!(
            "{:<35} {:>8} {:>8} {:>8}  {}",
            label,
            before,
            after,
            delta,
            if fired { "✓ FIRED" } else { "✗ NOT FIRED" }
        );
        if fired {
            // Show what changed
            let hist_after = op_histogram(&g);
            let hist_before = op_histogram(&graph);
            println!("    Ops added:");
            for (op, &count) in &hist_after {
                let before_count = hist_before.get(op).copied().unwrap_or(0);
                if count > before_count {
                    println!("      +{} {}", count - before_count, op);
                }
            }
            println!("    Ops removed:");
            for (op, &count) in &hist_before {
                let after_count = hist_after.get(op).copied().unwrap_or(0);
                if count > after_count {
                    println!("      -{} {}", count - after_count, op);
                }
            }
        }
    }

    // --- Phase 3: Run FULL production pipeline (session + EP passes) ---
    println!("\n\n=== Full production pipeline (session-level + CPU EP passes) ===");

    // Session-level passes first (like production)
    println!("Session-level passes:");
    let session_passes: Vec<Box<dyn OptimizationPass>> =
        vec![Box::new(ConstantFolding), Box::new(DeadNodeElimination)];
    for pass in &session_passes {
        let before = graph.num_nodes();
        pass.run(&mut graph, &ctx).unwrap();
        let after = graph.num_nodes();
        let delta = after as i64 - before as i64;
        println!("  {} → {} nodes (delta {})", pass.name(), after, delta);
    }

    // CPU EP passes (with initializer resolver for weight-access passes)
    println!("CPU EP passes:");
    let cpu_passes = onnx_runtime_ep_cpu::cpu_optimization_passes();
    for pass in &cpu_passes {
        let before = graph.num_nodes();
        pass.run(&mut graph, &ep_ctx).unwrap();
        let after = graph.num_nodes();
        let delta = after as i64 - before as i64;
        println!("  {} → {} nodes (delta {})", pass.name(), after, delta);
    }
    let nodes_after_opt = graph.num_nodes();
    println!(
        "\nFull pipeline: {} → {} nodes (delta {})",
        nodes_raw,
        nodes_after_opt,
        nodes_after_opt as i64 - nodes_raw as i64
    );

    // Final histogram
    let hist_final = op_histogram(&graph);
    print_histogram(&hist_final, "after full CPU EP optimization");

    // --- Phase 4: Key diagnostics ---
    println!("\n\n=== Key Diagnostics ===");
    let softmax_count = hist_final.get("Softmax").copied().unwrap_or(0);
    let fused_attn = hist_final
        .get("com.microsoft::FusedAttention")
        .copied()
        .unwrap_or(0);
    let matmul_count = hist_final.get("MatMul").copied().unwrap_or(0);
    let fused_matmul_bias = hist_final
        .get("com.microsoft::FusedMatMulBias")
        .copied()
        .unwrap_or(0);
    let layernorm_count = hist_final
        .get("com.microsoft::LayerNormalization")
        .copied()
        .unwrap_or(0);
    let gelu_count = hist_final.get("com.microsoft::Gelu").copied().unwrap_or(0);
    let matmul_nbits = hist_final
        .get("com.microsoft::MatMulNBits")
        .copied()
        .unwrap_or(0);

    println!("FusedAttention ops (SDPA):      {fused_attn}");
    println!("Remaining Softmax ops:          {softmax_count}");
    println!("Remaining MatMul ops:           {matmul_count}");
    println!("FusedMatMulBias ops:            {fused_matmul_bias}");
    println!("LayerNormalization (fused) ops: {layernorm_count}");
    println!("Gelu (fused) ops:               {gelu_count}");
    println!("MatMulNBits ops:                {matmul_nbits}");
    println!("Total ops after optimization:   {nodes_after_opt}");

    if fused_attn == 0 && softmax_count > 0 {
        println!("\n⚠️  SDPA FUSION IS NOT FIRING despite {softmax_count} Softmax nodes in graph!");
        println!("    This is the primary investigation target.");
        println!("\n    Softmax node details:");
        for (id, node) in graph.nodes.iter() {
            if node.op_type == "Softmax" {
                println!(
                    "      NodeId({}) axis={:?} inputs={:?} outputs={:?}",
                    id.0,
                    node.attr("axis"),
                    node.inputs,
                    node.outputs,
                );
            }
        }
    }

    // --- Phase 5: Topology analysis for fusion opportunities ---
    println!("\n\n=== Topology Analysis: Fusion Opportunities ===");

    // Count sibling projections sharing same input (gate/up merge candidates)
    let mut input_to_matmul: HashMap<ValueId, Vec<NodeId>> = HashMap::new();
    for (id, node) in graph.nodes.iter() {
        if ((node.op_type == "MatMul" && node.is_default_domain())
            || (node.op_type == "FusedMatMulBias" && node.domain == "com.microsoft"))
            && let Some(Some(input)) = node.inputs.first()
        {
            input_to_matmul.entry(*input).or_default().push(id);
        }
    }
    let mut sibling_groups: Vec<_> = input_to_matmul
        .iter()
        .filter(|(_, nodes)| nodes.len() >= 2)
        .collect();
    sibling_groups.sort_by_key(|(_, nodes)| std::cmp::Reverse(nodes.len()));

    println!("\nSibling projection groups (same input, multiple MatMul/FusedMatMulBias):");
    println!(
        "  Groups of 3+ (Q/K/V merge candidates): {}",
        sibling_groups.iter().filter(|(_, n)| n.len() >= 3).count()
    );
    println!(
        "  Groups of 2 (gate/up merge candidates): {}",
        sibling_groups.iter().filter(|(_, n)| n.len() == 2).count()
    );

    for (input_vid, nodes) in sibling_groups.iter().take(5) {
        println!(
            "\n  Input value {:?} → {} consumers:",
            input_vid,
            nodes.len()
        );
        for &nid in nodes.iter() {
            let n = graph.node(nid);
            let out_shape: Vec<_> = n
                .outputs
                .iter()
                .map(|&o| {
                    let v = graph.value(o);
                    format!("{:?}", v.shape)
                })
                .collect();
            println!(
                "    {} {} (domain={}) → shape {}",
                nid.0,
                n.op_type,
                n.domain,
                out_shape.join(", ")
            );
        }
    }

    // Count plain MatMul ops that could be gate/up merge targets
    let gate_up_candidates: usize = input_to_matmul
        .values()
        .filter(|nodes| {
            nodes.len() == 2
                && nodes.iter().all(|&nid| {
                    let n = graph.node(nid);
                    n.op_type == "MatMul" && n.is_default_domain()
                })
        })
        .count();
    println!("\n  Plain MatMul pairs (gate/up merge candidates): {gate_up_candidates}");
    println!("  → Would save {gate_up_candidates} dispatches if all merged");
}
