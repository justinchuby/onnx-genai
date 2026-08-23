//! End-to-end regression for the deterministic tiny DeepSeek-V4 fixture
//! (Mobius PR #550's dense-MoE -> single `com.microsoft::QMoE` export),
//! proving the hash-routed QMoE graph loads and decodes correctly through
//! onnx-genai's native runtime as well as stock ONNX Runtime.
//!
//! Scope: this fixture deliberately targets DeepSeek-V4's *dense* CSA
//! schedule (`compress_ratios=None`, i.e. every layer at ratio 0) and
//! excludes the MTP sidecar (`num_nextn_predict_layers=0`). The compressed/
//! indexer CSA path (ratios 4/128) needs the same native-only sparse-
//! attention primitives GLM-5.2's DSA IndexShare needs and is tracked
//! separately as its own multi-phase effort in
//! `docs/models/DEEPSEEK_CSA_MTP_RUNTIME.md`; it is out of scope here. With
//! CSA dense and MTP absent, the built graph uses **only standard ONNX ops
//! plus two real ORT contrib ops** (`com.microsoft::QMoE` for the routed
//! experts, `com.microsoft::MatMulNBits` for every other quantized Linear) --
//! no native-only custom op at all, so unlike the GLM-5.2 DSA/IndexShare
//! fixture, this graph is loadable and runnable by native CPU, native CUDA,
//! *and* stock ORT alike. (`onnx_runtime_loader::load_model` further
//! function-inlines `MatMulNBits` into its primitive decomposition at load
//! time -- see `assert_current_emission`'s doc comment below -- so the
//! *loaded* native-engine graph never actually contains a `MatMulNBits` node,
//! while stock ORT of course executes the fused contrib op directly from the
//! file as written.)
//!
//! Coherence is checked at three levels: structural (no native-only
//! `pkg.nxrt::*` op, has fused QMoE and MatMulNBits), a locked native
//! CPU/CUDA anchor (regression), and stock-ORT-vs-native-CPU token agreement
//! (proves the graph is executable and numerically consistent with no
//! native-only op dependency at all -- mirroring
//! `glm_tiny_full_attention_e2e.rs`'s stock-ORT proof for GLM-5.2).
//!
//! This native-CUDA build requires both `cuda` and `native-backend`:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=1 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend deepseek_v4_tiny
//! ```
//!
//! The committed fixture is reproducible with:
//!
//! ```bash
//! python3 tests/fixtures/tiny-deepseek-v4-qmoe/generate.py \
//!   --mobius-root /path/to/mobius
//! ```
//!
//! `DEEPSEEK_V4_TINY_E2E_DIR` may override the committed fixture. Missing
//! fixture files skip cleanly so source packages that omit binary fixtures
//! remain green.
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

/// Locked native CPU/CUDA decode anchor for the committed fixture (seed 0,
/// prompt `[123]`), determined empirically and pinned as a regression guard
/// -- see `glm_tiny_qmoe_native_cuda_e2e.rs::ANCHOR_IDS` for the pattern this
/// mirrors.
const ANCHOR_IDS: &[u32] = &[66, 125, 15, 171, 76, 23, 202, 54, 231, 131, 147, 88];

fn resolve_model_path(dir: &Path) -> Option<PathBuf> {
    let onnx = dir.join("model.onnx");
    if onnx.is_file() {
        return Some(onnx);
    }
    let textproto = dir.join("model.onnx.textproto");
    if textproto.is_file() {
        return Some(textproto);
    }
    None
}

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("DEEPSEEK_V4_TINY_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-deepseek-v4-qmoe")
        });
    let mut missing: Vec<String> = Vec::new();
    if resolve_model_path(&dir).is_none() {
        missing.push("model.onnx or model.onnx.textproto".to_string());
    }
    for name in ["inference_metadata.yaml", "tokenizer.json"] {
        if !dir.join(name).is_file() {
            missing.push(name.to_string());
        }
    }
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping DeepSeek-V4 tiny QMoE regression: fixture {} is missing {}",
            dir.display(),
            missing.join(", ")
        );
        None
    }
}

fn engine(
    dir: &Path,
    backend: EngineDecodeBackend,
    device: Option<NativeDecodeDevice>,
) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device: device,
            ..EngineConfig::default()
        },
    )
}

fn generate(engine: &mut Engine) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![123]));
    request.options.max_new_tokens = ANCHOR_IDS.len();
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

/// Structural gate: no native-only `pkg.nxrt::*` op may appear (this
/// fixture's entire claim to stock-ORT executability depends on it), while
/// routed experts must be fused into one `com.microsoft::QMoE` per MoE layer
/// (mobius#550's export, not a per-expert dense loop).
///
/// Note: `onnx_runtime_loader::load_model` function-inlines `MatMulNBits`
/// into its primitive decomposition (`BitwiseAnd`/`BitShift` unpack +
/// `DequantizeLinear` + `MatMul`) at load time -- confirmed empirically: the
/// *loaded* graph has zero `MatMulNBits` nodes of any domain even though the
/// committed `model.onnx.textproto` file contains 18 of them, while
/// `com.microsoft::QMoE` is left untouched (2 nodes, matching layer count).
/// So the dense-per-expert-loop check below reads the raw textproto text
/// directly for the routed-expert weight-naming pattern
/// (`mlp.moe.experts.`) instead of asserting on loaded-graph node domains,
/// which would not distinguish the two cases post-inlining.
fn assert_current_emission(dir: &Path) -> anyhow::Result<()> {
    let model = resolve_model_path(dir)
        .ok_or_else(|| anyhow::anyhow!("{} has no model.onnx(.textproto)", dir.display()))?;
    let graph = onnx_runtime_loader::load_model(&model)?;
    assert_eq!(
        graph
            .nodes
            .values()
            .filter(|node| node.domain == "pkg.nxrt")
            .count(),
        0,
        "{} claims stock-ORT executability and must contain zero native-only \
         pkg.nxrt::* nodes",
        model.display(),
    );
    assert!(
        graph
            .nodes
            .values()
            .any(|node| node.domain == "com.microsoft" && node.op_type == "QMoE"),
        "{} does not contain fused QMoE",
        model.display()
    );
    if model.extension().and_then(|ext| ext.to_str()) == Some("textproto") {
        let text = std::fs::read_to_string(&model)?;
        assert!(
            !text.contains("mlp.moe.experts."),
            "{} unexpectedly contains a dense per-expert MoE weight (mlp.moe.experts.*) \
             instead of the fused QMoE path",
            model.display()
        );
    }
    Ok(())
}

#[test]
fn deepseek_v4_tiny_structural_emission_is_stock_ort_executable() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)
}

#[test]
fn deepseek_v4_tiny_native_cpu_eager_decode_locks_anchor_ids() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut cpu = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let tokens = generate(&mut cpu)?;
    eprintln!("deepseek-v4 tiny native CPU eager tokens: {tokens:?}");
    assert_eq!(tokens, ANCHOR_IDS);
    Ok(())
}

/// Native CUDA decode for DeepSeek-V4's decomposed (non-fused-`Attention`)
/// attention export. Previously a real, reproduced runtime gap; now fixed by
/// generalizing the capacity-substitution invariant instead of adding a
/// DeepSeek-specific allowlist branch. History kept below for context.
///
/// Root cause: DeepSeek-V4's smallest (dense CSA) config exports fully
/// **decomposed/manual** attention -- plain `MatMul`/`Softmax`/`Concat`/
/// `Unsqueeze`/`Expand`, no fused `Attention`/`GroupQueryAttention` op and no
/// native-only kernel (confirmed: `model.onnx.textproto` has zero `Attention`-
/// family nodes). Every other model this native CUDA decoder previously
/// supported (GLM-5.2 full-attention, GLM-5.2 IndexShare, DeepSeek-V2/V3-Lite)
/// instead terminates its KV-consuming subgraph at a single kernel (a fused
/// `Attention` op, or the native `pkg.nxrt::IndexShare` op) that has its own
/// internal logic for reconciling a physically-wider capacity KV buffer
/// against the logical valid length. DeepSeek-V4's decomposed graph has no
/// such kernel: `present.0.key` is a plain `Concat` output consumed further
/// **in-graph** by an `Unsqueeze`/`Expand` GQA `repeat_kv` broadcast, then
/// combined (via `MatMul`+`Add`) with a causal-mask bias built from the
/// *separate* `attention_mask` input.
///
/// Originally reproduced failure (this binary, `CUDA_VISIBLE_DEVICES=1`):
/// ```text
/// node 90 ("model/layers.0/self_attn/Unsqueeze_node_94", op '::Unsqueeze',
/// inputs ["present.0.key", "const_1d_4"] [Float32, Int64]
/// [[1, 1, 256, 16], [1]], outputs ["...Unsqueeze_94"] [Float32]
/// [[1, 1, 1, 1, 16]]) failed: kernel execution failed:
/// cuda_ep Unsqueeze: input/output dtype and element count must match
/// ```
/// `present.0.key`'s declared shape was already the physical KV capacity
/// (256) by the time `Unsqueeze` read it in-graph, but `Unsqueeze`'s own
/// output was sized from the *logical* (small) value, so their element
/// counts disagreed.
///
/// A first fix attempt widened the `Concat`'s own `output_shapes[0]` (mirroring
/// `Attention`'s widening block in `dispatch.rs`) so `Unsqueeze` would see a
/// consistent physical shape. That traded one crash for another: a plain
/// `Concat` *kernel* independently validates `output.shape[axis] ==
/// past.shape[axis] + current.shape[axis]`, and correctly writes only that
/// (small) delta relying on present==past device buffer aliasing for the
/// append -- widening its own declared output shape made that arithmetic
/// check fail (`cuda_ep Concat: output dtype or shape mismatch`).
///
/// **Actual fix** (`onnx-runtime-session/src/executor/{geometry,dispatch}.rs`):
/// decouple the two concerns the first attempt conflated. `geometry.rs` gained
/// `is_kv_cache_growth_concat`/`derives_from_kv_cache_growth` -- a structural
/// (not model-specific) recognizer for "a plain `Concat` whose input 0 is a
/// graph input and output 0 is a graph output" (a decomposed KV-cache append),
/// and used it to extend `classify_mask_consumer`'s existing invariant (an
/// axis may be substituted with physical capacity iff every consumer either
/// sources that axis from the same substitution or is neutralized before a
/// non-padding-aware sink) to a decomposed `Add(score, mask)` -> `Softmax`
/// chain, exactly mirroring the fused-`Attention` case it already handled.
/// `dispatch.rs`'s widening block gained an `is_kv_cache_growth_concat` branch
/// that -- unlike `Attention`'s, which needs its physical shape for its own
/// kernel's in-place-append addressing -- leaves this node's own
/// `output_shapes[0]` naive (so the `Concat` kernel's arithmetic check keeps
/// passing) and instead corrects only `resolved`, the map same-step downstream
/// consumers (`Shape`/`Unsqueeze` reading via `refill_input_shapes`) actually
/// read from -- which is what was stale before this fix.
///
/// CPU decode and stock ORT (both exercised by the other tests in this file)
/// already executed this exact fixture correctly end-to-end throughout, so
/// this was strictly a native-CUDA-decode-engine gap, not a fixture, export,
/// or numerics bug.
///
/// **CUDA-graph capture stays declined for this fixture, correctly and by a
/// separate mechanism than the crash above.** Whole-step capture requires the
/// persistent `past_key_values.*` **input** bindings to be pinned at physical
/// capacity from the very first (empty) prefill step
/// (`GraphCaptureDecision`'s `persistent_inputs_have_fixed_logical_shapes`
/// predicate, `native_decode/cuda.rs`); that in turn requires every *direct*
/// consumer of that input to be capacity-aware
/// (`build.rs::binding_consumers_use_physical_capacity`,
/// `geometry.rs::kernel_input_uses_physical_capacity`), which today only
/// recognizes `Attention`/`GroupQueryAttention`/`pkg.nxrt::IndexShare` --
/// kernels deliberately designed with an auxiliary total-length signal so a
/// physical/padded cache extent never corrupts their arithmetic. A plain
/// `Concat` (this fixture's cache-growth op) has no such duality: its kernel
/// computes `output.shape[axis] = past.shape[axis] + current.shape[axis]`
/// literally, so feeding it a physical-capacity past shape would silently
/// reproduce a variant of the original crash (an ever-growing, wrong output
/// length) rather than becoming capture-safe. Reaching `captures>0` for a
/// `Concat`-based cache append therefore needs a genuinely different decode-
/// time write mechanism (e.g. a fixed-slot copy/scatter op in place of
/// `Concat`) -- a substantial new capability, not a classifier generalization
/// -- matching the same conclusion `glm_tiny_qmoe_native_cuda_e2e.rs`'s
/// `..._declines_capture` test already documents for GLM-5.2's own
/// concat/logical form ("must remain eager before S3 capacity emission").
/// Building that write mechanism is out of scope here (explicit user
/// direction: extend the existing classifier, do not modify the Mobius
/// exporter or invent a DeepSeek-specific path) and is the concrete follow-up
/// tracked for whole-step capture on this fixture. This test instead asserts
/// what the fix actually delivers: correct, eager, non-fallback CUDA decode
/// that is token-identical to CPU.
#[test]
fn deepseek_v4_tiny_native_cuda_matches_cpu() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping DeepSeek-V4 tiny native CUDA regression: CUDA is unavailable: {error}");
        return Ok(());
    }
    assert_current_emission(&dir)?;

    let mut cpu = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let cpu_tokens = generate(&mut cpu)?;
    assert_eq!(cpu_tokens, ANCHOR_IDS);

    let mut cuda = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
    )?;
    let cuda_tokens = generate(&mut cuda)?;
    let stats = cuda
        .native_cuda_debug_stats()
        .expect("native CUDA engine exposes decode diagnostics");
    eprintln!(
        "deepseek-v4 tiny native CUDA tokens: {cuda_tokens:?}; \
         captures={} replays={} fallbacks={} decline_reason={:?}",
        stats.graph.captures,
        stats.graph.replays,
        stats.graph.fallbacks,
        stats.graph.decline_reason
    );
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native CUDA diverged from native CPU"
    );
    assert_eq!(
        stats.graph.fallbacks, 0,
        "eager decode on a capacity-ineligible fixture must not additionally suffer runtime \
         capture-attempt fallbacks"
    );
    // Whole-step capture is not expected here -- see the doc comment above --
    // but the reason must be exactly the known, pre-existing structural one
    // (a `past_key_values.*` input consumed by a plain `Concat`), not some new
    // regression the classifier fix introduced.
    assert_eq!(
        stats.graph.captures, 0,
        "unexpected CUDA graph capture for a Concat-cache-growth fixture -- \
         if a capacity-write mechanism now makes this fixture capture-eligible, \
         update this assertion and the doc comment above deliberately"
    );
    assert!(
        stats
            .graph
            .decline_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("persistent_inputs_have_fixed_logical_shapes")),
        "expected capture to be declined specifically by the persistent-input capacity \
         predicate, got: {:?}",
        stats.graph.decline_reason
    );
    Ok(())
}

/// No success-shaped skip: this is the one test proving the dense
/// DeepSeek-V4 export is executable on **stock ONNX Runtime** with no
/// native-only op dependency at all. Its tokens must agree with the native
/// CPU path exactly -- both run the same graph and the same greedy/
/// deterministic decode, so any divergence is a real numeric or op-semantics
/// bug, not an expected difference.
#[test]
fn deepseek_v4_tiny_stock_ort_matches_native_cpu() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    assert_current_emission(&dir)?;

    let mut native = engine(
        &dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cpu),
    )?;
    let native_tokens = generate(&mut native)?;
    assert_eq!(native_tokens, ANCHOR_IDS);

    let mut ort = engine(&dir, EngineDecodeBackend::Ort, None)?;
    let ort_tokens = generate(&mut ort)?;
    eprintln!("deepseek-v4 tiny stock ORT tokens: {ort_tokens:?}");
    assert_eq!(
        ort_tokens, native_tokens,
        "stock ORT execution diverged from the native CPU backend"
    );
    Ok(())
}
