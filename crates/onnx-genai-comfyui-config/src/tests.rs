//! Conversion tests: graph walking, fail-closed refusals, and canonical shape.

use serde_json::{Value, json};

use crate::{
    ComfyUiConfigError, ComponentLayout, Conditioning, ConvertOptions, LatentSource, Prediction,
    Solver, Spacing, convert, strength_to_start_step,
};

/// The canonical KSampler-centric text-to-image graph ComfyUI exports.
fn txt2img() -> Value {
    json!({
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 7.5, "sampler_name": "euler",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}
    })
}

fn options() -> ConvertOptions {
    ConvertOptions::default()
}

fn convert_ok(workflow: &Value) -> (Value, crate::ConversionReport) {
    let (_, document, report) = convert(workflow, &options()).expect("conversion should succeed");
    (document, report)
}

fn workflow_of(document: &Value) -> &Value {
    &document["pipeline"]["workflow"]
}

fn step_kinds(document: &Value) -> Vec<String> {
    workflow_of(document)["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .map(|step| step["kind"].as_str().unwrap_or_default().to_owned())
        .collect()
}

/// Every component a loop body invokes, in order.
fn body_components(document: &Value) -> Vec<String> {
    workflow_of(document)["steps"][0]["steps"]
        .as_array()
        .expect("loop body")
        .iter()
        .filter_map(|step| step["component"].as_str().map(str::to_owned))
        .collect()
}

fn setup_components(document: &Value) -> Vec<String> {
    workflow_of(document)["steps"][0]["setup"]
        .as_array()
        .expect("loop setup")
        .iter()
        .filter_map(|step| step["component"].as_str().map(str::to_owned))
        .collect()
}

#[test]
fn converts_core_txt2img_into_canonical_workflow() {
    let (document, report) = convert_ok(&txt2img());
    assert_eq!(report.plan.steps, 20);
    assert_eq!(report.plan.solver, Solver::Euler);
    assert_eq!(report.plan.spacing, Spacing::Karras);
    assert_eq!(report.plan.prediction, Prediction::Epsilon);
    assert_eq!(report.plan.seed, 42);
    assert_eq!(report.plan.prompt.as_deref(), Some("a fox"));
    assert_eq!(report.plan.negative_prompt.as_deref(), Some("blurry"));
    assert_eq!(report.plan.checkpoint.as_deref(), Some("sd15.safetensors"));
    assert!(matches!(
        report.plan.latent,
        LatentSource::Noise {
            width: 512,
            height: 512,
            batch_size: 1
        }
    ));

    // The emitted document is a canonical workflow: a loop, then the decode and
    // the emits. Nothing about it is ComfyUI-shaped.
    assert_eq!(step_kinds(&document), ["loop", "invoke", "emit", "emit"]);
    let workflow = workflow_of(&document);
    assert!(workflow["manifest"].get("ir_version").is_none());
    assert!(workflow["manifest"].get("onnx_opsets").is_none());
    assert_eq!(
        workflow["steps"][0]["max_iterations"],
        json!("request.max_iterations")
    );
    assert_eq!(
        workflow["inputs"]["request.max_iterations"]["default"],
        json!(20)
    );
    assert_eq!(workflow["outputs"]["image"]["role"], json!("image"));
    assert_eq!(
        workflow["outputs"]["image"]["value_range"],
        json!("negative_one_to_one")
    );
}

#[test]
fn guidance_becomes_two_encoder_passes_and_a_combine() {
    let (document, report) = convert_ok(&txt2img());
    assert_eq!(
        report
            .plan
            .guidance
            .expect("cfg 7.5 enables guidance")
            .scale,
        7.5
    );

    // Two text-encoder invocations, one per conditioning branch.
    let setup = setup_components(&document);
    assert_eq!(
        setup.iter().filter(|name| *name == "text_encoder").count(),
        2
    );

    // Two denoiser passes and one guidance combine, in that order.
    let body = body_components(&document);
    assert_eq!(body.iter().filter(|name| *name == "denoiser").count(), 2);
    let combine = body.iter().position(|name| name == "guidance_combine");
    let solver = body.iter().position(|name| name == "solver_step");
    assert!(
        combine < solver,
        "guidance must combine before the solver steps"
    );

    let workflow = workflow_of(&document);
    assert_eq!(
        workflow["components"]["guidance_combine"]["contract"]["id"],
        json!("onnx-genai.guidance-combine")
    );
    assert_eq!(
        workflow["inputs"]["request.guidance_scale"]["default"],
        json!(7.5)
    );
    assert_eq!(
        workflow["inputs"]["request.guidance_scale"]["role"]["role"],
        json!("guidance_scale")
    );
    assert_eq!(
        workflow["inputs"]["request.negative_input_ids"]["role"]["role"],
        json!("negative_prompt_tokens")
    );
}

#[test]
fn cfg_of_one_emits_a_single_unguided_denoiser_pass() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["cfg"] = json!(1.0);
    let (document, report) = convert_ok(&workflow);
    assert!(report.plan.guidance.is_none());
    let body = body_components(&document);
    assert_eq!(body.iter().filter(|name| *name == "denoiser").count(), 1);
    assert!(!body.iter().any(|name| name == "guidance_combine"));
    assert!(
        workflow_of(&document)["inputs"]
            .get("request.negative_input_ids")
            .is_none()
    );
}

#[test]
fn scheduler_choice_reaches_the_solver_contract() {
    for (sampler, spacing, solver, expected_spacing) in [
        ("euler", "normal", Solver::Euler, "linear"),
        ("ddim", "ddim_uniform", Solver::Ddim, "ddim_uniform"),
        ("dpmpp_2m", "karras", Solver::DpmSolverPlusPlus2M, "karras"),
        (
            "euler_ancestral",
            "exponential",
            Solver::EulerAncestral,
            "exponential",
        ),
    ] {
        let mut workflow = txt2img();
        workflow["3"]["inputs"]["sampler_name"] = json!(sampler);
        workflow["3"]["inputs"]["scheduler"] = json!(spacing);
        let (document, report) = convert_ok(&workflow);
        assert_eq!(report.plan.solver, solver, "{sampler}");
        let contract = &workflow_of(&document)["components"]["solver_step"]["contract"];
        assert_eq!(contract["id"], json!("onnx-genai.solver-step"));
        assert_eq!(contract["parameters"]["solver"], json!(solver.as_str()));
        assert_eq!(contract["parameters"]["spacing"], json!(expected_spacing));
    }
}

#[test]
fn multistep_solver_carries_a_history_cell() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["sampler_name"] = json!("dpmpp_2m");
    let (document, _) = convert_ok(&workflow);
    let workflow_ir = workflow_of(&document);
    assert!(workflow_ir["state"]["history"].is_object());
    assert_eq!(
        workflow_ir["components"]["solver_step"]["contract"]["bindings"]["history"],
        json!("history")
    );
    let carried = workflow_ir["steps"][0]["carried"]
        .as_array()
        .expect("carried");
    assert!(
        carried
            .iter()
            .any(|entry| entry["cell"] == json!("history"))
    );
    assert!(setup_components(&document).contains(&"history_initializer".to_owned()));
}

#[test]
fn ancestral_solver_draws_fresh_noise_each_step() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["sampler_name"] = json!("euler_ancestral");
    let (document, _) = convert_ok(&workflow);
    let body = body_components(&document);
    assert!(body.contains(&"latent_noise".to_owned()));
    let workflow_ir = workflow_of(&document);
    assert!(workflow_ir["state"]["rng_offset"].is_object());
    assert_eq!(
        workflow_ir["components"]["solver_step"]["contract"]["bindings"]["noise"],
        json!("noise")
    );
}

#[test]
fn unsupported_sampler_fails_closed_with_a_remedy() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["sampler_name"] = json!("uni_pc_bh2");
    let error = convert(&workflow, &options()).expect_err("unknown sampler must fail");
    let message = error.to_string();
    assert!(message.contains("uni_pc_bh2"), "{message}");
    assert!(
        message.contains('3'),
        "the node id must be named: {message}"
    );
    assert!(
        message.contains("euler"),
        "the remedy must list what works: {message}"
    );
}

#[test]
fn unsupported_spacing_fails_closed_rather_than_falling_back() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["scheduler"] = json!("sgm_uniform");
    let error = convert(&workflow, &options()).expect_err("unknown spacing must fail");
    let message = error.to_string();
    assert!(message.contains("sgm_uniform"), "{message}");
    assert!(
        message.contains("silently produce a different image"),
        "{message}"
    );
}

// ── image-to-image ──────────────────────────────────────────────────────────

fn img2img(denoise: f64) -> Value {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["denoise"] = json!(denoise);
    workflow["3"]["inputs"]["latent_image"] = json!(["10", 0]);
    workflow["10"] = json!({
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["11", 0], "vae": ["4", 2]}
    });
    workflow["11"] = json!({"class_type": "LoadImage", "inputs": {"image": "cat.png"}});
    workflow
}

#[test]
fn strength_maps_to_the_diffusers_start_step() {
    assert_eq!(strength_to_start_step(1.0, 20), 0);
    assert_eq!(strength_to_start_step(0.75, 20), 5);
    assert_eq!(strength_to_start_step(0.5, 20), 10);
    assert_eq!(strength_to_start_step(0.0, 20), 20);
    // Round half to even, matching numpy and diffusers `get_timesteps`.
    assert_eq!(strength_to_start_step(0.25, 10), 8);
    assert_eq!(strength_to_start_step(0.25, 6), 4);
}

#[test]
fn img2img_encodes_the_source_and_renoises_at_the_start_step() {
    let (document, report) = convert_ok(&img2img(0.75));
    assert_eq!(report.plan.start_step, 5);
    assert_eq!(report.plan.end_step, 20);
    assert_eq!(report.plan.iterations(), 15);
    assert!(matches!(
        report.plan.latent,
        LatentSource::Image { strength, ref image }
            if (strength - 0.75).abs() < 1e-9 && image.as_deref() == Some("cat.png")
    ));

    let setup = setup_components(&document);
    assert!(setup.contains(&"vae_encoder".to_owned()));
    assert!(setup.contains(&"add_noise".to_owned()));

    let workflow = workflow_of(&document);
    // The loop runs only the remaining steps, and the schedule index is offset
    // so step 0 of the loop is step 5 of the schedule.
    assert_eq!(
        workflow["inputs"]["request.max_iterations"]["default"],
        json!(15)
    );
    assert_eq!(
        workflow["inputs"]["package.start_step"]["default"],
        json!(5)
    );
    assert!(body_components(&document).contains(&"step_offset".to_owned()));
    assert_eq!(
        workflow["state"]["latent_state"]["initializer"],
        json!("diffusion.initial_latent")
    );
}

#[test]
fn full_strength_img2img_runs_every_step_without_an_offset() {
    let (document, report) = convert_ok(&img2img(1.0));
    assert_eq!(report.plan.start_step, 0);
    assert!(!body_components(&document).contains(&"step_offset".to_owned()));
    assert!(
        workflow_of(&document)["inputs"]
            .get("package.start_step")
            .is_none()
    );
}

#[test]
fn zero_strength_fails_closed_instead_of_emitting_an_empty_loop() {
    let error = convert(&img2img(0.0), &options()).expect_err("no steps must fail");
    assert!(error.to_string().contains("executes no steps"), "{error}");
}

// ── inpainting ──────────────────────────────────────────────────────────────

fn inpaint() -> Value {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["denoise"] = json!(1.0);
    workflow["3"]["inputs"]["latent_image"] = json!(["10", 0]);
    workflow["10"] = json!({
        "class_type": "VAEEncodeForInpaint",
        "inputs": {"pixels": ["11", 0], "mask": ["12", 0], "vae": ["4", 2], "grow_mask_by": 6}
    });
    workflow["11"] = json!({"class_type": "LoadImage", "inputs": {"image": "room.png"}});
    workflow["12"] = json!({"class_type": "LoadImageMask", "inputs": {"image": "room_mask.png", "channel": "red"}});
    workflow
}

#[test]
fn inpainting_keeps_the_mask_as_a_per_step_blend() {
    let (document, report) = convert_ok(&inpaint());
    assert!(matches!(
        report.plan.latent,
        LatentSource::Inpaint { ref mask, grow_mask_by: 6, .. }
            if mask.as_deref() == Some("room_mask.png")
    ));

    let workflow = workflow_of(&document);
    // The mask is a typed workflow input, not a run parameter the converter
    // folded away, and it gates the latent every step rather than once at the end.
    assert_eq!(
        workflow["inputs"]["request.mask"]["contract"]["rank"],
        json!(4)
    );
    let body = body_components(&document);
    let blend = body
        .iter()
        .position(|name| name == "masked_blend")
        .expect("blend");
    let solver = body
        .iter()
        .position(|name| name == "solver_step")
        .expect("solver");
    assert!(
        solver < blend,
        "the mask must be applied to the solver's output"
    );
    assert_eq!(
        workflow["components"]["masked_blend"]["contract"]["id"],
        json!("onnx-genai.masked-blend")
    );
    let carried = workflow["steps"][0]["carried"].as_array().expect("carried");
    let latent = carried
        .iter()
        .find(|entry| entry["cell"] == json!("latent_state"))
        .expect("latent carry");
    assert_eq!(latent["next"], json!("latent.blended"));
}

#[test]
fn set_latent_noise_mask_is_recognized_as_inpainting() {
    let mut workflow = img2img(0.6);
    workflow["3"]["inputs"]["latent_image"] = json!(["13", 0]);
    workflow["13"] = json!({
        "class_type": "SetLatentNoiseMask",
        "inputs": {"samples": ["10", 0], "mask": ["12", 0]}
    });
    workflow["12"] =
        json!({"class_type": "LoadImageMask", "inputs": {"image": "m.png", "channel": "red"}});
    let (document, report) = convert_ok(&workflow);
    assert!(report.plan.is_inpainting());
    assert!(body_components(&document).contains(&"masked_blend".to_owned()));
}

// ── SDXL ────────────────────────────────────────────────────────────────────

fn sdxl() -> Value {
    let mut workflow = txt2img();
    for id in ["6", "7"] {
        workflow[id] = json!({
            "class_type": "CLIPTextEncodeSDXL",
            "inputs": {
                "width": 1024, "height": 1024, "crop_w": 0, "crop_h": 0,
                "target_width": 1024, "target_height": 1024,
                "text_g": "a fox", "text_l": "a fox", "clip": ["4", 1]
            }
        });
    }
    workflow
}

#[test]
fn sdxl_emits_dual_encoders_and_routes_time_ids() {
    let (document, report) = convert_ok(&sdxl());
    assert_eq!(report.plan.conditioning, Conditioning::SdxlDual);
    let workflow = workflow_of(&document);
    assert!(workflow["components"]["text_encoder_2"].is_object());
    assert_eq!(
        workflow["inputs"]["request.time_ids"]["contract"]["shape"],
        json!(["batch", 6])
    );
    let denoiser_inputs = &workflow["components"]["denoiser"]["ports"]["inputs"];
    assert!(denoiser_inputs["text_embeds"].is_object());
    assert!(denoiser_inputs["time_ids"].is_object());
    let setup = setup_components(&document);
    assert_eq!(
        setup
            .iter()
            .filter(|name| *name == "text_encoder_2")
            .count(),
        2
    );
}

#[test]
fn mismatched_encoder_families_fail_closed() {
    let mut workflow = sdxl();
    workflow["7"] =
        json!({"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}});
    let error = convert(&workflow, &options()).expect_err("mixed encoders must fail");
    assert!(error.to_string().contains("SDXL dual encoder"), "{error}");
}

// ── ControlNet ──────────────────────────────────────────────────────────────

fn controlnet(class: &str) -> Value {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["positive"] = json!(["20", 0]);
    let mut inputs = json!({
        "strength": 0.7,
        "control_net": ["21", 0],
        "image": ["22", 0]
    });
    if class == "ControlNetApplyAdvanced" {
        inputs["positive"] = json!(["6", 0]);
        inputs["negative"] = json!(["7", 0]);
        inputs["start_percent"] = json!(0.0);
        inputs["end_percent"] = json!(1.0);
        workflow["3"]["inputs"]["negative"] = json!(["20", 1]);
    } else {
        inputs["conditioning"] = json!(["6", 0]);
    }
    workflow["20"] = json!({"class_type": class, "inputs": inputs});
    workflow["21"] = json!({"class_type": "ControlNetLoader", "inputs": {"control_net_name": "canny.safetensors"}});
    workflow["22"] = json!({"class_type": "LoadImage", "inputs": {"image": "edges.png"}});
    workflow
}

#[test]
fn single_controlnet_becomes_a_component_invocation_with_runtime_strength() {
    let (document, report) = convert_ok(&controlnet("ControlNetApply"));
    assert_eq!(report.plan.controlnets.len(), 1);
    assert_eq!(report.plan.controlnets[0].name, "canny.safetensors");
    assert_eq!(report.plan.controlnets[0].strength, 0.7);
    assert_eq!(
        report.plan.controlnets[0].image.as_deref(),
        Some("edges.png")
    );

    let workflow = workflow_of(&document);
    assert_eq!(
        workflow["components"]["controlnet"]["contract"]["id"],
        json!("onnx-genai.controlnet-residual")
    );
    // The strength stays a typed runtime input; it is never folded into a constant.
    assert_eq!(
        workflow["inputs"]["request.control_strength"]["default"],
        json!(0.7)
    );
    assert!(workflow["components"]["denoiser"]["ports"]["inputs"]["control"].is_object());
    // A basic apply patches one branch, so one ControlNet invocation per step.
    assert_eq!(
        body_components(&document)
            .iter()
            .filter(|name| *name == "controlnet")
            .count(),
        1
    );
}

#[test]
fn advanced_controlnet_patches_both_conditioning_branches() {
    let (document, report) = convert_ok(&controlnet("ControlNetApplyAdvanced"));
    assert!(report.plan.controlnets[0].applies_to_negative);
    assert_eq!(
        body_components(&document)
            .iter()
            .filter(|name| *name == "controlnet")
            .count(),
        2
    );
}

#[test]
fn chained_controlnets_fail_closed() {
    let mut workflow = controlnet("ControlNetApply");
    workflow["23"] = json!({
        "class_type": "ControlNetApply",
        "inputs": {"strength": 0.4, "control_net": ["24", 0], "image": ["22", 0], "conditioning": ["20", 0]}
    });
    workflow["24"] = json!({"class_type": "ControlNetLoader", "inputs": {"control_net_name": "depth.safetensors"}});
    workflow["3"]["inputs"]["positive"] = json!(["23", 0]);
    let error = convert(&workflow, &options()).expect_err("multi-controlnet must fail");
    let message = error.to_string();
    assert!(message.contains("chained ControlNets"), "{message}");
    assert!(message.contains("silently drop"), "{message}");
}

#[test]
fn step_windowed_controlnet_fails_closed() {
    let mut workflow = controlnet("ControlNetApplyAdvanced");
    workflow["20"]["inputs"]["end_percent"] = json!(0.6);
    let error = convert(&workflow, &options()).expect_err("windowed controlnet must fail");
    assert!(
        error.to_string().contains("step-windowed ControlNet"),
        "{error}"
    );
}

#[test]
fn a_preprocessed_hint_image_fails_closed() {
    let mut workflow = controlnet("ControlNetApply");
    workflow["22"] = json!({"class_type": "CannyEdgePreprocessor", "inputs": {"image": ["25", 0]}});
    workflow["25"] = json!({"class_type": "LoadImage", "inputs": {"image": "photo.png"}});
    let error = convert(&workflow, &options()).expect_err("preprocessor must fail");
    assert!(
        error.to_string().contains("run the preprocessor offline"),
        "{error}"
    );
}

// ── LoRA ────────────────────────────────────────────────────────────────────

fn lora_workflow() -> Value {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["model"] = json!(["30", 0]);
    workflow["6"]["inputs"]["clip"] = json!(["30", 1]);
    workflow["7"]["inputs"]["clip"] = json!(["30", 1]);
    workflow["30"] = json!({
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "detail.safetensors", "strength_model": 0.8, "strength_clip": 0.6,
            "model": ["4", 0], "clip": ["4", 1]
        }
    });
    workflow
}

fn adapter_contract() -> Value {
    json!({
        "target_manifest": {"targets": [{
            "id": "denoiser.block0.to_q",
            "component": "denoiser",
            "initializer": "block0.to_q.weight",
            "node_name": "/block0/to_q/MatMul",
            "output_name": "/block0/to_q/MatMul_output_0",
            "activation_dtype": "float32",
            "input_features": 8,
            "output_features": 8
        }]},
        "application_capability": "onnx-genai.adapters@1",
        "artifacts": {"detail": {
            "index": 0,
            "identity": "detail",
            "version": "1",
            "rank": 4,
            "alpha": 8.0,
            "dtype": "float32",
            "weights": [{
                "location": "adapters/detail/weights.json",
                "loader_capability": "onnx-genai.adapters.json@1",
                "scale_encoding": "alpha_over_rank",
                "format": "json"
            }],
            "bindings": [{"target": "denoiser.block0.to_q", "weight_key": "block0.to_q"}]
        }},
        "selection": {
            "segments": "request.adapter_segments",
            "adapter_counts": "request.adapter_counts",
            "scales": "request.adapter_scales",
            "max_adapters": 2
        }
    })
}

#[test]
fn lora_without_a_package_adapter_contract_fails_closed() {
    let error = convert(&lora_workflow(), &options()).expect_err("bare LoRA must fail");
    let message = error.to_string();
    assert!(message.contains("detail.safetensors"), "{message}");
    assert!(message.contains("base-model fingerprint"), "{message}");
}

#[test]
fn lora_routes_through_canonical_adapter_selection() {
    let options = ConvertOptions {
        adapters: Some(adapter_contract()),
        ..ConvertOptions::default()
    };
    let (_, document, report) = convert(&lora_workflow(), &options).expect("lora conversion");
    assert_eq!(report.adapters, ["detail.safetensors"]);
    assert_eq!(report.plan.loras[0].model_strength, 0.8);
    assert_eq!(report.plan.loras[0].clip_strength, Some(0.6));

    let workflow = workflow_of(&document);
    for name in [
        "request.adapter_segments",
        "request.adapter_counts",
        "request.adapter_scales",
    ] {
        assert!(workflow["inputs"][name].is_object(), "missing {name}");
    }
    let capabilities = workflow["manifest"]["capabilities"]
        .as_array()
        .expect("capabilities");
    assert!(capabilities.contains(&json!("parameter_adapters")));
    assert!(capabilities.contains(&json!("heterogeneous_adapter_batching")));
    assert_eq!(
        document["adapters"]["selection"]["segments"],
        json!("request.adapter_segments")
    );
}

#[test]
fn a_lora_the_package_does_not_declare_fails_closed() {
    let mut workflow = lora_workflow();
    workflow["30"]["inputs"]["lora_name"] = json!("unknown.safetensors");
    let options = ConvertOptions {
        adapters: Some(adapter_contract()),
        ..ConvertOptions::default()
    };
    let error = convert(&workflow, &options).expect_err("undeclared lora must fail");
    assert!(error.to_string().contains("unknown.safetensors"), "{error}");
}

// ── fail-closed topology ────────────────────────────────────────────────────

#[test]
fn an_unknown_node_on_the_output_path_fails_closed() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["model"] = json!(["40", 0]);
    workflow["40"] = json!({"class_type": "MysteryCustomNode", "inputs": {"model": ["4", 0]}});
    let error = convert(&workflow, &options()).expect_err("unknown node must fail");
    let message = error.to_string();
    assert!(message.contains("MysteryCustomNode"), "{message}");
    assert!(message.contains("40"), "{message}");
    assert!(message.contains("fail-closed"), "{message}");
}

#[test]
fn an_unknown_node_off_the_output_path_is_reported_and_ignored() {
    let mut workflow = txt2img();
    workflow["50"] = json!({"class_type": "NoteNode", "inputs": {"text": "scratch"}});
    let (_, report) = convert_ok(&workflow);
    assert!(
        report
            .ignored_nodes
            .iter()
            .any(|node| node.contains("NoteNode")),
        "{:?}",
        report.ignored_nodes
    );
}

#[test]
fn two_image_sinks_are_ambiguous() {
    let mut workflow = txt2img();
    workflow["51"] = json!({"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}});
    let error = convert(&workflow, &options()).expect_err("two sinks must fail");
    assert!(error.to_string().contains("image sinks"), "{error}");
}

#[test]
fn a_workflow_with_no_image_sink_fails_closed() {
    let mut workflow = txt2img();
    workflow.as_object_mut().expect("object").remove("9");
    let error = convert(&workflow, &options()).expect_err("no sink must fail");
    assert!(
        matches!(error, ComfyUiConfigError::NoOutputPath { .. }),
        "{error}"
    );
}

#[test]
fn a_dangling_link_fails_closed() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["model"] = json!(["99", 0]);
    let error = convert(&workflow, &options()).expect_err("dangling link must fail");
    assert!(
        matches!(error, ComfyUiConfigError::DanglingLink { .. }),
        "{error}"
    );
}

#[test]
fn a_patched_denoiser_fails_closed() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["model"] = json!(["60", 0]);
    workflow["60"] = json!({
        "class_type": "FreeU_V2",
        "inputs": {"model": ["4", 0], "b1": 1.1, "b2": 1.2, "s1": 0.9, "s2": 0.2}
    });
    let error = convert(&workflow, &options()).expect_err("FreeU must fail");
    assert!(
        error.to_string().contains("patches the denoiser"),
        "{error}"
    );
}

#[test]
fn merged_conditioning_fails_closed() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["positive"] = json!(["61", 0]);
    workflow["61"] = json!({
        "class_type": "ConditioningCombine",
        "inputs": {"conditioning_1": ["6", 0], "conditioning_2": ["7", 0]}
    });
    let error = convert(&workflow, &options()).expect_err("combine must fail");
    assert!(
        error.to_string().contains("merged before the sampler"),
        "{error}"
    );
}

#[test]
fn a_truncated_clip_fails_closed() {
    let mut workflow = txt2img();
    workflow["6"]["inputs"]["clip"] = json!(["62", 0]);
    workflow["62"] = json!({
        "class_type": "CLIPSetLastLayer",
        "inputs": {"clip": ["4", 1], "stop_at_clip_layer": -2}
    });
    let error = convert(&workflow, &options()).expect_err("clip skip must fail");
    assert!(
        error.to_string().contains("truncated at CLIP layer -2"),
        "{error}"
    );
}

#[test]
fn a_no_op_clip_set_last_layer_is_accepted() {
    let mut workflow = txt2img();
    workflow["6"]["inputs"]["clip"] = json!(["62", 0]);
    workflow["62"] = json!({
        "class_type": "CLIPSetLastLayer",
        "inputs": {"clip": ["4", 1], "stop_at_clip_layer": -1}
    });
    convert_ok(&workflow);
}

// ── flow matching ───────────────────────────────────────────────────────────

#[test]
fn flow_matching_reaches_the_solver_contract() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["model"] = json!(["70", 0]);
    workflow["3"]["inputs"]["latent_image"] = json!(["71", 0]);
    workflow["70"] = json!({
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["4", 0], "shift": 3.0}
    });
    workflow["71"] = json!({
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
    });
    let (document, report) = convert_ok(&workflow);
    assert_eq!(report.plan.prediction, Prediction::FlowVelocity);
    assert_eq!(report.plan.latent_channels, 16);
    assert_eq!(
        workflow_of(&document)["components"]["solver_step"]["contract"]["parameters"]["prediction"],
        json!("flow_velocity")
    );
}

#[test]
fn ddim_over_a_flow_matching_model_fails_closed() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["sampler_name"] = json!("ddim");
    workflow["3"]["inputs"]["model"] = json!(["70", 0]);
    workflow["70"] = json!({
        "class_type": "ModelSamplingFlux",
        "inputs": {"model": ["4", 0], "max_shift": 1.15, "base_shift": 0.5, "width": 1024, "height": 1024}
    });
    let error = convert(&workflow, &options()).expect_err("ddim + flow must fail");
    assert!(error.to_string().contains("flow-matching"), "{error}");
}

#[test]
fn a_qwen_image_specific_node_fails_closed_by_name() {
    let mut workflow = txt2img();
    workflow["3"]["inputs"]["positive"] = json!(["80", 0]);
    workflow["80"] = json!({
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {"clip": ["4", 1], "prompt": "make it blue", "vae": ["4", 2], "image": ["81", 0]}
    });
    workflow["81"] = json!({"class_type": "LoadImage", "inputs": {"image": "src.png"}});
    let error = convert(&workflow, &options()).expect_err("qwen edit must fail");
    let message = error.to_string();
    assert!(message.contains("TextEncodeQwenImageEdit"), "{message}");
    assert!(message.contains("80"), "{message}");
}

// ── determinism, identity, and layout ───────────────────────────────────────

#[test]
fn conversion_is_deterministic() {
    let (first, _) = convert_ok(&txt2img());
    let (second, _) = convert_ok(&txt2img());
    assert_eq!(first, second);
    assert_eq!(
        crate::to_yaml(&first).expect("yaml"),
        crate::to_yaml(&second).expect("yaml")
    );
}

#[test]
fn semantically_identical_workflows_convert_to_one_identity() {
    // Renumbering nodes and reordering the document changes nothing about what
    // the workflow computes, so the canonical metadata must be identical.
    let renumbered = json!({
        "b": {"class_type": "SaveImage", "inputs": {"images": ["a", 0]}},
        "a": {"class_type": "VAEDecode", "inputs": {"samples": ["s", 0], "vae": ["ck", 2]}},
        "s": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 7.5, "sampler_name": "euler",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["ck", 0], "positive": ["p", 0], "negative": ["n", 0],
            "latent_image": ["l", 0]}},
        "ck": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
        "l": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "p": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["ck", 1]}},
        "n": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["ck", 1]}}
    });
    let (baseline, _) = convert_ok(&txt2img());
    let (permuted, _) = convert_ok(&renumbered);
    assert_eq!(
        onnx_genai_metadata::semantic_identity(&baseline),
        onnx_genai_metadata::semantic_identity(&permuted),
        "node ids are an import detail and must not reach the canonical identity"
    );
}

#[test]
fn a_changed_run_parameter_changes_the_identity() {
    let (baseline, _) = convert_ok(&txt2img());
    let mut other = txt2img();
    other["3"]["inputs"]["steps"] = json!(30);
    let (changed, _) = convert_ok(&other);
    assert_ne!(
        onnx_genai_metadata::semantic_identity(&baseline),
        onnx_genai_metadata::semantic_identity(&changed)
    );
}

#[test]
fn the_prompt_wrapper_form_converts_identically() {
    let wrapped = json!({ "prompt": txt2img() });
    let (baseline, _) = convert_ok(&txt2img());
    let (from_wrapper, _) = convert_ok(&wrapped);
    assert_eq!(baseline, from_wrapper);
}

#[test]
fn the_artifact_layout_is_overridable() {
    let options = ConvertOptions {
        layout: ComponentLayout::textproto(),
        ..ConvertOptions::default()
    };
    let (_, document, _) = convert(&txt2img(), &options).expect("conversion");
    let components = &workflow_of(&document)["components"];
    assert_eq!(
        components["denoiser"]["implementation"]["artifact"],
        json!("denoiser/model.onnx.textproto")
    );
    assert_eq!(
        components["solver_step"]["implementation"]["artifact"],
        json!("policies/solver_step.onnx.textproto")
    );
}

#[test]
fn the_emitted_document_is_valid_inference_metadata() {
    // `convert` validates before returning, so reaching this point already
    // proves it. Re-parsing from the serialized YAML proves the serialized form
    // is what a package can carry.
    let (document, _) = convert_ok(&txt2img());
    let yaml = crate::to_yaml(&document).expect("yaml");
    let reparsed: Value = serde_yaml::from_str(&yaml).expect("reparse");
    assert_eq!(reparsed, document);
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_json::from_value(reparsed).expect("typed metadata");
    onnx_genai_metadata::validate_metadata(&metadata).expect("valid metadata");
}
