//! Lower a [`WorkflowPlan`] into the canonical `pipeline.workflow` IR.
//!
//! The output of this module is an ordinary inference-metadata document: typed
//! SSA inputs, component declarations with explicit ports and contracts, state
//! cells, and a `loop`/`invoke`/`emit` step tree. It is the same IR the
//! Mobius-exported diffusion packages under
//! `tests/fixtures/onnx_genai_workflows/diffusion*` carry, produced by the same
//! rules, so a converted ComfyUI workflow is indistinguishable from a natively
//! exported one once the conversion is done. That is the point: ComfyUI is an
//! import source, and the emitted metadata is the sole source of execution
//! truth afterwards.

use onnx_genai_metadata::capabilities as capability;
use serde_json::{Map, Value, json};

use crate::ComfyUiConfigError;
use crate::layout::ComponentLayout;
use crate::plan::{Conditioning, LatentSource, Prediction, Solver, WorkflowPlan};

/// Canonical latent scale factor applied per denoiser call is package data, so
/// the emitted workflow only ever names ports, never numbers like this.
/// A dimension symbol shared by every request-aligned latent tensor.
const LATENT_HEIGHT: &str = "latent_height";
const LATENT_WIDTH: &str = "latent_width";

pub(crate) struct Lowering<'a> {
    plan: &'a WorkflowPlan,
    layout: &'a ComponentLayout,
    inputs: Map<String, Value>,
    components: Map<String, Value>,
    state: Map<String, Value>,
    setup: Vec<Value>,
    body: Vec<Value>,
    carried: Vec<Value>,
    tail: Vec<Value>,
    outputs: Map<String, Value>,
    capabilities: Vec<&'static str>,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(plan: &'a WorkflowPlan, layout: &'a ComponentLayout) -> Self {
        Self {
            plan,
            layout,
            inputs: Map::new(),
            components: Map::new(),
            state: Map::new(),
            setup: Vec::new(),
            body: Vec::new(),
            carried: Vec::new(),
            tail: Vec::new(),
            outputs: Map::new(),
            capabilities: vec![
                capability::WORKFLOW_SSA,
                capability::LINEAR_EFFECTS,
                capability::NESTED_CONTROL_FLOW,
                capability::LOOP_INDUCTION_VALUES,
                capability::TYPED_EMIT,
            ],
        }
    }

    /// Build the whole `pipeline.workflow` document.
    pub(crate) fn build(mut self, adapters: Option<&Value>) -> Result<Value, ComfyUiConfigError> {
        self.declare_inputs();
        self.declare_schedule();
        self.declare_conditioning();
        let initial = self.declare_initial_latent()?;
        self.declare_state(&initial);
        self.build_loop_body()?;
        self.build_tail();

        let mut workflow = Map::new();
        workflow.insert(
            "manifest".to_owned(),
            json!({
                "capabilities": self.capabilities,
            }),
        );
        workflow.insert("inputs".to_owned(), Value::Object(self.inputs));
        workflow.insert("outputs".to_owned(), Value::Object(self.outputs));
        workflow.insert("components".to_owned(), Value::Object(self.components));
        workflow.insert("state".to_owned(), Value::Object(self.state));

        let mut steps = vec![json!({
            "kind": "loop",
            "setup": self.setup,
            "steps": self.body,
            "continue_when": "loop_0_active",
            "max_iterations": "request.max_iterations",
            "carried": self.carried,
            "iteration": {
                "value": "loop.iteration",
                "contract": scalar_contract("int64", "batch"),
            },
        })];
        steps.extend(self.tail);
        workflow.insert("steps".to_owned(), Value::Array(steps));

        let mut document = Map::new();
        document.insert(
            "schema_version".to_owned(),
            json!(onnx_genai_metadata::SCHEMA_VERSION),
        );
        document.insert(
            "pipeline".to_owned(),
            json!({ "workflow": Value::Object(workflow) }),
        );
        if let Some(adapters) = adapters {
            document.insert("adapters".to_owned(), adapters.clone());
        }
        Ok(Value::Object(document))
    }

    // ── inputs ──────────────────────────────────────────────────────────────

    fn declare_inputs(&mut self) {
        let plan = self.plan;
        self.request_input(
            "request.input_ids",
            token_contract(),
            json!({"kind": "runtime", "version": "1.0", "role": "prompt_tokens"}),
            json!({"kind": "request"}),
            true,
            None,
        );
        self.request_input(
            "request.max_iterations",
            json!({"dtype": "int64", "shape": [1]}),
            json!({"kind": "runtime", "version": "1.0", "role": "max_iterations"}),
            json!({"kind": "request"}),
            false,
            Some(json!(plan.iterations())),
        );
        self.request_input(
            "request.seed",
            row_contract("int64"),
            json!({"kind": "runtime", "version": "1.0", "role": "seed"}),
            json!({"kind": "request"}),
            false,
            Some(json!(plan.seed)),
        );
        self.literal_input("package.rng_offset", row_contract("int64"), json!(0));
        self.literal_input("package.false", row_contract("bool"), json!(false));
        self.literal_input(
            "package.loop_0_active",
            json!({"dtype": "bool", "shape": [1]}),
            json!(true),
        );

        if plan.uses_guidance() {
            self.request_input(
                "request.negative_input_ids",
                token_contract(),
                json!({
                    "kind": "runtime",
                    "version": "1.0",
                    "role": "negative_prompt_tokens"
                }),
                json!({"kind": "application", "name": "negative_input_ids"}),
                true,
                None,
            );
            let scale = plan
                .guidance
                .as_ref()
                .map_or(1.0, |guidance| guidance.scale);
            self.request_input(
                "request.guidance_scale",
                row_contract("float32"),
                json!({"kind": "runtime", "version": "1.0", "role": "guidance_scale"}),
                json!({"kind": "application", "name": "guidance_scale"}),
                false,
                Some(json!(scale)),
            );
        }

        if plan.conditioning == Conditioning::SdxlDual {
            self.application_input(
                "request.input_ids_2",
                token_contract(),
                "input_ids_2",
                true,
                None,
            );
            if plan.uses_guidance() {
                self.application_input(
                    "request.negative_input_ids_2",
                    token_contract(),
                    "negative_input_ids_2",
                    true,
                    None,
                );
            }
            // SDXL micro-conditioning: (original h, original w, crop top, crop
            // left, target h, target w) per request row.
            self.application_input(
                "request.time_ids",
                json!({
                    "dtype": "float32",
                    "shape": ["batch", 6],
                    "batch_layout": {"kind": "request_aligned", "axis": 0},
                }),
                "time_ids",
                true,
                None,
            );
        }

        if plan.is_image_to_image() {
            self.application_input(
                "request.source_image",
                image_contract(3),
                "source_image",
                true,
                None,
            );
        }
        if plan.is_inpainting() {
            // The mask lives in latent space, because it gates the latent the
            // solver produces, not the pixels the VAE later decodes.
            self.application_input("request.mask", latent_mask_contract(), "mask", true, None);
        }
        if !plan.controlnets.is_empty() {
            self.application_input(
                "request.control_hint",
                image_contract(3),
                "control_hint",
                true,
                None,
            );
            let strength = plan.controlnets[0].strength;
            self.application_input(
                "request.control_strength",
                row_contract("float32"),
                "control_strength",
                false,
                Some(json!(strength)),
            );
        }
        if plan.start_step > 0 {
            self.literal_input(
                "package.start_step",
                row_contract("int64"),
                json!(plan.start_step),
            );
        }
    }

    fn request_input(
        &mut self,
        name: &str,
        contract: Value,
        role: Value,
        source: Value,
        required: bool,
        default: Option<Value>,
    ) {
        let mut input = Map::new();
        input.insert("contract".to_owned(), contract);
        input.insert("role".to_owned(), role);
        input.insert("source".to_owned(), source);
        input.insert("required".to_owned(), json!(required));
        if let Some(default) = default {
            input.insert("default".to_owned(), default);
        }
        self.inputs.insert(name.to_owned(), Value::Object(input));
    }

    fn application_input(
        &mut self,
        name: &str,
        contract: Value,
        application_name: &str,
        required: bool,
        default: Option<Value>,
    ) {
        self.request_input(
            name,
            contract,
            json!({"kind": "opaque"}),
            json!({"kind": "application", "name": application_name}),
            required,
            default,
        );
    }

    fn literal_input(&mut self, name: &str, contract: Value, default: Value) {
        self.request_input(
            name,
            contract,
            json!({"kind": "opaque"}),
            json!({"kind": "literal"}),
            false,
            Some(default),
        );
    }

    // ── components and setup ────────────────────────────────────────────────

    fn declare_schedule(&mut self) {
        let plan = self.plan;
        let sigmas = plan.steps as i64 + 1;
        let scheduling = json!({
            "solver": plan.solver.as_str(),
            "spacing": plan.spacing.as_str(),
            "prediction": plan.prediction.as_str(),
            "steps": plan.steps as i64,
        });

        self.component(
            "diffusion_schedule",
            self.layout.policy("diffusion_schedule"),
            json!({}),
            json!({"schedule": {"dtype": "float32", "shape": [sigmas]}}),
            Some(json!({
                "id": "onnx-genai.diffusion-schedule",
                "version": "1",
                "bindings": {"schedule": "schedule"},
                "parameters": scheduling,
            })),
        );
        self.component(
            "diffusion_timesteps",
            self.layout.policy("diffusion_timesteps"),
            json!({}),
            json!({"schedule": {"dtype": "float32", "shape": [plan.steps as i64]}}),
            Some(json!({
                "id": "onnx-genai.diffusion-schedule",
                "version": "1",
                "bindings": {"schedule": "schedule"},
                "parameters": scheduling,
            })),
        );
        self.component(
            "schedule_lookup",
            self.layout.policy("schedule_lookup"),
            json!({
                "schedule": {"dtype": "float32", "shape": ["schedule_length"]},
                "step": row_contract("int64"),
            }),
            json!({"timestep": row_contract("float32")}),
            None,
        );
        self.component(
            "model_input",
            self.layout.policy("model_input"),
            json!({
                "sample": latent_contract(),
                "step": row_contract("int64"),
                "schedule": {"dtype": "float32", "shape": ["schedule_length"]},
            }),
            json!({"model_input": latent_contract()}),
            None,
        );
        self.component(
            "continue_predicate",
            self.layout.policy("continue_predicate"),
            json!({"done": row_contract("bool")}),
            json!({"continue": {"dtype": "bool", "shape": [1]}}),
            None,
        );
        self.component(
            "latent_row_shape",
            self.layout.policy("latent_row_shape"),
            json!({}),
            json!({"shape": {"dtype": "int64", "shape": [3]}}),
            None,
        );
        self.component(
            "latent_noise",
            self.layout.policy("latent_noise"),
            json!({
                "seed": row_contract("int64"),
                "offset": row_contract("int64"),
                "row_shape": {"dtype": "int64", "shape": ["row_rank"]},
            }),
            json!({"noise": latent_contract(), "next_offset": row_contract("int64")}),
            Some(json!({
                "id": "onnx-genai.counter-rng",
                "version": "1",
                "bindings": {
                    "seed": "seed",
                    "offset": "offset",
                    "row_shape": "row_shape",
                    "noise": "noise",
                    "next_offset": "next_offset",
                },
            })),
        );

        self.setup.push(invoke(
            "diffusion_schedule",
            json!({}),
            json!({"schedule": "diffusion.schedule"}),
        ));
        self.setup.push(invoke(
            "diffusion_timesteps",
            json!({}),
            json!({"schedule": "diffusion.timesteps"}),
        ));
        self.setup.push(invoke(
            "latent_row_shape",
            json!({}),
            json!({"shape": "diffusion.latent_row_shape"}),
        ));
        self.setup.push(invoke(
            "latent_noise",
            json!({
                "seed": "request.seed",
                "offset": "package.rng_offset",
                "row_shape": "diffusion.latent_row_shape",
            }),
            json!({"noise": "diffusion.noise", "next_offset": "diffusion.rng_offset"}),
        ));
    }

    fn declare_conditioning(&mut self) {
        let plan = self.plan;
        let mut encoder_outputs = json!({"encoder_hidden_states": hidden_states_contract()});
        if plan.conditioning == Conditioning::SdxlDual {
            encoder_outputs = json!({
                "encoder_hidden_states": hidden_states_contract(),
                "pooled_embeds": {
                    "dtype": "float32",
                    "shape": ["batch", "pooled"],
                    "batch_layout": {"kind": "request_aligned", "axis": 0},
                },
            });
        }
        self.component(
            "text_encoder",
            self.layout.artifact(&self.layout.text_encoder),
            json!({"input_ids": token_contract()}),
            encoder_outputs.clone(),
            None,
        );
        self.setup.push(invoke(
            "text_encoder",
            json!({"input_ids": "request.input_ids"}),
            conditioning_outputs(plan, "conditional"),
        ));
        if plan.uses_guidance() {
            self.setup.push(invoke(
                "text_encoder",
                json!({"input_ids": "request.negative_input_ids"}),
                conditioning_outputs(plan, "unconditional"),
            ));
        }
        if plan.conditioning == Conditioning::SdxlDual {
            self.component(
                "text_encoder_2",
                self.layout.artifact(&self.layout.text_encoder_2),
                json!({"input_ids": token_contract()}),
                encoder_outputs,
                None,
            );
            self.setup.push(invoke(
                "text_encoder_2",
                json!({"input_ids": "request.input_ids_2"}),
                json!({
                    "encoder_hidden_states": "conditioning.conditional_2",
                    "pooled_embeds": "conditioning.conditional_pooled_2",
                }),
            ));
            if plan.uses_guidance() {
                self.setup.push(invoke(
                    "text_encoder_2",
                    json!({"input_ids": "request.negative_input_ids_2"}),
                    json!({
                        "encoder_hidden_states": "conditioning.unconditional_2",
                        "pooled_embeds": "conditioning.unconditional_pooled_2",
                    }),
                ));
            }
        }
    }

    fn declare_initial_latent(&mut self) -> Result<String, ComfyUiConfigError> {
        let plan = self.plan;
        match &plan.latent {
            LatentSource::Noise { .. } => {
                if !plan.add_initial_noise {
                    return Err(ComfyUiConfigError::AmbiguousTopology {
                        detail: "the sampler disables noise but starts from an empty latent, so \
                                 it would denoise an all-zero tensor"
                            .to_owned(),
                        remedy: "enable 'add_noise', or start the sampler from a VAE-encoded \
                                 latent to continue an existing trajectory"
                            .to_owned(),
                    });
                }
                Ok("diffusion.noise".to_owned())
            }
            LatentSource::Image { .. } | LatentSource::Inpaint { .. } => {
                self.component(
                    "vae_encoder",
                    self.layout.artifact(&self.layout.vae_encoder),
                    json!({"image": image_contract(3)}),
                    json!({"latent": latent_contract()}),
                    None,
                );
                self.setup.push(invoke(
                    "vae_encoder",
                    json!({"image": "request.source_image"}),
                    json!({"latent": "diffusion.encoded"}),
                ));
                if !plan.add_initial_noise {
                    return Ok("diffusion.encoded".to_owned());
                }
                self.component(
                    "add_noise",
                    self.layout.policy("add_noise"),
                    json!({
                        "sample": latent_contract(),
                        "noise": latent_contract(),
                        "step": row_contract("int64"),
                        "schedule": {"dtype": "float32", "shape": ["schedule_length"]},
                    }),
                    json!({"noisy": latent_contract()}),
                    Some(json!({
                        "id": "onnx-genai.add-noise",
                        "version": "1",
                        "bindings": {
                            "sample": "sample",
                            "noise": "noise",
                            "step": "step",
                            "schedule": "schedule",
                            "noisy": "noisy",
                        },
                        "parameters": {"prediction": plan.prediction.as_str()},
                    })),
                );
                let start = if plan.start_step > 0 {
                    "package.start_step"
                } else {
                    "package.rng_offset"
                };
                self.setup.push(invoke(
                    "add_noise",
                    json!({
                        "sample": "diffusion.encoded",
                        "noise": "diffusion.noise",
                        "step": start,
                        "schedule": "diffusion.schedule",
                    }),
                    json!({"noisy": "diffusion.initial_latent"}),
                ));
                Ok("diffusion.initial_latent".to_owned())
            }
        }
    }

    fn declare_state(&mut self, initial: &str) {
        let plan = self.plan;
        self.state.insert(
            "latent_state".to_owned(),
            json!({
                "contract": latent_contract(),
                "scope": "invocation",
                "initializer": initial,
                "recurrence": {"kind": "invariant"},
            }),
        );
        self.state.insert(
            "loop_0_active".to_owned(),
            json!({
                "contract": {"dtype": "bool", "shape": [1]},
                "scope": "invocation",
                "initializer": "package.loop_0_active",
                "recurrence": {"kind": "invariant"},
            }),
        );
        if plan.solver.needs_history() {
            self.component(
                "history_initializer",
                self.layout.policy("history_initializer"),
                json!({"reference": latent_contract()}),
                json!({"zeros": latent_contract()}),
                None,
            );
            self.setup.push(invoke(
                "history_initializer",
                json!({"reference": initial}),
                json!({"zeros": "diffusion.initial_history"}),
            ));
            self.state.insert(
                "history".to_owned(),
                json!({
                    "contract": latent_contract(),
                    "scope": "invocation",
                    "initializer": "diffusion.initial_history",
                    "recurrence": {"kind": "invariant"},
                }),
            );
        }
        if plan.solver.needs_step_noise() {
            self.state.insert(
                "rng_offset".to_owned(),
                json!({
                    "contract": row_contract("int64"),
                    "scope": "invocation",
                    "initializer": "diffusion.rng_offset",
                    "recurrence": {"kind": "invariant"},
                }),
            );
        }
        self.setup.push(invoke(
            "continue_predicate",
            json!({"done": "package.false"}),
            json!({"continue": "setup.continue"}),
        ));
    }

    // ── loop body ───────────────────────────────────────────────────────────

    fn build_loop_body(&mut self) -> Result<(), ComfyUiConfigError> {
        let plan = self.plan;
        let step_value = if plan.start_step > 0 {
            self.component(
                "step_offset",
                self.layout.policy("step_offset"),
                json!({"iteration": row_contract("int64"), "offset": row_contract("int64")}),
                json!({"step": row_contract("int64")}),
                None,
            );
            self.body.push(invoke(
                "step_offset",
                json!({"iteration": "loop.iteration", "offset": "package.start_step"}),
                json!({"step": "diffusion.step"}),
            ));
            "diffusion.step"
        } else {
            "loop.iteration"
        };

        self.body.push(invoke(
            "schedule_lookup",
            json!({"schedule": "diffusion.timesteps", "step": step_value}),
            json!({"timestep": "diffusion.timestep"}),
        ));
        self.body.push(invoke(
            "model_input",
            json!({
                "sample": "latent_state",
                "step": step_value,
                "schedule": "diffusion.schedule",
            }),
            json!({"model_input": "diffusion.model_input"}),
        ));

        self.declare_controlnet(step_value)?;
        self.declare_denoiser();
        self.declare_solver(step_value);
        self.declare_mask_blend(step_value);

        self.body.push(invoke(
            "continue_predicate",
            json!({"done": "package.false"}),
            json!({"continue": "loop.continue"}),
        ));
        self.carried
            .push(json!({"cell": "loop_0_active", "next": "loop.continue"}));
        Ok(())
    }

    fn declare_controlnet(&mut self, step_value: &str) -> Result<(), ComfyUiConfigError> {
        let plan = self.plan;
        let Some(controlnet) = plan.controlnets.first() else {
            return Ok(());
        };
        if plan.controlnets.len() > 1 {
            return Err(ComfyUiConfigError::UnsupportedFeature {
                feature: format!("{} chained ControlNets", plan.controlnets.len()),
                detail: "each ControlNet contributes its own residual tensor, and the canonical \
                         denoiser ABI accepts exactly one `control` input. Combining residuals \
                         needs a residual-merge contract that this metadata schema does not \
                         define, so importing the chain would silently drop every ControlNet \
                         after the first"
                    .to_owned(),
                remedy: "import a workflow with a single ControlNetApply, or extend the \
                         canonical component vocabulary with a residual-merge contract first"
                    .to_owned(),
            });
        }
        if controlnet.start_percent > f64::EPSILON
            || (controlnet.end_percent - 1.0).abs() > f64::EPSILON
        {
            return Err(ComfyUiConfigError::UnsupportedFeature {
                feature: "a step-windowed ControlNet".to_owned(),
                detail: format!(
                    "the ControlNet applies only between {:.3} and {:.3} of the schedule, which \
                     makes the denoiser's inputs differ between steps",
                    controlnet.start_percent, controlnet.end_percent
                ),
                remedy: "set start_percent to 0 and end_percent to 1; a step window needs a \
                         branch on the loop induction value, which this importer will not \
                         fabricate from an approximate percentage"
                    .to_owned(),
            });
        }

        self.component(
            "controlnet",
            self.layout.artifact(&self.layout.controlnet),
            json!({
                "sample": latent_contract(),
                "timestep": row_contract("float32"),
                "encoder_hidden_states": hidden_states_contract(),
                "hint": image_contract(3),
                "conditioning_scale": row_contract("float32"),
            }),
            json!({"control": control_contract()}),
            Some(json!({
                "id": "onnx-genai.controlnet-residual",
                "version": "1",
                "bindings": {
                    "sample": "sample",
                    "timestep": "timestep",
                    "conditioning": "encoder_hidden_states",
                    "hint": "hint",
                    "scale": "conditioning_scale",
                    "control": "control",
                },
            })),
        );
        self.body.push(invoke(
            "controlnet",
            json!({
                "sample": "diffusion.model_input",
                "timestep": "diffusion.timestep",
                "encoder_hidden_states": "conditioning.conditional",
                "hint": "request.control_hint",
                "conditioning_scale": "request.control_strength",
            }),
            json!({"control": "controlnet.conditional"}),
        ));
        if controlnet.applies_to_negative && plan.uses_guidance() {
            self.body.push(invoke(
                "controlnet",
                json!({
                    "sample": "diffusion.model_input",
                    "timestep": "diffusion.timestep",
                    "encoder_hidden_states": "conditioning.unconditional",
                    "hint": "request.control_hint",
                    "conditioning_scale": "request.control_strength",
                }),
                json!({"control": "controlnet.unconditional"}),
            ));
        }
        let _ = step_value;
        Ok(())
    }

    fn declare_denoiser(&mut self) {
        let plan = self.plan;
        let mut ports = Map::new();
        ports.insert("sample".to_owned(), latent_contract());
        ports.insert("timestep".to_owned(), row_contract("float32"));
        ports.insert("encoder_hidden_states".to_owned(), hidden_states_contract());
        if plan.conditioning == Conditioning::SdxlDual {
            ports.insert(
                "text_embeds".to_owned(),
                json!({
                    "dtype": "float32",
                    "shape": ["batch", "pooled"],
                    "batch_layout": {"kind": "request_aligned", "axis": 0},
                }),
            );
            ports.insert(
                "time_ids".to_owned(),
                json!({
                    "dtype": "float32",
                    "shape": ["batch", 6],
                    "batch_layout": {"kind": "request_aligned", "axis": 0},
                }),
            );
        }
        if !plan.controlnets.is_empty() {
            ports.insert("control".to_owned(), control_contract());
        }
        self.component(
            "denoiser",
            self.layout.artifact(&self.layout.denoiser),
            Value::Object(ports),
            json!({"noise_pred": latent_contract()}),
            None,
        );

        let branch = |plan: &WorkflowPlan, kind: &str, out: &str| -> Value {
            let mut inputs = Map::new();
            inputs.insert("sample".to_owned(), json!("diffusion.model_input"));
            inputs.insert("timestep".to_owned(), json!("diffusion.timestep"));
            inputs.insert(
                "encoder_hidden_states".to_owned(),
                json!(format!("conditioning.{kind}")),
            );
            if plan.conditioning == Conditioning::SdxlDual {
                inputs.insert(
                    "text_embeds".to_owned(),
                    json!(format!("conditioning.{kind}_pooled_2")),
                );
                inputs.insert("time_ids".to_owned(), json!("request.time_ids"));
            }
            if let Some(controlnet) = plan.controlnets.first() {
                // A basic ControlNetApply patches only the branch it sits on,
                // so the unconditional pass reuses the conditional residual
                // only when the workflow explicitly applied it to both.
                let residual = if kind == "unconditional" && !controlnet.applies_to_negative {
                    "controlnet.conditional".to_owned()
                } else {
                    format!("controlnet.{kind}")
                };
                inputs.insert("control".to_owned(), json!(residual));
            }
            invoke(
                "denoiser",
                Value::Object(inputs),
                json!({ "noise_pred": out }),
            )
        };

        if plan.uses_guidance() {
            let unconditional = branch(plan, "unconditional", "denoiser.unconditional");
            let conditional = branch(plan, "conditional", "denoiser.conditional");
            self.body.push(unconditional);
            self.body.push(conditional);
            self.component(
                "guidance_combine",
                self.layout.policy("guidance_combine"),
                json!({
                    "unconditional": latent_contract(),
                    "conditional": latent_contract(),
                    "scale": row_contract("float32"),
                }),
                json!({"estimate": latent_contract()}),
                Some(json!({
                    "id": "onnx-genai.guidance-combine",
                    "version": "1",
                    "bindings": {
                        "unconditional": "unconditional",
                        "conditional": "conditional",
                        "scale": "scale",
                        "estimate": "estimate",
                    },
                })),
            );
            self.body.push(invoke(
                "guidance_combine",
                json!({
                    "unconditional": "denoiser.unconditional",
                    "conditional": "denoiser.conditional",
                    "scale": "request.guidance_scale",
                }),
                json!({"estimate": "denoiser.estimate"}),
            ));
        } else {
            let conditional = branch(plan, "conditional", "denoiser.estimate");
            self.body.push(conditional);
        }
    }

    fn declare_solver(&mut self, step_value: &str) {
        let plan = self.plan;
        let mut inputs = Map::new();
        inputs.insert("sample".to_owned(), latent_contract());
        inputs.insert("estimate".to_owned(), latent_contract());
        inputs.insert("step".to_owned(), row_contract("int64"));
        inputs.insert(
            "schedule".to_owned(),
            json!({"dtype": "float32", "shape": ["schedule_length"]}),
        );
        let mut outputs = Map::new();
        outputs.insert("next_state".to_owned(), latent_contract());
        let mut bindings = Map::new();
        bindings.insert("state".to_owned(), json!("sample"));
        bindings.insert("estimate".to_owned(), json!("estimate"));
        bindings.insert("step".to_owned(), json!("step"));
        bindings.insert("schedule".to_owned(), json!("schedule"));
        bindings.insert("next_state".to_owned(), json!("next_state"));

        let mut call = Map::new();
        call.insert("sample".to_owned(), json!("latent_state"));
        call.insert("estimate".to_owned(), json!("denoiser.estimate"));
        call.insert("step".to_owned(), json!(step_value));
        call.insert("schedule".to_owned(), json!("diffusion.schedule"));
        let mut results = Map::new();
        results.insert("next_state".to_owned(), json!("latent.body"));

        if plan.solver.needs_history() {
            inputs.insert("history".to_owned(), latent_contract());
            outputs.insert("next_history".to_owned(), latent_contract());
            bindings.insert("history".to_owned(), json!("history"));
            bindings.insert("next_history".to_owned(), json!("next_history"));
            call.insert("history".to_owned(), json!("history"));
            results.insert("next_history".to_owned(), json!("history.body"));
            self.carried
                .push(json!({"cell": "history", "next": "history.body"}));
        }
        if plan.solver.needs_step_noise() {
            inputs.insert("noise".to_owned(), latent_contract());
            bindings.insert("noise".to_owned(), json!("noise"));
            call.insert("noise".to_owned(), json!("diffusion.step_noise"));
            self.body.push(invoke(
                "latent_noise",
                json!({
                    "seed": "request.seed",
                    "offset": "rng_offset",
                    "row_shape": "diffusion.latent_row_shape",
                }),
                json!({"noise": "diffusion.step_noise", "next_offset": "diffusion.next_rng_offset"}),
            ));
            self.carried
                .push(json!({"cell": "rng_offset", "next": "diffusion.next_rng_offset"}));
        }

        self.component(
            "solver_step",
            self.layout.policy("solver_step"),
            Value::Object(inputs),
            Value::Object(outputs),
            Some(json!({
                "id": "onnx-genai.solver-step",
                "version": "1",
                "bindings": Value::Object(bindings),
                "parameters": {
                    "solver": plan.solver.as_str(),
                    "spacing": plan.spacing.as_str(),
                    "prediction": plan.prediction.as_str(),
                },
            })),
        );
        self.body.push(invoke(
            "solver_step",
            Value::Object(call),
            Value::Object(results),
        ));
    }

    fn declare_mask_blend(&mut self, step_value: &str) {
        if !self.plan.is_inpainting() {
            self.carried
                .push(json!({"cell": "latent_state", "next": "latent.body"}));
            return;
        }
        self.component(
            "masked_blend",
            self.layout.policy("masked_blend"),
            json!({
                "current": latent_contract(),
                "reference": latent_contract(),
                "noise": latent_contract(),
                "mask": latent_mask_contract(),
                "step": row_contract("int64"),
                "schedule": {"dtype": "float32", "shape": ["schedule_length"]},
            }),
            json!({"blended": latent_contract()}),
            Some(json!({
                "id": "onnx-genai.masked-blend",
                "version": "1",
                "bindings": {
                    "current": "current",
                    "reference": "reference",
                    "noise": "noise",
                    "mask": "mask",
                    "step": "step",
                    "schedule": "schedule",
                    "blended": "blended",
                },
                "parameters": {"prediction": self.plan.prediction.as_str()},
            })),
        );
        self.body.push(invoke(
            "masked_blend",
            json!({
                "current": "latent.body",
                "reference": "diffusion.encoded",
                "noise": "diffusion.noise",
                "mask": "request.mask",
                "step": step_value,
                "schedule": "diffusion.schedule",
            }),
            json!({"blended": "latent.blended"}),
        ));
        self.carried
            .push(json!({"cell": "latent_state", "next": "latent.blended"}));
    }

    fn build_tail(&mut self) {
        self.component(
            "vae_decoder",
            self.layout.artifact(&self.layout.vae_decoder),
            json!({"latent": latent_contract()}),
            json!({"image": image_contract(3)}),
            None,
        );
        self.tail.push(invoke(
            "vae_decoder",
            json!({"latent": "latent_state"}),
            json!({"image": "vae.image"}),
        ));
        self.tail.push(json!({
            "kind": "emit",
            "value": "vae.image",
            "output": "image",
            "mode": "replace",
        }));
        self.tail.push(json!({
            "kind": "emit",
            "value": "latent_state",
            "output": "latent",
            "mode": "replace",
        }));
        self.outputs.insert(
            "image".to_owned(),
            json!({
                "contract": image_contract(3),
                "role": "image",
                "value_range": "negative_one_to_one",
                "stage": "pre_adapter"
            }),
        );
        self.outputs.insert(
            "latent".to_owned(),
            json!({"contract": latent_contract(), "role": "tensor", "stage": "pre_adapter"}),
        );
    }

    fn component(
        &mut self,
        name: &str,
        artifact: String,
        inputs: Value,
        outputs: Value,
        contract: Option<Value>,
    ) {
        let mut component = Map::new();
        component.insert(
            "implementation".to_owned(),
            json!({"kind": "onnx", "artifact": artifact}),
        );
        component.insert(
            "ports".to_owned(),
            json!({"inputs": inputs, "outputs": outputs}),
        );
        if let Some(contract) = contract {
            component.insert("contract".to_owned(), contract);
        }
        self.components
            .insert(name.to_owned(), Value::Object(component));
    }

    /// Declare the request-scoped adapter selection inputs a LoRA package needs.
    pub(crate) fn declare_adapter_selection(&mut self, max_adapters: usize) {
        self.capabilities.push(capability::PARAMETER_ADAPTERS);
        self.capabilities
            .push(capability::HETEROGENEOUS_ADAPTER_BATCHING);
        let count = i64::try_from(max_adapters).unwrap_or(i64::MAX);
        self.request_input(
            "request.adapter_segments",
            json!({
                "dtype": "int64",
                "shape": ["batch", count],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            }),
            json!({"kind": "runtime", "version": "1.0", "role": "adapter_segments"}),
            json!({"kind": "request"}),
            true,
            None,
        );
        self.request_input(
            "request.adapter_counts",
            row_contract("int64"),
            json!({"kind": "runtime", "version": "1.0", "role": "adapter_counts"}),
            json!({"kind": "request"}),
            true,
            None,
        );
        self.request_input(
            "request.adapter_scales",
            json!({
                "dtype": "float32",
                "shape": ["batch", count],
                "batch_layout": {"kind": "request_aligned", "axis": 0},
            }),
            json!({"kind": "runtime", "version": "1.0", "role": "adapter_scales"}),
            json!({"kind": "request"}),
            true,
            None,
        );
    }
}

fn conditioning_outputs(plan: &WorkflowPlan, kind: &str) -> Value {
    if plan.conditioning == Conditioning::SdxlDual {
        json!({
            "encoder_hidden_states": format!("conditioning.{kind}"),
            "pooled_embeds": format!("conditioning.{kind}_pooled"),
        })
    } else {
        json!({"encoder_hidden_states": format!("conditioning.{kind}")})
    }
}

fn invoke(component: &str, inputs: Value, outputs: Value) -> Value {
    json!({
        "kind": "invoke",
        "component": component,
        "inputs": inputs,
        "outputs": outputs,
    })
}

fn row_contract(dtype: &str) -> Value {
    json!({
        "dtype": dtype,
        "shape": ["batch"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn scalar_contract(dtype: &str, dimension: &str) -> Value {
    json!({"dtype": dtype, "shape": [dimension]})
}

fn token_contract() -> Value {
    json!({
        "dtype": "int64",
        "shape": ["batch", "prompt_sequence"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn hidden_states_contract() -> Value {
    json!({
        "dtype": "float32",
        "shape": ["batch", "prompt_sequence", "hidden"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn latent_contract() -> Value {
    json!({
        "dtype": "float32",
        "shape": ["batch", "channels", LATENT_HEIGHT, LATENT_WIDTH],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn latent_mask_contract() -> Value {
    json!({
        "dtype": "float32",
        "shape": ["batch", 1, LATENT_HEIGHT, LATENT_WIDTH],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn image_contract(channels: i64) -> Value {
    json!({
        "dtype": "float32",
        "shape": ["batch", channels, "height", "width"],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

fn control_contract() -> Value {
    json!({
        "dtype": "float32",
        "shape": ["batch", "control_channels", LATENT_HEIGHT, LATENT_WIDTH],
        "batch_layout": {"kind": "request_aligned", "axis": 0},
    })
}

/// Prediction parameterizations that the canonical solver contract carries.
pub(crate) fn supported_prediction(
    prediction: Prediction,
    solver: Solver,
) -> Result<(), ComfyUiConfigError> {
    if prediction == Prediction::FlowVelocity && solver == Solver::Ddim {
        return Err(ComfyUiConfigError::UnsupportedFeature {
            feature: "DDIM over a flow-matching model".to_owned(),
            detail: "DDIM inverts a variance-preserving noise schedule, which a flow-matching \
                     velocity field does not define"
                .to_owned(),
            remedy: "select the euler sampler, which is the flow-matching solver ComfyUI itself \
                     uses for SD3/Flux-family models"
                .to_owned(),
        });
    }
    Ok(())
}
