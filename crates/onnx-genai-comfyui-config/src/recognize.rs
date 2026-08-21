//! Structural recognition of a ComfyUI graph into a Comfy-free [`WorkflowPlan`].
//!
//! Recognition walks *backwards from the image sink*, because that is the only
//! part of a ComfyUI document that provably decides the produced image. Every
//! node on that output path must be understood; a class this module does not
//! know is an error naming the node id, the class, and how to fix it. Nodes
//! that cannot reach the sink are reported and ignored, which is the single
//! case where skipping a node is sound.
//!
//! Nothing here emits metadata. The recognizer's whole output is a
//! [`WorkflowPlan`], so a ComfyUI class name can never reach the lowerer.

use std::collections::BTreeSet;

use crate::ComfyUiConfigError;
use crate::graph::{ComfyGraph, Node};
use crate::plan::{
    Conditioning, ControlNet, Guidance, LatentSource, Lora, Prediction, Solver, Spacing,
    WorkflowPlan, strength_to_start_step,
};

/// Nodes that consume the final image. Exactly one must be reachable.
const IMAGE_SINKS: &[&str] = &["SaveImage", "PreviewImage", "SaveImageWebsocket"];

/// Classes the importer recognizes by name but deliberately refuses, each with
/// the reason it cannot be lowered.
///
/// A generic "unknown node" message is correct but unhelpful for a class that is
/// well known and simply outside the canonical contract. Naming the reason here
/// is the difference between "add support for it" and "this model family needs a
/// different package shape".
const KNOWN_UNREPRESENTABLE: &[(&str, &str)] = &[
    (
        "TextEncodeQwenImageEdit",
        "Qwen-Image editing conditions the transformer on a vision-language encoder and a \
         reference image simultaneously, which is a multi-encoder package shape rather than \
         the single text-conditioning ABI this importer emits. Export the Qwen-Image-Edit \
         package natively and drive it with its own inference metadata",
    ),
    (
        "QwenImageDiffsynthControlnet",
        "the Qwen-Image ControlNet variant injects residuals into a DiT block layout that the \
         canonical `onnx-genai.controlnet-residual` contract does not describe",
    ),
    (
        "FluxGuidance",
        "Flux embeds its guidance value as a denoiser input rather than as a second \
         conditioned pass, so it needs a denoiser port this importer does not declare. \
         Export the package with the guidance port wired natively",
    ),
    (
        "CFGGuider",
        "the custom-sampler guider replaces the sampling loop itself, so the emitted workflow \
         would not describe the computation ComfyUI performs",
    ),
    (
        "SamplerCustom",
        "custom samplers compose sigmas and a guider outside KSampler, and the importer will \
         not infer a solver from an opaque sampler object",
    ),
    (
        "SamplerCustomAdvanced",
        "custom samplers compose sigmas and a guider outside KSampler, and the importer will \
         not infer a solver from an opaque sampler object",
    ),
    (
        "UpscaleModelLoader",
        "a second-stage upscaler is a distinct model package; convert it separately and run \
         the two packages in sequence",
    ),
];

/// A tailored refusal for a class the importer knows about but cannot lower.
fn known_refusal(node: &Node) -> Option<ComfyUiConfigError> {
    KNOWN_UNREPRESENTABLE
        .iter()
        .find(|(class, _)| *class == node.class)
        .map(|(_, reason)| ComfyUiConfigError::Unrepresentable {
            node: node.id.clone(),
            class: node.class.clone(),
            detail: (*reason).to_owned(),
            remedy: "remove the node from the path that produces the saved image, or export a \
                     native package for this model family instead of importing its ComfyUI graph"
                .to_owned(),
        })
}

/// Recognize `graph`, or fail closed naming the node that stopped recognition.
pub fn recognize(graph: &ComfyGraph) -> Result<WorkflowPlan, ComfyUiConfigError> {
    let sink = single_sink(graph)?;
    let closure = graph.upstream_closure(std::slice::from_ref(&sink.id))?;
    let mut walker = Walker {
        graph,
        consumed: BTreeSet::new(),
        controlnets: Vec::new(),
        controlnet_ids: BTreeSet::new(),
    };
    let plan = walker.recognize_from(sink, &closure)?;
    Ok(plan)
}

fn single_sink(graph: &ComfyGraph) -> Result<&Node, ComfyUiConfigError> {
    let sinks: Vec<&Node> = graph.by_class(IMAGE_SINKS).collect();
    match sinks.as_slice() {
        [] => Err(ComfyUiConfigError::NoOutputPath {
            expected: IMAGE_SINKS.join(" / "),
        }),
        [only] => Ok(only),
        many => Err(ComfyUiConfigError::AmbiguousTopology {
            detail: format!(
                "the workflow has {} image sinks ({}); a single canonical workflow produces \
                 exactly one image output",
                many.len(),
                many.iter()
                    .map(|node| format!("{} ({})", node.id, node.class))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            remedy: "split the workflow so each converted package saves one image, or delete \
                     the extra save nodes"
                .to_owned(),
        }),
    }
}

struct Walker<'a> {
    graph: &'a ComfyGraph,
    consumed: BTreeSet<String>,
    /// ControlNet applications recorded once each, in discovery order.
    ///
    /// `ControlNetApplyAdvanced` is reachable from both conditioning branches,
    /// so the apply node — not the branch that found it — is what identifies
    /// one application. Counting per branch would read a single ControlNet as
    /// a chain and refuse a workflow that is in fact supported.
    controlnets: Vec<ControlNet>,
    controlnet_ids: BTreeSet<String>,
}

/// Everything the conditioning chain contributed.
#[derive(Default)]
struct ConditioningChain {
    prompt: Option<String>,
    prompt_2: Option<String>,
    sdxl: bool,
    inpaint: Option<InpaintConditioning>,
}

#[derive(Clone)]
struct InpaintConditioning {
    image: Option<String>,
    mask: Option<String>,
}

impl<'a> Walker<'a> {
    fn take(&mut self, node: &Node) {
        self.consumed.insert(node.id.clone());
    }

    fn recognize_from(
        &mut self,
        sink: &'a Node,
        closure: &BTreeSet<String>,
    ) -> Result<WorkflowPlan, ComfyUiConfigError> {
        self.take(sink);
        let (decoder, _) = self.graph.follow(sink, "images")?;
        self.take(decoder);
        match decoder.class.as_str() {
            // Tiling is a memory strategy over an identical decode, so it does
            // not change what the package means.
            "VAEDecode" | "VAEDecodeTiled" => {}
            _ => {
                return Err(decoder.unsupported(
                    "the image sink is not fed by a VAE decode",
                    "route the sampler's latent through VAEDecode before saving the image",
                ));
            }
        }
        let (vae, _) = self.graph.follow(decoder, "vae")?;
        self.consume_loader(
            vae,
            &["VAELoader", "CheckpointLoaderSimple", "CheckpointLoader"],
        )?;

        let (sampler, _) = self.graph.follow(decoder, "samples")?;
        self.take(sampler);
        let sampling = self.read_sampler(sampler)?;

        let (positive_node, positive_slot) = self.graph.follow(sampler, "positive")?;
        let positive = self.walk_conditioning(positive_node, positive_slot)?;
        let (negative_node, negative_slot) = self.graph.follow(sampler, "negative")?;
        let negative = self.walk_conditioning(negative_node, negative_slot)?;

        let (latent_node, _) = self.graph.follow(sampler, "latent_image")?;
        let latent = self.walk_latent(latent_node, sampling.denoise, &positive, &negative)?;

        let (model_node, _) = self.graph.follow(sampler, "model")?;
        let model = self.walk_model(model_node)?;

        if positive.sdxl != negative.sdxl {
            return Err(ComfyUiConfigError::AmbiguousTopology {
                detail: "one conditioning branch uses the SDXL dual encoder and the other does \
                         not, so the denoiser's conditioning ABI is not decidable"
                    .to_owned(),
                remedy: "encode both the positive and the negative prompt with the same text \
                         encoder node class"
                    .to_owned(),
            });
        }

        let steps =
            u32::try_from(sampling.steps).map_err(|_| ComfyUiConfigError::Unrepresentable {
                node: sampler.id.clone(),
                class: sampler.class.clone(),
                detail: format!("step count {} is not a positive step count", sampling.steps),
                remedy: "set 'steps' to a positive integer".to_owned(),
            })?;
        if steps == 0 {
            return Err(sampler.unsupported(
                "the sampler declares zero steps, so the workflow denoises nothing",
                "set 'steps' to at least 1",
            ));
        }

        let start_step = sampling
            .start_at_step
            .unwrap_or_else(|| strength_to_start_step(sampling.denoise, steps));
        let end_step = sampling.end_at_step.unwrap_or(steps).min(steps);
        if start_step >= end_step {
            return Err(sampler.unsupported(
                format!(
                    "the sampler executes no steps: start step {start_step} is not before end \
                     step {end_step} of {steps}"
                ),
                "raise 'denoise' (or lower 'start_at_step') so at least one solver step runs",
            ));
        }

        let guidance = (sampling.cfg - 1.0)
            .abs()
            .gt(&f64::EPSILON)
            .then_some(Guidance {
                scale: sampling.cfg,
            });

        let prediction = model.prediction.unwrap_or(Prediction::Epsilon);
        let (latent_channels, latent) = latent;

        let unrecognized: Vec<&Node> = closure
            .iter()
            .filter(|id| !self.consumed.contains(*id))
            .map(|id| self.graph.node(id))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(node) = unrecognized.first() {
            if let Some(refusal) = known_refusal(node) {
                return Err(refusal);
            }
            return Err(ComfyUiConfigError::UnknownNode {
                node: node.id.clone(),
                class: node.class.clone(),
                remedy: "this node sits on the path that produces the saved image, so the \
                         converter cannot ignore it. Remove it from the workflow, replace it \
                         with a class the importer models, or add support for it before \
                         importing"
                    .to_owned(),
            });
        }

        let ignored = self
            .graph
            .nodes()
            .filter(|node| !closure.contains(&node.id))
            .map(|node| format!("{} ({})", node.id, node.class))
            .collect();
        let recognized = self
            .consumed
            .iter()
            .map(|id| {
                self.graph
                    .node(id)
                    .map(|node| format!("{} ({})", node.id, node.class))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(WorkflowPlan {
            steps,
            start_step,
            end_step,
            add_initial_noise: sampling.add_noise,
            solver: sampling.solver,
            spacing: sampling.spacing,
            prediction,
            guidance,
            conditioning: if positive.sdxl {
                Conditioning::SdxlDual
            } else {
                Conditioning::Single
            },
            prompt: positive.prompt,
            negative_prompt: negative.prompt,
            prompt_2: positive.prompt_2,
            negative_prompt_2: negative.prompt_2,
            latent,
            seed: sampling.seed,
            checkpoint: model.checkpoint,
            loras: model.loras,
            controlnets: self.controlnets.clone(),
            latent_channels,
            recognized_nodes: recognized,
            ignored_nodes: ignored,
        })
    }

    /// Read one loader node that only names a file the package already carries.
    fn consume_loader(&mut self, node: &Node, allowed: &[&str]) -> Result<(), ComfyUiConfigError> {
        if !allowed.contains(&node.class.as_str()) {
            return Err(node.unsupported(
                format!("expected one of {} on this port", allowed.join(" / ")),
                "connect the port to a plain loader node; a converted package resolves weights \
                 from its own ONNX components, not from the ComfyUI model directory",
            ));
        }
        self.take(node);
        Ok(())
    }

    fn read_sampler(&mut self, node: &Node) -> Result<Sampling, ComfyUiConfigError> {
        let advanced = match node.class.as_str() {
            "KSampler" => false,
            "KSamplerAdvanced" => true,
            _ => {
                return Err(node.unsupported(
                    "the latent is produced by a sampler class the importer does not model",
                    "use KSampler or KSamplerAdvanced; custom samplers define their own \
                     dynamics, which a structural import cannot infer",
                ));
            }
        };
        let steps = node.integer("steps")?;
        let cfg = node.float("cfg")?;
        let solver = solver_of(node, &node.string("sampler_name")?)?;
        let spacing = spacing_of(node, &node.string("scheduler")?)?;
        let seed = if advanced {
            node.integer("noise_seed")?
        } else {
            node.integer("seed")?
        };
        let (denoise, start_at_step, end_at_step, add_noise) = if advanced {
            let start = u32::try_from(node.integer("start_at_step")?).map_err(|_| {
                node.unsupported(
                    "'start_at_step' is negative",
                    "set 'start_at_step' to a non-negative step index",
                )
            })?;
            let end = u32::try_from(node.integer("end_at_step")?).map_err(|_| {
                node.unsupported(
                    "'end_at_step' is negative",
                    "set 'end_at_step' to a non-negative step index",
                )
            })?;
            let add_noise = match node.string("add_noise")?.as_str() {
                "enable" => true,
                "disable" => false,
                other => {
                    return Err(node.unsupported(
                        format!("'add_noise' is {other:?}"),
                        "set 'add_noise' to 'enable' or 'disable'",
                    ));
                }
            };
            (1.0, Some(start), Some(end), add_noise)
        } else {
            (node.float("denoise")?, None, None, true)
        };
        Ok(Sampling {
            steps,
            cfg,
            solver,
            spacing,
            seed,
            denoise,
            start_at_step,
            end_at_step,
            add_noise,
        })
    }

    /// Walk one conditioning chain from the sampler back to its text encoders.
    ///
    /// `slot` is the output slot the consumer read, which is what distinguishes
    /// the positive from the negative output of a dual-branch node.
    fn walk_conditioning(
        &mut self,
        node: &'a Node,
        slot: usize,
    ) -> Result<ConditioningChain, ComfyUiConfigError> {
        self.take(node);
        match node.class.as_str() {
            "CLIPTextEncode" => {
                let (clip, _) = self.graph.follow(node, "clip")?;
                self.consume_clip(clip)?;
                Ok(ConditioningChain {
                    prompt: node
                        .literal("text")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned),
                    ..ConditioningChain::default()
                })
            }
            "CLIPTextEncodeSDXL" => {
                let (clip, _) = self.graph.follow(node, "clip")?;
                self.consume_clip(clip)?;
                Ok(ConditioningChain {
                    prompt: node
                        .literal("text_g")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned),
                    prompt_2: node
                        .literal("text_l")
                        .and_then(|value| value.as_str())
                        .map(str::to_owned),
                    sdxl: true,
                    ..ConditioningChain::default()
                })
            }
            "ControlNetApply" | "ControlNetApplyAdvanced" => self.walk_controlnet(node, slot),
            "InpaintModelConditioning" => {
                let (positive, positive_slot) = self.graph.follow(node, "positive")?;
                let mut chain = self.walk_conditioning(positive, positive_slot)?;
                let (pixels, _) = self.graph.follow(node, "pixels")?;
                let (mask, _) = self.graph.follow(node, "mask")?;
                let (vae, _) = self.graph.follow(node, "vae")?;
                self.consume_loader(
                    vae,
                    &["VAELoader", "CheckpointLoaderSimple", "CheckpointLoader"],
                )?;
                chain.inpaint = Some(InpaintConditioning {
                    image: self.image_name(pixels)?,
                    mask: self.image_name(mask)?,
                });
                Ok(chain)
            }
            "ConditioningZeroOut" => Err(node.unsupported(
                "a zeroed conditioning branch is a distinct conditioning value, not the \
                 encoding of an empty prompt",
                "encode the unconditional branch with CLIPTextEncode and an empty prompt, \
                 which the canonical workflow represents as a second text-encoder invocation",
            )),
            "ConditioningCombine" | "ConditioningAverage" | "ConditioningConcat" => Err(node
                .unsupported(
                    "multiple conditionings are merged before the sampler",
                    "the canonical denoiser ABI takes one conditioning tensor per branch; \
                     merge the prompts into a single CLIPTextEncode instead",
                )),
            "ConditioningSetArea"
            | "ConditioningSetAreaPercentage"
            | "ConditioningSetMask"
            | "ConditioningSetTimestepRange" => Err(node.unsupported(
                "region- or step-scoped conditioning changes what the denoiser sees per step \
                 or per pixel",
                "remove the scoping node; the canonical workflow has no contract that carries \
                 per-region or per-step conditioning windows",
            )),
            _ => Err(
                known_refusal(node).unwrap_or_else(|| ComfyUiConfigError::UnknownNode {
                    node: node.id.clone(),
                    class: node.class.clone(),
                    remedy:
                        "this node is on the conditioning path that reaches the sampler, so it \
                         changes the produced image and cannot be ignored"
                            .to_owned(),
                }),
            ),
        }
    }

    fn walk_controlnet(
        &mut self,
        node: &'a Node,
        slot: usize,
    ) -> Result<ConditioningChain, ComfyUiConfigError> {
        let advanced = node.class == "ControlNetApplyAdvanced";
        // The advanced node carries both branches: slot 0 is the positive
        // conditioning and slot 1 the negative one. Reading the slot is what
        // keeps the negative branch from inheriting the positive prompt.
        let (requested, other) = if advanced && slot == 1 {
            ("negative", Some("positive"))
        } else if advanced {
            ("positive", Some("negative"))
        } else {
            ("conditioning", None)
        };
        let (upstream, upstream_slot) = self.graph.follow(node, requested)?;
        let chain = self.walk_conditioning(upstream, upstream_slot)?;
        if let Some(other) = other {
            // Consume the sibling branch too. It is normally walked from the
            // sampler as well, but a workflow may route only one output here.
            let (sibling, sibling_slot) = self.graph.follow(node, other)?;
            self.walk_conditioning(sibling, sibling_slot)?;
        }
        let (loader, _) = self.graph.follow(node, "control_net")?;
        self.consume_loader(loader, &["ControlNetLoader", "DiffControlNetLoader"])?;
        let name = loader.string("control_net_name")?;
        let (hint, _) = self.graph.follow(node, "image")?;
        let image = self.image_name(hint)?;
        let strength = node.float("strength")?;
        let (start_percent, end_percent) = if advanced {
            (node.float("start_percent")?, node.float("end_percent")?)
        } else {
            (0.0, 1.0)
        };
        if self.controlnet_ids.insert(node.id.clone()) {
            self.controlnets.push(ControlNet {
                name,
                strength,
                start_percent,
                end_percent,
                applies_to_negative: advanced,
                image,
            });
        }
        Ok(chain)
    }

    fn consume_clip(&mut self, node: &'a Node) -> Result<(), ComfyUiConfigError> {
        match node.class.as_str() {
            "CheckpointLoaderSimple" | "CheckpointLoader" | "CLIPLoader" | "DualCLIPLoader" => {
                self.take(node);
                Ok(())
            }
            "LoraLoader" => {
                // A LoRA that also patches CLIP is recorded by the model walk;
                // consuming it here keeps the text branch from failing closed
                // while leaving the adapter facts to a single owner.
                self.take(node);
                let (upstream, _) = self.graph.follow(node, "clip")?;
                self.consume_clip(upstream)
            }
            "CLIPSetLastLayer" => {
                let stop = node.integer("stop_at_clip_layer")?;
                if stop != -1 {
                    return Err(node.unsupported(
                        format!(
                            "the text encoder is truncated at CLIP layer {stop}, which changes \
                             the conditioning the denoiser receives"
                        ),
                        "export the text-encoder ONNX component already truncated to that \
                         layer, then remove CLIPSetLastLayer from the workflow",
                    ));
                }
                self.take(node);
                let (upstream, _) = self.graph.follow(node, "clip")?;
                self.consume_clip(upstream)
            }
            _ => Err(
                known_refusal(node).unwrap_or_else(|| ComfyUiConfigError::UnknownNode {
                    node: node.id.clone(),
                    class: node.class.clone(),
                    remedy: "this node feeds the text encoder that produces conditioning, so it \
                         changes the produced image"
                        .to_owned(),
                }),
            ),
        }
    }

    /// Walk the latent chain, returning the latent channel count and source.
    fn walk_latent(
        &mut self,
        node: &'a Node,
        denoise: f64,
        positive: &ConditioningChain,
        negative: &ConditioningChain,
    ) -> Result<(u32, LatentSource), ComfyUiConfigError> {
        self.take(node);
        match node.class.as_str() {
            "EmptyLatentImage" | "EmptySD3LatentImage" => {
                let channels = if node.class == "EmptySD3LatentImage" {
                    16
                } else {
                    4
                };
                Ok((
                    channels,
                    LatentSource::Noise {
                        width: dimension(node, "width")?,
                        height: dimension(node, "height")?,
                        batch_size: node
                            .optional_integer("batch_size")
                            .and_then(|value| u32::try_from(value).ok())
                            .unwrap_or(1)
                            .max(1),
                    },
                ))
            }
            "VAEEncode" => {
                let (pixels, _) = self.graph.follow(node, "pixels")?;
                let (vae, _) = self.graph.follow(node, "vae")?;
                self.consume_loader(
                    vae,
                    &["VAELoader", "CheckpointLoaderSimple", "CheckpointLoader"],
                )?;
                Ok((
                    4,
                    LatentSource::Image {
                        strength: denoise,
                        image: self.image_name(pixels)?,
                    },
                ))
            }
            "VAEEncodeForInpaint" => {
                let (pixels, _) = self.graph.follow(node, "pixels")?;
                let (mask, _) = self.graph.follow(node, "mask")?;
                let (vae, _) = self.graph.follow(node, "vae")?;
                self.consume_loader(
                    vae,
                    &["VAELoader", "CheckpointLoaderSimple", "CheckpointLoader"],
                )?;
                let grow = node
                    .optional_integer("grow_mask_by")
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0);
                Ok((
                    4,
                    LatentSource::Inpaint {
                        strength: denoise,
                        image: self.image_name(pixels)?,
                        mask: self.image_name(mask)?,
                        grow_mask_by: grow,
                    },
                ))
            }
            "SetLatentNoiseMask" => {
                let (mask, _) = self.graph.follow(node, "mask")?;
                let mask_name = self.image_name(mask)?;
                let (samples, _) = self.graph.follow(node, "samples")?;
                let (channels, inner) = self.walk_latent(samples, denoise, positive, negative)?;
                let (strength, image) = match inner {
                    LatentSource::Image { strength, image } => (strength, image),
                    LatentSource::Noise { .. } => (denoise, None),
                    LatentSource::Inpaint {
                        strength, image, ..
                    } => (strength, image),
                };
                Ok((
                    channels,
                    LatentSource::Inpaint {
                        strength,
                        image,
                        mask: mask_name,
                        grow_mask_by: 0,
                    },
                ))
            }
            "InpaintModelConditioning" => {
                // The conditioning walk already consumed this node's prompt and
                // image ports; reuse what it recovered so the mask is recorded
                // exactly once.
                let inpaint = positive
                    .inpaint
                    .clone()
                    .or_else(|| negative.inpaint.clone())
                    .ok_or_else(|| {
                        node.unsupported(
                            "the inpaint conditioning node is not reachable from either \
                             conditioning branch",
                            "connect InpaintModelConditioning's positive output to the \
                             sampler's positive input",
                        )
                    })?;
                Ok((
                    4,
                    LatentSource::Inpaint {
                        strength: denoise,
                        image: inpaint.image,
                        mask: inpaint.mask,
                        grow_mask_by: 0,
                    },
                ))
            }
            "LatentUpscale" | "LatentUpscaleBy" | "LatentComposite" | "LatentBlend"
            | "RepeatLatentBatch" | "LatentFromBatch" => Err(node.unsupported(
                "the initial latent is transformed by a latent-space operation",
                "produce the starting latent with EmptyLatentImage, VAEEncode, or \
                 VAEEncodeForInpaint; the canonical workflow has no contract for latent \
                 resampling or compositing",
            )),
            _ => Err(
                known_refusal(node).unwrap_or_else(|| ComfyUiConfigError::UnknownNode {
                    node: node.id.clone(),
                    class: node.class.clone(),
                    remedy: "this node produces the sampler's starting latent, so it decides what \
                         the workflow denoises"
                        .to_owned(),
                }),
            ),
        }
    }

    fn walk_model(&mut self, node: &'a Node) -> Result<ModelChain, ComfyUiConfigError> {
        self.take(node);
        match node.class.as_str() {
            "CheckpointLoaderSimple" | "CheckpointLoader" => Ok(ModelChain {
                checkpoint: Some(node.string("ckpt_name")?),
                ..ModelChain::default()
            }),
            "UNETLoader" | "DiffusersLoader" => Ok(ModelChain {
                checkpoint: Some(
                    node.string("unet_name")
                        .or_else(|_| node.string("model_path"))?,
                ),
                ..ModelChain::default()
            }),
            "LoraLoader" | "LoraLoaderModelOnly" => {
                let (upstream, _) = self.graph.follow(node, "model")?;
                let mut chain = self.walk_model(upstream)?;
                chain.loras.push(Lora {
                    name: node.string("lora_name")?,
                    model_strength: node.float("strength_model")?,
                    clip_strength: node.optional_float("strength_clip"),
                });
                Ok(chain)
            }
            "ModelSamplingDiscrete" => {
                let (upstream, _) = self.graph.follow(node, "model")?;
                let mut chain = self.walk_model(upstream)?;
                let sampling = node.string("sampling")?;
                chain.prediction = Some(match sampling.as_str() {
                    "eps" => Prediction::Epsilon,
                    "v_prediction" => Prediction::VPrediction,
                    other => {
                        return Err(node.unsupported(
                            format!("model sampling {other:?} redefines the denoiser output"),
                            "use 'eps' or 'v_prediction'; other modes change the solver's \
                             inversion, which the canonical solver contract does not carry",
                        ));
                    }
                });
                if node
                    .optional_integer("zsnr")
                    .is_some_and(|value| value != 0)
                    || node
                        .literal("zsnr")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                {
                    return Err(node.unsupported(
                        "zero-terminal-SNR rescaling changes the noise schedule",
                        "bake the rescaled schedule into the exported schedule component and \
                         disable zsnr in the workflow",
                    ));
                }
                Ok(chain)
            }
            "ModelSamplingSD3" | "ModelSamplingAuraFlow" | "ModelSamplingFlux" => {
                let (upstream, _) = self.graph.follow(node, "model")?;
                let mut chain = self.walk_model(upstream)?;
                chain.prediction = Some(Prediction::FlowVelocity);
                Ok(chain)
            }
            "FreeU"
            | "FreeU_V2"
            | "PatchModelAddDownscale"
            | "RescaleCFG"
            | "SelfAttentionGuidance" => Err(node.unsupported(
                "this node patches the denoiser's forward function",
                "export the patched denoiser as its own ONNX component; a converted \
                     package invokes the graph it is given and cannot re-patch it at runtime",
            )),
            _ => Err(
                known_refusal(node).unwrap_or_else(|| ComfyUiConfigError::UnknownNode {
                    node: node.id.clone(),
                    class: node.class.clone(),
                    remedy:
                        "this node produces the model the sampler denoises with, so it decides \
                         the denoiser's behaviour"
                            .to_owned(),
                }),
            ),
        }
    }

    /// Read the file name an image path names, failing closed on any transform.
    fn image_name(&mut self, node: &'a Node) -> Result<Option<String>, ComfyUiConfigError> {
        self.take(node);
        match node.class.as_str() {
            "LoadImage" | "LoadImageMask" => Ok(node
                .literal("image")
                .and_then(|value| value.as_str())
                .map(str::to_owned)),
            "ImageToMask" | "MaskToImage" | "InvertMask" => {
                let port = if node.class == "ImageToMask" {
                    "image"
                } else {
                    "mask"
                };
                let (upstream, _) = self.graph.follow(node, port)?;
                self.image_name(upstream)
            }
            _ => Err(node.unsupported(
                "an image reaching the model is produced by a transform the importer cannot \
                 reproduce",
                "run the preprocessor offline and feed its result through LoadImage; a \
                 converted package declares no image transform program for diffusion hints, \
                 so importing the graph as-is would silently change the pixels the model sees",
            )),
        }
    }
}

#[derive(Default)]
struct ModelChain {
    checkpoint: Option<String>,
    loras: Vec<Lora>,
    prediction: Option<Prediction>,
}

struct Sampling {
    steps: i64,
    cfg: f64,
    solver: Solver,
    spacing: Spacing,
    seed: i64,
    denoise: f64,
    start_at_step: Option<u32>,
    end_at_step: Option<u32>,
    add_noise: bool,
}

fn dimension(node: &Node, port: &str) -> Result<u32, ComfyUiConfigError> {
    let value = node.integer(port)?;
    u32::try_from(value).map_err(|_| {
        node.unsupported(
            format!("'{port}' is {value}, which is not a pixel dimension"),
            format!("set '{port}' to a positive pixel size"),
        )
    })
}

fn solver_of(node: &Node, name: &str) -> Result<Solver, ComfyUiConfigError> {
    Ok(match name {
        "euler" => Solver::Euler,
        "euler_ancestral" => Solver::EulerAncestral,
        "ddim" => Solver::Ddim,
        "dpmpp_2m" | "dpm_2m" => Solver::DpmSolverPlusPlus2M,
        other => {
            return Err(node.unsupported(
                format!("sampler {other:?} has no canonical solver contract"),
                "select one of: ddim, dpmpp_2m, euler, euler_ancestral. A solver the runtime \
                 cannot reproduce would change every step of the trajectory, so it is refused \
                 rather than approximated",
            ));
        }
    })
}

fn spacing_of(node: &Node, name: &str) -> Result<Spacing, ComfyUiConfigError> {
    Ok(match name {
        "normal" | "simple" => Spacing::Linear,
        "ddim_uniform" => Spacing::DdimUniform,
        "karras" => Spacing::Karras,
        "exponential" => Spacing::Exponential,
        "beta" => Spacing::Beta,
        other => {
            return Err(node.unsupported(
                format!("sigma spacing {other:?} has no canonical schedule contract"),
                "select one of: normal, simple, ddim_uniform, karras, exponential, beta. The \
                 spacing decides every sigma, so falling back to a linear schedule would \
                 silently produce a different image",
            ));
        }
    })
}
