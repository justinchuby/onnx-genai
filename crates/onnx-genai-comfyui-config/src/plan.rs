//! The Comfy-free import plan.
//!
//! Recognition produces a [`WorkflowPlan`]: a complete, normalized statement of
//! what the imported workflow *does*, with no ComfyUI vocabulary left in it.
//! Lowering consumes only this. That split is what keeps Comfy-specific
//! execution dispatch out of the emitted metadata: by the time the lowerer
//! runs, `KSampler`, `CLIPTextEncode`, and slot indices no longer exist.

/// The sampling algorithm the workflow selected.
///
/// Each variant names a solver *shape* the canonical `onnx-genai.solver-step`
/// contract can carry, not a ComfyUI spelling. `history` is what decides
/// whether the workflow needs a loop-carried solver history cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solver {
    /// First-order explicit step. No history.
    Euler,
    /// Ancestral Euler: a fresh noise draw is added after each step.
    EulerAncestral,
    /// DDIM deterministic step. No history.
    Ddim,
    /// Second-order multistep DPM-Solver++. Carries the previous estimate.
    DpmSolverPlusPlus2M,
}

impl Solver {
    /// Stable parameter spelling recorded on the emitted solver contract.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::EulerAncestral => "euler_ancestral",
            Self::Ddim => "ddim",
            Self::DpmSolverPlusPlus2M => "dpmpp_2m",
        }
    }

    /// Whether the solver carries state between steps.
    pub fn needs_history(self) -> bool {
        matches!(self, Self::DpmSolverPlusPlus2M)
    }

    /// Whether the solver draws fresh noise inside the loop.
    pub fn needs_step_noise(self) -> bool {
        matches!(self, Self::EulerAncestral)
    }
}

/// Sigma spacing of the noise schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spacing {
    /// Evenly spaced training timesteps (ComfyUI `normal` / `simple`).
    Linear,
    /// Uniform DDIM stride.
    DdimUniform,
    /// Karras rho-spaced sigmas.
    Karras,
    /// Exponentially spaced sigmas.
    Exponential,
    /// Beta-distributed spacing.
    Beta,
}

impl Spacing {
    /// Stable parameter spelling recorded on the emitted schedule contract.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::DdimUniform => "ddim_uniform",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::Beta => "beta",
        }
    }
}

/// What the denoiser predicts, which the solver needs in order to invert it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prediction {
    /// The added noise (Stable Diffusion 1.x / XL).
    Epsilon,
    /// Velocity parameterization.
    VPrediction,
    /// A flow-matching velocity field (SD3, Flux, Qwen-Image).
    FlowVelocity,
}

impl Prediction {
    /// Stable parameter spelling recorded on the emitted contracts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Epsilon => "epsilon",
            Self::VPrediction => "v_prediction",
            Self::FlowVelocity => "flow_velocity",
        }
    }
}

/// Which text-conditioning shape the workflow uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conditioning {
    /// One text encoder producing one hidden-state tensor (SD 1.x, SD 2.x).
    Single,
    /// Two text encoders plus the SDXL micro-conditioning `time_ids` vector.
    ///
    /// The two encoders are distinct components, so the plan carries both and
    /// lowering emits two invocations rather than one flagged invocation.
    SdxlDual,
}

/// Where the loop's initial latent comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum LatentSource {
    /// Pure noise at the requested resolution (text-to-image).
    Noise {
        /// Pixel width the workflow asked for.
        width: u32,
        /// Pixel height the workflow asked for.
        height: u32,
        /// Latent rows produced per request.
        batch_size: u32,
    },
    /// A VAE-encoded source image, renoised to the start step (image-to-image).
    Image {
        /// Denoise strength the sampler declared.
        strength: f64,
        /// Source image the workflow named, when it named a file.
        image: Option<String>,
    },
    /// A VAE-encoded source image plus a mask that pins the kept region.
    Inpaint {
        /// Denoise strength the sampler declared.
        strength: f64,
        /// Source image the workflow named, when it named a file.
        image: Option<String>,
        /// Mask image the workflow named, when it named a file.
        mask: Option<String>,
        /// Whether the workflow asked for the masked pixels to be erased before
        /// encoding, which changes what the VAE sees and therefore the result.
        grow_mask_by: u32,
    },
}

/// One ControlNet application recovered from the conditioning chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlNet {
    /// ControlNet checkpoint the loader named.
    pub name: String,
    /// Conditioning strength the apply node declared. This is a *runtime* input
    /// of the emitted workflow, never a constant folded away at import.
    pub strength: f64,
    /// Fraction of the schedule at which the ControlNet starts applying.
    pub start_percent: f64,
    /// Fraction of the schedule at which the ControlNet stops applying.
    pub end_percent: f64,
    /// Whether the hint also conditions the unconditional (negative) pass.
    ///
    /// `ControlNetApply` patches one branch; `ControlNetApplyAdvanced` patches
    /// both, which costs a second ControlNet invocation per step.
    pub applies_to_negative: bool,
    /// Hint image the workflow named, when it named a file.
    pub image: Option<String>,
}

/// One LoRA selected along the model chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Lora {
    /// LoRA file the loader named. This becomes the adapter artifact identity.
    pub name: String,
    /// Strength applied to the denoiser weights.
    pub model_strength: f64,
    /// Strength applied to the text-encoder weights, when the node applies one.
    pub clip_strength: Option<f64>,
}

/// Classifier-free guidance settings.
#[derive(Debug, Clone, PartialEq)]
pub struct Guidance {
    /// Guidance scale. `1.0` means the workflow disabled CFG.
    pub scale: f64,
}

/// A complete, normalized statement of an imported workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowPlan {
    /// Number of solver steps the schedule spans.
    pub steps: u32,
    /// First executed step index, non-zero for a partial (img2img) denoise.
    pub start_step: u32,
    /// One past the last executed step index.
    ///
    /// Below `steps` when the workflow deliberately returns a partially
    /// denoised latent (`KSamplerAdvanced.return_with_leftover_noise`).
    pub end_step: u32,
    /// Whether the workflow draws noise into the initial latent.
    ///
    /// `KSamplerAdvanced` can disable this to continue an existing trajectory,
    /// which is a different computation, not a tuning knob.
    pub add_initial_noise: bool,
    /// Solver algorithm.
    pub solver: Solver,
    /// Sigma spacing.
    pub spacing: Spacing,
    /// Denoiser output parameterization.
    pub prediction: Prediction,
    /// Guidance, absent when the workflow set `cfg` to 1.
    pub guidance: Option<Guidance>,
    /// Text-conditioning shape.
    pub conditioning: Conditioning,
    /// Positive prompt text, when the workflow spelled it as a literal.
    pub prompt: Option<String>,
    /// Negative prompt text, when the workflow spelled it as a literal.
    pub negative_prompt: Option<String>,
    /// Second positive prompt for the SDXL dual encoder.
    pub prompt_2: Option<String>,
    /// Second negative prompt for the SDXL dual encoder.
    pub negative_prompt_2: Option<String>,
    /// Initial-latent source.
    pub latent: LatentSource,
    /// Seed the workflow declared.
    pub seed: i64,
    /// Base checkpoint the model chain resolved to.
    pub checkpoint: Option<String>,
    /// LoRAs in application order, base checkpoint first.
    pub loras: Vec<Lora>,
    /// ControlNets recovered from the conditioning chain.
    pub controlnets: Vec<ControlNet>,
    /// Latent channel count implied by the recognized latent node.
    pub latent_channels: u32,
    /// Classes on the output path that the recognizer read and consumed.
    pub recognized_nodes: Vec<String>,
    /// Classes present in the document but provably off the output path.
    pub ignored_nodes: Vec<String>,
}

impl WorkflowPlan {
    /// Whether classifier-free guidance is active.
    pub fn uses_guidance(&self) -> bool {
        self.guidance.is_some()
    }

    /// Number of solver iterations the emitted loop runs.
    pub fn iterations(&self) -> u32 {
        self.end_step.saturating_sub(self.start_step)
    }

    /// Whether the workflow denoises an existing latent rather than pure noise.
    pub fn is_image_to_image(&self) -> bool {
        matches!(
            self.latent,
            LatentSource::Image { .. } | LatentSource::Inpaint { .. }
        )
    }

    /// Whether the workflow pins a kept region with a mask.
    pub fn is_inpainting(&self) -> bool {
        matches!(self.latent, LatentSource::Inpaint { .. })
    }

    /// Denoise strength, `1.0` for a full text-to-image run.
    pub fn strength(&self) -> f64 {
        match &self.latent {
            LatentSource::Noise { .. } => 1.0,
            LatentSource::Image { strength, .. } | LatentSource::Inpaint { strength, .. } => {
                *strength
            }
        }
    }
}

/// Convert a denoise strength to the first executed step.
///
/// This is `steps - round_ties_even(steps * strength)`, matching diffusers
/// `get_timesteps` and ComfyUI's own partial-denoise slicing. A strength of
/// zero therefore executes no steps at all, which the caller rejects rather
/// than emitting an empty loop.
pub fn strength_to_start_step(strength: f64, steps: u32) -> u32 {
    if steps == 0 {
        return 0;
    }
    let strength = strength.clamp(0.0, 1.0);
    let rounded = round_ties_even(f64::from(steps) * strength) as i64;
    i64::from(steps)
        .saturating_sub(rounded)
        .clamp(0, i64::from(steps)) as u32
}

/// Round half to even, matching numpy's `round`.
fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    if (value - floor - 0.5).abs() < f64::EPSILON {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}
